// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
//! Unless required by applicable law or agreed to in writing,
//   software distributed under the License is distributed on an
//   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied.  See the License for the
//   specific language governing permissions and limitations
//   under the License.

//! Partitioned TopK Sort: Sort operator that applies TopK limit per partition
//!
//! This operator is used to optimize queries like:
//! ```sql
//! SELECT * FROM (
//!   SELECT *, ROW_NUMBER() OVER (PARTITION BY col1 ORDER BY col2) as rn
//! ) WHERE rn <= 10
//! ```
//!
//! Instead of sorting all data, it only keeps the top K rows for each partition,
//! significantly reducing memory usage and improving performance.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use crate::execution_plan::{Boundedness, CardinalityEffect, EmissionType};
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, SendableRecordBatchStream, Statistics,
};
use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::{concat_batches, lexsort_to_indices, take};
use arrow::datatypes::SchemaRef;
use arrow::row::{RowConverter, SortField};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::LexOrdering;

use futures::{StreamExt, TryStreamExt};

/// Partitioned TopK Sort execution plan.
///
/// This operator sorts data and applies a TopK limit per logical partition,
/// where partitions are defined by a prefix of the sort key.
///
/// For example, with sort key `(a, b)` and partition prefix length 1,
/// the data is partitioned by column `a`, and within each partition,
/// rows are sorted by `(a, b)` with only the top K rows retained.
#[derive(Debug, Clone)]
pub struct PartitionedTopKSortExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// Sort expressions (full sort key)
    expr: LexOrdering,
    /// Number of sort key columns that define the partition
    /// For PARTITION BY (col1) ORDER BY (col2), this would be 1
    partition_prefix_len: usize,
    /// Maximum number of rows to keep per partition
    fetch: usize,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Cache holding plan properties
    cache: PlanProperties,
}

impl PartitionedTopKSortExec {
    /// Create a new PartitionedTopKSortExec
    ///
    /// # Arguments
    ///
    /// * `input` - The input execution plan
    /// * `expr` - Complete sort expressions (partition columns + order columns)
    /// * `partition_prefix_len` - Number of leading sort expressions that define partitions
    /// * `fetch` - Maximum rows to keep per partition
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        expr: LexOrdering,
        partition_prefix_len: usize,
        fetch: usize,
    ) -> Result<Self> {
        if partition_prefix_len == 0 {
            return Err(DataFusionError::Plan(
                "partition_prefix_len must be greater than 0 for PartitionedTopKSortExec"
                    .to_string(),
            ));
        }
        if partition_prefix_len > expr.len() {
            return Err(DataFusionError::Plan(format!(
                "partition_prefix_len ({}) cannot exceed sort expression length ({})",
                partition_prefix_len,
                expr.len()
            )));
        }
        if fetch == 0 {
            return Err(DataFusionError::Plan(
                "fetch must be greater than 0".to_string(),
            ));
        }

        let cache = Self::compute_properties(&input, &expr)?;

        Ok(Self {
            input,
            expr,
            partition_prefix_len,
            fetch,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    /// Get the input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Get the sort expressions
    pub fn expr(&self) -> &LexOrdering {
        &self.expr
    }

    /// Get the partition prefix length
    pub fn partition_prefix_len(&self) -> usize {
        self.partition_prefix_len
    }

    /// Get the fetch limit
    pub fn fetch(&self) -> usize {
        self.fetch
    }

    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        expr: &LexOrdering,
    ) -> Result<PlanProperties> {
        let mut eq_properties = input.equivalence_properties().clone();
        // The output is sorted according to the sort expressions
        eq_properties.reorder(expr.clone())?;

        // Preserve the input partitioning since we can process each partition independently
        let output_partitioning = input.output_partitioning().clone();

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            EmissionType::Final,
            Boundedness::Bounded,
        ))
    }
}

impl DisplayAs for PartitionedTopKSortExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let partition_exprs = &self.expr[..self.partition_prefix_len];
                let order_exprs = &self.expr[self.partition_prefix_len..];
                write!(
                    f,
                    "PartitionedTopKSortExec: partition_by=[{:?}], order_by=[{:?}], fetch={}",
                    partition_exprs, order_exprs, self.fetch
                )
            }
            DisplayFormatType::TreeRender => Ok(()),
        }
    }
}

impl ExecutionPlan for PartitionedTopKSortExec {
    fn name(&self) -> &'static str {
        "PartitionedTopKSortExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(PartitionedTopKSortExec::try_new(
            Arc::clone(&children[0]),
            self.expr.clone(),
            self.partition_prefix_len,
            self.fetch,
        )?))
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let schema = input.schema();
        let sort_exprs = self.expr.clone();
        let partition_prefix_len = self.partition_prefix_len;
        let fetch_limit = self.fetch;
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            futures::stream::once(async move {
                partitioned_topk_sort(
                    input,
                    schema,
                    sort_exprs,
                    partition_prefix_len,
                    fetch_limit,
                    baseline_metrics,
                )
                .await
            })
            .try_flatten(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        // The output statistics depend on the number of distinct partition values
        // which we don't know statically. Return unknown for now.
        Ok(Statistics::new_unknown(&self.schema()))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }
}

/// Core algorithm for partitioned topk sort
///
/// Groups input batches by partition prefix, sorts each group,
/// and keeps only top K rows per partition.
async fn partitioned_topk_sort(
    mut input: SendableRecordBatchStream,
    schema: SchemaRef,
    sort_exprs: LexOrdering,
    partition_prefix_len: usize,
    fetch: usize,
    _baseline_metrics: BaselineMetrics,
) -> Result<SendableRecordBatchStream> {
    // Collect all input batches
    let mut batches = Vec::new();
    while let Some(batch) = input.next().await {
        let batch = batch?;
        if batch.num_rows() > 0 {
            batches.push(batch);
        }
    }

    if batches.is_empty() {
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            futures::stream::iter(vec![]),
        )));
    }

    // Concatenate all batches
    let combined_batch = concat_batches(&schema, &batches)?;
    if combined_batch.num_rows() == 0 {
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            futures::stream::iter(vec![]),
        )));
    }

    // Extract partition columns (prefix of sort key)
    let partition_sort_exprs = &sort_exprs[..partition_prefix_len];

    // Build a row converter for partition keys to enable grouping
    let partition_fields: Vec<SortField> = partition_sort_exprs
        .iter()
        .map(|e| SortField::new(e.expr.data_type(&schema).unwrap()))
        .collect();
    let partition_row_converter = RowConverter::new(partition_fields)?;

    // Evaluate partition key columns
    let partition_arrays: Vec<ArrayRef> = partition_sort_exprs
        .iter()
        .map(|e| {
            e.expr
                .evaluate(&combined_batch)
                .and_then(|v| v.into_array(combined_batch.num_rows()))
        })
        .collect::<Result<Vec<_>>>()?;

    // Convert partition keys to rows for grouping
    let partition_rows = partition_row_converter.convert_columns(&partition_arrays)?;

    // Group row indices by partition
    let mut partition_groups: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
    for (idx, row) in partition_rows.iter().enumerate() {
        partition_groups
            .entry(row.as_ref().to_vec())
            .or_default()
            .push(idx);
    }

    // Sort partition keys to ensure deterministic output order
    let mut partition_keys: Vec<Vec<u8>> = partition_groups.keys().cloned().collect();
    partition_keys.sort();

    // For each partition (in sorted order), sort and take top K
    let mut result_indices = Vec::new();

    for partition_key in partition_keys {
        let indices = partition_groups.get(&partition_key).unwrap();
        // Create a batch with only rows from this partition
        let partition_batch = take_record_batch(&combined_batch, indices)?;

        // Sort the partition batch
        let sort_columns: Vec<_> = sort_exprs
            .iter()
            .map(|expr| expr.evaluate_to_sort_column(&partition_batch))
            .collect::<Result<Vec<_>>>()?;

        let sorted_indices = lexsort_to_indices(
            &sort_columns,
            Some(fetch.min(partition_batch.num_rows())),
        )?;

        // Map sorted indices back to original combined_batch indices
        for &sorted_idx in sorted_indices.values().iter() {
            result_indices.push(indices[sorted_idx as usize]);
        }
    }

    // Take the selected rows from the combined batch
    let result_batch = take_record_batch(&combined_batch, &result_indices)?;

    // Sort the final result by the full sort key to ensure output ordering
    let sort_columns: Vec<_> = sort_exprs
        .iter()
        .map(|expr| expr.evaluate_to_sort_column(&result_batch))
        .collect::<Result<Vec<_>>>()?;

    let final_sorted_indices = lexsort_to_indices(&sort_columns, None)?;
    let final_result_batch = take_record_batch(
        &result_batch,
        &final_sorted_indices
            .values()
            .iter()
            .map(|&i| i as usize)
            .collect::<Vec<_>>(),
    )?;

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        Arc::clone(&schema),
        futures::stream::iter(vec![Ok(final_result_batch)]),
    )))
}

/// Helper function to take rows from a RecordBatch by indices
fn take_record_batch(batch: &RecordBatch, indices: &[usize]) -> Result<RecordBatch> {
    let indices_array =
        arrow::array::UInt32Array::from_iter_values(indices.iter().map(|&i| i as u32));

    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|col| take(col.as_ref(), &indices_array, None))
        .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>()
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

    RecordBatch::try_new(batch.schema(), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collect;
    use crate::test::TestMemoryExec;
    use arrow::array::Int32Array;
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

    #[tokio::test]
    async fn test_partitioned_topk_sort_basic() -> Result<()> {
        // Create test data:
        // partition | value
        // ---------+-------
        //    1     |   10
        //    1     |   20
        //    1     |   30
        //    2     |   15
        //    2     |   25
        //    2     |   35
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let partition_col = Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 2]));
        let value_col = Arc::new(Int32Array::from(vec![30, 10, 20, 35, 15, 25]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![partition_col, value_col])?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        // Sort by (partition, value) and keep top 2 per partition
        let partition_expr = col("partition", &schema)?;
        let value_expr = col("value", &schema)?;
        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(partition_expr, SortOptions::default()),
            PhysicalSortExpr::new(value_expr, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        // Expected output: top 2 rows per partition (partition 1: 10, 20; partition 2: 15, 25)
        assert_eq!(result.len(), 1);
        let result_batch = &result[0];
        assert_eq!(result_batch.num_rows(), 4);

        // Verify we got the right rows
        let partition_result = result_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let value_result = result_batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // Should have 2 rows from partition 1 and 2 from partition 2
        let partition_1_count = partition_result
            .values()
            .iter()
            .filter(|&&x| x == 1)
            .count();
        let partition_2_count = partition_result
            .values()
            .iter()
            .filter(|&&x| x == 2)
            .count();
        assert_eq!(partition_1_count, 2);
        assert_eq!(partition_2_count, 2);

        // Collect values by partition
        let mut p1_values = Vec::new();
        let mut p2_values = Vec::new();
        for i in 0..result_batch.num_rows() {
            if partition_result.value(i) == 1 {
                p1_values.push(value_result.value(i));
            } else {
                p2_values.push(value_result.value(i));
            }
        }
        p1_values.sort();
        p2_values.sort();

        // Should get smallest 2 values from each partition
        assert_eq!(p1_values, vec![10, 20]);
        assert_eq!(p2_values, vec![15, 25]);

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_single_partition() -> Result<()> {
        // Test with only one partition value
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let partition_col = Arc::new(Int32Array::from(vec![1, 1, 1, 1]));
        let value_col = Arc::new(Int32Array::from(vec![40, 10, 30, 20]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![partition_col, value_col])?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let partition_expr = col("partition", &schema)?;
        let value_expr = col("value", &schema)?;
        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(partition_expr, SortOptions::default()),
            PhysicalSortExpr::new(value_expr, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        assert_eq!(result.len(), 1);
        let result_batch = &result[0];
        assert_eq!(result_batch.num_rows(), 2); // Only top 2

        let value_result = result_batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // Should get the two smallest values: 10 and 20
        let mut values = vec![value_result.value(0), value_result.value(1)];
        values.sort();
        assert_eq!(values, vec![10, 20]);

        Ok(())
    }
}
