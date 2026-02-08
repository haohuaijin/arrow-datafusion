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
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

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
//!
//! # Structure
//!
//! The implementation follows the same `insert_batch` / `emit` pattern as
//! [`TopK`](crate::topk::TopK):
//!
//! - [`PartitionedTopK`]: The core processor that maintains per-partition
//!   buckets with periodic compaction to bound memory usage.
//! - [`PartitionedTopKSortExec`]: The [`ExecutionPlan`] wrapper that drives
//!   the processor from an input stream.

use std::any::Any;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

use crate::execution_plan::{Boundedness, CardinalityEffect, EmissionType};
use crate::expressions::PhysicalSortExpr;
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use crate::sorts::sort::sort_batch;
use crate::spill::get_record_batch_memory_size;
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, SendableRecordBatchStream, Statistics,
};

use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow::compute::{concat_batches, take_arrays};
use arrow::datatypes::SchemaRef;
use arrow::row::{RowConverter, SortField};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_expr::LexOrdering;

use futures::{StreamExt, TryStreamExt};
use log::trace;

/// Builds [`SortField`] descriptors from physical sort expressions,
/// preserving each expression's sort options (ASC/DESC, NULLS FIRST/LAST).
fn build_sort_fields(
    ordering: &[PhysicalSortExpr],
    schema: &SchemaRef,
) -> Result<Vec<SortField>> {
    ordering
        .iter()
        .map(|e| {
            Ok(SortField::new_with_options(
                e.expr.data_type(schema)?,
                e.options,
            ))
        })
        .collect::<Result<_>>()
}

// ============================================================================
// PartitionBucket: per-partition row accumulator with compaction
// ============================================================================

/// Per-partition bucket that accumulates rows and supports compaction
/// (sort + truncate to K rows) to bound memory usage.
struct PartitionBucket {
    /// Accumulated record batches for this partition
    batches: Vec<RecordBatch>,
    /// Total number of rows across all batches
    num_rows: usize,
    /// Whether the current state is already sorted and truncated
    is_sorted: bool,
}

impl PartitionBucket {
    fn new() -> Self {
        Self {
            batches: Vec::new(),
            num_rows: 0,
            is_sorted: true,
        }
    }

    fn add_batch(&mut self, batch: RecordBatch) {
        self.num_rows += batch.num_rows();
        self.batches.push(batch);
        self.is_sorted = false;
    }

    /// Sort by ORDER BY expressions and keep only the top `k` rows.
    ///
    /// Within a partition all partition-key columns are constant, so
    /// sorting by only the ORDER BY columns is sufficient and avoids
    /// redundant comparisons on the partition prefix.
    fn compact(
        &mut self,
        schema: &SchemaRef,
        order_expr: Option<&LexOrdering>,
        k: usize,
    ) -> Result<()> {
        // Already sorted and within limit – nothing to do
        if self.is_sorted && self.num_rows <= k {
            return Ok(());
        }
        if self.batches.is_empty() || self.num_rows == 0 {
            return Ok(());
        }

        // Combine batches (skip concat for the single-batch case)
        let combined = if self.batches.len() == 1 {
            self.batches.remove(0)
        } else {
            let result = concat_batches(schema, &self.batches)?;
            self.batches.clear();
            result
        };

        let n = k.min(combined.num_rows());
        let result = match order_expr {
            // Sort by ORDER BY columns and take top n
            Some(expr) => sort_batch(&combined, expr, Some(n))?,
            // No ORDER BY within partition – take first n rows
            None => combined.slice(0, n),
        };

        self.num_rows = result.num_rows();
        self.batches = vec![result];
        self.is_sorted = true;
        Ok(())
    }

    /// Returns the estimated memory size of this bucket in bytes.
    fn size(&self) -> usize {
        size_of::<Self>()
            + self
                .batches
                .iter()
                .map(get_record_batch_memory_size)
                .sum::<usize>()
    }
}

// ============================================================================
// PartitionedTopK: the core processor (insert_batch / emit)
// ============================================================================

/// Partitioned TopK processor.
///
/// Maintains separate TopK state per partition value, following the same
/// `insert_batch` / `emit` lifecycle as [`TopK`](crate::topk::TopK).
///
/// On each [`insert_batch`](Self::insert_batch) call the incoming rows are
/// grouped by partition key and appended to per-partition
/// [`PartitionBucket`]s. When a bucket exceeds the compaction threshold
/// it is sorted and truncated to `k` rows, bounding memory to
/// `O(k × num_partitions)`.
///
/// [`emit`](Self::emit) performs a final compaction of every bucket, then
/// concatenates the results in partition-key order and breaks them into
/// `batch_size` chunks.
pub(crate) struct PartitionedTopK {
    /// Schema of the output (and the input)
    schema: SchemaRef,
    /// Runtime metrics
    metrics: BaselineMetrics,
    /// Memory reservation tracked through the memory pool
    reservation: MemoryReservation,
    /// The target number of rows for output batches
    batch_size: usize,
    /// Full sort expressions (partition prefix + order by)
    expr: LexOrdering,
    /// Number of leading sort expressions that define partitions
    partition_prefix_len: usize,
    /// Maximum rows to keep per partition
    k: usize,
    /// Row converter for partition keys (encodes sort options so that
    /// bytewise comparison matches the requested partition ordering)
    partition_converter: RowConverter,
    /// ORDER BY expressions (sort key minus partition prefix), if any.
    /// `None` when `partition_prefix_len == expr.len()`.
    order_expr: Option<LexOrdering>,
    /// Per-partition buckets keyed by the binary partition-key
    /// produced by [`partition_converter`](Self::partition_converter)
    partitions: HashMap<Vec<u8>, PartitionBucket>,
    /// Row count threshold above which a partition bucket is compacted
    compact_threshold: usize,
}

impl PartitionedTopK {
    /// Create a new [`PartitionedTopK`].
    pub fn try_new(
        partition_id: usize,
        schema: SchemaRef,
        expr: LexOrdering,
        partition_prefix_len: usize,
        k: usize,
        batch_size: usize,
        runtime: Arc<RuntimeEnv>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Self> {
        let reservation = MemoryConsumer::new(format!("PartitionedTopK[{partition_id}]"))
            .register(&runtime.memory_pool);

        let partition_sort_exprs = &expr[..partition_prefix_len];
        let partition_fields = build_sort_fields(partition_sort_exprs, &schema)?;
        let partition_converter = RowConverter::new(partition_fields)?;

        // Pre-build the ORDER BY LexOrdering (None when there are no
        // order-by columns beyond the partition prefix)
        let order_expr = LexOrdering::new(expr[partition_prefix_len..].iter().cloned());

        let compact_threshold = k.saturating_mul(2).max(k.saturating_add(128));

        Ok(Self {
            schema: Arc::clone(&schema),
            metrics: BaselineMetrics::new(metrics, partition_id),
            reservation,
            batch_size,
            expr,
            partition_prefix_len,
            k,
            partition_converter,
            order_expr,
            partitions: HashMap::new(),
            compact_threshold,
        })
    }

    /// Insert a [`RecordBatch`], distributing its rows into per-partition
    /// buckets. Buckets that exceed the compaction threshold are
    /// automatically compacted.
    pub fn insert_batch(&mut self, batch: RecordBatch) -> Result<()> {
        // Updates on drop
        let _timer = self.metrics.elapsed_compute().timer();

        if batch.num_rows() == 0 {
            return Ok(());
        }

        let partition_sort_exprs = &self.expr[..self.partition_prefix_len];

        // Evaluate partition key columns
        let partition_arrays: Vec<ArrayRef> = partition_sort_exprs
            .iter()
            .map(|e| {
                let value = e.expr.evaluate(&batch)?;
                value.into_array(batch.num_rows())
            })
            .collect::<Result<Vec<_>>>()?;

        let partition_rows = self
            .partition_converter
            .convert_columns(&partition_arrays)?;

        // Group row indices by partition key
        let mut batch_groups: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
        for (idx, row) in partition_rows.iter().enumerate() {
            batch_groups
                .entry(row.as_ref().to_vec())
                .or_default()
                .push(idx as u32);
        }

        // Distribute rows to per-partition buckets
        for (partition_key, indices) in batch_groups {
            let indices_array = UInt32Array::from(indices);
            let partition_batch = take_batch(&batch, &indices_array)?;

            let bucket = self
                .partitions
                .entry(partition_key)
                .or_insert_with(PartitionBucket::new);
            bucket.add_batch(partition_batch);

            // Compact when accumulated rows exceed threshold to bound memory
            if bucket.num_rows >= self.compact_threshold {
                bucket.compact(&self.schema, self.order_expr.as_ref(), self.k)?;
            }
        }

        // Update memory reservation
        self.reservation.try_resize(self.size())?;

        Ok(())
    }

    /// Returns the top-k results for every partition as a sorted stream
    /// of [`RecordBatch`]es, consuming the processor.
    ///
    /// Partitions are emitted in partition-key order. Within each
    /// partition, rows are sorted by the ORDER BY columns. The
    /// combined output therefore satisfies the full sort key
    /// (partition columns ++ order columns).
    pub fn emit(self) -> Result<SendableRecordBatchStream> {
        let Self {
            schema,
            metrics,
            reservation: _,
            batch_size,
            expr: _,
            partition_prefix_len: _,
            k,
            partition_converter: _,
            order_expr,
            mut partitions,
            compact_threshold: _,
        } = self;

        let _timer = metrics.elapsed_compute().timer(); // time updated on drop

        // Sort partition keys – RowConverter encodes sort options into
        // the binary representation so bytewise sort gives the correct
        // logical order.
        let mut partition_keys: Vec<Vec<u8>> = partitions.keys().cloned().collect();
        partition_keys.sort();

        // Final compact for each partition and collect results
        let mut partition_batches = Vec::with_capacity(partition_keys.len());
        for key in &partition_keys {
            if let Some(mut bucket) = partitions.remove(key) {
                bucket.compact(&schema, order_expr.as_ref(), k)?;
                partition_batches.extend(bucket.batches);
            }
        }

        if partition_batches.is_empty() {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                schema,
                futures::stream::iter(vec![]),
            )));
        }

        // Combine all partition results into a single sorted batch
        let result = if partition_batches.len() == 1 {
            partition_batches.pop().unwrap()
        } else {
            concat_batches(&schema, &partition_batches)?
        };
        drop(partition_batches);

        metrics.record_output(result.num_rows());

        // Break into batch_size chunks (following TopK pattern)
        let mut batches = vec![];
        let mut batch = result;
        loop {
            if batch.num_rows() <= batch_size {
                batches.push(Ok(batch));
                break;
            } else {
                batches.push(Ok(batch.slice(0, batch_size)));
                let remaining_length = batch.num_rows() - batch_size;
                batch = batch.slice(batch_size, remaining_length);
            }
        }

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches),
        )))
    }

    /// Return the estimated memory used by this operator, in bytes.
    fn size(&self) -> usize {
        size_of::<Self>()
            + self.partition_converter.size()
            + self.partitions.values().map(|b| b.size()).sum::<usize>()
    }
}

// ============================================================================
// Helper
// ============================================================================

/// Take rows from a [`RecordBatch`] using a [`UInt32Array`] of indices.
fn take_batch(batch: &RecordBatch, indices: &UInt32Array) -> Result<RecordBatch> {
    let columns = take_arrays(batch.columns(), indices, None)?;
    let options = RecordBatchOptions::new().with_row_count(Some(indices.len()));
    Ok(RecordBatch::try_new_with_options(
        batch.schema(),
        columns,
        &options,
    )?)
}

// ============================================================================
// PartitionedTopKSortExec: the ExecutionPlan wrapper
// ============================================================================

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
        trace!(
            "Start PartitionedTopKSortExec::execute for partition {}",
            partition,
        );

        let mut input = self.input.execute(partition, Arc::clone(&context))?;

        let mut partitioned_topk = PartitionedTopK::try_new(
            partition,
            input.schema(),
            self.expr.clone(),
            self.partition_prefix_len,
            self.fetch,
            context.session_config().batch_size(),
            context.runtime_env(),
            &self.metrics,
        )?;

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            futures::stream::once(async move {
                while let Some(batch) = input.next().await {
                    let batch = batch?;
                    partitioned_topk.insert_batch(batch)?;
                }
                partitioned_topk.emit()
            })
            .try_flatten(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.schema()))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collect;
    use crate::test::TestMemoryExec;
    use arrow::array::Int32Array;
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::assert_batches_eq;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

    #[tokio::test]
    async fn test_partitioned_topk_sort_basic() -> Result<()> {
        // Create test data:
        // partition | value
        // ---------+-------
        //    1     |   30
        //    1     |   10
        //    1     |   20
        //    2     |   35
        //    2     |   15
        //    2     |   25
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

        // Sort by (partition ASC, value ASC) and keep top 2 per partition
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

        // Verify output: partition 1 → (10, 20), partition 2 → (15, 25)
        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 10    |",
                "| 1         | 20    |",
                "| 2         | 15    |",
                "| 2         | 25    |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_single_partition() -> Result<()> {
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

        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 10    |",
                "| 1         | 20    |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_multiple_batches() -> Result<()> {
        // Test with input split across multiple batches
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 1])),
                Arc::new(Int32Array::from(vec![30, 35, 10])),
            ],
        )?;
        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![2, 1, 2])),
                Arc::new(Int32Array::from(vec![15, 20, 25])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch1, batch2]],
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

        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 10    |",
                "| 1         | 20    |",
                "| 2         | 15    |",
                "| 2         | 25    |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_desc_order() -> Result<()> {
        // Test with DESC ordering within partitions
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let partition_col = Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 2]));
        let value_col = Arc::new(Int32Array::from(vec![10, 30, 20, 15, 35, 25]));
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
            PhysicalSortExpr::new(
                value_expr,
                SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            ),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        // DESC order: top 2 means largest values
        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 30    |",
                "| 1         | 20    |",
                "| 2         | 35    |",
                "| 2         | 25    |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }
}
