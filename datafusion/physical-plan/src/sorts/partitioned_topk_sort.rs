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
//!   heaps using [`GroupValues`] for grouping and [`TopKHeap`] for
//!   efficient top-k tracking.
//! - [`PartitionedTopKSortExec`]: The [`ExecutionPlan`] wrapper that drives
//!   the processor from an input stream.

use std::any::Any;
use std::mem::size_of;
use std::sync::Arc;

use crate::aggregates::group_values::{GroupValues, new_group_values};
use crate::aggregates::order::GroupOrdering;
use crate::execution_plan::{Boundedness, CardinalityEffect, EmissionType};
use crate::expressions::PhysicalSortExpr;
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use crate::sorts::sort::ExternalSorter;
use crate::stream::RecordBatchStreamAdapter;
use crate::topk::TopKHeap;
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, SendableRecordBatchStream,
};

use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::concat_batches;
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::EmitTo;
use datafusion_physical_expr::LexOrdering;

use futures::{StreamExt, TryStreamExt};
use log::trace;

/// Guesstimate for memory allocation: estimated number of bytes used per row
/// in the RowConverter
const ESTIMATED_BYTES_PER_ROW: usize = 20;

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
// PartitionedTopK: the core processor (insert_batch / emit)
// ============================================================================

/// Partitioned TopK processor.
///
/// Maintains separate TopK state per partition value, following the same
/// `insert_batch` / `emit` lifecycle as [`TopK`](crate::topk::TopK).
///
/// On each [`insert_batch`](Self::insert_batch) call the incoming rows are
/// grouped by partition key using [`GroupValues`] and fed into per-partition
/// [`TopKHeap`]s. Each heap maintains the top `k` rows using a binary heap
/// with efficient memory management and compaction.
///
/// [`emit`](Self::emit) emits each partition's heap in partition-key order,
/// then concatenates the results and breaks them into `batch_size` chunks.
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
    /// GroupValues for efficient partition key interning
    group_values: Box<dyn GroupValues>,
    /// RowConverter for ORDER BY sort keys (feeds TopKHeap)
    order_row_converter: RowConverter,
    /// Scratch space for ORDER BY row conversion
    scratch_rows: Rows,
    /// Per-partition TopKHeap, indexed by group_id from GroupValues
    partition_heaps: Vec<TopKHeap>,
    /// RowConverter for partition key ordering at emit time
    /// (encodes sort options so bytewise comparison gives correct order)
    partition_order_converter: RowConverter,
    /// Reusable buffer for group indices from GroupValues::intern
    current_group_indices: Vec<usize>,
    /// Reusable per-group row index buffers, indexed by group_id.
    /// Avoids HashMap allocation overhead that is significant when there
    /// are many distinct partition values (e.g., 100k+).
    scratch_group_rows: Vec<Vec<usize>>,
    /// Reusable buffer tracking which groups were touched in the current batch
    scratch_touched_groups: Vec<usize>,
}

impl PartitionedTopK {
    /// Create a new [`PartitionedTopK`].
    #[expect(clippy::too_many_arguments)]
    pub fn try_new(
        partition_id: usize,
        schema: &SchemaRef,
        expr: LexOrdering,
        partition_prefix_len: usize,
        k: usize,
        batch_size: usize,
        runtime: &Arc<RuntimeEnv>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Self> {
        let reservation = MemoryConsumer::new(format!("PartitionedTopK[{partition_id}]"))
            .register(&runtime.memory_pool);

        let partition_sort_exprs = &expr[..partition_prefix_len];

        // Build a schema for just the partition key columns (for GroupValues)
        let partition_fields: Vec<Field> = partition_sort_exprs
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let dt = e.expr.data_type(schema)?;
                let nullable = e.expr.nullable(schema)?;
                Ok(Field::new(format!("partition_{i}"), dt, nullable))
            })
            .collect::<Result<_>>()?;
        let partition_schema = Arc::new(Schema::new(partition_fields));

        // Create GroupValues for efficient partition key interning
        let group_values = new_group_values(partition_schema, &GroupOrdering::None)?;

        // RowConverter for partition key ordering at emit time
        let partition_order_fields = build_sort_fields(partition_sort_exprs, schema)?;
        let partition_order_converter = RowConverter::new(partition_order_fields)?;

        // RowConverter for ORDER BY sort keys (used by TopKHeap)
        let order_sort_exprs = &expr[partition_prefix_len..];
        let order_fields = build_sort_fields(order_sort_exprs, schema)?;
        let order_row_converter = RowConverter::new(order_fields)?;
        let scratch_rows = order_row_converter
            .empty_rows(batch_size, ESTIMATED_BYTES_PER_ROW * batch_size);

        Ok(Self {
            schema: Arc::clone(schema),
            metrics: BaselineMetrics::new(metrics, partition_id),
            reservation,
            batch_size,
            expr,
            partition_prefix_len,
            k,
            group_values,
            order_row_converter,
            scratch_rows,
            partition_heaps: Vec::new(),
            partition_order_converter,
            current_group_indices: Vec::new(),
            scratch_group_rows: Vec::new(),
            scratch_touched_groups: Vec::new(),
        })
    }

    /// Insert a [`RecordBatch`], distributing its rows into per-partition
    /// heaps. Each heap maintains the top `k` rows using a binary heap.
    pub fn insert_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        // Updates on drop
        let _timer = self.metrics.elapsed_compute().timer();

        if batch.num_rows() == 0 {
            return Ok(());
        }

        let partition_sort_exprs = &self.expr[..self.partition_prefix_len];
        let order_sort_exprs = &self.expr[self.partition_prefix_len..];

        // Evaluate partition key columns
        let partition_arrays: Vec<ArrayRef> = partition_sort_exprs
            .iter()
            .map(|e| {
                let value = e.expr.evaluate(batch)?;
                value.into_array(batch.num_rows())
            })
            .collect::<Result<Vec<_>>>()?;

        // Intern partition keys to get group IDs (vectorized)
        self.current_group_indices.clear();
        self.group_values
            .intern(&partition_arrays, &mut self.current_group_indices)?;

        // Ensure we have heaps for any newly discovered groups
        while self.partition_heaps.len() < self.group_values.len() {
            self.partition_heaps
                .push(TopKHeap::new(self.k, self.batch_size));
        }

        // Convert ORDER BY columns to row format for heap comparison
        let order_arrays: Vec<ArrayRef> = order_sort_exprs
            .iter()
            .map(|e| {
                let value = e.expr.evaluate(batch)?;
                value.into_array(batch.num_rows())
            })
            .collect::<Result<Vec<_>>>()?;

        let rows = &mut self.scratch_rows;
        rows.clear();
        self.order_row_converter.append(rows, &order_arrays)?;

        // Group row indices by group_id using reusable Vec-based buffers.
        // This avoids HashMap allocation overhead that is significant when
        // there are many distinct partition values (e.g., 100k+).
        while self.scratch_group_rows.len() < self.group_values.len() {
            self.scratch_group_rows.push(Vec::new());
        }
        self.scratch_touched_groups.clear();

        for (row_idx, &group_id) in self.current_group_indices.iter().enumerate() {
            if self.scratch_group_rows[group_id].is_empty() {
                self.scratch_touched_groups.push(group_id);
            }
            self.scratch_group_rows[group_id].push(row_idx);
        }

        // For each touched partition, register batch and add qualifying rows
        for &group_id in &self.scratch_touched_groups {
            let row_indices = std::mem::take(&mut self.scratch_group_rows[group_id]);
            let heap = &mut self.partition_heaps[group_id];
            // RecordBatch::clone is cheap (Arc refs on column arrays)
            let mut batch_entry = heap.register_batch(batch.clone());

            for &row_idx in &row_indices {
                let row = rows.row(row_idx);
                match heap.max() {
                    // Heap full and row >= max: not a new top-k item
                    Some(max_row) if row.as_ref() >= max_row.row() => {}
                    // Heap not full or row < max: add it
                    None | Some(_) => {
                        heap.add(&mut batch_entry, row, row_idx);
                    }
                }
            }

            heap.insert_batch_entry(batch_entry);
            heap.maybe_compact()?;

            // Return the Vec for reuse (preserves capacity)
            self.scratch_group_rows[group_id] = row_indices;
            self.scratch_group_rows[group_id].clear();
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
    pub fn emit(mut self) -> Result<SendableRecordBatchStream> {
        let _timer = self.metrics.elapsed_compute().timer(); // time updated on drop

        let num_groups = self.group_values.len();
        if num_groups == 0 {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema,
                futures::stream::iter(vec![]),
            )));
        }

        // Get partition key arrays to determine ordering
        let partition_key_arrays = self.group_values.emit(EmitTo::All)?;

        // Convert to rows for ordering (respects ASC/DESC, NULLS FIRST/LAST)
        let partition_rows = self
            .partition_order_converter
            .convert_columns(&partition_key_arrays)?;

        // Sort group IDs by partition key order
        let mut group_order: Vec<usize> = (0..num_groups).collect();
        group_order.sort_by(|&a, &b| {
            partition_rows
                .row(a)
                .as_ref()
                .cmp(partition_rows.row(b).as_ref())
        });

        // Emit each partition's heap in partition-key order
        let mut partition_batches = Vec::with_capacity(num_groups);
        for group_id in group_order {
            if let Some(batch) = self.partition_heaps[group_id].emit()? {
                partition_batches.push(batch);
            }
        }

        if partition_batches.is_empty() {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema,
                futures::stream::iter(vec![]),
            )));
        }

        // Combine all partition results into a single sorted batch
        let result = if partition_batches.len() == 1 {
            partition_batches.pop().unwrap()
        } else {
            concat_batches(&self.schema, &partition_batches)?
        };
        drop(partition_batches);

        self.metrics.record_output(result.num_rows());

        // Break into batch_size chunks (following TopK pattern)
        let mut batches = vec![];
        let mut batch = result;
        loop {
            if batch.num_rows() <= self.batch_size {
                batches.push(Ok(batch));
                break;
            } else {
                batches.push(Ok(batch.slice(0, self.batch_size)));
                let remaining_length = batch.num_rows() - self.batch_size;
                batch = batch.slice(self.batch_size, remaining_length);
            }
        }

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema,
            futures::stream::iter(batches),
        )))
    }

    /// Returns the total number of rows currently stored across all partition heaps.
    fn total_heap_rows(&self) -> usize {
        self.partition_heaps.iter().map(|h| h.len()).sum()
    }

    /// Return the estimated memory used by this operator, in bytes.
    fn size(&self) -> usize {
        size_of::<Self>()
            + self.group_values.size()
            + self.order_row_converter.size()
            + self.scratch_rows.size()
            + self.partition_order_converter.size()
            + self.partition_heaps.iter().map(|h| h.size()).sum::<usize>()
            + self
                .scratch_group_rows
                .iter()
                .map(|v| v.capacity() * size_of::<usize>())
                .sum::<usize>()
            + self.scratch_touched_groups.capacity() * size_of::<usize>()
    }
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
                let partition_exprs: Vec<String> = self.expr[..self.partition_prefix_len]
                    .iter()
                    .map(|e| e.to_string())
                    .collect();
                let order_exprs: Vec<String> = self.expr[self.partition_prefix_len..]
                    .iter()
                    .map(|e| e.to_string())
                    .collect();
                write!(
                    f,
                    "PartitionedTopKSortExec: partition_by=[{}], order_by=[{}], fetch={}",
                    partition_exprs.join(", "),
                    order_exprs.join(", "),
                    self.fetch
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
        trace!("Start PartitionedTopKSortExec::execute for partition {partition}",);

        let mut input = self.input.execute(partition, Arc::clone(&context))?;

        let schema = input.schema();
        let expr = self.expr.clone();
        let partition_prefix_len = self.partition_prefix_len;
        let fetch = self.fetch;
        let batch_size = context.session_config().batch_size();
        let runtime = context.runtime_env();
        let metrics = self.metrics.clone();
        let output_schema = self.schema();
        let context = Arc::clone(&context);

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            futures::stream::once(async move {
                // Read the first batch to probe filtering effectiveness
                let Some(first_result) = input.next().await else {
                    return Ok(Box::pin(RecordBatchStreamAdapter::new(
                        schema,
                        futures::stream::iter(vec![]),
                    )) as SendableRecordBatchStream);
                };
                let first_batch = first_result?;
                let first_batch_rows = first_batch.num_rows();

                let mut partitioned_topk = PartitionedTopK::try_new(
                    partition,
                    &schema,
                    expr.clone(),
                    partition_prefix_len,
                    fetch,
                    batch_size,
                    &runtime,
                    &metrics,
                )?;

                partitioned_topk.insert_batch(&first_batch)?;
                let rows_in_heaps = partitioned_topk.total_heap_rows();

                // Adaptive check: if heaps accepted >= 50% of input rows,
                // the TopK filtering benefit is outweighed by the per-heap
                // overhead (hash lookups, heap maintenance for many partitions).
                // Fall back to plain sort which has lower constant overhead.
                //
                // Benchmarks (10000 categories, K=10) show:
                //   ~10% accepted (100 rows/cat) → TopK 37% faster
                //   ~20% accepted ( 50 rows/cat) → TopK  6% faster
                //   ~50% accepted ( 20 rows/cat) → TopK 33% SLOWER
                //
                // Only trigger when the first batch is at least `batch_size`
                // rows, so we have enough data for a reliable decision.
                // The sort fallback does NOT truncate per-partition (that is
                // handled by the downstream BoundedWindowAggExec + Filter),
                // so it must only be used when the full plan is present.
                let filtering_ineffective = first_batch_rows >= batch_size
                    && rows_in_heaps * 2 >= first_batch_rows;

                if filtering_ineffective {
                    trace!(
                        "PartitionedTopKSortExec: filtering ineffective \
                         ({rows_in_heaps}/{first_batch_rows} rows accepted), \
                         falling back to sort"
                    );
                    // Fall back to plain sort using ExternalSorter (same as
                    // SortExec). This gives us memory tracking, spill-to-disk
                    // support, and adaptive sort strategies. The downstream
                    // BoundedWindowAggExec + FilterExec will handle the
                    // per-partition truncation.
                    drop(partitioned_topk);

                    let execution_options = &context.session_config().options().execution;
                    let mut sorter = ExternalSorter::new(
                        partition,
                        Arc::clone(&schema),
                        expr,
                        batch_size,
                        execution_options.sort_spill_reservation_bytes,
                        execution_options.sort_in_place_threshold_bytes,
                        context.session_config().spill_compression(),
                        &metrics,
                        Arc::clone(&runtime),
                    )?;

                    sorter.insert_batch(first_batch).await?;
                    while let Some(batch) = input.next().await {
                        sorter.insert_batch(batch?).await?;
                    }
                    sorter.sort().await
                } else {
                    // TopK is effective, continue with heap approach
                    while let Some(batch) = input.next().await {
                        let batch = batch?;
                        partitioned_topk.insert_batch(&batch)?;
                    }
                    partitioned_topk.emit()
                }
            })
            .try_flatten(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
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
    use arrow::array::{Int32Array, StringArray};
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

    #[tokio::test]
    async fn test_partitioned_topk_sort_empty_input() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_k_equals_one() -> Result<()> {
        // K=1: only keep the single top row per partition
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2])),
                Arc::new(Int32Array::from(vec![30, 10, 20, 50, 40])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 1)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        // ASC order with k=1: smallest value per partition
        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 10    |",
                "| 2         | 40    |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_k_larger_than_partition() -> Result<()> {
        // K larger than partition size: return all rows in that partition
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2])),
                Arc::new(Int32Array::from(vec![20, 10, 30, 10, 20])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        // k=10 but partition 1 only has 2 rows and partition 2 only has 3
        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 10)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 10    |",
                "| 1         | 20    |",
                "| 2         | 10    |",
                "| 2         | 20    |",
                "| 2         | 30    |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_with_ties() -> Result<()> {
        // Test with duplicate values in the order column
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 1, 2, 2, 2])),
                Arc::new(Int32Array::from(vec![10, 10, 20, 20, 5, 5, 5])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        // Partition 1: two 10s are smallest, keep 2
        // Partition 2: three 5s are all tied, keep 2
        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 10    |",
                "| 1         | 10    |",
                "| 2         | 5     |",
                "| 2         | 5     |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_string_partition_keys() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("category", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![
                    "apple", "apple", "apple", "banana", "banana", "banana",
                ])),
                Arc::new(Int32Array::from(vec![30, 10, 20, 60, 40, 50])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("category", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        assert_batches_eq!(
            &[
                "+----------+-------+",
                "| category | value |",
                "+----------+-------+",
                "| apple    | 10    |",
                "| apple    | 20    |",
                "| banana   | 40    |",
                "| banana   | 50    |",
                "+----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_multiple_partition_columns() -> Result<()> {
        // Partition by two columns: (region, category)
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Int32, false),
            Field::new("category", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 1, 2, 2, 2, 2])),
                Arc::new(Int32Array::from(vec![10, 10, 20, 20, 10, 10, 20, 20])),
                Arc::new(Int32Array::from(vec![
                    100, 200, 300, 400, 500, 600, 700, 800,
                ])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("region", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("category", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        // partition_prefix_len = 2 means we partition by (region, category)
        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 2, 1)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        // 4 partitions (1,10), (1,20), (2,10), (2,20), keep top 1 (smallest) each
        assert_batches_eq!(
            &[
                "+--------+----------+-------+",
                "| region | category | value |",
                "+--------+----------+-------+",
                "| 1      | 10       | 100   |",
                "| 1      | 20       | 300   |",
                "| 2      | 10       | 500   |",
                "| 2      | 20       | 700   |",
                "+--------+----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_nullable_values() -> Result<()> {
        // Test with NULL values in the order column
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 2])),
                Arc::new(Int32Array::from(vec![
                    Some(30),
                    None,
                    Some(10),
                    Some(20),
                    None,
                    Some(40),
                ])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        // Default sort: ASC, nulls_first=true
        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        // Default SortOptions: ASC with nulls_first=true
        // Partition 1: NULL < 10 < 30 → top 2: [NULL, 10]
        // Partition 2: NULL < 20 < 40 → top 2: [NULL, 20]
        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         |       |",
                "| 1         | 10    |",
                "| 2         |       |",
                "| 2         | 20    |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_nullable_nulls_last() -> Result<()> {
        // Test NULLs with nulls_last sort option
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 1])),
                Arc::new(Int32Array::from(vec![Some(30), None, Some(10), Some(20)])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        // ASC with nulls_first=false (nulls last)
        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(
                col("value", &schema)?,
                SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            ),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        // ASC nulls_last: 10 < 20 < 30 < NULL → top 2: [10, 20]
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
    async fn test_partitioned_topk_sort_many_partitions() -> Result<()> {
        // Stress test with many partitions
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let num_partitions = 50;
        let rows_per_partition = 5;
        let mut partition_vals = Vec::new();
        let mut value_vals = Vec::new();
        for p in 0..num_partitions {
            for v in 0..rows_per_partition {
                partition_vals.push(p);
                value_vals.push((rows_per_partition - v) * 10); // descending
            }
        }

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(partition_vals)),
                Arc::new(Int32Array::from(value_vals)),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        // Keep top 1 per partition
        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 1)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, num_partitions as usize);

        // Each partition should have value 10 (smallest)
        for batch in &result {
            let values = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..values.len() {
                assert_eq!(values.value(i), 10);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_single_row_partitions() -> Result<()> {
        // Each partition has only one row
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![100, 200, 300])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
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
                "| 1         | 100   |",
                "| 2         | 200   |",
                "| 3         | 300   |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_no_input_batches() -> Result<()> {
        // Zero batches from input
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0);

        Ok(())
    }

    #[test]
    fn test_partitioned_topk_sort_error_partition_prefix_zero() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let memory_exec = Arc::new(
            TestMemoryExec::try_new(&[vec![]], Arc::clone(&schema), None).unwrap(),
        );

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(
                col("partition", &schema).unwrap(),
                SortOptions::default(),
            ),
            PhysicalSortExpr::new(col("value", &schema).unwrap(), SortOptions::default()),
        ])
        .unwrap();

        let result = PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 0, 2);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("partition_prefix_len must be greater than 0")
        );
    }

    #[test]
    fn test_partitioned_topk_sort_error_fetch_zero() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let memory_exec = Arc::new(
            TestMemoryExec::try_new(&[vec![]], Arc::clone(&schema), None).unwrap(),
        );

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(
                col("partition", &schema).unwrap(),
                SortOptions::default(),
            ),
            PhysicalSortExpr::new(col("value", &schema).unwrap(), SortOptions::default()),
        ])
        .unwrap();

        let result = PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("fetch must be greater than 0")
        );
    }

    #[test]
    fn test_partitioned_topk_sort_error_prefix_exceeds_exprs() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let memory_exec = Arc::new(
            TestMemoryExec::try_new(&[vec![]], Arc::clone(&schema), None).unwrap(),
        );

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(
                col("partition", &schema).unwrap(),
                SortOptions::default(),
            ),
            PhysicalSortExpr::new(col("value", &schema).unwrap(), SortOptions::default()),
        ])
        .unwrap();

        // partition_prefix_len = 3, but only 2 sort expressions
        let result = PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 3, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(
            "partition_prefix_len (3) cannot exceed sort expression length (2)"
        ));
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_desc_partition_order() -> Result<()> {
        // Test with DESC partition key ordering
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        // Partition by DESC, order by ASC
        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(
                col("partition", &schema)?,
                SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            ),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 1)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        // Partitions ordered DESC: 3, 2, 1
        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 3         | 50    |",
                "| 2         | 30    |",
                "| 1         | 10    |",
                "+-----------+-------+",
            ],
            &result
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_topk_sort_preserves_extra_columns() -> Result<()> {
        // Ensure non-partition, non-order columns are preserved correctly
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
            Field::new("label", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2])),
                Arc::new(Int32Array::from(vec![30, 10, 20, 50, 40])),
                Arc::new(StringArray::from(vec!["c", "a", "b", "e", "d"])),
            ],
        )?;

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);

        let sort_exprs = LexOrdering::new(vec![
            PhysicalSortExpr::new(col("partition", &schema)?, SortOptions::default()),
            PhysicalSortExpr::new(col("value", &schema)?, SortOptions::default()),
        ])
        .unwrap();

        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 2)?;

        let task_ctx = Arc::new(TaskContext::default());
        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;

        assert_batches_eq!(
            &[
                "+-----------+-------+-------+",
                "| partition | value | label |",
                "+-----------+-------+-------+",
                "| 1         | 10    | a     |",
                "| 1         | 20    | b     |",
                "| 2         | 40    | d     |",
                "| 2         | 50    | e     |",
                "+-----------+-------+-------+",
            ],
            &result
        );

        Ok(())
    }
}
