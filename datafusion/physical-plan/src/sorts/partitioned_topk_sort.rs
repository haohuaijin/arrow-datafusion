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
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::sync::Arc;

use crate::aggregates::group_values::{GroupValues, new_group_values};
use crate::aggregates::order::GroupOrdering;
use crate::execution_plan::{Boundedness, CardinalityEffect, EmissionType};
use crate::expressions::PhysicalSortExpr;
use crate::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, RecordOutput,
};

use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, SendableRecordBatchStream,
};

use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::interleave_record_batch;
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::EmitTo;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr};

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
/// lightweight heaps. Unlike the single-partition [`TopK`](crate::topk::TopK),
/// this uses a **shared batch store** so that input batches are stored once
/// regardless of how many groups reference them. This is critical for
/// performance when there are many distinct partition values (e.g., 100k+).
///
/// [`emit`](Self::emit) emits each partition's heap in partition-key order,
/// then concatenates the results and breaks them into `batch_size` chunks.
pub(crate) struct PartitionedTopK {
    // ---- Plan configuration ----
    /// Schema of the output (and the input)
    schema: SchemaRef,
    /// Full sort expressions (partition prefix + order by)
    expr: LexOrdering,
    /// Number of leading sort expressions that define partitions
    partition_prefix_len: usize,
    /// Maximum rows to keep per partition
    k: usize,
    /// The target number of rows for output batches
    batch_size: usize,

    // ---- Row converters ----
    /// RowConverter for ORDER BY sort keys
    order_row_converter: RowConverter,
    /// RowConverter for partition key ordering at emit time
    /// (encodes sort options so bytewise comparison gives correct order)
    partition_order_converter: RowConverter,
    /// Scratch space for ORDER BY row conversion
    scratch_rows: Rows,

    // ---- Partition state ----
    /// GroupValues for efficient partition key interning
    group_values: Box<dyn GroupValues>,
    /// Per-partition lightweight heaps, indexed by group_id from GroupValues.
    /// Each heap stores sort-key bytes and references into `shared_batches`.
    group_heaps: Vec<GroupHeap>,
    /// Reusable buffer for group indices from GroupValues::intern
    current_group_indices: Vec<usize>,

    // ---- Shared batch store ----
    /// Shared batch storage: input batches stored once, referenced by all heaps
    shared_batches: Vec<RecordBatch>,
    /// Number of heap entries referencing each shared batch (for compaction)
    shared_batch_uses: Vec<usize>,

    // ---- Resource tracking ----
    /// Runtime metrics
    metrics: BaselineMetrics,
    /// Memory reservation tracked through the memory pool
    reservation: MemoryReservation,
    /// Total bytes owned by all heap entries (sort key Vec allocations)
    heap_owned_bytes: usize,
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
            // Plan configuration
            schema: Arc::clone(schema),
            expr,
            partition_prefix_len,
            k,
            batch_size,
            // Row converters
            order_row_converter,
            partition_order_converter,
            scratch_rows,
            // Partition state
            group_values,
            group_heaps: Vec::new(),
            current_group_indices: Vec::new(),
            // Shared batch store
            shared_batches: Vec::new(),
            shared_batch_uses: Vec::new(),
            // Resource tracking
            metrics: BaselineMetrics::new(metrics, partition_id),
            reservation,
            heap_owned_bytes: 0,
        })
    }

    /// Insert a [`RecordBatch`], distributing its rows into per-partition
    /// heaps. Each heap maintains the top `k` rows using a binary heap.
    ///
    /// Input batches are stored once in a shared store and referenced by
    /// all per-group heaps, avoiding per-group batch cloning and HashMap
    /// overhead that is significant with many distinct partition values.
    pub fn insert_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let baseline = self.metrics.clone();
        let _timer = baseline.elapsed_compute().timer(); // time updated on drop

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
        let k = self.k;
        while self.group_heaps.len() < self.group_values.len() {
            self.group_heaps.push(GroupHeap::new(k));
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

        // Register batch once in shared store (lazy: only if any row qualifies)
        let batch_idx = self.shared_batches.len() as u32;
        let mut batch_registered = false;

        // Process each row directly against its group's heap.
        // This avoids the scatter-gather pattern (grouping rows into per-group
        // Vecs then iterating groups) which has poor cache locality when there
        // are many distinct groups (e.g., 100k+ categories).
        for (row_idx, &group_id) in self.current_group_indices.iter().enumerate() {
            let row = rows.row(row_idx);

            // Check if this row qualifies for the heap
            let heap = &self.group_heaps[group_id];
            let dominated = match heap.max() {
                Some(max_entry) => row.as_ref() >= max_entry.row.as_slice(),
                None => false, // heap not full, always accept
            };

            if !dominated {
                if !batch_registered {
                    self.shared_batches.push(batch.clone());
                    self.shared_batch_uses.push(0);
                    batch_registered = true;
                }

                // Track eviction from shared store use counts
                let heap = &self.group_heaps[group_id];
                if heap.inner.len() == heap.k {
                    let evicted_batch_idx = heap.inner.peek().unwrap().batch_idx as usize;
                    self.shared_batch_uses[evicted_batch_idx] -= 1;
                }

                // add() reuses the evicted entry's Vec allocation,
                // so the returned delta is the net change in owned bytes.
                let heap = &mut self.group_heaps[group_id];
                self.heap_owned_bytes += heap.add(row.as_ref(), batch_idx, row_idx);
                self.shared_batch_uses[batch_idx as usize] += 1;
            }
        }

        // Compact shared store if too much memory is wasted on unused rows
        self.maybe_compact()?;

        // Update memory reservation
        self.reservation.try_resize(self.size())?;

        Ok(())
    }

    /// Compact the shared batch store when too many unused rows accumulate.
    fn maybe_compact(&mut self) -> Result<()> {
        if self.shared_batches.len() <= 2 {
            return Ok(());
        }

        let total_batch_rows: usize =
            self.shared_batches.iter().map(|b| b.num_rows()).sum();
        let total_used: usize = self.shared_batch_uses.iter().sum();
        let unused = total_batch_rows.saturating_sub(total_used);

        let max_unused = (20 * self.batch_size) + (self.k * self.group_heaps.len());
        if unused < max_unused {
            return Ok(());
        }

        // Collect all entries from all heaps with their group IDs
        let mut all_entries: Vec<(usize, GroupHeapEntry)> = Vec::new();
        for (group_id, heap) in self.group_heaps.iter_mut().enumerate() {
            for entry in heap.inner.drain() {
                all_entries.push((group_id, entry));
            }
        }

        if all_entries.is_empty() {
            self.shared_batches.clear();
            self.shared_batch_uses.clear();
            return Ok(());
        }

        // Build interleave indices from old shared batches
        let batch_refs: Vec<&RecordBatch> = self.shared_batches.iter().collect();
        let indices: Vec<(usize, usize)> = all_entries
            .iter()
            .map(|(_, e)| (e.batch_idx as usize, e.row_idx))
            .collect();

        let new_batch = interleave_record_batch(&batch_refs, &indices)?;
        let num_entries = all_entries.len();

        // Replace shared store with compacted batch
        self.shared_batches.clear();
        self.shared_batch_uses.clear();
        self.shared_batches.push(new_batch);
        self.shared_batch_uses.push(num_entries);

        // Restore heap entries with updated references
        for (new_row_idx, (group_id, mut entry)) in all_entries.into_iter().enumerate() {
            entry.batch_idx = 0;
            entry.row_idx = new_row_idx;
            self.group_heaps[group_id].inner.push(entry);
        }

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

        // Collect batch references for interleaving
        let batch_refs: Vec<&RecordBatch> = self.shared_batches.iter().collect();

        // Collect ALL indices across all groups in partition-key order,
        // with each group's entries sorted by ORDER BY columns.
        // This allows a single interleave_record_batch call instead of
        // one per group, which is critical when there are many groups.
        let mut all_indices: Vec<(usize, usize)> = Vec::new();
        for group_id in group_order {
            let heap = &mut self.group_heaps[group_id];
            if heap.inner.is_empty() {
                continue;
            }

            // Drain and sort heap entries (low to high)
            let entries = std::mem::take(&mut heap.inner).into_sorted_vec();

            for e in &entries {
                all_indices.push((e.batch_idx as usize, e.row_idx));
            }
        }

        if all_indices.is_empty() {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema,
                futures::stream::iter(vec![]),
            )));
        }

        let result = interleave_record_batch(&batch_refs, &all_indices)?;

        (&result).record_output(&self.metrics);

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

    /// Return the estimated memory used by this operator, in bytes.
    fn size(&self) -> usize {
        size_of::<Self>()
            + self.group_values.size()
            + self.order_row_converter.size()
            + self.scratch_rows.size()
            + self.partition_order_converter.size()
            + self.group_heaps.iter().map(|h| h.size()).sum::<usize>()
            + self
                .shared_batches
                .iter()
                .map(|b| b.get_array_memory_size())
                .sum::<usize>()
            + self.heap_owned_bytes
    }
}

// ============================================================================
// GroupHeap: lightweight per-group heap for PartitionedTopK
// ============================================================================

/// A single entry in a [`GroupHeap`], storing the sort key and a reference
/// to the source row in the shared batch store.
#[derive(Debug, PartialEq, Eq)]
struct GroupHeapEntry {
    /// Sort key in arrow Row format (for comparison)
    row: Vec<u8>,
    /// Index into the shared `shared_batches` Vec in [`PartitionedTopK`]
    batch_idx: u32,
    /// Row index within the batch
    row_idx: usize,
}

impl Ord for GroupHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.row.cmp(&other.row)
    }
}

impl PartialOrd for GroupHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A lightweight max-heap that maintains the top K rows for a single
/// partition group. Unlike [`TopKHeap`](crate::topk::TopKHeap), this does
/// not own batch data — it references a shared batch store managed by
/// [`PartitionedTopK`].
struct GroupHeap {
    /// Max-heap of entries (worst/largest at top for easy eviction)
    inner: BinaryHeap<GroupHeapEntry>,
    /// Maximum number of elements
    k: usize,
}

impl GroupHeap {
    fn new(k: usize) -> Self {
        Self {
            inner: BinaryHeap::new(),
            k,
        }
    }

    /// Returns the largest (worst) value if the heap is full.
    fn max(&self) -> Option<&GroupHeapEntry> {
        if self.inner.len() < self.k {
            None
        } else {
            self.inner.peek()
        }
    }

    /// Add a row to the heap. If full, evicts the worst entry (caller
    /// must handle use-count bookkeeping before calling this).
    /// Returns the number of newly allocated bytes for the sort key.
    fn add(&mut self, row: &[u8], batch_idx: u32, row_idx: usize) -> usize {
        if self.inner.len() == self.k {
            // Replace worst entry, reusing its Vec allocation
            let mut max = self.inner.peek_mut().unwrap();
            let old_cap = max.row.capacity();
            max.row.clear();
            max.row.extend_from_slice(row);
            max.batch_idx = batch_idx;
            max.row_idx = row_idx;
            // Return the net change in owned bytes
            max.row.capacity() - old_cap
        } else {
            let entry = GroupHeapEntry {
                row: row.to_vec(),
                batch_idx,
                row_idx,
            };
            let cap = entry.row.capacity();
            self.inner.push(entry);
            cap
        }
    }

    /// Return the size of memory used by this heap's metadata (not
    /// including owned row bytes, which are tracked separately).
    fn size(&self) -> usize {
        size_of::<Self>() + self.inner.capacity() * size_of::<GroupHeapEntry>()
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
    cache: Arc<PlanProperties>,
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
    ) -> Result<Arc<PlanProperties>> {
        let mut eq_properties = input.equivalence_properties().clone();
        // The output is sorted according to the sort expressions
        eq_properties.reorder(expr.clone())?;

        // Preserve the input partitioning since we can process each partition independently
        let output_partitioning = input.output_partitioning().clone();

        Ok(Arc::new(PlanProperties::new(
            eq_properties,
            output_partitioning,
            EmissionType::Final,
            Boundedness::Bounded,
        )))
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

    fn properties(&self) -> &Arc<PlanProperties> {
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
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            futures::stream::once(async move {
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

                while let Some(batch) = input.next().await {
                    let batch = batch?;
                    partitioned_topk.insert_batch(&batch)?;
                }
                partitioned_topk.emit()
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
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

    /// Standard (partition INT32, value INT32) schema used by most tests.
    fn partition_value_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]))
    }

    /// Build default ASC sort expressions for the given column names.
    fn default_sort_exprs(names: &[&str], schema: &SchemaRef) -> LexOrdering {
        LexOrdering::new(
            names
                .iter()
                .map(|name| {
                    PhysicalSortExpr::new(
                        col(name, schema).unwrap(),
                        SortOptions::default(),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// Build sort expressions with per-column options.
    fn sort_exprs_with_options(
        cols: &[(&str, SortOptions)],
        schema: &SchemaRef,
    ) -> LexOrdering {
        LexOrdering::new(
            cols.iter()
                .map(|(name, opts)| {
                    PhysicalSortExpr::new(col(name, schema).unwrap(), *opts)
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// Run partitioned topk on batches and return results.
    async fn run_partitioned_topk(
        partitions: &[Vec<RecordBatch>],
        schema: &SchemaRef,
        sort_exprs: LexOrdering,
        partition_prefix_len: usize,
        k: usize,
    ) -> Result<Vec<RecordBatch>> {
        let memory_exec = Arc::new(TestMemoryExec::try_new(
            partitions,
            Arc::clone(schema),
            None,
        )?);
        let partitioned_topk = PartitionedTopKSortExec::try_new(
            memory_exec,
            sort_exprs,
            partition_prefix_len,
            k,
        )?;
        let task_ctx = Arc::new(TaskContext::default());
        collect(Arc::new(partitioned_topk), task_ctx).await
    }

    /// Helper: create a single RecordBatch with two Int32 columns.
    fn int32_batch(schema: &SchemaRef, col1: Vec<i32>, col2: Vec<i32>) -> RecordBatch {
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(Int32Array::from(col1)),
                Arc::new(Int32Array::from(col2)),
            ],
        )
        .unwrap()
    }

    fn total_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    // ==================== Basic functionality ====================

    #[tokio::test]
    async fn test_basic() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(
            &schema,
            vec![1, 1, 1, 2, 2, 2],
            vec![30, 10, 20, 35, 15, 25],
        );
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

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
    async fn test_single_partition() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(&schema, vec![1, 1, 1, 1], vec![40, 10, 30, 20]);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

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
    async fn test_multiple_batches() -> Result<()> {
        let schema = partition_value_schema();
        let batch1 = int32_batch(&schema, vec![1, 2, 1], vec![30, 35, 10]);
        let batch2 = int32_batch(&schema, vec![2, 1, 2], vec![15, 20, 25]);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch1, batch2]], &schema, sort_exprs, 1, 2)
                .await?;

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

    // ==================== Sort options ====================

    #[tokio::test]
    async fn test_desc_order() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(
            &schema,
            vec![1, 1, 1, 2, 2, 2],
            vec![10, 30, 20, 15, 35, 25],
        );
        let desc = SortOptions {
            descending: true,
            nulls_first: false,
        };
        let sort_exprs = sort_exprs_with_options(
            &[("partition", SortOptions::default()), ("value", desc)],
            &schema,
        );
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

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
    async fn test_desc_partition_order() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(
            &schema,
            vec![1, 1, 2, 2, 3, 3],
            vec![10, 20, 30, 40, 50, 60],
        );
        let desc = SortOptions {
            descending: true,
            nulls_first: false,
        };
        let sort_exprs = sort_exprs_with_options(
            &[("partition", desc), ("value", SortOptions::default())],
            &schema,
        );
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 1).await?;

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

    // ==================== Edge cases ====================

    #[tokio::test]
    async fn test_empty_input() -> Result<()> {
        let schema = partition_value_schema();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
            ],
        )?;
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;
        assert_eq!(total_rows(&result), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_no_input_batches() -> Result<()> {
        let schema = partition_value_schema();
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result = run_partitioned_topk(&[vec![]], &schema, sort_exprs, 1, 2).await?;
        assert_eq!(total_rows(&result), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_k_equals_one() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(&schema, vec![1, 1, 1, 2, 2], vec![30, 10, 20, 50, 40]);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 1).await?;

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
    async fn test_k_larger_than_partition() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(&schema, vec![1, 1, 2, 2, 2], vec![20, 10, 30, 10, 20]);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 10).await?;

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
    async fn test_with_ties() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(
            &schema,
            vec![1, 1, 1, 1, 2, 2, 2],
            vec![10, 10, 20, 20, 5, 5, 5],
        );
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

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
    async fn test_single_row_partitions() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(&schema, vec![1, 2, 3], vec![100, 200, 300]);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

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
    async fn test_large_k_single_partition() -> Result<()> {
        let schema = partition_value_schema();
        let batch = int32_batch(&schema, vec![1, 1, 1], vec![30, 10, 20]);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 1000).await?;

        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 10    |",
                "| 1         | 20    |",
                "| 1         | 30    |",
                "+-----------+-------+",
            ],
            &result
        );
        Ok(())
    }

    // ==================== Data types ====================

    #[tokio::test]
    async fn test_string_partition_keys() -> Result<()> {
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
        let sort_exprs = default_sort_exprs(&["category", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

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

    // ==================== Nullable ====================

    #[tokio::test]
    async fn test_nullable_values() -> Result<()> {
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
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

        // Default: ASC nulls_first → NULL < 10 < 30, NULL < 20 < 40
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
    async fn test_nullable_nulls_last() -> Result<()> {
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
        let nulls_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let sort_exprs = sort_exprs_with_options(
            &[("partition", SortOptions::default()), ("value", nulls_last)],
            &schema,
        );
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

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
    async fn test_nullable_partition_keys() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, true),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![
                    Some(1),
                    Some(1),
                    None,
                    None,
                    Some(2),
                    Some(2),
                    None,
                ])),
                Arc::new(Int32Array::from(vec![30, 10, 50, 20, 40, 60, 10])),
            ],
        )?;
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "|           | 10    |",
                "|           | 20    |",
                "| 1         | 10    |",
                "| 1         | 30    |",
                "| 2         | 40    |",
                "| 2         | 60    |",
                "+-----------+-------+",
            ],
            &result
        );
        Ok(())
    }

    // ==================== Multi-column ====================

    #[tokio::test]
    async fn test_multiple_partition_columns() -> Result<()> {
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
        let sort_exprs = default_sort_exprs(&["region", "category", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 2, 1).await?;

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
    async fn test_multiple_order_columns() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int32, false),
            Field::new("priority", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 1, 2, 2, 2, 2])),
                Arc::new(Int32Array::from(vec![2, 1, 1, 2, 1, 2, 1, 2])),
                Arc::new(Int32Array::from(vec![10, 20, 10, 20, 30, 10, 20, 30])),
            ],
        )?;
        let sort_exprs = default_sort_exprs(&["partition", "priority", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

        // Partition 1: (1,10),(1,20),(2,10),(2,20) → top 2: (1,10),(1,20)
        // Partition 2: (1,20),(1,30),(2,10),(2,30) → top 2: (1,20),(1,30)
        assert_batches_eq!(
            &[
                "+-----------+----------+-------+",
                "| partition | priority | value |",
                "+-----------+----------+-------+",
                "| 1         | 1        | 10    |",
                "| 1         | 1        | 20    |",
                "| 2         | 1        | 20    |",
                "| 2         | 1        | 30    |",
                "+-----------+----------+-------+",
            ],
            &result
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_multi_partition_multi_order() -> Result<()> {
        // PARTITION BY (region, category) ORDER BY (priority ASC, value ASC)
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("priority", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])),
                Arc::new(StringArray::from(vec![
                    "a", "a", "a", "b", "b", "b", "a", "a", "a", "b", "b", "b",
                ])),
                Arc::new(Int32Array::from(vec![2, 1, 1, 3, 1, 2, 1, 2, 1, 2, 1, 1])),
                Arc::new(Int32Array::from(vec![
                    100, 200, 50, 300, 150, 250, 400, 350, 100, 500, 200, 150,
                ])),
            ],
        )?;
        let sort_exprs =
            default_sort_exprs(&["region", "category", "priority", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 2, 2).await?;

        // (1,"a"): (1,50),(1,200)  (1,"b"): (1,150),(2,250)
        // (2,"a"): (1,100),(1,400) (2,"b"): (1,150),(1,200)
        assert_batches_eq!(
            &[
                "+--------+----------+----------+-------+",
                "| region | category | priority | value |",
                "+--------+----------+----------+-------+",
                "| 1      | a        | 1        | 50    |",
                "| 1      | a        | 1        | 200   |",
                "| 1      | b        | 1        | 150   |",
                "| 1      | b        | 2        | 250   |",
                "| 2      | a        | 1        | 100   |",
                "| 2      | a        | 1        | 400   |",
                "| 2      | b        | 1        | 150   |",
                "| 2      | b        | 1        | 200   |",
                "+--------+----------+----------+-------+",
            ],
            &result
        );
        Ok(())
    }

    // ==================== Extra columns ====================

    #[tokio::test]
    async fn test_preserves_extra_columns() -> Result<()> {
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
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 2).await?;

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

    // ==================== Multi-batch stress ====================

    #[tokio::test]
    async fn test_multiple_input_batches() -> Result<()> {
        let schema = partition_value_schema();
        let batch1 = int32_batch(&schema, vec![1, 1, 2, 2], vec![50, 30, 90, 70]);
        let batch2 = int32_batch(&schema, vec![1, 1, 2, 2], vec![10, 40, 60, 80]);
        let batch3 = int32_batch(&schema, vec![1, 2], vec![20, 55]);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result = run_partitioned_topk(
            &[vec![batch1, batch2, batch3]],
            &schema,
            sort_exprs,
            1,
            3,
        )
        .await?;

        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 10    |",
                "| 1         | 20    |",
                "| 1         | 30    |",
                "| 2         | 55    |",
                "| 2         | 60    |",
                "| 2         | 70    |",
                "+-----------+-------+",
            ],
            &result
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_interleaved_partitions_across_batches() -> Result<()> {
        let schema = partition_value_schema();
        let batches: Vec<RecordBatch> = (0..5)
            .map(|i| {
                int32_batch(
                    &schema,
                    vec![1, 2, 3, 1, 2, 3],
                    vec![
                        i * 10 + 5,
                        i * 10 + 3,
                        i * 10 + 8,
                        i * 10 + 1,
                        i * 10 + 9,
                        i * 10 + 2,
                    ],
                )
            })
            .collect();
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result = run_partitioned_topk(&[batches], &schema, sort_exprs, 1, 2).await?;

        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 1         | 1     |",
                "| 1         | 5     |",
                "| 2         | 3     |",
                "| 2         | 9     |",
                "| 3         | 2     |",
                "| 3         | 8     |",
                "+-----------+-------+",
            ],
            &result
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_many_partitions() -> Result<()> {
        let schema = partition_value_schema();
        let num_partitions = 50i32;
        let rows_per_partition = 5i32;
        let mut pvals = Vec::new();
        let mut vvals = Vec::new();
        for p in 0..num_partitions {
            for v in 0..rows_per_partition {
                pvals.push(p);
                vvals.push((rows_per_partition - v) * 10);
            }
        }
        let batch = int32_batch(&schema, pvals, vvals);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result =
            run_partitioned_topk(&[vec![batch]], &schema, sort_exprs, 1, 1).await?;

        assert_eq!(total_rows(&result), num_partitions as usize);
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
    async fn test_compaction() -> Result<()> {
        let schema = partition_value_schema();
        let batches: Vec<RecordBatch> = (0..200)
            .map(|i| int32_batch(&schema, vec![i % 3], vec![(200 - i) * 10]))
            .collect();
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result = run_partitioned_topk(&[batches], &schema, sort_exprs, 1, 2).await?;

        assert_eq!(total_rows(&result), 6);
        assert_batches_eq!(
            &[
                "+-----------+-------+",
                "| partition | value |",
                "+-----------+-------+",
                "| 0         | 20    |",
                "| 0         | 50    |",
                "| 1         | 10    |",
                "| 1         | 40    |",
                "| 2         | 30    |",
                "| 2         | 60    |",
                "+-----------+-------+",
            ],
            &result
        );
        Ok(())
    }

    // ==================== Error cases ====================

    #[test]
    fn test_error_partition_prefix_zero() {
        let schema = partition_value_schema();
        let memory_exec = Arc::new(
            TestMemoryExec::try_new(&[vec![]], Arc::clone(&schema), None).unwrap(),
        );
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
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
    fn test_error_fetch_zero() {
        let schema = partition_value_schema();
        let memory_exec = Arc::new(
            TestMemoryExec::try_new(&[vec![]], Arc::clone(&schema), None).unwrap(),
        );
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
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
    fn test_error_prefix_exceeds_exprs() {
        let schema = partition_value_schema();
        let memory_exec = Arc::new(
            TestMemoryExec::try_new(&[vec![]], Arc::clone(&schema), None).unwrap(),
        );
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let result = PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 3, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(
            "partition_prefix_len (3) cannot exceed sort expression length (2)"
        ));
    }

    // ==================== Memory leak ====================

    #[tokio::test]
    async fn test_memory_released_after_execution() -> Result<()> {
        let schema = partition_value_schema();
        let num_partitions = 20i32;
        let mut pvals = Vec::new();
        let mut vvals = Vec::new();
        for p in 0..num_partitions {
            for v in 0..10 {
                pvals.push(p);
                vvals.push(v * 10);
            }
        }
        let batch = int32_batch(&schema, pvals, vvals);

        let memory_exec = Arc::new(TestMemoryExec::try_new(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?);
        let sort_exprs = default_sort_exprs(&["partition", "value"], &schema);
        let partitioned_topk =
            PartitionedTopKSortExec::try_new(memory_exec, sort_exprs, 1, 3)?;

        let task_ctx = Arc::new(TaskContext::default());
        let pool = task_ctx.runtime_env().memory_pool.clone();
        let reserved_before = pool.reserved();

        let result = collect(Arc::new(partitioned_topk), task_ctx).await?;
        assert_eq!(total_rows(&result), (num_partitions * 3) as usize);
        drop(result);

        let reserved_after = pool.reserved();
        assert_eq!(
            reserved_after, reserved_before,
            "Memory leak: {reserved_after} bytes reserved (was {reserved_before})"
        );
        Ok(())
    }
}
