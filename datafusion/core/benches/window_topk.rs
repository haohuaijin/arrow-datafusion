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

//! Benchmark for window TopK optimization with ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)

use std::any::Any;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Int64Builder, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, Constraints, Statistics};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::SessionContext;
use datafusion_common::config::ConfigOptions;
use datafusion_common::internal_err;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_expr::dml::InsertOp;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
use datafusion_physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation,
};
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream, SortOrderPushdownResult,
};
use datafusion_session::Session;
use rand::Rng;
use rand::SeedableRng;
use rand::prelude::SliceRandom;
use tokio::runtime::Runtime;

const LIMIT: usize = 10;
const PARTITIONS: usize = 10;

/// Remap aggregate statistics when the memory scan applies a column projection.
fn project_scan_statistics(stats: &Statistics, projection: &[usize]) -> Statistics {
    let column_statistics: Vec<ColumnStatistics> = projection
        .iter()
        .map(|&i| {
            stats
                .column_statistics
                .get(i)
                .cloned()
                .unwrap_or_else(ColumnStatistics::new_unknown)
        })
        .collect();
    Statistics {
        num_rows: stats.num_rows,
        total_byte_size: stats.total_byte_size,
        column_statistics,
    }
}

/// Wraps [`DataSourceExec`] so aggregate [`partition_statistics`](ExecutionPlan::partition_statistics)
/// (`partition == None`) returns injected stats; per-partition stats and execution delegate to the inner node.
#[derive(Debug)]
struct DataSourceExecWithScanStatistics {
    inner: Arc<DataSourceExec>,
    aggregate_stats: Arc<Statistics>,
}

impl DataSourceExecWithScanStatistics {
    fn rewrap_as_stats_overlay(
        stats: Arc<Statistics>,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        if let Some(ds) = plan.as_any().downcast_ref::<DataSourceExec>() {
            Arc::new(Self {
                inner: Arc::new(ds.clone()),
                aggregate_stats: stats,
            }) as Arc<dyn ExecutionPlan>
        } else {
            plan
        }
    }
}

impl DisplayAs for DataSourceExecWithScanStatistics {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        self.inner.fmt_as(t, f)
    }
}

impl ExecutionPlan for DataSourceExecWithScanStatistics {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.inner.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inner.children()
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        self.inner.apply_expressions(f)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!(
                "DataSourceExecWithScanStatistics does not support children"
            );
        }
        Ok(self)
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        config: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let stats = Arc::clone(&self.aggregate_stats);
        Ok(self
            .inner
            .repartitioned(target_partitions, config)?
            .map(|p| Self::rewrap_as_stats_overlay(stats, p)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.inner.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.inner.metrics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        match partition {
            None => Ok(Arc::clone(&self.aggregate_stats)),
            Some(_) => self.inner.partition_statistics(partition),
        }
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let stats = Arc::clone(&self.aggregate_stats);
        self.inner
            .with_fetch(limit)
            .map(|p| Self::rewrap_as_stats_overlay(stats, p))
    }

    fn fetch(&self) -> Option<usize> {
        self.inner.fetch()
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let stats = Arc::clone(&self.aggregate_stats);
        Ok(self
            .inner
            .try_swapping_with_projection(projection)?
            .map(|p| Self::rewrap_as_stats_overlay(stats, p)))
    }

    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let stats = Arc::clone(&self.aggregate_stats);
        let mut out = self.inner.handle_child_pushdown_result(
            phase,
            child_pushdown_result,
            config,
        )?;
        if let Some(node) = out.updated_node.take() {
            out.updated_node = Some(Self::rewrap_as_stats_overlay(stats, node));
        }
        Ok(out)
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        let stats = Arc::clone(&self.aggregate_stats);
        Ok(self
            .inner
            .try_pushdown_sort(order)?
            .map(|p| Self::rewrap_as_stats_overlay(stats, p)))
    }

    fn with_preserve_order(
        &self,
        preserve_order: bool,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        let stats = Arc::clone(&self.aggregate_stats);
        self.inner
            .with_preserve_order(preserve_order)
            .map(|p| Self::rewrap_as_stats_overlay(stats, p))
    }

    fn with_new_state(
        &self,
        state: Arc<dyn Any + Send + Sync>,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        let stats = Arc::clone(&self.aggregate_stats);
        self.inner
            .with_new_state(state)
            .map(|p| Self::rewrap_as_stats_overlay(stats, p))
    }
}

/// If `plan` is a single [`DataSourceExec`] over [`MemorySourceConfig`], attach aggregate stats.
fn attach_memory_scan_statistics(
    plan: Arc<dyn ExecutionPlan>,
    full_stats: &Statistics,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(ds) = plan.as_any().downcast_ref::<DataSourceExec>() {
        let src = ds.data_source();
        if let Some(mem) = src.as_any().downcast_ref::<MemorySourceConfig>() {
            let stats = match mem.projection() {
                None => full_stats.clone(),
                Some(proj) => project_scan_statistics(full_stats, proj),
            };
            return Ok(Arc::new(DataSourceExecWithScanStatistics {
                inner: Arc::new(ds.clone()),
                aggregate_stats: Arc::new(stats),
            }) as Arc<dyn ExecutionPlan>);
        }
    }
    Ok(plan)
}

/// Wraps a plain [`MemTable`] and injects table statistics into the memory scan plan only.
#[derive(Debug)]
struct MemTableWithScanStatistics {
    inner: Arc<MemTable>,
    statistics: Statistics,
}

#[async_trait]
impl TableProvider for MemTableWithScanStatistics {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.inner.constraints()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    fn statistics(&self) -> Option<Statistics> {
        Some(self.statistics.clone())
    }

    fn get_column_default(&self, column: &str) -> Option<&Expr> {
        self.inner.get_column_default(column)
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = self.inner.scan(state, projection, filters, limit).await?;
        attach_memory_scan_statistics(plan, &self.statistics)
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.insert_into(state, input, insert_op).await
    }

    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.delete_from(state, filters).await
    }

    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.update(state, assignments, filters).await
    }
}

/// Create test data for window TopK benchmarks
/// Each category has `rows_per_category` rows with random revenue values
fn make_window_data(
    rows_per_category: usize,
    num_categories: usize,
) -> Result<(SchemaRef, Vec<Vec<RecordBatch>>)> {
    let num_partitions = PARTITIONS;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("product_id", DataType::Int64, false),
        Field::new("revenue", DataType::Int64, false),
    ]));

    let total_rows = num_categories * rows_per_category;

    // Step 1: Generate all rows
    let mut all_rows: Vec<(String, i64, i64)> = Vec::with_capacity(total_rows);
    for category_id in 0..num_categories {
        let category = format!("category_{:04}", category_id);
        for row in 0..rows_per_category {
            let product_id = (category_id * rows_per_category + row) as i64;
            let revenue = rng.random_range(1000..10000000);
            all_rows.push((category.clone(), product_id, revenue));
        }
    }

    // Step 2: Shuffle all rows randomly
    all_rows.shuffle(&mut rng);

    // Step 3: Split shuffled rows into partitions
    let rows_per_partition = total_rows / num_partitions;
    let mut partitions = Vec::new();

    for chunk in all_rows.chunks(rows_per_partition) {
        let mut category_builder = Vec::new();
        let mut product_id_builder = Int64Builder::new();
        let mut revenue_builder = Int64Builder::new();

        for (category, product_id, revenue) in chunk {
            category_builder.push(category.clone());
            product_id_builder.append_value(*product_id);
            revenue_builder.append_value(*revenue);
        }

        let category_array = Arc::new(StringArray::from(category_builder));
        let product_id_array = Arc::new(product_id_builder.finish());
        let revenue_array = Arc::new(revenue_builder.finish());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![category_array, product_id_array, revenue_array],
        )?;
        partitions.push(vec![batch]);
    }

    Ok((schema, partitions))
}

/// Aggregate statistics for the `sales` scan: row count and exact distinct categories.
fn make_table_statistics(
    schema: &SchemaRef,
    total_rows: usize,
    num_categories: usize,
) -> Statistics {
    let mut column_statistics = Statistics::unknown_column(schema);
    column_statistics[0] = column_statistics[0]
        .clone()
        .with_distinct_count(Precision::Exact(num_categories));
    Statistics {
        num_rows: Precision::Exact(total_rows),
        total_byte_size: Precision::Absent,
        column_statistics,
    }
}

async fn create_context(
    rows_per_category: usize,
    num_categories: usize,
    use_window_topk: bool,
    window_topk_min_partition_ndv: usize,
) -> Result<SessionContext> {
    let (schema, parts) = make_window_data(rows_per_category, num_categories)?;
    let total_rows = num_categories * rows_per_category;
    let stats = make_table_statistics(&schema, total_rows, num_categories);
    let mem = MemTable::try_new(schema, parts)?;
    let table = Arc::new(MemTableWithScanStatistics {
        inner: Arc::new(mem),
        statistics: stats,
    });

    let mut cfg = SessionConfig::new();
    let opts = cfg.options_mut();
    opts.optimizer.enable_window_topk = use_window_topk;
    opts.optimizer.window_topk_min_partition_ndv = window_topk_min_partition_ndv;
    let ctx = SessionContext::new_with_config(cfg);
    ctx.register_table("sales", table)?;

    Ok(ctx)
}

fn run_window_topk(
    rt: &Runtime,
    ctx: SessionContext,
    limit: usize,
    use_window_topk: bool,
    num_categories: usize,
    rows_per_category: usize,
    window_topk_min_partition_ndv: usize,
) {
    black_box(rt.block_on(async {
        window_topk_query(
            ctx,
            limit,
            use_window_topk,
            num_categories,
            rows_per_category,
            window_topk_min_partition_ndv,
        )
        .await
    }))
    .unwrap();
}

async fn window_topk_query(
    ctx: SessionContext,
    limit: usize,
    use_window_topk: bool,
    _num_categories: usize,
    rows_per_category: usize,
    window_topk_min_partition_ndv: usize,
) -> Result<()> {
    let sql = format!(
        "SELECT category, product_id, revenue
         FROM (
             SELECT
                 category,
                 product_id,
                 revenue,
                 ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) as rn
             FROM sales
         ) t
         WHERE rn <= {limit}"
    );

    let df = ctx.sql(&sql).await?;
    let plan = df.create_physical_plan().await?;
    let actual_phys_plan = displayable(plan.as_ref()).indent(true).to_string();

    // Mirrors `should_apply_partitioned_topk_sort`: with our stats, G ≈ num_categories and
    // N = num_categories * rows_per_category, so G * limit * T <= N iff limit * T <= rows_per_category
    // when num_categories > 0; T <= 1 disables the gate.
    let expect_partitioned_topk = use_window_topk
        && (window_topk_min_partition_ndv <= 1
            || limit * window_topk_min_partition_ndv <= rows_per_category);
    let has_partitioned_topk = actual_phys_plan.contains("PartitionedTopKSortExec");
    assert_eq!(
        has_partitioned_topk, expect_partitioned_topk,
        "Expected PartitionedTopKSortExec: {}, Found: {} (plan:\n{actual_phys_plan})",
        expect_partitioned_topk, has_partitioned_topk
    );

    let batches = collect(plan, ctx.task_ctx()).await?;

    // Verify we get the correct number of rows (limit per category)
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(
        total_rows > 0,
        "Expected at least some rows, got {}",
        total_rows
    );

    Ok(())
}

/// Helper to benchmark a single scenario with and without optimization
fn bench_case(
    c: &mut Criterion,
    rt: &Runtime,
    label: &str,
    num_categories: usize,
    rows_per_category: usize,
    window_topk_min_partition_ndv: usize,
) {
    let ctx_no_opt = rt
        .block_on(create_context(
            rows_per_category,
            num_categories,
            false,
            window_topk_min_partition_ndv,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "{label} ({num_categories} categories, {rows_per_category} rows per category) [no optimization, window_topk_min_partition_ndv={window_topk_min_partition_ndv}]"
        ),
        |b| {
            b.iter(|| {
                run_window_topk(
                    rt,
                    ctx_no_opt.clone(),
                    LIMIT,
                    false,
                    num_categories,
                    rows_per_category,
                    window_topk_min_partition_ndv,
                );
            })
        },
    );

    let ctx_with_opt = rt
        .block_on(create_context(
            rows_per_category,
            num_categories,
            true,
            window_topk_min_partition_ndv,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "{label} ({num_categories} categories, {rows_per_category} rows per category) [window_topk on, window_topk_min_partition_ndv={window_topk_min_partition_ndv}]"
        ),
        |b| {
            b.iter(|| {
                run_window_topk(
                    rt,
                    ctx_with_opt.clone(),
                    LIMIT,
                    true,
                    num_categories,
                    rows_per_category,
                    window_topk_min_partition_ndv,
                );
            })
        },
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    // `datafusion.optimizer.window_topk_min_partition_ndv` (T): with table stats on this bench,
    // the rule uses G * K * T <= N (K = LIMIT below, N from stats). Here G ≈ num_categories,
    // N = num_categories * rows_per_category, so PartitionedTopK applies when K * T <= rows_per_category.
    const WINDOW_TOPK_MIN_PARTITION_NDV: usize = 10;

    // Case 1: Small number of categories with many rows
    bench_case(
        c,
        &rt,
        "few categories, many rows",
        10,
        100_000,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );
    bench_case(
        c,
        &rt,
        "few categories, many rows",
        50,
        100_000,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );
    bench_case(
        c,
        &rt,
        "few categories, many rows",
        100,
        100_000,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );
    bench_case(
        c,
        &rt,
        "few categories, many rows",
        1_000,
        100_000,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );

    // Case 2: Many number of categories with many rows
    bench_case(
        c,
        &rt,
        "many categories, many rows",
        10_000,
        10_000,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );

    // Case 2: Large number of categories with few rows per category
    bench_case(
        c,
        &rt,
        "many categories, few rows",
        100_000,
        1_000,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );
    bench_case(
        c,
        &rt,
        "many categories, few rows",
        100_000,
        100,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );
    bench_case(
        c,
        &rt,
        "many categories, few rows",
        100_000,
        50,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );
    bench_case(
        c,
        &rt,
        "many categories, few rows",
        100_000,
        10,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );
    bench_case(
        c,
        &rt,
        "many categories, few rows",
        100_000,
        1,
        WINDOW_TOPK_MIN_PARTITION_NDV,
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
}
criterion_main!(benches);
