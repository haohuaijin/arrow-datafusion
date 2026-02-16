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

use arrow::array::{Int64Builder, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::SessionContext;
use datafusion::{datasource::MemTable, error::Result};
use datafusion_execution::config::SessionConfig;
use rand::Rng;
use rand::SeedableRng;
use rand::prelude::SliceRandom;
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::Runtime;

const LIMIT: usize = 10;
const PARTITIONS: usize = 10;

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

async fn create_context(
    rows_per_category: usize,
    num_categories: usize,
    use_window_topk: bool,
) -> Result<SessionContext> {
    let (schema, parts) = make_window_data(rows_per_category, num_categories)?;
    let mem_table = Arc::new(MemTable::try_new(schema, parts)?);

    let mut cfg = SessionConfig::new();
    let opts = cfg.options_mut();
    opts.optimizer.enable_window_topk_pushdown = use_window_topk;
    let ctx = SessionContext::new_with_config(cfg);
    ctx.register_table("sales", mem_table)?;

    Ok(ctx)
}

fn run_window_topk(
    rt: &Runtime,
    ctx: SessionContext,
    limit: usize,
    use_window_topk: bool,
) {
    black_box(
        rt.block_on(async { window_topk_query(ctx, limit, use_window_topk).await }),
    )
    .unwrap();
}

async fn window_topk_query(
    ctx: SessionContext,
    limit: usize,
    use_window_topk: bool,
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

    // Verify that PartitionedTopKSortExec is used when optimization is enabled
    let has_partitioned_topk = actual_phys_plan.contains("PartitionedTopKSortExec");
    assert_eq!(
        has_partitioned_topk, use_window_topk,
        "Expected PartitionedTopKSortExec: {}, Found: {}",
        use_window_topk, has_partitioned_topk
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
) {
    let ctx_no_opt = rt
        .block_on(create_context(rows_per_category, num_categories, false))
        .unwrap();
    c.bench_function(
        &format!(
            "{label} ({num_categories} categories, {rows_per_category} rows per category) [no optimization]"
        ),
        |b| {
            b.iter(|| {
                run_window_topk(rt, ctx_no_opt.clone(), LIMIT, false);
            })
        },
    );

    let ctx_with_opt = rt
        .block_on(create_context(rows_per_category, num_categories, true))
        .unwrap();
    c.bench_function(
        &format!(
            "{label} ({num_categories} categories, {rows_per_category} rows per category) [PartitionedTopKSortExec]"
        ),
        |b| {
            b.iter(|| {
                run_window_topk(rt, ctx_with_opt.clone(), LIMIT, true);
            })
        },
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Case 1: Small number of categories, each with many rows
    // 10 categories × 1,000,000 rows = 10M total rows
    bench_case(c, &rt, "few categories, many rows", 10, 1_000_000);

    // Case 2: Large number of categories, each with only 10 rows
    // 100,000 categories × 10 rows = 1M total rows
    bench_case(c, &rt, "many categories, few rows", 100_000, 10);

    // Case 3: Large number of categories with varying rows per category
    // 100,000 categories × {500, 100, 50, 20} rows = {50M, 10M, 5M, 2M} total rows
    // Tests how the optimization scales as rows-per-category decreases
    bench_case(c, &rt, "many categories, many rows", 10_000, 500);
    bench_case(c, &rt, "many categories, many rows", 10_000, 200);
    bench_case(c, &rt, "many categories, many rows", 10_000, 100);
    bench_case(c, &rt, "many categories, many rows", 10_000, 50);
    bench_case(c, &rt, "many categories, many rows", 10_000, 20);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
