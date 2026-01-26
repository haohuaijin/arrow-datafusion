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
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::SessionContext;
use datafusion::{datasource::MemTable, error::Result};
use datafusion_execution::config::SessionConfig;
use rand::SeedableRng;
use rand::Rng;
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::Runtime;

const LIMIT: usize = 10;
const NUM_CATEGORIES: usize = 100;

/// Create test data for window TopK benchmarks
/// Each category has `rows_per_category` rows with random revenue values
fn make_window_data(
    num_partitions: usize,
    rows_per_category: usize,
    num_categories: usize,
) -> Result<(SchemaRef, Vec<Vec<RecordBatch>>)> {
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("product_id", DataType::Int64, false),
        Field::new("revenue", DataType::Int64, false),
    ]));

    let total_rows = num_categories * rows_per_category;
    let rows_per_partition = total_rows / num_partitions;

    let mut partitions = Vec::new();
    let mut global_row = 0;

    for _ in 0..num_partitions {
        let mut category_builder = Vec::new();
        let mut product_id_builder = Int64Builder::new();
        let mut revenue_builder = Int64Builder::new();

        for _ in 0..rows_per_partition {
            let category_id = global_row / rows_per_category;
            let category = format!("category_{:04}", category_id % num_categories);
            category_builder.push(category);

            product_id_builder.append_value(global_row as i64);
            // Generate revenue values that create interesting TopK scenarios
            let revenue = rng.random_range(1000..1000000);
            revenue_builder.append_value(revenue);

            global_row += 1;
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
    num_partitions: usize,
    rows_per_category: usize,
    num_categories: usize,
    use_window_topk: bool,
) -> Result<SessionContext> {
    let (schema, parts) = make_window_data(num_partitions, rows_per_category, num_categories)?;
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

fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let limit = LIMIT;

    // Small dataset: 10 partitions, 10,000 rows per category, 100 categories = 1M rows
    let partitions_small = 10;
    let rows_per_category_small = 10_000;
    let categories = NUM_CATEGORIES;

    let ctx_no_opt = rt
        .block_on(create_context(
            partitions_small,
            rows_per_category_small,
            categories,
            false,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "window topk {} rows [no optimization]",
            partitions_small * rows_per_category_small * categories
        ),
        |b| {
            b.iter(|| {
                run_window_topk(&rt, ctx_no_opt.clone(), limit, false);
            })
        },
    );

    let ctx_with_opt = rt
        .block_on(create_context(
            partitions_small,
            rows_per_category_small,
            categories,
            true,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "window topk {} rows [PartitionedTopKSortExec]",
            partitions_small * rows_per_category_small * categories
        ),
        |b| {
            b.iter(|| {
                run_window_topk(&rt, ctx_with_opt.clone(), limit, true);
            })
        },
    );

    // Medium dataset: 10 partitions, 50,000 rows per category, 100 categories = 5M rows
    let rows_per_category_medium = 50_000;

    let ctx_no_opt_medium = rt
        .block_on(create_context(
            partitions_small,
            rows_per_category_medium,
            categories,
            false,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "window topk {} rows [no optimization]",
            partitions_small * rows_per_category_medium * categories
        ),
        |b| {
            b.iter(|| {
                run_window_topk(&rt, ctx_no_opt_medium.clone(), limit, false);
            })
        },
    );

    let ctx_with_opt_medium = rt
        .block_on(create_context(
            partitions_small,
            rows_per_category_medium,
            categories,
            true,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "window topk {} rows [PartitionedTopKSortExec]",
            partitions_small * rows_per_category_medium * categories
        ),
        |b| {
            b.iter(|| {
                run_window_topk(&rt, ctx_with_opt_medium.clone(), limit, true);
            })
        },
    );

    // Test with fewer categories but more rows per category
    let categories_few = 10;
    let rows_per_category_many = 100_000;

    let ctx_few_categories_no_opt = rt
        .block_on(create_context(
            partitions_small,
            rows_per_category_many,
            categories_few,
            false,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "window topk {} rows, {} categories [no optimization]",
            partitions_small * rows_per_category_many * categories_few,
            categories_few
        ),
        |b| {
            b.iter(|| {
                run_window_topk(&rt, ctx_few_categories_no_opt.clone(), limit, false);
            })
        },
    );

    let ctx_few_categories_with_opt = rt
        .block_on(create_context(
            partitions_small,
            rows_per_category_many,
            categories_few,
            true,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "window topk {} rows, {} categories [PartitionedTopKSortExec]",
            partitions_small * rows_per_category_many * categories_few,
            categories_few
        ),
        |b| {
            b.iter(|| {
                run_window_topk(&rt, ctx_few_categories_with_opt.clone(), limit, true);
            })
        },
    );

    // Test with many categories but fewer rows per category
    let categories_many = 1000;
    let rows_per_category_few = 1000;

    let ctx_many_categories_no_opt = rt
        .block_on(create_context(
            partitions_small,
            rows_per_category_few,
            categories_many,
            false,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "window topk {} rows, {} categories [no optimization]",
            partitions_small * rows_per_category_few * categories_many,
            categories_many
        ),
        |b| {
            b.iter(|| {
                run_window_topk(&rt, ctx_many_categories_no_opt.clone(), limit, false);
            })
        },
    );

    let ctx_many_categories_with_opt = rt
        .block_on(create_context(
            partitions_small,
            rows_per_category_few,
            categories_many,
            true,
        ))
        .unwrap();
    c.bench_function(
        &format!(
            "window topk {} rows, {} categories [PartitionedTopKSortExec]",
            partitions_small * rows_per_category_few * categories_many,
            categories_many
        ),
        |b| {
            b.iter(|| {
                run_window_topk(&rt, ctx_many_categories_with_opt.clone(), limit, true);
            })
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
