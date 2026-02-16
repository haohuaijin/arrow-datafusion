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

//! Window TopK optimization benchmark.
//!
//! Run with:
//!   cargo run --release --example window_topk -p datafusion

use arrow::array::{Int64Builder, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::SessionContext;
use datafusion_execution::config::SessionConfig;
use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::sync::Arc;

const LIMIT: usize = 10;
const PARTITIONS: usize = 10;

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
    num_categories: usize,
    rows_per_category: usize,
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

async fn run_query(
    ctx: SessionContext,
    limit: usize,
    use_window_topk: bool,
) -> Result<usize> {
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
    let has_partitioned_topk = actual_phys_plan.contains("PartitionedTopKSortExec");
    assert_eq!(
        has_partitioned_topk, use_window_topk,
        "Expected PartitionedTopKSortExec: {}, Found: {}",
        use_window_topk, has_partitioned_topk
    );

    let batches = collect(plan, ctx.task_ctx()).await?;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(
        total_rows > 0,
        "Expected at least some rows, got {}",
        total_rows
    );

    Ok(total_rows)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cases: Vec<(&str, usize, usize)> = vec![
        // Case 1: Small number of categories, each with many rows
        ("few categories, many rows", 10, 1_000_000), //0
        // Case 2: Large number of categories, each with only 10 rows
        ("many categories, few rows", 100_000, 10), //1
        // Case 3: Large number of categories with varying rows per category
        ("many categories, many rows", 100_000, 1_000), //2
        ("many categories, many rows", 100_000, 100),   //3
        ("many categories, many rows", 100_000, 50),    //4
        ("many categories, many rows", 100_000, 20),    //5
        ("many categories, many rows", 100_000, 10_000), //6
        ("many categories, many rows", 100_000, 500),   //7
    ];

    let use_window_topk = std::env::args()
        .nth(1)
        .unwrap_or("true".to_string())
        .parse::<bool>()
        .unwrap();
    let (desc, num_categories, rows_per_category) = cases[2];
    let ctx = create_context(num_categories, rows_per_category, use_window_topk).await?;
    let opt_rows = run_query(ctx, LIMIT, use_window_topk).await?;
    println!("{}: {} rows", desc, opt_rows);

    Ok(())
}
