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

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::MemTable;
use datafusion::common::{Result, assert_batches_eq};
use datafusion::prelude::SessionContext;
use std::sync::Arc;

pub async fn window_topk_optimizer() -> Result<()> {
    let ctx = SessionContext::new();

    let mem_table = Arc::new(
        MemTable::try_new(
            sales_data_partition_0().schema(),
            vec![
                vec![sales_data_partition_0()],
                vec![sales_data_partition_1()],
            ],
        )
        .unwrap(),
    );

    ctx.register_table("sales", Arc::clone(&mem_table) as _)?;

    let sql = r#"
        SELECT category, product, revenue
        FROM (
            SELECT
                category,
                product,
                revenue,
                ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) as rn
            FROM sales
        ) t
        WHERE rn <= 3
        ORDER BY category, revenue DESC
    "#;

    let dataframe = ctx.sql(sql).await?;
    let physical_plan = dataframe.clone().create_physical_plan().await?;
    println!("Physical Plan (optimized):");
    println!(
        "{}\n",
        datafusion::physical_plan::displayable(physical_plan.as_ref()).indent(true)
    );

    // Execute and show results
    let results = dataframe.collect().await?;
    println!("Results:");
    assert_batches_eq!(
        [
            "+-------------+---------+---------+",
            "| category    | product | revenue |",
            "+-------------+---------+---------+",
            "| Clothing    | Jacket  | 120     |",
            "| Clothing    | Shoes   | 90      |",
            "| Clothing    | Pants   | 70      |",
            "| Electronics | Laptop  | 1000    |",
            "| Electronics | Phone   | 800     |",
            "| Electronics | Tablet  | 600     |",
            "| Furniture   | Desk    | 500     |",
            "| Furniture   | Cabinet | 400     |",
            "| Furniture   | Chair   | 300     |",
            "+-------------+---------+---------+",
        ],
        &results
    );

    // Demonstrate disabling the optimization
    println!("\n=== Configuration Demo ===\n");
    println!("Disabling window TopK pushdown optimization...\n");

    let ctx_no_opt = SessionContext::new();
    ctx_no_opt
        .state_ref()
        .write()
        .config_mut()
        .options_mut()
        .optimizer
        .enable_window_topk_pushdown = false;

    ctx_no_opt.register_table("sales", Arc::clone(&mem_table) as _)?;

    let dataframe_no_opt = ctx_no_opt.sql(sql).await?;
    let physical_plan_no_opt = dataframe_no_opt.clone().create_physical_plan().await?;
    println!("Physical Plan (optimization disabled):");
    println!(
        "{}\n",
        datafusion::physical_plan::displayable(physical_plan_no_opt.as_ref())
            .indent(true)
    );
    println!(
        "Note: When disabled, the plan uses regular SortExec instead of PartitionedTopKSortExec"
    );

    Ok(())
}

/// Create sample sales data for partition 0
fn sales_data_partition_0() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("product", DataType::Utf8, false),
        Field::new("revenue", DataType::Int32, false),
    ]));

    let category = StringArray::from(vec![
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Furniture",
        "Furniture",
    ]);

    let product = StringArray::from(vec![
        "Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Desk", "Chair",
    ]);

    let revenue = Int32Array::from(vec![1000, 800, 600, 400, 200, 500, 300]);

    RecordBatch::try_new(
        schema,
        vec![Arc::new(category), Arc::new(product), Arc::new(revenue)],
    )
    .unwrap()
}

/// Create sample sales data for partition 1
fn sales_data_partition_1() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("product", DataType::Utf8, false),
        Field::new("revenue", DataType::Int32, false),
    ]));

    let category = StringArray::from(vec![
        "Furniture",
        "Furniture",
        "Furniture",
        "Clothing",
        "Clothing",
        "Clothing",
        "Clothing",
        "Clothing",
    ]);

    let product = StringArray::from(vec![
        "Lamp", "Shelf", "Cabinet", "Shirt", "Pants", "Shoes", "Hat", "Jacket",
    ]);

    let revenue = Int32Array::from(vec![150, 250, 400, 50, 70, 90, 30, 120]);

    RecordBatch::try_new(
        schema,
        vec![Arc::new(category), Arc::new(product), Arc::new(revenue)],
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_window_topk_optimizer() -> Result<()> {
        window_topk_optimizer().await
    }
}
