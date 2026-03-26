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

//! Compare window TopK execution with and without the optimizer pushdown.
//!
//! Usage:
//! `cargo run -p datafusion-examples --example query -- [num_categories] [rows_per_category] [use_window_topk]`
//!
//! Defaults to the benchmark scenario that previously lived in
//! `datafusion/core/benches/window_topk.rs`:
//! `100000 categories, 1000 rows/category, limit 10`.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Int64Builder, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::basic::Compression;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::prelude::ParquetReadOptions;
use datafusion::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand::prelude::SliceRandom;

const DEFAULT_NUM_CATEGORIES: usize = 1_000_000;
const DEFAULT_ROWS_PER_CATEGORY: usize = 1;
const DEFAULT_LIMIT: usize = 10;
const PARTITIONS: usize = 10;

#[tokio::main]
async fn main() -> Result<()> {
    let config = ExampleConfig::from_args(std::env::args().skip(1))?;

    println!(
        "Window TopK enable {}, {} categories, {} rows/category, limit {}",
        config.use_window_topk,
        config.num_categories,
        config.rows_per_category,
        config.limit
    );

    let ctx_with_opt = create_context(
        config.rows_per_category,
        config.num_categories,
        config.use_window_topk,
    )
    .await?;
    run_window_topk(&ctx_with_opt, config.limit).await?;

    Ok(())
}

async fn create_context(
    rows_per_category: usize,
    num_categories: usize,
    use_window_topk: bool,
) -> Result<SessionContext> {
    let mut cfg = SessionConfig::from_env()?;
    cfg.options_mut().optimizer.enable_window_topk = use_window_topk;

    let ctx = SessionContext::new_with_config(cfg);
    let parquet_dir = ensure_parquet_data(rows_per_category, num_categories)?;
    ctx.register_parquet(
        "sales",
        parquet_dir.to_string_lossy().as_ref(),
        ParquetReadOptions::default(),
    )
    .await?;

    Ok(ctx)
}

async fn run_window_topk(ctx: &SessionContext, limit: usize) -> Result<()> {
    let sql = format!(
        "explain analyze SELECT category, product_id, revenue
         FROM (
             SELECT
                 category,
                 product_id,
                 revenue,
                 ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) AS rn
             FROM sales
         ) t
         WHERE rn <= {limit}"
    );

    let start = std::time::Instant::now();
    let _ = ctx.sql(&sql).await?.show().await?;
    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);
    Ok(())
}

fn ensure_parquet_data(
    rows_per_category: usize,
    num_categories: usize,
) -> Result<PathBuf> {
    let parquet_dir = parquet_data_dir(rows_per_category, num_categories);
    let expected_files = partition_paths(&parquet_dir);

    if expected_files.iter().all(|path| path.exists()) {
        println!("Reusing cached Parquet data from {}", parquet_dir.display());
        return Ok(parquet_dir);
    }

    println!("Generating Parquet data in {}", parquet_dir.display());
    fs::create_dir_all(&parquet_dir)?;

    let (schema, partitions) = make_window_data(rows_per_category, num_categories)?;
    for parquet_file in &expected_files {
        if parquet_file.exists() {
            fs::remove_file(parquet_file)?;
        }
    }

    for (partition_idx, batches) in partitions.iter().enumerate() {
        write_partition(&expected_files[partition_idx], Arc::clone(&schema), batches)?;
    }

    Ok(parquet_dir)
}

fn parquet_data_dir(rows_per_category: usize, num_categories: usize) -> PathBuf {
    PathBuf::from(format!(
        "/Users/huaijinhao/gitrepo/window_topk/categories_{num_categories}_rows_per_category_{rows_per_category}"
    ))
}

fn partition_paths(parquet_dir: &Path) -> Vec<PathBuf> {
    (0..PARTITIONS)
        .map(|idx| parquet_dir.join(format!("partition-{idx:02}.parquet")))
        .collect()
}

fn write_partition(
    path: &Path,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> Result<()> {
    let file = File::create(path)?;
    let options = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(options))
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    for batch in batches {
        writer
            .write(batch)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
    }

    writer
        .close()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    Ok(())
}

fn make_window_data(
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
    let mut all_rows: Vec<(String, i64, i64)> = Vec::with_capacity(total_rows);

    for category_id in 0..num_categories {
        let category = format!("category_{category_id:04}");
        for row in 0..rows_per_category {
            let product_id = (category_id * rows_per_category + row) as i64;
            let revenue = rng.random_range(1000..10_000_000);
            all_rows.push((category.clone(), product_id, revenue));
        }
    }

    all_rows.shuffle(&mut rng);

    let rows_per_partition = total_rows.div_ceil(PARTITIONS);
    let partitions = all_rows
        .chunks(rows_per_partition)
        .map(|chunk| {
            let mut category_builder = Vec::with_capacity(chunk.len());
            let mut product_id_builder = Int64Builder::new();
            let mut revenue_builder = Int64Builder::new();

            for (category, product_id, revenue) in chunk {
                category_builder.push(category.clone());
                product_id_builder.append_value(*product_id);
                revenue_builder.append_value(*revenue);
            }

            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringArray::from(category_builder)),
                    Arc::new(product_id_builder.finish()),
                    Arc::new(revenue_builder.finish()),
                ],
            )?;
            Ok(vec![batch])
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((schema, partitions))
}

// Example config
struct ExampleConfig {
    num_categories: usize,
    rows_per_category: usize,
    limit: usize,
    use_window_topk: bool,
}

impl ExampleConfig {
    fn from_args(args: impl Iterator<Item = String>) -> Result<Self> {
        let values = args.collect::<Vec<_>>();
        if values.len() > 3 {
            return Err(DataFusionError::Execution(Self::usage().to_string()));
        }

        Ok(Self {
            num_categories: parse_arg(
                values.first(),
                DEFAULT_NUM_CATEGORIES,
                "num_categories",
            )?,
            rows_per_category: parse_arg(
                values.get(1),
                DEFAULT_ROWS_PER_CATEGORY,
                "rows_per_category",
            )?,
            limit: DEFAULT_LIMIT,
            use_window_topk: parse_bool_arg(values.get(2), true, "use_window_topk")?,
        })
    }

    fn usage() -> &'static str {
        "Usage: cargo run -p datafusion-examples --example query -- [num_categories] [rows_per_category] [use_window_topk]\nDefault limit: 10\nDefault use_window_topk: true"
    }
}

fn parse_arg(arg: Option<&String>, default: usize, name: &str) -> Result<usize> {
    match arg {
        Some(value) => value.parse::<usize>().map_err(|e| {
            DataFusionError::Execution(format!(
                "Invalid {name} value '{value}': {e}. {}",
                ExampleConfig::usage()
            ))
        }),
        None => Ok(default),
    }
}

fn parse_bool_arg(arg: Option<&String>, default: bool, name: &str) -> Result<bool> {
    match arg {
        Some(value) => parse_bool_value(value, name),
        None => Ok(default),
    }
}

fn parse_bool_value(value: &str, name: &str) -> Result<bool> {
    if value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("1")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
    {
        return Ok(true);
    }

    if value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("0")
        || value.eq_ignore_ascii_case("no")
        || value.eq_ignore_ascii_case("off")
    {
        return Ok(false);
    }

    Err(DataFusionError::Execution(format!(
        "Invalid {name} value '{value}'. Expected one of true/false, 1/0, yes/no, or on/off."
    )))
}
