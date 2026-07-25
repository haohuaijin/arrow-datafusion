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

//! Benchmark for [`ParquetAccessPlan`]-aware repartitioning.
//!
//! Simulates the external-index use case: an index has narrowed a scan down
//! to a sparse set of rows (attached to the files as `ParquetAccessPlan`s),
//! and the scan decodes a wide row for each selected row — e.g. a
//! `SELECT * ... ORDER BY ts DESC LIMIT n` over indexed log data.
//!
//! Compares executing the same scan under two partitionings:
//!
//! * `byte_range`: the plan-blind [`FileGroupPartitioner`] byte split
//!   (the previous behavior)
//! * `access_plan`: [`ParquetSource::repartitioned`], which splits by the
//!   scanned row groups, balanced by estimated scan bytes
//!
//! Both partitionings run with DataFusion's default
//! `repartition_file_min_size` (10 MiB). The generated file is larger than
//! that total-input threshold, so both strategies are eligible to repartition;
//! individual output chunks may be smaller than the threshold.
//!
//! Four selection shapes are measured:
//!
//! * `clustered`: all selected rows in the first quarter of the row groups
//!   (a top-N over time-ordered data), which the byte split serializes
//!   into few partitions
//! * `uniform`: the same number of selected rows spread evenly, where both
//!   partitionings should perform about the same
//! * `sparse`: a handful of rows per row group (~0.01% of the file) — the
//!   external-index point-lookup shape; a byte-fraction estimate of this
//!   scan is a few KiB, yet the decode cost is a full page set per row
//! * `mixed`: two hot row groups with dense selections plus a sparse tail
//!   across the rest, exercising the weight-balanced packing

use std::fs::File;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::object_store::ObjectStoreUrl;
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionContext;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision;
use datafusion_datasource::file_groups::{FileGroup, FileGroupPartitioner};
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_parquet::{ParquetAccessPlan, RowGroupAccess};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

const ROW_GROUPS: usize = 16;
const ROWS_PER_RG: usize = 16384;
const PAYLOAD_BYTES: usize = 256;
const TARGET_PARTITIONS: usize = 8;
/// selected rows per scanned row group, scattered so that every data page
/// of the row group is touched
const SELECTED_PER_RG: usize = 32;
/// DataFusion's default `datafusion.optimizer.repartition_file_min_size`
const DEFAULT_MIN_SIZE: usize = 10 * 1024 * 1024;

fn file_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]))
}

/// Write one parquet file with `ROW_GROUPS` row groups of `ROWS_PER_RG`
/// rows. The payload column is incompressible so decode cost resembles a
/// wide log row.
fn write_test_file(path: &std::path::Path) {
    let schema = file_schema();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_RG))
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    for rg in 0..ROW_GROUPS {
        let ts: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (0..ROWS_PER_RG).map(|i| (rg * ROWS_PER_RG + i) as i64),
        ));
        let payload: ArrayRef =
            Arc::new(StringArray::from_iter_values((0..ROWS_PER_RG).map(|_| {
                (0..PAYLOAD_BYTES / 2)
                    .map(|_| format!("{:02x}", rng.random::<u8>()))
                    .collect::<String>()
            })));
        let batch = RecordBatch::try_new(schema.clone(), vec![ts, payload]).unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap(); // one row group per batch
    }
    writer.close().unwrap();
}

/// A selection of `rows` rows evenly scattered within one row group, so
/// each selected row lands on a different region of the pages.
fn scattered_selection(rows: usize) -> RowGroupAccess {
    let step = ROWS_PER_RG / rows;
    let mut selectors = Vec::new();
    for _ in 0..rows {
        selectors.push(RowSelector::skip(step - 1));
        selectors.push(RowSelector::select(1));
    }
    RowGroupAccess::Selection(RowSelection::from(selectors))
}

/// See the module docs for the scenario shapes. Returns the access plan and
/// the total number of selected rows.
fn scenario_plan(scenario: &str) -> (ParquetAccessPlan, usize) {
    let mut plan = ParquetAccessPlan::new_none(ROW_GROUPS);
    let mut selected = 0;
    match scenario {
        "clustered" => {
            for rg in 0..ROW_GROUPS / 4 {
                plan.set(rg, scattered_selection(SELECTED_PER_RG));
                selected += SELECTED_PER_RG;
            }
        }
        "uniform" => {
            for rg in 0..ROW_GROUPS {
                plan.set(rg, scattered_selection(SELECTED_PER_RG / 4));
                selected += SELECTED_PER_RG / 4;
            }
        }
        "sparse" => {
            // ~0.01% of the file: a byte-fraction estimate of this scan is
            // a few KiB, but every selected row decodes a full page set
            for rg in 0..ROW_GROUPS {
                plan.set(rg, scattered_selection(2));
                selected += 2;
            }
        }
        "mixed" => {
            // two hot row groups plus a sparse tail
            for rg in 0..ROW_GROUPS {
                let rows = if rg < 2 { SELECTED_PER_RG * 2 } else { 2 };
                plan.set(rg, scattered_selection(rows));
                selected += rows;
            }
        }
        other => unreachable!("unknown scenario {other}"),
    }
    (plan, selected)
}

fn base_config(path: &std::path::Path, plan: ParquetAccessPlan) -> FileScanConfig {
    let size = std::fs::metadata(path).unwrap().len();
    let mut file = PartitionedFile::new(path.to_str().unwrap().to_string(), size);
    let mut stats = Statistics::new_unknown(&file_schema());
    stats.num_rows = Precision::Exact(ROW_GROUPS * ROWS_PER_RG);
    file.statistics = Some(Arc::new(stats));
    file.extensions.insert(plan);

    let source = Arc::new(ParquetSource::new(file_schema()));
    FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
        .with_file_group(FileGroup::new(vec![file]))
        .with_projection_indices(Some(vec![0, 1]))
        .expect("projection pushdown")
        .build()
}

/// The previous behavior: plan-blind byte-range split.
fn byte_range_split(config: &FileScanConfig) -> FileScanConfig {
    let file_groups = FileGroupPartitioner::new()
        .with_target_partitions(TARGET_PARTITIONS)
        .with_repartition_file_min_size(DEFAULT_MIN_SIZE)
        .repartition_file_groups(&config.file_groups)
        .expect("byte split should apply");
    let mut config = config.clone();
    config.file_groups = file_groups;
    config
}

/// The new behavior: split by the scanned row groups of the access plan.
fn access_plan_split(config: &FileScanConfig) -> FileScanConfig {
    config
        .file_source()
        .repartitioned(TARGET_PARTITIONS, DEFAULT_MIN_SIZE, None, config)
        .unwrap()
        .expect("access plan split should apply")
}

/// Execute one freshly built scan (mirroring a real query lifecycle, where
/// every query plans its own exec) and return the decoded row count.
fn run_scan(
    rt: &tokio::runtime::Runtime,
    task_ctx: &Arc<datafusion::execution::TaskContext>,
    config: &FileScanConfig,
) -> usize {
    let exec = DataSourceExec::from_data_source(config.clone());
    rt.block_on(collect(exec, Arc::clone(task_ctx)))
        .unwrap()
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

fn criterion_benchmark(c: &mut Criterion) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("data.parquet");
    write_test_file(&path);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(TARGET_PARTITIONS)
        .build()
        .unwrap();
    let task_ctx = SessionContext::new().task_ctx();

    let mut group = c.benchmark_group("parquet_scan_access_plan");
    group.sample_size(20);

    for scenario in ["clustered", "uniform", "sparse", "mixed"] {
        let (plan, selected) = scenario_plan(scenario);
        let config = base_config(&path, plan);
        let byte_config = byte_range_split(&config);
        let plan_config = access_plan_split(&config);

        // both partitionings must decode exactly the selected rows
        assert_eq!(run_scan(&rt, &task_ctx, &byte_config), selected);
        assert_eq!(run_scan(&rt, &task_ctx, &plan_config), selected);

        group.bench_function(format!("{scenario}/byte_range"), |b| {
            b.iter(|| {
                assert_eq!(run_scan(&rt, &task_ctx, &byte_config), selected);
            })
        });
        group.bench_function(format!("{scenario}/access_plan"), |b| {
            b.iter(|| {
                assert_eq!(run_scan(&rt, &task_ctx, &plan_config), selected);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
