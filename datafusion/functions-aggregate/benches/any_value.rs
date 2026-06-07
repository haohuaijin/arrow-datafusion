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

//! Compares the grouped-aggregation cost of three ways to pull one value per
//! group out of a `GROUP BY`:
//!
//! * `any_value(tag)` — the lightweight `GroupsAccumulator` added here. Captures
//!   the first non-null value per group and short-circuits every later row.
//! * `max(tag)` — a fast specialized `PrimitiveGroupsAccumulator`, but it still
//!   does a compare-and-maybe-store for *every* row.
//! * `first_value(tag)` *without* an `ORDER BY` — has no `GroupsAccumulator`, so
//!   real execution wraps a per-group `Accumulator` in a
//!   `GroupsAccumulatorAdapter`. This is the heavyweight path `any_value` aims
//!   to replace.
//!
//! All three are driven through the same `GroupsAccumulator::update_batch`
//! interface so the numbers are directly comparable.

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray};
use arrow::datatypes::{DataType, Field, Int64Type, Schema};
use arrow::util::bench_util::{create_boolean_array, create_primitive_array};

use datafusion_expr::{AggregateUDFImpl, GroupsAccumulator, function::AccumulatorArgs};
use datafusion_functions_aggregate::any_value::AnyValue;
use datafusion_functions_aggregate::first_last::TrivialFirstValueAccumulator;
use datafusion_functions_aggregate::min_max::Max;
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::GroupsAccumulatorAdapter;
use datafusion_physical_expr::expressions::col;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

/// Builds `AccumulatorArgs` for `agg(value)` with no `ORDER BY`, over an
/// `Int64` value column.
fn value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        true,
    )]))
}

fn any_value_accumulator(schema: &Arc<Schema>) -> Box<dyn GroupsAccumulator> {
    let value_field: Arc<Field> = Field::new("value", DataType::Int64, true).into();
    let args = AccumulatorArgs {
        return_field: Arc::clone(&value_field),
        schema,
        expr_fields: std::slice::from_ref(&value_field),
        ignore_nulls: false,
        order_bys: &[],
        is_reversed: false,
        name: "any_value(value)",
        is_distinct: false,
        exprs: &[col("value", schema).unwrap()],
    };
    AnyValue::new().create_groups_accumulator(args).unwrap()
}

fn max_accumulator(schema: &Arc<Schema>) -> Box<dyn GroupsAccumulator> {
    let value_field: Arc<Field> = Field::new("value", DataType::Int64, true).into();
    let args = AccumulatorArgs {
        return_field: Arc::clone(&value_field),
        schema,
        expr_fields: std::slice::from_ref(&value_field),
        ignore_nulls: false,
        order_bys: &[],
        is_reversed: false,
        name: "max(value)",
        is_distinct: false,
        exprs: &[col("value", schema).unwrap()],
    };
    Max::new().create_groups_accumulator(args).unwrap()
}

/// `first_value` without `ORDER BY` has no native `GroupsAccumulator`, so model
/// what the planner actually does: wrap the trivial per-group accumulator in a
/// `GroupsAccumulatorAdapter`.
fn first_value_no_order_accumulator() -> Box<dyn GroupsAccumulator> {
    Box::new(GroupsAccumulatorAdapter::new(|| {
        Ok(Box::new(TrivialFirstValueAccumulator::try_new(
            &DataType::Int64,
            // first_value defaults to RESPECT NULLS; any_value ignores nulls.
            // Use ignore_nulls=false here to reflect first_value's default.
            false,
        )?))
    }))
}

#[expect(clippy::needless_pass_by_value)]
fn update_bench(
    c: &mut Criterion,
    name: &str,
    mut make: impl FnMut() -> Box<dyn GroupsAccumulator>,
    values: ArrayRef,
    opt_filter: Option<&BooleanArray>,
    num_groups: usize,
) {
    let n = values.len();
    let group_indices: Vec<usize> = (0..n).map(|i| i % num_groups).collect();

    c.bench_function(name, |b| {
        b.iter_batched(
            &mut make,
            |mut accumulator| {
                // Feed the same batch 100 times: after the first pass every group
                // is populated, which is exactly where `any_value`'s short-circuit
                // and `first_value`'s adapter overhead diverge most.
                for _ in 0..100 {
                    #[expect(clippy::unit_arg)]
                    black_box(
                        accumulator
                            .update_batch(
                                std::slice::from_ref(&values),
                                &group_indices,
                                opt_filter,
                                num_groups,
                            )
                            .unwrap(),
                    );
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn any_value_benchmark(c: &mut Criterion) {
    const N: usize = 65536;

    let schema = value_schema();

    for num_groups in [128_usize, 1024, 8192] {
        for pct in [0, 90] {
            let null_density = (pct as f32) / 100.0;
            let values = Arc::new(create_primitive_array::<Int64Type>(N, null_density))
                as ArrayRef;

            for with_filter in [false, true] {
                let filter = create_boolean_array(N, 0.0, 0.5);
                let opt_filter = if with_filter { Some(&filter) } else { None };
                let suffix =
                    format!("groups={num_groups}, nulls={pct}%, filter={with_filter}");

                update_bench(
                    c,
                    &format!("any_value update_bench {suffix}"),
                    {
                        let schema = Arc::clone(&schema);
                        move || any_value_accumulator(&schema)
                    },
                    Arc::clone(&values),
                    opt_filter,
                    num_groups,
                );
                update_bench(
                    c,
                    &format!("max update_bench {suffix}"),
                    {
                        let schema = Arc::clone(&schema);
                        move || max_accumulator(&schema)
                    },
                    Arc::clone(&values),
                    opt_filter,
                    num_groups,
                );
                update_bench(
                    c,
                    &format!("first_value(no order) update_bench {suffix}"),
                    first_value_no_order_accumulator,
                    Arc::clone(&values),
                    opt_filter,
                    num_groups,
                );
            }
        }
    }
}

criterion_group!(benches, any_value_benchmark);
criterion_main!(benches);
