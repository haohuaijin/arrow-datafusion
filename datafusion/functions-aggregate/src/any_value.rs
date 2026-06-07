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

//! Defines the ANY_VALUE aggregation.
//!
//! `any_value` returns an arbitrary (non-deterministic) value from each
//! aggregation group, ignoring `NULL`s. Unlike `first_value`/`last_value`, it
//! carries no notion of ordering, so its state is just "one value, set once":
//! the per-group accumulator captures the first non-null value it sees and
//! short-circuits all subsequent rows for that group. This makes it
//! significantly lighter than `first_value` without an `ORDER BY`, which falls
//! back to a per-group [`Accumulator`] wrapped in a `GroupsAccumulatorAdapter`.

use std::fmt::Debug;
use std::hash::Hash;
use std::mem::size_of_val;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, BooleanBufferBuilder};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, Decimal32Type, Decimal64Type, Decimal128Type,
    Decimal256Type, Field, FieldRef, Float16Type, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use datafusion_common::{Result, ScalarValue, internal_err, not_impl_err};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, EmitTo, Expr, GroupsAccumulator,
    ReversedUDAF, Signature, Volatility,
};
use datafusion_macros::user_doc;

// Reuse the per-group value storage that `first_last` already implements.
use crate::first_last::state::{
    BytesValueState, PrimitiveValueState, ValueState, take_need,
};

create_func!(AnyValue, any_value_udaf);

/// Returns an arbitrary value from a group of values.
pub fn any_value(expression: Expr) -> Expr {
    any_value_udaf().call(vec![expression])
}

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns an arbitrary (non-deterministic) value from the group. \
        `NULL` values are ignored, so a `NULL` is only returned when every value \
        in the group is `NULL`. Unlike `first_value`, no ordering is tracked, \
        which makes it cheaper for cases where any single value will do.",
    syntax_example = "any_value(expression)",
    sql_example = r#"```sql
> SELECT any_value(column_name) FROM table_name;
+--------------------------+
| any_value(column_name)   |
+--------------------------+
| some_value               |
+--------------------------+
```"#,
    standard_argument(name = "expression",)
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AnyValue {
    signature: Signature,
}

impl Default for AnyValue {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyValue {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for AnyValue {
    fn name(&self) -> &str {
        "any_value"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        not_impl_err!("Not called because return_field is implemented")
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        // Preserve metadata from the input field; always nullable since the
        // group may be empty or contain only NULLs.
        Ok(Arc::new(
            Field::new(self.name(), arg_fields[0].data_type().clone(), true)
                .with_metadata(arg_fields[0].metadata().clone()),
        ))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        AnyValueAccumulator::try_new(acc_args.return_field.data_type())
            .map(|acc| Box::new(acc) as _)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new(
                format_state_name(args.name, "any_value"),
                args.return_type().clone(),
                true,
            )
            .into(),
            Field::new(
                format_state_name(args.name, "any_value_is_set"),
                DataType::Boolean,
                true,
            )
            .into(),
        ])
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        groups_accumulator_supported(args.return_field.data_type())
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        create_groups_accumulator(args.return_field.data_type(), self.name())
    }

    fn reverse_expr(&self) -> ReversedUDAF {
        // Order does not matter for an arbitrary value.
        ReversedUDAF::Identical
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// Returns `true` if [`AnyValueGroupsAccumulator`] supports `data_type`.
fn groups_accumulator_supported(data_type: &DataType) -> bool {
    use DataType::*;
    matches!(
        data_type,
        Int8 | Int16
            | Int32
            | Int64
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | Float16
            | Float32
            | Float64
            | Decimal32(_, _)
            | Decimal64(_, _)
            | Decimal128(_, _)
            | Decimal256(_, _)
            | Date32
            | Date64
            | Time32(_)
            | Time64(_)
            | Timestamp(_, _)
            | Utf8
            | LargeUtf8
            | Utf8View
            | Binary
            | LargeBinary
            | BinaryView
    )
}

fn create_groups_accumulator(
    data_type: &DataType,
    function_name: &str,
) -> Result<Box<dyn GroupsAccumulator>> {
    macro_rules! instantiate_primitive {
        ($t:ty) => {
            Ok(Box::new(AnyValueGroupsAccumulator::new(
                PrimitiveValueState::<$t>::new(data_type.clone()),
            )))
        };
    }

    match data_type {
        DataType::Int8 => instantiate_primitive!(Int8Type),
        DataType::Int16 => instantiate_primitive!(Int16Type),
        DataType::Int32 => instantiate_primitive!(Int32Type),
        DataType::Int64 => instantiate_primitive!(Int64Type),
        DataType::UInt8 => instantiate_primitive!(UInt8Type),
        DataType::UInt16 => instantiate_primitive!(UInt16Type),
        DataType::UInt32 => instantiate_primitive!(UInt32Type),
        DataType::UInt64 => instantiate_primitive!(UInt64Type),
        DataType::Float16 => instantiate_primitive!(Float16Type),
        DataType::Float32 => instantiate_primitive!(Float32Type),
        DataType::Float64 => instantiate_primitive!(Float64Type),

        DataType::Decimal32(_, _) => instantiate_primitive!(Decimal32Type),
        DataType::Decimal64(_, _) => instantiate_primitive!(Decimal64Type),
        DataType::Decimal128(_, _) => instantiate_primitive!(Decimal128Type),
        DataType::Decimal256(_, _) => instantiate_primitive!(Decimal256Type),

        DataType::Timestamp(TimeUnit::Second, _) => {
            instantiate_primitive!(TimestampSecondType)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            instantiate_primitive!(TimestampMillisecondType)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            instantiate_primitive!(TimestampMicrosecondType)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            instantiate_primitive!(TimestampNanosecondType)
        }

        DataType::Date32 => instantiate_primitive!(Date32Type),
        DataType::Date64 => instantiate_primitive!(Date64Type),
        DataType::Time32(TimeUnit::Second) => instantiate_primitive!(Time32SecondType),
        DataType::Time32(TimeUnit::Millisecond) => {
            instantiate_primitive!(Time32MillisecondType)
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            instantiate_primitive!(Time64MicrosecondType)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            instantiate_primitive!(Time64NanosecondType)
        }

        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => Ok(Box::new(AnyValueGroupsAccumulator::new(
            BytesValueState::try_new(data_type.clone())?,
        ))),

        _ => internal_err!(
            "GroupsAccumulator not supported for {}({})",
            function_name,
            data_type
        ),
    }
}

/// A plain [`Accumulator`] that keeps the first non-null value it sees.
///
/// Once a value has been captured (`is_set == true`) every later batch is
/// skipped entirely.
#[derive(Debug)]
pub struct AnyValueAccumulator {
    value: ScalarValue,
    is_set: bool,
}

impl AnyValueAccumulator {
    pub fn try_new(data_type: &DataType) -> Result<Self> {
        ScalarValue::try_from(data_type).map(|value| Self {
            value,
            is_set: false,
        })
    }
}

impl Accumulator for AnyValueAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.value.clone(), ScalarValue::from(self.is_set)])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if self.is_set {
            return Ok(());
        }
        let array = &values[0];
        for i in 0..array.len() {
            if !array.is_null(i) {
                self.value = ScalarValue::try_from_array(array, i)?;
                self.value.compact();
                self.is_set = true;
                break;
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if self.is_set {
            return Ok(());
        }
        let values = &states[0];
        let is_set = states[1].as_boolean();
        for i in 0..is_set.len() {
            if is_set.value(i) && !values.is_null(i) {
                self.value = ScalarValue::try_from_array(values, i)?;
                self.value.compact();
                self.is_set = true;
                break;
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(self.value.clone())
    }

    fn size(&self) -> usize {
        size_of_val(self) - size_of_val(&self.value) + self.value.size()
    }
}

/// A lightweight [`GroupsAccumulator`] for `any_value`.
///
/// Compared to `first_last`'s groups accumulator, this keeps no ordering
/// columns, no lexicographical comparator and no per-group ordering buffers —
/// just the captured value (`state`) and a single "have we captured anything
/// for this group yet" bit (`is_sets`).
///
/// Two properties keep it cheap, especially in steady state:
///
/// * A group's value never changes once captured, so we track `num_set` and
///   skip a batch entirely the moment every group is populated.
/// * Only non-null rows can ever set a group, so for nullable input we walk the
///   value array's null buffer and visit *only* the non-null positions — a
///   90%-null column does ~10% of the per-row work.
struct AnyValueGroupsAccumulator<S: ValueState> {
    /// Per-group captured value.
    state: S,
    /// `is_sets[g]` is `true` once group `g` has captured a (non-null) value.
    is_sets: BooleanBufferBuilder,
    /// Number of groups whose `is_sets` bit is `true`. Lets us short-circuit a
    /// batch once `num_set == total_num_groups`.
    num_set: usize,
}

impl<S: ValueState> AnyValueGroupsAccumulator<S> {
    fn new(state: S) -> Self {
        Self {
            state,
            is_sets: BooleanBufferBuilder::new(0),
            num_set: 0,
        }
    }

    fn resize(&mut self, total_num_groups: usize) {
        self.state.resize(total_num_groups);
        if self.is_sets.len() < total_num_groups {
            // New groups start out unset, so `num_set` is unchanged.
            self.is_sets.resize(total_num_groups);
        }
    }

    /// Emits the `is_sets` bits for `emit_to` and keeps `num_set` consistent
    /// with the groups that remain.
    fn take_is_sets(&mut self, emit_to: EmitTo) -> BooleanBuffer {
        let emitted = take_need(&mut self.is_sets, emit_to);
        match emit_to {
            EmitTo::All => self.num_set = 0,
            EmitTo::First(_) => self.num_set -= emitted.count_set_bits(),
        }
        emitted
    }
}

impl<S: ValueState + 'static> GroupsAccumulator for AnyValueGroupsAccumulator<S> {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.resize(total_num_groups);
        // Every group already has a value; nothing in this batch can change it.
        if self.num_set == total_num_groups {
            return Ok(());
        }
        let vals = &values[0];

        // Capture the value at `idx` for its group if that group is still unset
        // and the row passes the filter. Expands inline so the borrow checker is
        // happy mutating `self` across the sequential statements. The all-set
        // `break` lives inside the set branch because `num_set` can only change
        // there — checking it on every (mostly skipped) row would be pure
        // overhead when some groups never get a value.
        macro_rules! capture {
            ($idx:expr) => {{
                let idx = $idx;
                let group_idx = group_indices[idx];
                if !self.is_sets.get_bit(group_idx)
                    && opt_filter.is_none_or(|f| f.value(idx))
                {
                    self.state.update(group_idx, vals, idx)?;
                    self.is_sets.set_bit(group_idx, true);
                    self.num_set += 1;
                    if self.num_set == total_num_groups {
                        break;
                    }
                }
            }};
        }

        match vals.nulls() {
            // Nullable input: only non-null rows can set a group, so visit just
            // those positions and skip the null-check (and the nulls) entirely.
            Some(nulls) => {
                for idx in nulls.valid_indices() {
                    capture!(idx);
                }
            }
            // No nulls: scan rows directly, no per-row null check needed.
            None => {
                for idx in 0..group_indices.len() {
                    capture!(idx);
                }
            }
        }
        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        _opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.resize(total_num_groups);
        if self.num_set == total_num_groups {
            return Ok(());
        }
        let vals = &values[0];
        let in_is_set = values[1].as_boolean();

        // A partial state only contributes where it actually captured a value,
        // i.e. where `is_set` is true; drive iteration off those positions.
        for idx in in_is_set.values().set_indices() {
            let group_idx = group_indices[idx];
            if self.is_sets.get_bit(group_idx) {
                continue;
            }
            self.state.update(group_idx, vals, idx)?;
            self.is_sets.set_bit(group_idx, true);
            self.num_set += 1;
            if self.num_set == total_num_groups {
                break;
            }
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        // Keep `is_sets` aligned with `state` across streaming emits.
        let _ = self.take_is_sets(emit_to);
        self.state.take(emit_to)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let is_set: BooleanBuffer = self.take_is_sets(emit_to);
        let values = self.state.take(emit_to)?;
        Ok(vec![values, Arc::new(BooleanArray::new(is_set, None))])
    }

    fn size(&self) -> usize {
        self.state.size() + self.is_sets.capacity() / 8 + size_of_val(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};

    fn int64(values: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(values))
    }

    #[test]
    fn accumulator_picks_first_non_null() -> Result<()> {
        let mut acc = AnyValueAccumulator::try_new(&DataType::Int64)?;
        acc.update_batch(&[int64(vec![None, None, Some(7), Some(9)])])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int64(Some(7)));
        // Once set, later batches are ignored.
        acc.update_batch(&[int64(vec![Some(1)])])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int64(Some(7)));
        Ok(())
    }

    #[test]
    fn accumulator_all_nulls_is_null() -> Result<()> {
        let mut acc = AnyValueAccumulator::try_new(&DataType::Int64)?;
        acc.update_batch(&[int64(vec![None, None])])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int64(None));
        Ok(())
    }

    #[test]
    fn accumulator_merge() -> Result<()> {
        // Two partial states: first all-null (is_set=false), second set to 5.
        let mut acc = AnyValueAccumulator::try_new(&DataType::Int64)?;
        let values = int64(vec![None, Some(5)]);
        let is_set: ArrayRef = Arc::new(BooleanArray::from(vec![false, true]));
        acc.merge_batch(&[values, is_set])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int64(Some(5)));
        Ok(())
    }

    #[test]
    fn groups_accumulator_basic() -> Result<()> {
        let mut acc = AnyValueGroupsAccumulator::new(
            PrimitiveValueState::<Int64Type>::new(DataType::Int64),
        );
        // groups: row 0->g0, 1->g1, 2->g0, 3->g1
        let values = int64(vec![None, Some(20), Some(10), Some(21)]);
        acc.update_batch(&[values], &[0, 1, 0, 1], None, 2)?;
        let out = acc.evaluate(EmitTo::All)?;
        let out = out.as_primitive::<Int64Type>();
        // g0 skips the leading null and captures 10; g1 captures 20.
        assert_eq!(out.value(0), 10);
        assert_eq!(out.value(1), 20);
        Ok(())
    }

    #[test]
    fn groups_accumulator_all_null_group_is_null() -> Result<()> {
        let mut acc = AnyValueGroupsAccumulator::new(
            PrimitiveValueState::<Int64Type>::new(DataType::Int64),
        );
        let values = int64(vec![None, Some(20), None]);
        acc.update_batch(&[values], &[0, 1, 0], None, 2)?;
        let out = acc.evaluate(EmitTo::All)?;
        let out = out.as_primitive::<Int64Type>();
        assert!(out.is_null(0));
        assert_eq!(out.value(1), 20);
        Ok(())
    }

    #[test]
    fn groups_accumulator_respects_filter() -> Result<()> {
        let mut acc = AnyValueGroupsAccumulator::new(
            PrimitiveValueState::<Int64Type>::new(DataType::Int64),
        );
        let values = int64(vec![Some(1), Some(2), Some(3)]);
        // Filter out the first row, so g0 should capture 3 (row 2), not 1.
        let filter = BooleanArray::from(vec![false, true, true]);
        acc.update_batch(&[values], &[0, 1, 0], Some(&filter), 2)?;
        let out = acc.evaluate(EmitTo::All)?;
        let out = out.as_primitive::<Int64Type>();
        assert_eq!(out.value(0), 3);
        assert_eq!(out.value(1), 2);
        Ok(())
    }

    #[test]
    fn groups_accumulator_merge() -> Result<()> {
        let mut acc = AnyValueGroupsAccumulator::new(
            PrimitiveValueState::<Int64Type>::new(DataType::Int64),
        );
        let values = int64(vec![None, Some(42)]);
        let is_set: ArrayRef = Arc::new(BooleanArray::from(vec![false, true]));
        acc.merge_batch(&[values, is_set], &[0, 0], None, 1)?;
        let out = acc.evaluate(EmitTo::All)?;
        assert_eq!(out.as_primitive::<Int64Type>().value(0), 42);
        Ok(())
    }

    #[test]
    fn groups_accumulator_all_set_short_circuits() -> Result<()> {
        let mut acc = AnyValueGroupsAccumulator::new(
            PrimitiveValueState::<Int64Type>::new(DataType::Int64),
        );
        acc.update_batch(&[int64(vec![Some(1), Some(2)])], &[0, 1], None, 2)?;
        assert_eq!(acc.num_set, 2);
        // All groups already set: a later batch must not overwrite anything.
        acc.update_batch(&[int64(vec![Some(9), Some(9)])], &[0, 1], None, 2)?;
        let out = acc.evaluate(EmitTo::All)?;
        let out = out.as_primitive::<Int64Type>();
        assert_eq!(out.value(0), 1);
        assert_eq!(out.value(1), 2);
        Ok(())
    }

    #[test]
    fn groups_accumulator_emit_first_tracks_num_set() -> Result<()> {
        let mut acc = AnyValueGroupsAccumulator::new(
            PrimitiveValueState::<Int64Type>::new(DataType::Int64),
        );
        // groups 0,1 set; group 2 stays unset (all null).
        acc.update_batch(
            &[int64(vec![Some(10), Some(11), None])],
            &[0, 1, 2],
            None,
            3,
        )?;
        assert_eq!(acc.num_set, 2);

        // Emit the first group; one set bit leaves, so num_set drops to 1.
        let emitted = acc.evaluate(EmitTo::First(1))?;
        assert_eq!(emitted.as_primitive::<Int64Type>().value(0), 10);
        assert_eq!(acc.num_set, 1);

        // Remaining groups shift down: old group 1 -> 0 (set), old group 2 -> 1.
        let rest = acc.evaluate(EmitTo::All)?;
        let rest = rest.as_primitive::<Int64Type>();
        assert_eq!(rest.value(0), 11);
        assert!(rest.is_null(1));
        assert_eq!(acc.num_set, 0);
        Ok(())
    }

    #[test]
    fn groups_accumulator_state_roundtrip_bytes() -> Result<()> {
        let mut acc =
            AnyValueGroupsAccumulator::new(BytesValueState::try_new(DataType::Utf8)?);
        let values: ArrayRef =
            Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
        acc.update_batch(&[values], &[0, 1, 1], None, 2)?;
        let state = acc.state(EmitTo::All)?;
        assert_eq!(state.len(), 2);
        let vals = state[0].as_string::<i32>();
        let is_set = state[1].as_boolean();
        assert_eq!(vals.value(0), "a");
        assert!(is_set.value(0));
        assert_eq!(vals.value(1), "b");
        assert!(is_set.value(1));
        Ok(())
    }
}
