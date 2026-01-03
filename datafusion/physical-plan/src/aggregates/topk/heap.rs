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

//! A custom binary heap implementation for performant top K aggregation

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, PrimitiveArray, downcast_primitive,
};
use arrow::array::{
    cast::AsArray,
    types::{IntervalDayTime, IntervalMonthDayNano},
};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, i256};
use datafusion_common::Result;
use datafusion_common::exec_datafusion_err;

use half::f16;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

/// A custom version of `Ord` that only exists to we can implement it for the Values in our heap
pub trait Comparable {
    fn comp(&self, other: &Self) -> Ordering;
}

impl Comparable for Option<String> {
    fn comp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

/// A "type alias" for Values which are stored in our heap
pub trait ValueType: Comparable + Clone + Debug {}

impl<T> ValueType for T where T: Comparable + Clone + Debug {}

/// An entry in our heap, which contains both the value and a index into an external HashTable
struct HeapItem<VAL: ValueType> {
    val: VAL,
    map_idx: usize,
}

/// A custom heap implementation that allows several things that couldn't be achieved with
/// `collections::BinaryHeap`:
/// 1. It allows values to be updated at arbitrary positions (when group values change)
/// 2. It can be either a min or max heap
/// 3. It can use our `HeapItem` type & `Comparable` trait
/// 4. It is specialized to grow to a certain limit, then always replace without grow & shrink
struct TopKHeap<VAL: ValueType> {
    desc: bool,
    len: usize,
    capacity: usize,
    heap: Vec<Option<HeapItem<VAL>>>,
}

/// An interface to hide the generic type signature of TopKHeap behind arrow arrays
pub trait ArrowHeap {
    fn set_batch(&mut self, vals: ArrayRef);
    fn is_worse(&self, idx: usize) -> bool;
    fn worst_map_idx(&self) -> usize;
    fn insert(&mut self, row_idx: usize, map_idx: usize, map: &mut Vec<(usize, usize)>);
    fn replace_if_better(
        &mut self,
        heap_idx: usize,
        row_idx: usize,
        map: &mut Vec<(usize, usize)>,
    );
    fn drain(&mut self) -> (ArrayRef, Vec<usize>);
}

/// An implementation of `ArrowHeap` that deals with primitive values
pub struct PrimitiveHeap<VAL: ArrowPrimitiveType>
where
    Option<<VAL as ArrowPrimitiveType>::Native>: Comparable,
{
    batch: ArrayRef,
    heap: TopKHeap<Option<VAL::Native>>,
    sort_options: SortOptions,
    data_type: DataType,
}

impl<VAL: ArrowPrimitiveType> PrimitiveHeap<VAL>
where
    Option<<VAL as ArrowPrimitiveType>::Native>: Comparable,
{
    pub fn new(limit: usize, sort_options: SortOptions, data_type: DataType) -> Self {
        let owned: ArrayRef = Arc::new(PrimitiveArray::<VAL>::builder(0).finish());
        Self {
            batch: owned,
            heap: TopKHeap::new(limit, sort_options.descending),
            sort_options,
            data_type,
        }
    }
}

impl<VAL: ArrowPrimitiveType> ArrowHeap for PrimitiveHeap<VAL>
where
    Option<<VAL as ArrowPrimitiveType>::Native>: Comparable,
{
    fn set_batch(&mut self, vals: ArrayRef) {
        self.batch = vals;
    }

    fn is_worse(&self, row_idx: usize) -> bool {
        if !self.heap.is_full() {
            return false;
        }
        let vals = self.batch.as_primitive::<VAL>();
        let new_val = if vals.is_null(row_idx) {
            None
        } else {
            Some(vals.value(row_idx))
        };
        let worst_val = self.heap.worst_val().expect("Missing root");

        // Apply NULL ordering based on sort_options
        // For SQL semantics:
        // - NULLS FIRST means NULL < any value (NULL is "smaller")
        // - NULLS LAST means NULL > any value (NULL is "larger")
        let cmp = match (&new_val, worst_val) {
            (Some(a), Some(b)) => Some(*a).comp(&Some(*b)),
            (Some(_), None) => {
                // new_val is non-NULL, worst_val is NULL
                if self.sort_options.nulls_first {
                    Ordering::Greater // non-NULL > NULL
                } else {
                    Ordering::Less // non-NULL < NULL
                }
            }
            (None, Some(_)) => {
                // new_val is NULL, worst_val is non-NULL
                if self.sort_options.nulls_first {
                    Ordering::Less // NULL < non-NULL
                } else {
                    Ordering::Greater // NULL > non-NULL
                }
            }
            (None, None) => Ordering::Equal,
        };

        (!self.sort_options.descending && cmp == Ordering::Greater)
            || (self.sort_options.descending && cmp == Ordering::Less)
    }

    fn worst_map_idx(&self) -> usize {
        self.heap.worst_map_idx()
    }

    fn insert(&mut self, row_idx: usize, map_idx: usize, map: &mut Vec<(usize, usize)>) {
        let vals = self.batch.as_primitive::<VAL>();
        let new_val = if vals.is_null(row_idx) {
            None
        } else {
            Some(vals.value(row_idx))
        };
        self.heap.append_or_replace(new_val, map_idx, map);
    }

    fn replace_if_better(
        &mut self,
        heap_idx: usize,
        row_idx: usize,
        map: &mut Vec<(usize, usize)>,
    ) {
        let vals = self.batch.as_primitive::<VAL>();
        let new_val = if vals.is_null(row_idx) {
            None
        } else {
            Some(vals.value(row_idx))
        };

        // For MIN/MAX aggregation: non-NULL values always replace NULL values
        // This implements standard SQL semantics where MIN/MAX ignore NULL values
        let existing_item = self.heap.heap[heap_idx]
            .as_ref()
            .expect("Missing heap item");
        let existing_val = &existing_item.val;

        // Apply SQL NULL ordering for comparison
        let should_replace = match (&new_val, existing_val) {
            (Some(a), Some(b)) => {
                let cmp = Some(*a).comp(&Some(*b));
                if self.sort_options.descending {
                    cmp == Ordering::Greater
                } else {
                    cmp == Ordering::Less
                }
            }
            (Some(_), None) => {
                // new is non-NULL, existing is NULL
                // For MIN/MAX: non-NULL always better than NULL
                // But we also need to check sort order
                // For MIN (ASC): non-NULL can be better if it's smaller than other values
                // For MAX (DESC): non-NULL can be better if it's larger than other values
                // Since existing is NULL, non-NULL is always "better" for aggregation
                true
            }
            (None, Some(_)) => {
                // new is NULL, existing is non-NULL
                // For MIN/MAX: NULL never replaces non-NULL
                false
            }
            (None, None) => false, // Both NULL, no need to replace
        };

        if should_replace {
            let existing = &mut self.heap.heap[heap_idx]
                .as_mut()
                .expect("Missing heap item")
                .val;
            *existing = new_val;
            self.heap.heapify_down(heap_idx, map);
        }
    }

    fn drain(&mut self) -> (ArrayRef, Vec<usize>) {
        let (vals, map_idxs) = self.heap.drain();
        let mut builder = PrimitiveArray::<VAL>::builder(vals.len());
        for val in vals {
            match val {
                Some(v) => builder.append_value(v),
                None => builder.append_null(),
            }
        }
        let arr = builder.finish().with_data_type(self.data_type.clone());
        (Arc::new(arr), map_idxs)
    }
}

impl<VAL: ValueType> TopKHeap<VAL> {
    pub fn new(limit: usize, desc: bool) -> Self {
        Self {
            desc,
            capacity: limit,
            len: 0,
            heap: (0..=limit).map(|_| None).collect::<Vec<_>>(),
        }
    }

    pub fn worst_val(&self) -> Option<&VAL> {
        let root = self.heap.first()?;
        let hi = match root {
            None => return None,
            Some(hi) => hi,
        };
        Some(&hi.val)
    }

    pub fn worst_map_idx(&self) -> usize {
        self.heap[0].as_ref().map(|hi| hi.map_idx).unwrap_or(0)
    }

    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn append_or_replace(
        &mut self,
        new_val: VAL,
        map_idx: usize,
        map: &mut Vec<(usize, usize)>,
    ) {
        if self.is_full() {
            self.replace_root(new_val, map_idx, map);
        } else {
            self.append(new_val, map_idx, map);
        }
    }

    fn append(&mut self, new_val: VAL, map_idx: usize, mapper: &mut Vec<(usize, usize)>) {
        let hi = HeapItem::new(new_val, map_idx);
        self.heap[self.len] = Some(hi);
        self.heapify_up(self.len, mapper);
        self.len += 1;
    }

    fn pop(&mut self, map: &mut Vec<(usize, usize)>) -> Option<HeapItem<VAL>> {
        if self.len() == 0 {
            return None;
        }
        if self.len() == 1 {
            self.len = 0;
            return self.heap[0].take();
        }
        self.swap(0, self.len - 1, map);
        let former_root = self.heap[self.len - 1].take();
        self.len -= 1;
        self.heapify_down(0, map);
        former_root
    }

    pub fn drain(&mut self) -> (Vec<VAL>, Vec<usize>) {
        let mut map = Vec::with_capacity(self.len);
        let mut vals = Vec::with_capacity(self.len);
        let mut map_idxs = Vec::with_capacity(self.len);
        while let Some(worst_hi) = self.pop(&mut map) {
            vals.push(worst_hi.val);
            map_idxs.push(worst_hi.map_idx);
        }
        vals.reverse();
        map_idxs.reverse();
        (vals, map_idxs)
    }

    fn replace_root(
        &mut self,
        new_val: VAL,
        map_idx: usize,
        mapper: &mut Vec<(usize, usize)>,
    ) {
        let hi = self.heap[0].as_mut().expect("No root");
        hi.val = new_val;
        hi.map_idx = map_idx;
        self.heapify_down(0, mapper);
    }

    fn heapify_up(&mut self, mut idx: usize, mapper: &mut Vec<(usize, usize)>) {
        let desc = self.desc;
        while idx != 0 {
            let parent_idx = (idx - 1) / 2;
            let node = self.heap[idx].as_ref().expect("No heap item");
            let parent = self.heap[parent_idx].as_ref().expect("No heap item");
            if (!desc && node.val.comp(&parent.val) != Ordering::Greater)
                || (desc && node.val.comp(&parent.val) != Ordering::Less)
            {
                return;
            }
            self.swap(idx, parent_idx, mapper);
            idx = parent_idx;
        }
    }

    fn swap(&mut self, a_idx: usize, b_idx: usize, mapper: &mut Vec<(usize, usize)>) {
        let a_hi = self.heap[a_idx].take().expect("Missing heap entry");
        let b_hi = self.heap[b_idx].take().expect("Missing heap entry");

        mapper.push((a_hi.map_idx, b_idx));
        mapper.push((b_hi.map_idx, a_idx));

        self.heap[a_idx] = Some(b_hi);
        self.heap[b_idx] = Some(a_hi);
    }

    fn heapify_down(&mut self, node_idx: usize, mapper: &mut Vec<(usize, usize)>) {
        let left_child = node_idx * 2 + 1;
        let desc = self.desc;
        let entry = self.heap.get(node_idx).expect("Missing node!");
        let entry = entry.as_ref().expect("Missing node!");
        let mut best_idx = node_idx;
        let mut best_val = &entry.val;
        for child_idx in left_child..=left_child + 1 {
            if let Some(Some(child)) = self.heap.get(child_idx)
                && ((!desc && child.val.comp(best_val) == Ordering::Greater)
                    || (desc && child.val.comp(best_val) == Ordering::Less))
            {
                best_val = &child.val;
                best_idx = child_idx;
            }
        }
        if best_val.comp(&entry.val) != Ordering::Equal {
            self.swap(best_idx, node_idx, mapper);
            self.heapify_down(best_idx, mapper);
        }
    }

    fn _tree_print(&self, idx: usize, prefix: &str, is_tail: bool, output: &mut String) {
        if let Some(Some(hi)) = self.heap.get(idx) {
            let connector = if idx != 0 {
                if is_tail { "└── " } else { "├── " }
            } else {
                ""
            };
            output.push_str(&format!(
                "{}{}val={:?} idx={}, bucket={}\n",
                prefix, connector, hi.val, idx, hi.map_idx
            ));
            let new_prefix = if is_tail { "" } else { "│   " };
            let child_prefix = format!("{prefix}{new_prefix}");

            let left_idx = idx * 2 + 1;
            let right_idx = idx * 2 + 2;

            let left_exists = left_idx < self.len;
            let right_exists = right_idx < self.len;

            if left_exists {
                self._tree_print(left_idx, &child_prefix, !right_exists, output);
            }
            if right_exists {
                self._tree_print(right_idx, &child_prefix, true, output);
            }
        }
    }
}

impl<VAL: ValueType> Display for TopKHeap<VAL> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        if !self.heap.is_empty() {
            self._tree_print(0, "", true, &mut output);
        }
        write!(f, "{output}")
    }
}

impl<VAL: ValueType> HeapItem<VAL> {
    pub fn new(val: VAL, buk_idx: usize) -> Self {
        Self {
            val,
            map_idx: buk_idx,
        }
    }
}

impl<VAL: ValueType> Debug for HeapItem<VAL> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("bucket=")?;
        Debug::fmt(&self.map_idx, f)?;
        f.write_str(" val=")?;
        Debug::fmt(&self.val, f)?;
        f.write_str("\n")?;
        Ok(())
    }
}

impl<VAL: ValueType> Eq for HeapItem<VAL> {}

impl<VAL: ValueType> PartialEq<Self> for HeapItem<VAL> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<VAL: ValueType> PartialOrd<Self> for HeapItem<VAL> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<VAL: ValueType> Ord for HeapItem<VAL> {
    fn cmp(&self, other: &Self) -> Ordering {
        let res = self.val.comp(&other.val);
        if res != Ordering::Equal {
            return res;
        }
        self.map_idx.cmp(&other.map_idx)
    }
}

macro_rules! compare_float {
    ($($t:ty),+) => {
        $(impl Comparable for Option<$t> {
            fn comp(&self, other: &Self) -> Ordering {
                match (self, other) {
                    (Some(me), Some(other)) => me.total_cmp(other),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
                }
            }
        })+

        $(impl Comparable for $t {
            fn comp(&self, other: &Self) -> Ordering {
                self.total_cmp(other)
            }
        })+
    };
}

macro_rules! compare_integer {
    ($($t:ty),+) => {
        $(impl Comparable for Option<$t> {
            fn comp(&self, other: &Self) -> Ordering {
                match (self, other) {
                    (Some(me), Some(other)) => me.cmp(other),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
                }
            }
        })+

        $(impl Comparable for $t {
            fn comp(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        })+
    };
}

compare_integer!(i8, i16, i32, i64, i128, i256);
compare_integer!(u8, u16, u32, u64);
compare_integer!(IntervalDayTime, IntervalMonthDayNano);
compare_float!(f16, f32, f64);

pub fn new_heap(
    limit: usize,
    sort_options: SortOptions,
    vt: DataType,
) -> Result<Box<dyn ArrowHeap + Send>> {
    macro_rules! downcast_helper {
        ($vt:ty, $d:ident) => {
            return Ok(Box::new(PrimitiveHeap::<$vt>::new(limit, sort_options, vt)))
        };
    }

    downcast_primitive! {
        vt => (downcast_helper, vt),
        _ => {}
    }

    Err(exec_datafusion_err!("Can't group type: {vt:?}"))
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;

    use super::*;

    #[test]
    fn should_append() -> Result<()> {
        let mut map = vec![];
        let mut heap = TopKHeap::new(10, false);
        heap.append_or_replace(1, 1, &mut map);

        let actual = heap.to_string();
        assert_snapshot!(actual, @"val=1 idx=0, bucket=1");

        Ok(())
    }

    #[test]
    fn should_heapify_up() -> Result<()> {
        let mut map = vec![];
        let mut heap = TopKHeap::new(10, false);

        heap.append_or_replace(1, 1, &mut map);
        assert_eq!(map, vec![]);

        heap.append_or_replace(2, 2, &mut map);
        assert_eq!(map, vec![(2, 0), (1, 1)]);

        let actual = heap.to_string();
        assert_snapshot!(actual, @r"
        val=2 idx=0, bucket=2
        └── val=1 idx=1, bucket=1
        ");

        Ok(())
    }

    #[test]
    fn should_heapify_down() -> Result<()> {
        let mut map = vec![];
        let mut heap = TopKHeap::new(3, false);

        heap.append_or_replace(1, 1, &mut map);
        heap.append_or_replace(2, 2, &mut map);
        heap.append_or_replace(3, 3, &mut map);
        let actual = heap.to_string();
        assert_snapshot!(actual, @r"
        val=3 idx=0, bucket=3
        ├── val=1 idx=1, bucket=1
        └── val=2 idx=2, bucket=2
        ");

        let mut map = vec![];
        heap.append_or_replace(0, 0, &mut map);
        let actual = heap.to_string();
        assert_snapshot!(actual, @r"
        val=2 idx=0, bucket=2
        ├── val=1 idx=1, bucket=1
        └── val=0 idx=2, bucket=0
        ");
        assert_eq!(map, vec![(2, 0), (0, 2)]);

        Ok(())
    }

    #[test]
    fn should_find_worst() -> Result<()> {
        let mut map = vec![];
        let mut heap = TopKHeap::new(10, false);

        heap.append_or_replace(1, 1, &mut map);
        heap.append_or_replace(2, 2, &mut map);

        let actual = heap.to_string();
        assert_snapshot!(actual, @r"
        val=2 idx=0, bucket=2
        └── val=1 idx=1, bucket=1
        ");

        assert_eq!(heap.worst_val(), Some(&2));
        assert_eq!(heap.worst_map_idx(), 2);

        Ok(())
    }

    #[test]
    fn should_drain() -> Result<()> {
        let mut map = vec![];
        let mut heap = TopKHeap::new(10, false);

        heap.append_or_replace(1, 1, &mut map);
        heap.append_or_replace(2, 2, &mut map);

        let actual = heap.to_string();
        assert_snapshot!(actual, @r"
        val=2 idx=0, bucket=2
        └── val=1 idx=1, bucket=1
        ");

        let (vals, map_idxs) = heap.drain();
        assert_eq!(vals, vec![1, 2]);
        assert_eq!(map_idxs, vec![1, 2]);
        assert_eq!(heap.len(), 0);

        Ok(())
    }
}
