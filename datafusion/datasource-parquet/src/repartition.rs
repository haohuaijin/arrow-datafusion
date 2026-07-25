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

//! [`ParquetAccessPlan`]-aware repartitioning of parquet scans.
//!
//! The generic [`FileGroupPartitioner`] splits files into evenly sized byte
//! ranges, which is blind to any [`ParquetAccessPlan`] attached to the files:
//! partitions covering only skipped row groups open the file for nothing,
//! while the row groups that actually contain selected rows can end up
//! concentrated in a single partition, serializing the decode work.
//!
//! This module instead splits files *by their scanned row groups*, weighting
//! each row group by its estimated decoded bytes, so the decode work is
//! balanced across partitions and skipped row groups never contribute decode
//! work. A file may appear in multiple partitions, each carrying a sub-plan
//! that scans a disjoint subset of the original plan's row groups (all other
//! row groups are marked [`RowGroupAccess::Skip`]).
//!
//! # Design decisions
//!
//! * **Weights are estimated decoded bytes.** A
//!   [`RowGroupAccess::Selection`] weighs the selected fraction of its
//!   file's average row-group bytes plus a small fixed per-row-group cost (a
//!   sparse selection still decompresses whole pages around every selected
//!   row, so touching a row group is never free), a
//!   [`RowGroupAccess::Scan`] weighs the file's average row-group bytes, and
//!   a file without an access plan weighs its total byte size. This keeps
//!   planned and unplanned files comparable even when row-count statistics
//!   are unavailable.
//! * **Files without an access plan are still split by byte ranges**, with a
//!   chunk count proportional to their share of the total weight, so mixing
//!   indexed and unindexed files in one scan does not serialize the
//!   unindexed ones.
//! * **`repartition_file_min_size` retains its public meaning.** The custom
//!   repartitioning is attempted only when the total input file size reaches
//!   the configured threshold. Once that threshold is met, individual output
//!   chunks may be smaller, matching [`FileGroupPartitioner`].
//! * **Plans that produce no rows are retained once for validation.** This
//!   preserves the opener's invalid-plan and invalid-selection checks without
//!   allowing empty plans to fall back to repeated byte-range file opens.
//! * **Ordering**: when the scan declares an output ordering, every partition's
//!   row-producing work items come from a single original group, in their
//!   original (file order, ascending row group) order. A partition is then a
//!   *subsequence* of a sorted group, which is itself sorted, so the
//!   per-partition ordering claim still holds. Zero-output validation plans
//!   may be attached without affecting row order. Mirroring
//!   [`FileGroupPartitioner`], repartitioning is declined when there are
//!   already at least `target_partitions` active sorted groups.
//!
//! [`FileGroupPartitioner`]: datafusion_datasource::file_groups::FileGroupPartitioner

use datafusion_datasource::FileRange;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_groups::FileGroup;

use crate::access_plan::{ParquetAccessPlan, RowGroupAccess};

/// The fixed cost of scanning a row group at all, expressed as this fraction
/// of the row group's estimated bytes (see the module docs).
const PER_ROW_GROUP_COST_DIVISOR: u128 = 64;

/// A unit of scan work inside one original file group.
struct WorkItem<'a> {
    /// index of the file within its original group
    file_idx: usize,
    /// what this item scans
    kind: ItemKind<'a>,
    /// zero-output selections from the same file that must be included in
    /// this item's sub-plan so the opener validates their covered row count
    validation_accesses: Vec<(usize, &'a RowGroupAccess)>,
    /// estimated number of bytes this item decodes
    weight: u128,
}

enum ItemKind<'a> {
    /// one row group scanned by the file's access plan
    RowGroup(usize, &'a RowGroupAccess),
    /// a byte range of a file without an access plan; a range covering the
    /// whole file is materialized without a [`FileRange`]
    ByteRange { start: u64, end: u64 },
}

/// A zero-output plan that must still reach the opener so plan length and row
/// selections are validated against the parquet metadata.
struct ValidationFile {
    group_idx: usize,
    file_idx: usize,
    plan: ParquetAccessPlan,
}

/// Split file groups by the row groups their [`ParquetAccessPlan`]s scan,
/// balancing partitions by estimated decode cost. See the module docs for
/// the full design.
///
/// Returns `None` (caller should fall back to byte-range repartitioning) when
/// no file carries an access plan, when any file already has a byte range
/// assigned (row group byte offsets are unknown at planning time, so the
/// range cannot be intersected with the plan), or when `preserve_order` is set
/// and there are already enough active groups. When total input size is below
/// `repartition_file_min_size`, returns the original groups as handled so a
/// fallback cannot reinterpret the threshold.
pub(crate) fn repartition_by_access_plan(
    file_groups: &[FileGroup],
    target_partitions: usize,
    repartition_file_min_size: usize,
    preserve_order: bool,
) -> Option<Vec<FileGroup>> {
    if target_partitions <= 1 || file_groups.is_empty() {
        return None;
    }
    let mut has_plan = false;
    let mut total_file_size = 0_u128;
    for file in file_groups.iter().flat_map(|g| g.iter()) {
        if file.range.is_some() {
            return None;
        }
        total_file_size = total_file_size.saturating_add(file.object_meta.size as u128);
        if file.extensions.get::<ParquetAccessPlan>().is_some() {
            has_plan = true;
        }
    }
    if !has_plan {
        return None;
    }
    // Keep the public configuration contract: this threshold applies to the
    // total input file size, not to each produced chunk.
    if total_file_size == 0 || total_file_size < repartition_file_min_size as u128 {
        // Returning the original groups as handled is intentional: a generic
        // preserve-order fallback does not consistently apply this global
        // threshold for multi-group inputs.
        return Some(file_groups.to_vec());
    }

    // first pass: row-group items for planned files, whole-file items for
    // the rest
    let mut group_items: Vec<Vec<WorkItem<'_>>> = Vec::with_capacity(file_groups.len());
    let mut validation_files = Vec::new();
    let mut total_weight = 0_u128;
    for (group_idx, group) in file_groups.iter().enumerate() {
        let mut items = Vec::new();
        for (file_idx, file) in group.iter().enumerate() {
            let file_size = file.object_meta.size;
            match file.extensions.get::<ParquetAccessPlan>() {
                Some(plan) => {
                    let first_file_item = items.len();
                    let row_group_count = plan.len().max(1) as u128;
                    let avg_rg_bytes =
                        (file_size as u128).div_ceil(row_group_count).max(1);
                    // scanning a row group has a fixed cost regardless of
                    // how few rows it selects (the pages around every
                    // selected row are decompressed whole); approximate it
                    // as a fraction of the row group's bytes so very sparse
                    // row groups are not packed too densely
                    let fixed_rg_cost =
                        (avg_rg_bytes / PER_ROW_GROUP_COST_DIVISOR).max(1);
                    let mut active_items = 0;
                    let mut validation_plan = ParquetAccessPlan::new_none(plan.len());
                    let mut zero_row_selections = Vec::new();
                    for (rg, access) in plan.inner().iter().enumerate() {
                        let weight = match access {
                            RowGroupAccess::Skip => continue,
                            RowGroupAccess::Scan => avg_rg_bytes,
                            RowGroupAccess::Selection(selection) => {
                                let rows = selection.row_count() as u128;
                                if rows == 0 {
                                    // A zero-output selection still has to be
                                    // checked against the row-group row count.
                                    validation_plan.set(rg, access.clone());
                                    if plan.is_fully_matched(rg) {
                                        validation_plan.mark_fully_matched(rg);
                                    }
                                    zero_row_selections.push((rg, access));
                                    continue;
                                }
                                let covered_rows =
                                    selection.iter().fold(0_u128, |total, selector| {
                                        total.saturating_add(selector.row_count as u128)
                                    });
                                let proportional_bytes = avg_rg_bytes
                                    .saturating_mul(rows)
                                    .div_ceil(covered_rows.max(1));
                                proportional_bytes
                                    .saturating_add(fixed_rg_cost)
                                    .min(avg_rg_bytes)
                                    .max(1)
                            }
                        };
                        total_weight = total_weight.saturating_add(weight);
                        active_items += 1;
                        items.push(WorkItem {
                            file_idx,
                            kind: ItemKind::RowGroup(rg, access),
                            validation_accesses: Vec::new(),
                            weight,
                        });
                    }
                    // With no active items, retain an all-Skip plan once so
                    // its length is still validated by the opener.
                    if active_items == 0 {
                        validation_files.push(ValidationFile {
                            group_idx,
                            file_idx,
                            plan: validation_plan,
                        });
                    } else if !zero_row_selections.is_empty() {
                        items[first_file_item].validation_accesses = zero_row_selections;
                    }
                }
                None => {
                    let weight = (file_size as u128).max(1);
                    total_weight = total_weight.saturating_add(weight);
                    items.push(WorkItem {
                        file_idx,
                        kind: ItemKind::ByteRange {
                            start: 0,
                            end: file_size,
                        },
                        validation_accesses: Vec::new(),
                        weight,
                    });
                }
            }
        }
        group_items.push(items);
    }
    // Every planned file is empty. Keep exactly one validation partition and
    // do not let the caller fall back to target-sized byte-range splitting.
    if total_weight == 0 {
        let files = validation_files
            .into_iter()
            .map(|validation| materialize_validation_file(file_groups, validation))
            .collect();
        return Some(vec![FileGroup::new(files)]);
    }

    // Mirror `FileGroupPartitioner::repartition_preserving_order`, but count
    // only groups that can emit rows. All-Skip validation files do not provide
    // useful scan parallelism.
    let active_group_count = group_items.iter().filter(|items| !items.is_empty()).count();
    if preserve_order && active_group_count >= target_partitions {
        return None;
    }

    // second pass: split whole-file items of unplanned files into byte
    // chunks proportional to their share of the total weight. The global
    // `repartition_file_min_size` threshold was applied above; like the
    // default partitioner, individual chunks may be smaller than it.
    let ideal_weight = total_weight.div_ceil(target_partitions as u128).max(1);
    for items in &mut group_items {
        let mut expanded = Vec::with_capacity(items.len());
        for item in items.drain(..) {
            match item.kind {
                ItemKind::ByteRange { start: 0, end } => {
                    if end == 0 {
                        expanded.push(item);
                        continue;
                    }
                    let chunks = item
                        .weight
                        .saturating_add(ideal_weight / 2)
                        .div_euclid(ideal_weight)
                        .clamp(1, target_partitions as u128)
                        .min(end as u128);
                    if chunks == 1 {
                        expanded.push(item);
                        continue;
                    }
                    let chunks = u64::try_from(chunks)
                        .expect("chunk count is bounded by the u64 file size");
                    let chunk_bytes = end.div_ceil(chunks);
                    let mut offset = 0;
                    while offset < end {
                        let chunk_end = offset.saturating_add(chunk_bytes).min(end);
                        expanded.push(WorkItem {
                            file_idx: item.file_idx,
                            kind: ItemKind::ByteRange {
                                start: offset,
                                end: chunk_end,
                            },
                            validation_accesses: Vec::new(),
                            weight: (chunk_end - offset) as u128,
                        });
                        offset = chunk_end;
                    }
                }
                _ => expanded.push(item),
            }
        }
        *items = expanded;
    }

    // bins of (group index, item index)
    let bins: Vec<Vec<(usize, usize)>> = if preserve_order {
        // allocate partitions to groups proportionally to their total
        // weight, then bin-pack within each group; items of different
        // groups never share a partition, keeping every partition a sorted
        // subsequence of one group
        let weights: Vec<u128> = group_items
            .iter()
            .map(|items| {
                items
                    .iter()
                    .fold(0_u128, |sum, item| sum.saturating_add(item.weight))
            })
            .collect();
        let allocation = allocate_partitions(&weights, &group_items, target_partitions);
        group_items
            .iter()
            .enumerate()
            .flat_map(|(group_idx, items)| {
                let weights: Vec<u128> = items.iter().map(|i| i.weight).collect();
                lpt_pack(&weights, allocation[group_idx])
                    .into_iter()
                    .map(move |bin| bin.into_iter().map(|i| (group_idx, i)).collect())
            })
            .collect()
    } else {
        // one global bin-packing across all groups; a partition may mix
        // files of different groups
        let all_items: Vec<(usize, usize)> = group_items
            .iter()
            .enumerate()
            .flat_map(|(g, items)| (0..items.len()).map(move |i| (g, i)))
            .collect();
        let weights: Vec<u128> = all_items
            .iter()
            .map(|&(g, i)| group_items[g][i].weight)
            .collect();
        lpt_pack(&weights, target_partitions)
            .into_iter()
            .map(|bin| bin.into_iter().map(|idx| all_items[idx]).collect())
            .collect()
    };

    // Assign validation-only files before materialization so they can be put
    // back in original (group, file) order within the selected partition.
    let bin_original_groups: Vec<Vec<usize>> = bins
        .iter()
        .map(|bin| {
            let mut groups: Vec<usize> =
                bin.iter().map(|(group_idx, _)| *group_idx).collect();
            groups.sort_unstable();
            groups.dedup();
            groups
        })
        .collect();
    let mut validations_by_bin: Vec<Vec<ValidationFile>> =
        (0..bins.len()).map(|_| Vec::new()).collect();
    for validation in validation_files {
        let target = bin_original_groups
            .iter()
            .position(|groups| groups.contains(&validation.group_idx))
            .or_else(|| {
                validations_by_bin
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, validations)| validations.len())
                    .map(|(idx, _)| idx)
            })
            .expect("total_weight > 0 creates at least one output group");
        validations_by_bin[target].push(validation);
    }

    // Materialize each bin into a file group with sub-plans.
    let new_groups: Vec<FileGroup> = bins
        .into_iter()
        .zip(validations_by_bin)
        .map(|(mut bin, validations)| {
            debug_assert!(!bin.is_empty());
            // original order within each group; group runs stay contiguous
            bin.sort_unstable();
            build_partition(file_groups, &group_items, &bin, validations)
        })
        .collect();
    if new_groups.is_empty() {
        return None;
    }
    Some(new_groups)
}

fn materialize_validation_file(
    file_groups: &[FileGroup],
    validation: ValidationFile,
) -> PartitionedFile {
    let mut file = file_groups[validation.group_idx].files()[validation.file_idx].clone();
    file.extensions.insert(validation.plan);
    file
}

/// Allocate `target` partitions to groups proportionally to their weight,
/// with at least one partition per non-empty group and at most one per item.
fn allocate_partitions(
    weights: &[u128],
    group_items: &[Vec<WorkItem<'_>>],
    target: usize,
) -> Vec<usize> {
    let mut allocation: Vec<usize> = group_items
        .iter()
        .map(|items| usize::from(!items.is_empty()))
        .collect();
    let used: usize = allocation.iter().sum();
    debug_assert!(used <= target);
    // Starting from one partition per active group guarantees the allocation
    // can never exceed target. Give each remaining partition to the currently
    // most heavily loaded group that still has more than one item per bin.
    let mut remaining = target.saturating_sub(used);
    while remaining > 0 {
        let candidate = (0..weights.len())
            .filter(|&g| allocation[g] > 0 && allocation[g] < group_items[g].len())
            .max_by_key(|&g| weights[g].div_ceil(allocation[g] as u128));
        match candidate {
            Some(g) => {
                allocation[g] += 1;
                remaining -= 1;
            }
            None => break,
        }
    }
    allocation
}

/// Longest-processing-time bin packing: returns `k` bins of item indexes.
fn lpt_pack(weights: &[u128], k: usize) -> Vec<Vec<usize>> {
    if weights.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(weights.len());
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(weights[i]));
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut loads: Vec<u128> = vec![0; k];
    for i in order {
        let lightest = loads
            .iter()
            .enumerate()
            .min_by_key(|(_, l)| **l)
            .map(|(b, _)| b)
            .expect("k >= 1");
        bins[lightest].push(i);
        loads[lightest] = loads[lightest].saturating_add(weights[i]);
    }
    bins
}

/// The pending run while materializing one partition: consecutive items of
/// the same file coalesce into a single [`PartitionedFile`].
enum Run {
    /// row-group items of one file accumulating a sub-plan
    Plan(usize, usize, ParquetAccessPlan),
    /// byte-range items of one file accumulating a contiguous range
    Range(usize, usize, u64, u64),
}

/// Build one output partition from a sorted list of `(group, item)` indexes.
/// Consecutive row-group items of the same file coalesce into one
/// [`PartitionedFile`] carrying a sub-plan that scans exactly those row
/// groups; adjacent byte chunks of the same file coalesce into one range.
fn build_partition(
    file_groups: &[FileGroup],
    group_items: &[Vec<WorkItem<'_>>],
    bin: &[(usize, usize)],
    validations: Vec<ValidationFile>,
) -> FileGroup {
    // original group, original file, order within that file, materialized file
    let mut out: Vec<(usize, usize, usize, PartitionedFile)> = Vec::new();
    let mut current: Option<Run> = None;
    let mut next_order = 1;

    let flush = |current: &mut Option<Run>,
                 out: &mut Vec<(usize, usize, usize, PartitionedFile)>,
                 next_order: &mut usize| match current.take() {
        Some(Run::Plan(group_idx, file_idx, plan)) => {
            let mut file = file_groups[group_idx].files()[file_idx].clone();
            file.extensions.insert(plan);
            out.push((group_idx, file_idx, *next_order, file));
            *next_order += 1;
        }
        Some(Run::Range(group_idx, file_idx, start, end)) => {
            let mut file = file_groups[group_idx].files()[file_idx].clone();
            // a range covering the whole file scans without one
            if start > 0 || end < file.object_meta.size {
                file.range = Some(FileRange {
                    start: start as i64,
                    end: end as i64,
                });
            }
            out.push((group_idx, file_idx, *next_order, file));
            *next_order += 1;
        }
        None => {}
    };

    for &(group_idx, item_idx) in bin {
        let item = &group_items[group_idx][item_idx];
        let files = file_groups[group_idx].files();
        match item.kind {
            ItemKind::ByteRange { start, end } => match current.as_mut() {
                // extend a contiguous range of the same file
                Some(Run::Range(g, f, _, run_end))
                    if *g == group_idx && *f == item.file_idx && *run_end == start =>
                {
                    *run_end = end;
                }
                _ => {
                    flush(&mut current, &mut out, &mut next_order);
                    current = Some(Run::Range(group_idx, item.file_idx, start, end));
                }
            },
            ItemKind::RowGroup(rg, access) => {
                let same_file = matches!(&current,
                    Some(Run::Plan(g, f, _)) if *g == group_idx && *f == item.file_idx);
                if !same_file {
                    flush(&mut current, &mut out, &mut next_order);
                    let total = files[item.file_idx]
                        .extensions
                        .get::<ParquetAccessPlan>()
                        .expect("row group items only exist for planned files")
                        .len();
                    current = Some(Run::Plan(
                        group_idx,
                        item.file_idx,
                        ParquetAccessPlan::new_none(total),
                    ));
                }
                if let Some(Run::Plan(_, _, plan)) = current.as_mut() {
                    plan.set(rg, access.clone());
                    let original_plan = files[item.file_idx]
                        .extensions
                        .get::<ParquetAccessPlan>()
                        .expect("row group items only exist for planned files");
                    if original_plan.is_fully_matched(rg) {
                        plan.mark_fully_matched(rg);
                    }
                    for &(validation_rg, validation_access) in &item.validation_accesses {
                        plan.set(validation_rg, validation_access.clone());
                        if original_plan.is_fully_matched(validation_rg) {
                            plan.mark_fully_matched(validation_rg);
                        }
                    }
                }
            }
        }
    }
    flush(&mut current, &mut out, &mut next_order);
    for validation in validations {
        let group_idx = validation.group_idx;
        let file_idx = validation.file_idx;
        let file = materialize_validation_file(file_groups, validation);
        // Validate before emitting work from the same original file.
        out.push((group_idx, file_idx, 0, file));
    }
    out.sort_unstable_by_key(|(group_idx, file_idx, order, _)| {
        (*group_idx, *file_idx, *order)
    });
    FileGroup::new(out.into_iter().map(|(_, _, _, file)| file).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion_common::Statistics;
    use datafusion_common::stats::Precision;
    use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

    use super::*;

    fn selection(selected: usize, skipped: usize) -> RowGroupAccess {
        RowGroupAccess::Selection(RowSelection::from(vec![
            RowSelector::select(selected),
            RowSelector::skip(skipped),
        ]))
    }

    fn planned_file(
        name: &str,
        num_rows: usize,
        accesses: Vec<RowGroupAccess>,
    ) -> PartitionedFile {
        planned_file_with_size(name, num_rows, 1024 * 1024, accesses)
    }

    fn planned_file_with_size(
        name: &str,
        num_rows: usize,
        size: u64,
        accesses: Vec<RowGroupAccess>,
    ) -> PartitionedFile {
        let mut file = plain_file_with_size(name, num_rows, size);
        file.extensions.insert(ParquetAccessPlan::new(accesses));
        file
    }

    fn plain_file(name: &str, num_rows: usize) -> PartitionedFile {
        plain_file_with_size(name, num_rows, 1024 * 1024)
    }

    fn plain_file_with_size(name: &str, num_rows: usize, size: u64) -> PartitionedFile {
        let mut file = PartitionedFile::new(name.to_string(), size);
        let mut stats = Statistics::new_unknown(&arrow::datatypes::Schema::empty());
        stats.num_rows = Precision::Exact(num_rows);
        file.statistics = Some(Arc::new(stats));
        file
    }

    fn plain_file_without_statistics(name: &str, size: u64) -> PartitionedFile {
        PartitionedFile::new(name.to_string(), size)
    }

    fn scanned_row_groups(file: &PartitionedFile) -> Vec<usize> {
        file.extensions
            .get::<ParquetAccessPlan>()
            .expect("plan")
            .row_group_indexes()
    }

    fn partition_weight(group: &FileGroup) -> u64 {
        group
            .iter()
            .map(|f| {
                f.extensions
                    .get::<ParquetAccessPlan>()
                    .map(|plan| {
                        plan.inner()
                            .iter()
                            .map(|a| match a {
                                RowGroupAccess::Selection(s) => s.row_count() as u64,
                                _ => 0,
                            })
                            .sum::<u64>()
                    })
                    .unwrap_or(0)
            })
            .sum()
    }

    #[test]
    fn no_plans_returns_none() {
        let groups = vec![FileGroup::new(vec![plain_file("a.parquet", 100)])];
        assert!(repartition_by_access_plan(&groups, 4, 0, false).is_none());
    }

    #[test]
    fn preexisting_byte_range_returns_none() {
        let mut file = planned_file("a.parquet", 100, vec![selection(10, 90)]);
        file.range = Some(FileRange { start: 0, end: 10 });
        let groups = vec![FileGroup::new(vec![file])];
        assert!(repartition_by_access_plan(&groups, 4, 0, false).is_none());
    }

    #[test]
    fn min_size_keeps_original_groups() {
        // The threshold applies to the total input size. Returning the
        // original groups as handled prevents an ordered generic fallback
        // from splitting below the configured threshold.
        let file =
            planned_file("a.parquet", 200, vec![selection(10, 90), selection(10, 90)]);
        let groups = vec![FileGroup::new(vec![file])];
        let result = repartition_by_access_plan(&groups, 4, 100 * 1024 * 1024, false)
            .expect("access-plan path should handle the threshold");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1);
        assert!(result[0][0].range.is_none());
        assert_eq!(scanned_row_groups(&result[0][0]), vec![0, 1]);
    }

    #[test]
    fn min_size_is_not_a_byte_chunk_floor() {
        // Total input is 2 MiB, so a 1 MiB threshold allows repartitioning;
        // individual chunks may be smaller than 1 MiB, like the generic
        // FileGroupPartitioner.
        let planned =
            planned_file("a.parquet", 200, vec![selection(10, 90), selection(10, 90)]);
        let plain = plain_file("b.parquet", 50);
        let groups = vec![FileGroup::new(vec![planned, plain])];
        let result = repartition_by_access_plan(&groups, 3, 1024 * 1024, false)
            .expect("should split");

        let plain_out: Vec<&PartitionedFile> = result
            .iter()
            .flat_map(|g| g.iter())
            .filter(|f| f.path().as_ref().ends_with("b.parquet"))
            .collect();
        assert_eq!(plain_out.len(), 3, "plain file should use its full share");
        assert!(plain_out.iter().all(|file| file.range.is_some()));
    }

    #[test]
    fn preserve_order_with_enough_groups_returns_none() {
        // mirror the default partitioner: >= target_partitions sorted groups
        // already provide enough parallelism
        let groups: Vec<FileGroup> = (0..3)
            .map(|i| {
                FileGroup::new(vec![planned_file(
                    &format!("{i}.parquet"),
                    100,
                    vec![selection(10, 90)],
                )])
            })
            .collect();
        assert!(repartition_by_access_plan(&groups, 3, 0, true).is_none());
        assert!(repartition_by_access_plan(&groups, 4, 0, true).is_some());
    }

    #[test]
    fn splits_scanned_row_groups_across_partitions() {
        // 4 scanned row groups (10 rows each) + 2 skipped ones
        let file = planned_file(
            "a.parquet",
            600,
            vec![
                selection(10, 90),
                RowGroupAccess::Skip,
                selection(10, 90),
                selection(10, 90),
                RowGroupAccess::Skip,
                selection(10, 90),
            ],
        );
        let groups = vec![FileGroup::new(vec![file])];
        let result =
            repartition_by_access_plan(&groups, 4, 0, false).expect("should split");

        assert_eq!(result.len(), 4);
        let mut all_scanned: Vec<usize> = Vec::new();
        for group in &result {
            assert_eq!(group.len(), 1);
            let scanned = scanned_row_groups(&group.files()[0]);
            assert_eq!(scanned.len(), 1, "one row group per partition");
            // sub-plans keep the original row group count
            assert_eq!(
                group.files()[0]
                    .extensions
                    .get::<ParquetAccessPlan>()
                    .unwrap()
                    .len(),
                6
            );
            all_scanned.extend(scanned);
        }
        all_scanned.sort_unstable();
        // skipped row groups 1 and 4 are never assigned to any partition
        assert_eq!(all_scanned, vec![0, 2, 3, 5]);
    }

    #[test]
    fn balances_by_selected_rows() {
        // weights 30 / 10 / 10 / 10 into 2 partitions: LPT puts the heavy
        // row group alone and the three light ones together
        let file = planned_file(
            "a.parquet",
            400,
            vec![
                selection(30, 70),
                selection(10, 90),
                selection(10, 90),
                selection(10, 90),
            ],
        );
        let groups = vec![FileGroup::new(vec![file])];
        let result =
            repartition_by_access_plan(&groups, 2, 0, false).expect("should split");

        assert_eq!(result.len(), 2);
        let mut weights: Vec<u64> = result.iter().map(partition_weight).collect();
        weights.sort_unstable();
        assert_eq!(weights, vec![30, 30]);
    }

    #[test]
    fn preserve_order_keeps_partitions_within_one_group() {
        // two time-disjoint groups; partitions must not mix them and row
        // groups must stay in ascending order inside each partition
        let g1 = FileGroup::new(vec![planned_file(
            "new.parquet",
            300,
            vec![selection(20, 80), selection(20, 80), selection(20, 80)],
        )]);
        let g2 = FileGroup::new(vec![planned_file(
            "old.parquet",
            300,
            vec![selection(20, 80), selection(20, 80), selection(20, 80)],
        )]);
        let result =
            repartition_by_access_plan(&[g1, g2], 6, 0, true).expect("should split");

        assert_eq!(result.len(), 6);
        for group in &result {
            assert_eq!(group.len(), 1, "no partition mixes files of two groups");
            let scanned = scanned_row_groups(&group.files()[0]);
            assert!(scanned.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn non_preserve_order_may_mix_groups() {
        // without an ordering, one partition may hold files of different
        // original groups
        let g1 = FileGroup::new(vec![planned_file(
            "a.parquet",
            200,
            vec![selection(10, 90), selection(10, 90)],
        )]);
        let g2 = FileGroup::new(vec![planned_file(
            "b.parquet",
            200,
            vec![selection(10, 90), selection(10, 90)],
        )]);
        let result =
            repartition_by_access_plan(&[g1, g2], 2, 0, false).expect("should split");

        assert_eq!(result.len(), 2);
        assert!(
            result.iter().any(|g| {
                let has =
                    |name: &str| g.iter().any(|f| f.path().as_ref().ends_with(name));
                has("a.parquet") && has("b.parquet")
            }),
            "expected at least one partition mixing both groups"
        );
    }

    #[test]
    fn unplanned_file_splits_by_byte_ranges() {
        // Planned and plain work share one byte-based partition budget. The
        // full-size plain file receives three chunks in this shape.
        let planned =
            planned_file("a.parquet", 200, vec![selection(10, 90), selection(10, 90)]);
        let plain = plain_file("b.parquet", 50);
        let plain_size = plain.object_meta.size;
        let groups = vec![FileGroup::new(vec![planned, plain])];
        let result =
            repartition_by_access_plan(&groups, 3, 0, false).expect("should split");

        let mut chunks: Vec<(u64, u64)> = result
            .iter()
            .flat_map(|g| g.iter())
            .filter(|f| f.path().as_ref().ends_with("b.parquet"))
            .map(|f| {
                assert!(f.extensions.get::<ParquetAccessPlan>().is_none());
                let range = f.range.as_ref().expect("chunk should carry a range");
                (range.start as u64, range.end as u64)
            })
            .collect();
        chunks.sort_unstable();
        assert_eq!(chunks.len(), 3, "plain file split into 3 byte chunks");
        // chunks are disjoint and cover the whole file
        assert_eq!(chunks[0].0, 0);
        assert!(chunks.windows(2).all(|window| window[0].1 == window[1].0));
        assert_eq!(chunks.last().unwrap().1, plain_size);
    }

    #[test]
    fn small_unplanned_file_stays_whole() {
        // the plain file is a tiny share of the weight: one chunk, no range
        let planned = planned_file(
            "a.parquet",
            2000,
            vec![selection(500, 0), selection(500, 0)],
        );
        let plain = plain_file_with_size("b.parquet", 10, 64 * 1024);
        let groups = vec![FileGroup::new(vec![planned, plain])];
        let result =
            repartition_by_access_plan(&groups, 3, 0, false).expect("should split");

        let plain_out: Vec<&PartitionedFile> = result
            .iter()
            .flat_map(|g| g.iter())
            .filter(|f| f.path().as_ref().ends_with("b.parquet"))
            .collect();
        assert_eq!(plain_out.len(), 1, "tiny file appears exactly once");
        assert!(
            plain_out[0].range.is_none(),
            "whole file scans without a range"
        );
    }

    #[test]
    fn coalesces_row_groups_of_same_file_in_one_partition() {
        // 3 scanned row groups into 2 partitions: one partition holds two
        // row groups of the same file, coalesced into a single sub-plan
        let file = planned_file(
            "a.parquet",
            300,
            vec![selection(10, 90), selection(10, 90), selection(10, 90)],
        );
        let groups = vec![FileGroup::new(vec![file])];
        let result =
            repartition_by_access_plan(&groups, 2, 0, false).expect("should split");

        assert_eq!(result.len(), 2);
        let sizes: Vec<usize> = result
            .iter()
            .map(|g| {
                assert_eq!(g.len(), 1, "same-file row groups coalesce");
                scanned_row_groups(&g.files()[0]).len()
            })
            .collect();
        let mut sizes = sizes;
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 2]);
    }

    #[test]
    fn scan_access_uses_full_row_group_byte_estimate() {
        // A full Scan weighs the average row-group bytes, so it gets its own
        // partition against a 10%-selection row group.
        let file = planned_file(
            "a.parquet",
            1000,
            vec![RowGroupAccess::Scan, selection(10, 490)],
        );
        let groups = vec![FileGroup::new(vec![file])];
        let result =
            repartition_by_access_plan(&groups, 2, 0, false).expect("should split");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn allocation_never_exceeds_target() {
        let access = RowGroupAccess::Scan;
        let group_items = vec![
            vec![WorkItem {
                file_idx: 0,
                kind: ItemKind::RowGroup(0, &access),
                validation_accesses: Vec::new(),
                weight: 2,
            }],
            vec![WorkItem {
                file_idx: 0,
                kind: ItemKind::RowGroup(0, &access),
                validation_accesses: Vec::new(),
                weight: 2,
            }],
            (0..3)
                .map(|rg| WorkItem {
                    file_idx: 0,
                    kind: ItemKind::RowGroup(rg, &access),
                    validation_accesses: Vec::new(),
                    weight: 100,
                })
                .collect(),
        ];
        let allocation = allocate_partitions(&[2, 2, 300], &group_items, 4);
        assert_eq!(allocation, vec![1, 1, 2]);
        assert_eq!(allocation.iter().sum::<usize>(), 4);
    }

    #[test]
    fn all_skip_groups_do_not_consume_parallelism() {
        let mut groups: Vec<FileGroup> = (0..3)
            .map(|idx| {
                FileGroup::new(vec![planned_file(
                    &format!("empty-{idx}.parquet"),
                    100,
                    vec![RowGroupAccess::Skip],
                )])
            })
            .collect();
        groups.push(FileGroup::new(vec![planned_file(
            "hot.parquet",
            400,
            vec![
                selection(10, 90),
                selection(10, 90),
                selection(10, 90),
                selection(10, 90),
            ],
        )]));

        let result = repartition_by_access_plan(&groups, 4, 0, true)
            .expect("empty groups must not block hot-group splitting");
        assert_eq!(result.len(), 4);

        let hot_files: Vec<&PartitionedFile> = result
            .iter()
            .flat_map(|group| group.iter())
            .filter(|file| file.path().as_ref().ends_with("hot.parquet"))
            .collect();
        assert_eq!(hot_files.len(), 4);
        let mut hot_row_groups: Vec<usize> =
            hot_files.into_iter().flat_map(scanned_row_groups).collect();
        hot_row_groups.sort_unstable();
        assert_eq!(hot_row_groups, vec![0, 1, 2, 3]);

        for idx in 0..3 {
            let suffix = format!("empty-{idx}.parquet");
            let validation_files: Vec<&PartitionedFile> = result
                .iter()
                .flat_map(|group| group.iter())
                .filter(|file| file.path().as_ref().ends_with(&suffix))
                .collect();
            assert_eq!(validation_files.len(), 1);
            assert!(validation_files[0].range.is_none());
        }
    }

    #[test]
    fn all_empty_plans_use_one_validation_partition() {
        let groups = vec![FileGroup::new(vec![
            planned_file(
                "a.parquet",
                200,
                vec![RowGroupAccess::Skip, RowGroupAccess::Skip],
            ),
            planned_file("b.parquet", 100, vec![RowGroupAccess::Skip]),
        ])];

        let result = repartition_by_access_plan(&groups, 8, 0, false)
            .expect("empty plans are handled without byte fallback");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        for file in result[0].iter() {
            assert!(file.range.is_none());
            assert!(scanned_row_groups(file).is_empty());
        }
    }

    #[test]
    fn all_skip_file_is_retained_for_plan_validation() {
        // The first file's one-entry plan would be stale for a two-row-group
        // parquet file. Repartitioning must not silently remove it before the
        // opener can perform that metadata validation.
        let invalid = planned_file("invalid.parquet", 200, vec![RowGroupAccess::Skip]);
        let active = planned_file(
            "active.parquet",
            200,
            vec![selection(10, 90), selection(10, 90)],
        );
        let groups = vec![FileGroup::new(vec![invalid, active])];

        let result = repartition_by_access_plan(&groups, 2, 0, false)
            .expect("validation-only file must be retained");
        let invalid_out: Vec<&PartitionedFile> = result
            .iter()
            .flat_map(|group| group.iter())
            .filter(|file| file.path().as_ref().ends_with("invalid.parquet"))
            .collect();
        assert_eq!(invalid_out.len(), 1);
        assert_eq!(
            invalid_out[0]
                .extensions
                .get::<ParquetAccessPlan>()
                .unwrap()
                .len(),
            1
        );
        assert!(invalid_out[0].range.is_none());
    }

    #[test]
    fn zero_row_selection_is_retained_once_for_validation() {
        let zero_selection =
            RowGroupAccess::Selection(RowSelection::from(vec![RowSelector::skip(100)]));
        let file = planned_file(
            "a.parquet",
            200,
            vec![selection(10, 90), zero_selection.clone()],
        );
        let groups = vec![FileGroup::new(vec![file])];

        let result = repartition_by_access_plan(&groups, 2, 0, false)
            .expect("zero-row selection must reach the opener");
        let output_files: Vec<&PartitionedFile> =
            result.iter().flat_map(|group| group.iter()).collect();
        assert_eq!(output_files.len(), 1);
        let zero_selections = output_files
            .iter()
            .flat_map(|file| {
                file.extensions
                    .get::<ParquetAccessPlan>()
                    .unwrap()
                    .inner()
            })
            .filter(|access| {
                matches!(access, RowGroupAccess::Selection(selection) if selection.row_count() == 0)
            })
            .count();
        assert_eq!(zero_selections, 1);
        assert_eq!(scanned_row_groups(output_files[0]), vec![0, 1]);
    }

    #[test]
    fn fully_matched_flags_survive_subplans() {
        let mut plan = ParquetAccessPlan::new(vec![selection(10, 90), selection(10, 90)]);
        plan.mark_fully_matched(0);
        let mut file = plain_file("a.parquet", 200);
        file.extensions.insert(plan);
        let groups = vec![FileGroup::new(vec![file])];

        let result =
            repartition_by_access_plan(&groups, 2, 0, false).expect("should split");
        for file in result.iter().flat_map(|group| group.iter()) {
            let plan = file.extensions.get::<ParquetAccessPlan>().unwrap();
            for rg in plan.row_group_indexes() {
                assert_eq!(plan.is_fully_matched(rg), rg == 0);
            }
        }
    }

    #[test]
    fn mixed_files_without_statistics_are_weighted_by_bytes() {
        let planned = planned_file("planned.parquet", 100, vec![selection(10, 90)]);
        let large = plain_file_without_statistics("large.parquet", 16 * 1024 * 1024);
        let small = plain_file_without_statistics("small.parquet", 1024 * 1024);
        let groups = vec![FileGroup::new(vec![planned, large, small])];

        let result = repartition_by_access_plan(&groups, 4, 0, false)
            .expect("mixed scan should repartition");
        let count = |suffix: &str| {
            result
                .iter()
                .flat_map(|group| group.iter())
                .filter(|file| file.path().as_ref().ends_with(suffix))
                .count()
        };
        assert_eq!(count("large.parquet"), 4);
        assert_eq!(count("small.parquet"), 1);
        assert!(result.len() <= 4);
    }

    #[test]
    fn mixed_all_skip_plan_and_plain_file_keep_all_work() {
        let skipped = planned_file("skip.parquet", 100, vec![RowGroupAccess::Skip]);
        let plain = plain_file_with_size("plain.parquet", 100, 4 * 1024 * 1024);
        let plain_size = plain.object_meta.size;
        let groups = vec![FileGroup::new(vec![skipped, plain])];

        let result = repartition_by_access_plan(&groups, 4, 0, false)
            .expect("mixed scan should use the access-aware path");
        let skipped_files: Vec<&PartitionedFile> = result
            .iter()
            .flat_map(|group| group.iter())
            .filter(|file| file.path().as_ref().ends_with("skip.parquet"))
            .collect();
        assert_eq!(skipped_files.len(), 1, "validate the skipped plan once");
        assert!(skipped_files[0].range.is_none());

        let mut ranges: Vec<(u64, u64)> = result
            .iter()
            .flat_map(|group| group.iter())
            .filter(|file| file.path().as_ref().ends_with("plain.parquet"))
            .map(|file| {
                let range = file.range.as_ref().expect("plain file byte range");
                (range.start as u64, range.end as u64)
            })
            .collect();
        ranges.sort_unstable();
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0].0, 0);
        assert!(ranges.windows(2).all(|window| window[0].1 == window[1].0));
        assert_eq!(ranges.last().unwrap().1, plain_size);
    }
}
