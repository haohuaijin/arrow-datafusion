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
//   software distributed under the License is distributed on an
//   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied.  See the License for the
//   specific language governing permissions and limitations
//   under the License.

//! [`WindowTopKPushdown`] optimizes `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) < N`
//! patterns by replacing the [`SortExec`] with a [`PartitionedTopKSortExec`].
//!
//! For example, the following plan:
//! ```text
//! FilterExec: row_number < 10
//!   ProjectionExec
//!     BoundedWindowAggExec: ROW_NUMBER() PARTITION BY [a] ORDER BY [b]
//!       SortExec: [a, b]
//! ```
//!
//! Is rewritten to:
//! ```text
//! FilterExec: row_number < 10
//!   ProjectionExec
//!     BoundedWindowAggExec: ROW_NUMBER() PARTITION BY [a] ORDER BY [b]
//!       PartitionedTopKSortExec: partition_by=[a], order_by=[b], fetch=10
//! ```

use std::sync::Arc;

use crate::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_expr::utils::split_conjunction;
use datafusion_physical_expr::window::{StandardWindowExpr, WindowExpr};
use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::partitioned_topk_sort::PartitionedTopKSortExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowUDFExpr};

/// Result of extracting a TopK limit from a binary expression.
struct TopKInfo {
    /// The Column expression that references the ROW_NUMBER result
    col: Column,
    /// The extracted limit value (already adjusted for < vs <=)
    limit: usize,
}

/// Information about a ROW_NUMBER TopK pattern found in the filter predicate.
struct RowNumberTopK {
    /// The extracted limit (K)
    limit: usize,
    /// The index of the window expression that produces the ROW_NUMBER column
    window_expr_index: usize,
}

/// An optimizer rule that pushes TopK limits down past window functions
/// when it detects `ROW_NUMBER() OVER (...) < N` patterns
#[derive(Debug)]
pub struct WindowTopKPushdown {}

impl WindowTopKPushdown {
    pub fn new() -> Self {
        Self {}
    }

    /// Try to match Filter -> (Projection)? -> Window -> Sort and replace
    /// the SortExec with a PartitionedTopKSortExec.
    fn try_optimize(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        let filter = plan.as_any().downcast_ref::<FilterExec>()?;

        // Navigate through optional projection to find window
        let filter_input = filter.input();
        let (window, projection) =
            match filter_input.as_any().downcast_ref::<ProjectionExec>() {
                Some(proj) => {
                    let w = proj
                        .input()
                        .as_any()
                        .downcast_ref::<BoundedWindowAggExec>()?;
                    (w, Some(proj))
                }
                None => {
                    let w = filter_input
                        .as_any()
                        .downcast_ref::<BoundedWindowAggExec>()?;
                    (w, None)
                }
            };

        // Find the ROW_NUMBER limit from the predicate
        // (handles AND conjunctions, aliased columns, and any window expr position)
        let topk = Self::find_row_number_limit(filter.predicate(), window, projection)?;

        let sort = window.input().as_any().downcast_ref::<SortExec>()?;

        // Use the matched window expression (not hardcoded [0])
        let window_expr = &window.window_expr()[topk.window_expr_index];
        let partition_by = window_expr.partition_by();
        let order_by = window_expr.order_by();
        let partition_len = partition_by.len();

        // Verify sort expressions match window partition_by + order_by
        if sort.expr().len() != partition_len + order_by.len() {
            return None;
        }
        if !Self::partition_exprs_match(&sort.expr()[..partition_len], partition_by) {
            return None;
        }
        if !Self::sort_exprs_match(&sort.expr()[partition_len..], order_by) {
            return None;
        }

        // Replace SortExec with PartitionedTopKSortExec
        let partitioned_sort = Arc::new(
            PartitionedTopKSortExec::try_new(
                Arc::clone(sort.input()),
                sort.expr().clone(),
                partition_len,
                topk.limit,
            )
            .ok()?,
        ) as _;

        // Reconstruct the plan tree
        let new_window = (Arc::new(window.clone()) as Arc<dyn ExecutionPlan>)
            .with_new_children(vec![partitioned_sort])
            .ok()?;
        let new_filter_input = if let Some(proj) = projection {
            (Arc::new(proj.clone()) as Arc<dyn ExecutionPlan>)
                .with_new_children(vec![new_window])
                .ok()?
        } else {
            new_window
        };
        let new_plan = (Arc::new(filter.clone()) as Arc<dyn ExecutionPlan>)
            .with_new_children(vec![new_filter_input])
            .ok()?;

        Some(new_plan)
    }

    /// Search through the filter predicate (possibly an AND conjunction) to find
    /// a conjunct that limits a ROW_NUMBER() result to at most K rows.
    fn find_row_number_limit(
        predicate: &Arc<dyn PhysicalExpr>,
        window: &BoundedWindowAggExec,
        projection: Option<&ProjectionExec>,
    ) -> Option<RowNumberTopK> {
        let conjuncts = split_conjunction(predicate);

        for conjunct in &conjuncts {
            let Some(topk_info) = Self::extract_topk_from_binary(conjunct) else {
                continue;
            };

            let Some(window_expr_idx) = Self::resolve_column_to_window_expr_index(
                &topk_info.col,
                window,
                projection,
            ) else {
                continue;
            };

            let window_exprs = window.window_expr();
            if window_expr_idx >= window_exprs.len() {
                continue;
            }

            if Self::is_row_number_expr(&window_exprs[window_expr_idx]) {
                return Some(RowNumberTopK {
                    limit: topk_info.limit,
                    window_expr_index: window_expr_idx,
                });
            }
        }

        None
    }

    /// Try to extract a TopK limit from a single binary expression like `col <= N`.
    /// Does NOT validate that the column references a ROW_NUMBER expression.
    fn extract_topk_from_binary(expr: &Arc<dyn PhysicalExpr>) -> Option<TopKInfo> {
        let binary = expr.as_any().downcast_ref::<BinaryExpr>()?;

        let (col, limit_value, operator) =
            if let Some(col) = binary.left().as_any().downcast_ref::<Column>() {
                let scalar = binary
                    .right()
                    .as_any()
                    .downcast_ref::<datafusion_physical_expr::expressions::Literal>(
                )?;
                (col.clone(), scalar.value().clone(), *binary.op())
            } else if let Some(col) = binary.right().as_any().downcast_ref::<Column>() {
                let scalar = binary
                    .left()
                    .as_any()
                    .downcast_ref::<datafusion_physical_expr::expressions::Literal>(
                )?;
                let flipped_op = match *binary.op() {
                    Operator::Gt => Operator::Lt,
                    Operator::GtEq => Operator::LtEq,
                    Operator::Lt => Operator::Gt,
                    Operator::LtEq => Operator::GtEq,
                    _ => return None,
                };
                (col.clone(), scalar.value().clone(), flipped_op)
            } else {
                return None;
            };

        let limit = match &limit_value {
            ScalarValue::Int8(Some(v)) => *v as i64,
            ScalarValue::Int16(Some(v)) => *v as i64,
            ScalarValue::Int32(Some(v)) => *v as i64,
            ScalarValue::Int64(Some(v)) => *v,
            ScalarValue::UInt8(Some(v)) => *v as i64,
            ScalarValue::UInt16(Some(v)) => *v as i64,
            ScalarValue::UInt32(Some(v)) => *v as i64,
            ScalarValue::UInt64(Some(v)) => *v as i64,
            _ => return None,
        };

        let adjusted_limit = match operator {
            Operator::Lt => limit - 1,
            Operator::LtEq => limit,
            Operator::Eq => limit,
            _ => return None,
        };

        if adjusted_limit > 0 {
            Some(TopKInfo {
                col,
                limit: adjusted_limit as usize,
            })
        } else {
            None
        }
    }

    /// Returns true if the given window expression is a ROW_NUMBER() function.
    ///
    /// Uses the downcast chain: WindowExpr -> StandardWindowExpr ->
    /// StandardWindowFunctionExpr -> WindowUDFExpr -> fun().name()
    fn is_row_number_expr(expr: &Arc<dyn WindowExpr>) -> bool {
        let Some(std_expr) = expr.as_any().downcast_ref::<StandardWindowExpr>() else {
            return false;
        };
        let func_expr = std_expr.get_standard_func_expr();
        let Some(udf_expr) = func_expr.as_any().downcast_ref::<WindowUDFExpr>() else {
            return false;
        };
        udf_expr.fun().name() == "row_number"
    }

    /// Given a Column from the filter predicate, resolve it to a window expression index.
    ///
    /// Returns `Some(window_expr_index)` if the column refers to a window expression output,
    /// or `None` if it refers to an input column.
    fn resolve_column_to_window_expr_index(
        col: &Column,
        window: &BoundedWindowAggExec,
        projection: Option<&ProjectionExec>,
    ) -> Option<usize> {
        let input_col_count = window.input().schema().fields().len();

        // Determine the column index in the BoundedWindowAggExec output schema
        let window_output_col_index = if let Some(proj) = projection {
            // The column index refers to the projection's output.
            // Trace it back through the projection expression list.
            let proj_expr = proj.expr().get(col.index())?;
            let source_col = proj_expr.expr.as_any().downcast_ref::<Column>()?;
            source_col.index()
        } else {
            col.index()
        };

        // Window expression results start after the input columns
        if window_output_col_index >= input_col_count {
            Some(window_output_col_index - input_col_count)
        } else {
            None
        }
    }

    /// Returns true if sort expression columns match the given partition expressions.
    fn partition_exprs_match(
        sort_exprs: &[PhysicalSortExpr],
        partition_exprs: &[Arc<dyn PhysicalExpr>],
    ) -> bool {
        sort_exprs.len() == partition_exprs.len()
            && sort_exprs
                .iter()
                .zip(partition_exprs.iter())
                .all(|(sort_expr, partition_expr)| sort_expr.expr.eq(partition_expr))
    }

    /// Returns true if two sort expression slices are equivalent.
    fn sort_exprs_match(
        sort_exprs1: &[PhysicalSortExpr],
        sort_exprs2: &[PhysicalSortExpr],
    ) -> bool {
        sort_exprs1.len() == sort_exprs2.len()
            && sort_exprs1
                .iter()
                .zip(sort_exprs2.iter())
                .all(|(e1, e2)| e1.expr.eq(&e2.expr) && e1.options == e2.options)
    }
}

impl Default for WindowTopKPushdown {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizerRule for WindowTopKPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if config.optimizer.enable_window_topk_pushdown {
            plan.transform_down(|plan| {
                Ok(
                    if let Some(plan) = WindowTopKPushdown::try_optimize(&plan) {
                        Transformed::yes(plan)
                    } else {
                        Transformed::no(plan)
                    },
                )
            })
            .data()
        } else {
            Ok(plan)
        }
    }

    fn name(&self) -> &str {
        "WindowTopKPushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
