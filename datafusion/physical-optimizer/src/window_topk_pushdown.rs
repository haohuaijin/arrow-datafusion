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
//! Unless required by applicable law or agreed to in writing,
//   software distributed under the License is distributed on an
//   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied.  See the License for the
//   specific language governing permissions and limitations
//   under the License.

//! Window TopK Pushdown Optimizer
//!
//! Optimizes queries with `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) < N`
//! patterns by pushing the TopK limit down into the sort operation.
//!
//! ## Example
//!
//! Before optimization:
//! ```text
//! FilterExec: row_number < 10
//!   ProjectionExec
//!     WindowExec: ROW_NUMBER() PARTITION BY [a] ORDER BY [b]
//!       SortExec: [a, b]
//!         ...
//! ```
//!
//! After optimization:
//! ```text
//! FilterExec: row_number < 10
//!   ProjectionExec
//!     WindowExec: ROW_NUMBER() PARTITION BY [a] ORDER BY [b]
//!       PartitionedTopKSortExec: partition_by=[a], order_by=[b], fetch=10
//!         ...
//! ```

use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::partitioned_topk_sort::PartitionedTopKSortExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::windows::BoundedWindowAggExec;

use crate::PhysicalOptimizerRule;

/// Optimizer rule that pushes TopK limits down past window functions
/// when analyzing ROW_NUMBER() OVER (...) < N patterns.
#[derive(Default, Clone, Debug)]
pub struct WindowTopKPushdown;

impl WindowTopKPushdown {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for WindowTopKPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Check if the optimization is enabled
        if !config.optimizer.enable_window_topk_pushdown {
            return Ok(plan);
        }

        plan.transform_down(|node| {
            // Try to match the pattern: Filter -> (Projection)? -> Window -> Sort
            if let Some(optimization) = try_optimize_window_topk(&node)? {
                Ok(Transformed::yes(optimization))
            } else {
                Ok(Transformed::no(node))
            }
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "WindowTopKPushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Try to detect and optimize the window TopK pattern
fn try_optimize_window_topk(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    // Step 1: Check if this is a FilterExec
    let filter = match plan.as_any().downcast_ref::<FilterExec>() {
        Some(f) => f,
        None => return Ok(None),
    };

    // Step 2: Extract limit from filter predicate (e.g., row_number < 10)
    let limit = match extract_row_number_limit(filter.predicate()) {
        Some(l) => l,
        None => return Ok(None),
    };

    // Step 3: Navigate through optional projection to find window
    let filter_input = filter.input();
    let (window, projection) =
        match filter_input.as_any().downcast_ref::<ProjectionExec>() {
            Some(proj) => {
                // Filter -> Projection -> Window
                match proj.input().as_any().downcast_ref::<BoundedWindowAggExec>() {
                    Some(w) => (w, Some(proj)),
                    None => return Ok(None),
                }
            }
            None => {
                // Filter -> Window (no projection)
                match filter_input.as_any().downcast_ref::<BoundedWindowAggExec>() {
                    Some(w) => (w, None),
                    None => return Ok(None),
                }
            }
        };

    // Step 4: Check if window function is ROW_NUMBER
    if !is_row_number_window(window) {
        return Ok(None);
    }

    // Step 5: Find the SortExec below the window
    let sort = match window.input().as_any().downcast_ref::<SortExec>() {
        Some(s) => s,
        None => return Ok(None),
    };

    // Step 6: Verify that sort expressions match window partition_by + order_by
    let window_expr = &window.window_expr()[0];
    let partition_by = window_expr.partition_by();
    let order_by = window_expr.order_by();

    // Build expected sort expressions: partition_by columns followed by order_by columns
    let expected_partition_len = partition_by.len();
    let total_expected_len = expected_partition_len + order_by.len();

    if sort.expr().len() != total_expected_len {
        return Ok(None);
    }

    // Verify partition_by columns match sort prefix
    if !exprs_match(&sort.expr()[..expected_partition_len], partition_by) {
        return Ok(None);
    }

    // Verify order_by columns match remaining sort expressions
    if !sort_exprs_match(&sort.expr()[expected_partition_len..], order_by) {
        return Ok(None);
    }

    // Step 7: Replace SortExec with PartitionedTopKSortExec
    let partitioned_sort = Arc::new(PartitionedTopKSortExec::try_new(
        Arc::clone(sort.input()),
        sort.expr().clone(),
        expected_partition_len,
        limit,
    )?) as Arc<dyn ExecutionPlan>;

    // Reconstruct the plan with the new sort
    let window_arc = Arc::new(window.clone()) as Arc<dyn ExecutionPlan>;
    let new_window = window_arc.with_new_children(vec![partitioned_sort])?;

    let new_filter_input: Arc<dyn ExecutionPlan> = if let Some(proj) = projection {
        let proj_arc = Arc::new(proj.clone()) as Arc<dyn ExecutionPlan>;
        proj_arc.with_new_children(vec![new_window])?
    } else {
        new_window
    };

    let filter_arc = Arc::new(filter.clone()) as Arc<dyn ExecutionPlan>;
    let new_filter = filter_arc.with_new_children(vec![new_filter_input])?;

    Ok(Some(new_filter))
}

/// Extract the limit value from a filter predicate like "row_number < 10" or "row_number <= 10"
fn extract_row_number_limit(predicate: &Arc<dyn PhysicalExpr>) -> Option<usize> {
    let binary = predicate.as_any().downcast_ref::<BinaryExpr>()?;

    // Check if this is a comparison with row_number column
    let (column_name, limit_value, operator) = if let Some(col) =
        binary.left().as_any().downcast_ref::<Column>()
    {
        // row_number < N or row_number <= N
        let scalar = binary
            .right()
            .as_any()
            .downcast_ref::<datafusion_physical_expr::expressions::Literal>()?;
        (col.name().to_string(), scalar.value().clone(), *binary.op())
    } else if let Some(col) = binary.right().as_any().downcast_ref::<Column>() {
        // N > row_number or N >= row_number
        let scalar = binary
            .left()
            .as_any()
            .downcast_ref::<datafusion_physical_expr::expressions::Literal>()?;
        // Flip the operator
        let flipped_op = match *binary.op() {
            Operator::Gt => Operator::Lt,
            Operator::GtEq => Operator::LtEq,
            Operator::Lt => Operator::Gt,
            Operator::LtEq => Operator::GtEq,
            _ => return None,
        };
        (col.name().to_string(), scalar.value().clone(), flipped_op)
    } else {
        return None;
    };

    // Check if column name contains "row_number" (case insensitive)
    if !column_name.to_lowercase().contains("row_number") {
        return None;
    }

    // Extract the numeric limit
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

    // Adjust limit based on operator
    let adjusted_limit = match operator {
        Operator::Lt => limit - 1, // row_number < 10 means we want up to 9
        Operator::LtEq => limit,   // row_number <= 10 means we want up to 10
        Operator::Eq => limit,     // row_number = 10 is unusual but treat as <= 10
        _ => return None,
    };

    if adjusted_limit > 0 {
        Some(adjusted_limit as usize)
    } else {
        None
    }
}

/// Check if the window expression is ROW_NUMBER
fn is_row_number_window(window: &BoundedWindowAggExec) -> bool {
    if window.window_expr().is_empty() {
        return false;
    }
    // Check the window function name
    window.window_expr()[0].name().contains("row_number")
}

/// Check if physical expressions match (used for partition_by comparison)
fn exprs_match(
    sort_exprs: &[PhysicalSortExpr],
    partition_exprs: &[Arc<dyn PhysicalExpr>],
) -> bool {
    if sort_exprs.len() != partition_exprs.len() {
        return false;
    }

    sort_exprs
        .iter()
        .zip(partition_exprs.iter())
        .all(|(sort_expr, partition_expr)| {
            // Compare the underlying expressions
            sort_expr.expr.eq(partition_expr)
        })
}

/// Check if sort expressions match (used for order_by comparison)
fn sort_exprs_match(
    sort_exprs1: &[PhysicalSortExpr],
    sort_exprs2: &[PhysicalSortExpr],
) -> bool {
    if sort_exprs1.len() != sort_exprs2.len() {
        return false;
    }

    sort_exprs1
        .iter()
        .zip(sort_exprs2.iter())
        .all(|(e1, e2)| e1.expr.eq(&e2.expr) && e1.options == e2.options)
}
