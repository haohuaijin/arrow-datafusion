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

//! Hash aggregation

use std::sync::Arc;
use std::task::{Context, Poll};

use crate::aggregates::group_values::{new_group_values, GroupValues};
use crate::aggregates::{
    evaluate_group_by, evaluate_many, AggregateMode, PhysicalGroupBy,
};
use crate::metrics::{BaselineMetrics, RecordOutput};
use crate::{aggregates, PhysicalExpr};
use crate::{RecordBatchStream, SendableRecordBatchStream};

use arrow::array::*;
use arrow::datatypes::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::TaskContext;
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_physical_expr::GroupsAccumulatorAdapter;

use super::order::GroupOrdering;
use super::AggregateExec;
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use futures::ready;
use futures::stream::{Stream, StreamExt};
use log::debug;

#[derive(Debug, Clone)]
/// This object tracks the aggregation phase (input/output)
pub(crate) enum ExecutionState {
    ReadingInput,
    /// When producing output, the remaining rows to output are stored
    /// here and are sliced off as needed in batch_size chunks
    ProducingOutput(RecordBatch),
    /// All input has been consumed and all groups have been emitted
    Done,
}

pub struct GroupedHashAggregateStream {
    // ========================================================================
    // PROPERTIES:
    // These fields are initialized at the start and remain constant throughout
    // the execution.
    // ========================================================================
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    mode: AggregateMode,

    /// Arguments to pass to each accumulator.
    ///
    /// The arguments in `accumulator[i]` is passed `aggregate_arguments[i]`
    ///
    /// The argument to each accumulator is itself a `Vec` because
    /// some aggregates such as `CORR` can accept more than one
    /// argument.
    aggregate_arguments: Vec<Vec<Arc<dyn PhysicalExpr>>>,

    /// GROUP BY expressions
    group_by: PhysicalGroupBy,

    /// max rows in output RecordBatches
    batch_size: usize,

    // ========================================================================
    // STATE FLAGS:
    // These fields will be updated during the execution. And control the flow of
    // the execution.
    // ========================================================================
    /// Tracks if this stream is generating input or output
    exec_state: ExecutionState,

    /// Have we seen the end of the input
    input_done: bool,

    // ========================================================================
    // STATE BUFFERS:
    // These fields will accumulate intermediate results during the execution.
    // ========================================================================
    /// An interning store of group keys
    group_values: Box<dyn GroupValues>,

    /// scratch space for the current input [`RecordBatch`] being
    /// processed. Reused across batches here to avoid reallocations
    current_group_indices: Vec<usize>,

    /// Accumulators, one for each `AggregateFunctionExpr` in the query
    ///
    /// For example, if the query has aggregates, `SUM(x)`,
    /// `COUNT(y)`, there will be two accumulators, each one
    /// specialized for that particular aggregate and its input types
    accumulators: Vec<Box<dyn GroupsAccumulator>>,

    // ========================================================================
    // EXECUTION RESOURCES:
    // Fields related to managing execution resources and monitoring performance.
    // ========================================================================
    /// The memory reservation for this grouping
    reservation: MemoryReservation,

    /// Execution metrics
    baseline_metrics: BaselineMetrics,
}

impl GroupedHashAggregateStream {
    /// Create a new GroupedHashAggregateStream
    #[allow(dead_code)]
    pub fn new(
        agg: &AggregateExec,
        context: Arc<TaskContext>,
        input: SendableRecordBatchStream,
        schema: SchemaRef,
    ) -> Result<Self> {
        debug!("Creating GroupedHashAggregateStream");
        let agg_schema = schema;
        let agg_group_by = agg.group_by.clone();

        let batch_size = context.session_config().batch_size();
        // let input = agg.input.execute(partition, Arc::clone(&context))?;
        let baseline_metrics = BaselineMetrics::new(&agg.metrics, 0);

        let timer = baseline_metrics.elapsed_compute().timer();

        let aggregate_exprs = agg.aggr_expr.clone();

        let aggregate_arguments = aggregates::aggregate_expressions(
            &agg.aggr_expr,
            &agg.mode,
            agg_group_by.num_group_exprs(),
        )?;

        // Instantiate the accumulators
        let accumulators: Vec<_> = aggregate_exprs
            .iter()
            .map(create_group_accumulator)
            .collect::<Result<_>>()?;

        let group_schema = agg_group_by.group_schema(&agg.input().schema())?;

        let name = format!("GroupedHashAggregateStream[]");
        let reservation = MemoryConsumer::new(name)
            .with_can_spill(true)
            .register(context.memory_pool());

        let group_values = new_group_values(group_schema, &GroupOrdering::None)?;
        timer.done();

        let exec_state = ExecutionState::ReadingInput;

        Ok(GroupedHashAggregateStream {
            schema: agg_schema,
            input,
            mode: agg.mode,
            accumulators,
            aggregate_arguments,
            group_by: agg_group_by,
            reservation,
            group_values,
            current_group_indices: Default::default(),
            exec_state,
            baseline_metrics,
            batch_size,
            input_done: false,
        })
    }
}

/// Create an accumulator for `agg_expr` -- a [`GroupsAccumulator`] if
/// that is supported by the aggregate, or a
/// [`GroupsAccumulatorAdapter`] if not.
pub(crate) fn create_group_accumulator(
    agg_expr: &Arc<AggregateFunctionExpr>,
) -> Result<Box<dyn GroupsAccumulator>> {
    if agg_expr.groups_accumulator_supported() {
        agg_expr.create_groups_accumulator()
    } else {
        // Note in the log when the slow path is used
        debug!(
            "Creating GroupsAccumulatorAdapter for {}: {agg_expr:?}",
            agg_expr.name()
        );
        let agg_expr_captured = Arc::clone(agg_expr);
        let factory = move || agg_expr_captured.create_accumulator();
        Ok(Box::new(GroupsAccumulatorAdapter::new(factory)))
    }
}

/// Extracts a successful Ok(_) or returns Poll::Ready(Some(Err(e))) with errors
macro_rules! extract_ok {
    ($RES: expr) => {{
        match $RES {
            Ok(v) => v,
            Err(e) => return Poll::Ready(Some(Err(e))),
        }
    }};
}

impl Stream for GroupedHashAggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();

        loop {
            match &self.exec_state {
                ExecutionState::ReadingInput => {
                    match ready!(self.input.poll_next_unpin(cx)) {
                        // New batch to aggregate in partial aggregation operator
                        Some(Ok(batch)) if self.mode == AggregateMode::Partial => {
                            let timer = elapsed_compute.timer();

                            // Do the grouping
                            extract_ok!(self.group_aggregate_batch(batch));

                            // If we can begin emitting rows, do so,
                            // otherwise keep consuming input
                            assert!(!self.input_done);

                            extract_ok!(self.emit_early_if_necessary());

                            timer.done();
                        }

                        // New batch to aggregate in terminal aggregation operator
                        // (Final/FinalPartitioned/Single/SinglePartitioned)
                        Some(Ok(batch)) => {
                            let timer = elapsed_compute.timer();

                            // Do the grouping
                            extract_ok!(self.group_aggregate_batch(batch));

                            // If we can begin emitting rows, do so,
                            // otherwise keep consuming input
                            assert!(!self.input_done);

                            timer.done();
                        }

                        // Found error from input stream
                        Some(Err(e)) => {
                            // inner had error, return to caller
                            return Poll::Ready(Some(Err(e)));
                        }

                        // Found end from input stream
                        None => {
                            // inner is done, emit all rows and switch to producing output
                            extract_ok!(self.set_input_done_and_produce_output());
                        }
                    }
                }

                ExecutionState::ProducingOutput(batch) => {
                    // slice off a part of the batch, if needed
                    let output_batch;
                    let size = self.batch_size;
                    (self.exec_state, output_batch) = if batch.num_rows() <= size {
                        (
                            if self.input_done {
                                ExecutionState::Done
                            } else {
                                ExecutionState::ReadingInput
                            },
                            batch.clone(),
                        )
                    } else {
                        // output first batch_size rows
                        let size = self.batch_size;
                        let num_remaining = batch.num_rows() - size;
                        let remaining = batch.slice(size, num_remaining);
                        let output = batch.slice(0, size);
                        (ExecutionState::ProducingOutput(remaining), output)
                    };
                    // Empty record batches should not be emitted.
                    // They need to be treated as  [`Option<RecordBatch>`]es and handled separately
                    debug_assert!(output_batch.num_rows() > 0);
                    return Poll::Ready(Some(Ok(
                        output_batch.record_output(&self.baseline_metrics)
                    )));
                }

                ExecutionState::Done => {
                    // release the memory reservation since sending back output batch itself needs
                    // some memory reservation, so make some room for it.
                    self.clear_all();
                    let _ = self.update_memory_reservation();
                    return Poll::Ready(None);
                }
            }
        }
    }
}

impl RecordBatchStream for GroupedHashAggregateStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl GroupedHashAggregateStream {
    /// Perform group-by aggregation for the given [`RecordBatch`].
    fn group_aggregate_batch(&mut self, batch: RecordBatch) -> Result<()> {
        // Evaluate the grouping expressions
        let group_by_values = evaluate_group_by(&self.group_by, &batch)?;

        // Evaluate the aggregation expressions.
        let input_values = evaluate_many(&self.aggregate_arguments, &batch)?;

        for group_values in &group_by_values {
            // calculate the group indices for each input row
            self.group_values
                .intern(group_values, &mut self.current_group_indices)?;
            let group_indices = &self.current_group_indices;

            // Update ordering information if necessary
            let total_num_groups = self.group_values.len();

            // Gather the inputs to call the actual accumulator
            let t = self.accumulators.iter_mut().zip(input_values.iter());

            for (acc, values) in t {
                // Call the appropriate method on each aggregator with
                // the entire input row and the relevant group indexes
                match self.mode {
                    AggregateMode::Partial
                    | AggregateMode::Single
                    | AggregateMode::SinglePartitioned => {
                        acc.update_batch(values, group_indices, None, total_num_groups)?;
                    }
                    _ => {
                        // if aggregation is over intermediate states,
                        // use merge
                        acc.merge_batch(values, group_indices, None, total_num_groups)?;
                    }
                }
            }
        }

        match self.update_memory_reservation() {
            // Here we can ignore `insufficient_capacity_err` because we will spill later,
            // but at least one batch should fit in the memory
            Err(DataFusionError::ResourcesExhausted(_))
                if self.group_values.len() >= self.batch_size =>
            {
                Ok(())
            }
            other => other,
        }
    }

    fn update_memory_reservation(&mut self) -> Result<()> {
        let acc = self.accumulators.iter().map(|x| x.size()).sum::<usize>();
        let reservation_result = self.reservation.try_resize(
            acc + self.group_values.size() + self.current_group_indices.allocated_size(),
        );

        reservation_result
    }

    /// Create an output RecordBatch with the group keys and
    /// accumulator states/values specified in emit_to
    fn emit(&mut self, emit_to: EmitTo) -> Result<Option<RecordBatch>> {
        let schema = self.schema();
        if self.group_values.is_empty() {
            return Ok(None);
        }

        let mut output = self.group_values.emit(emit_to)?;

        // Next output each aggregate value
        for acc in self.accumulators.iter_mut() {
            // match self.mode {
            //     AggregateMode::Partial => output.extend(acc.state(emit_to)?),
            //     AggregateMode::Final
            //     | AggregateMode::FinalPartitioned
            //     | AggregateMode::Single
            //     | AggregateMode::SinglePartitioned => output.push(acc.evaluate(emit_to)?),
            // }
            output.extend(acc.state(emit_to)?);
        }

        // emit reduces the memory usage. Ignore Err from update_memory_reservation. Even if it is
        // over the target memory size after emission, we can emit again rather than returning Err.
        let _ = self.update_memory_reservation();
        let batch = RecordBatch::try_new(schema, output)?;
        debug_assert!(batch.num_rows() > 0);
        Ok(Some(batch))
    }

    /// Clear memory and shirk capacities to the size of the batch.
    fn clear_shrink(&mut self, batch: &RecordBatch) {
        self.group_values.clear_shrink(batch);
        self.current_group_indices.clear();
        self.current_group_indices.shrink_to(batch.num_rows());
    }

    /// Clear memory and shirk capacities to zero.
    fn clear_all(&mut self) {
        let s = self.schema();
        self.clear_shrink(&RecordBatch::new_empty(s));
    }

    /// Emit if the used memory exceeds the target for partial aggregation.
    /// Currently only [`GroupOrdering::None`] is supported for early emitting.
    /// TODO: support group_ordering for early emitting
    fn emit_early_if_necessary(&mut self) -> Result<()> {
        if self.group_values.len() >= self.batch_size
            && self.update_memory_reservation().is_err()
        {
            assert_eq!(self.mode, AggregateMode::Partial);
            let n = self.group_values.len() / self.batch_size * self.batch_size;
            if let Some(batch) = self.emit(EmitTo::First(n))? {
                self.exec_state = ExecutionState::ProducingOutput(batch);
            };
        }
        Ok(())
    }

    /// common function for signalling end of processing of the input stream
    fn set_input_done_and_produce_output(&mut self) -> Result<()> {
        self.input_done = true;
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();
        let timer = elapsed_compute.timer();
        let batch = self.emit(EmitTo::All)?;
        self.exec_state =
            batch.map_or(ExecutionState::Done, ExecutionState::ProducingOutput);
        timer.done();
        Ok(())
    }
}
