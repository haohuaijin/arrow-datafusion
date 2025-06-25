use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion, TreeNodeVisitor};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::RecordBatchStream;
use datafusion::physical_plan::aggregates::merge_phase::GroupedHashAggregateStream;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use futures::{Stream, StreamExt};
use std::fs::File;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = SessionConfig::from_env()?;
    let ctx = SessionContext::new_with_config(config);

    let table = create_memtable()?;
    ctx.register_table("default", Arc::new(table))?;

    let start = std::time::Instant::now();
    let sql = "select clientip, count(*) as cnt from default group by clientip order by cnt desc limit 3";
    let plan = ctx.state().create_logical_plan(&sql).await?;
    let physical_plan = ctx.state().create_physical_plan(&plan).await?;

    let partial_plan = get_partial_aggregate_plan(physical_plan.clone());
    let final_plan = get_final_aggregate_plan(physical_plan.clone());

    let stream = ArrowRecordBatchStream::new(
        get_schema(),
        "/Users/huaijinhao/Downloads/arrow/1750330800000000_1750334400000000.arrow",
    )?;
    let mut group_hash_aggregate_stream = GroupedHashAggregateStream::new(
        &final_plan,
        ctx.task_ctx(),
        Box::pin(stream),
        partial_plan.schema(),
    )?;

    // get result from group_hash_aggregate_stream
    let mut result = Vec::new();
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    while let Poll::Ready(Some(batch)) =
        group_hash_aggregate_stream.poll_next_unpin(&mut cx)
    {
        result.push(batch.unwrap());
    }

    // write to disk
    let file = File::create("/Users/huaijinhao/Downloads/arrow/result.arrow")?;
    let mut writer =
        arrow::ipc::writer::FileWriter::try_new(file, &partial_plan.schema())?;
    for batch in result {
        writer.write(&batch)?;
    }
    writer.finish()?;

    // let ret = datafusion::physical_plan::collect(physical_plan.clone(), ctx.task_ctx()).await?;
    println!("Frist Time: {:?}", start.elapsed());

    Ok(())
}

fn get_partial_aggregate_plan(plan: Arc<dyn ExecutionPlan>) -> AggregateExec {
    let mut visitor = AggregateVisitor::new(true);
    let _ = plan.visit(&mut visitor);
    let data = visitor.get_data();
    data.unwrap()
        .as_any()
        .downcast_ref::<AggregateExec>()
        .unwrap()
        .clone()
}

fn get_final_aggregate_plan(plan: Arc<dyn ExecutionPlan>) -> AggregateExec {
    let mut visitor = AggregateVisitor::new(false);
    let _ = plan.visit(&mut visitor);
    let data = visitor.get_data();
    data.unwrap()
        .as_any()
        .downcast_ref::<AggregateExec>()
        .unwrap()
        .clone()
}

pub struct AggregateVisitor {
    is_partial: bool,
    data: Option<Arc<dyn ExecutionPlan>>,
}

impl AggregateVisitor {
    pub fn new(is_partial: bool) -> Self {
        Self {
            is_partial,
            data: None,
        }
    }

    pub fn get_data(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        self.data.as_ref()
    }
}

impl Default for AggregateVisitor {
    fn default() -> Self {
        Self::new(false)
    }
}

impl<'n> TreeNodeVisitor<'n> for AggregateVisitor {
    type Node = Arc<dyn ExecutionPlan>;

    fn f_up(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
        if node.name() == "AggregateExec" {
            let agg = node.as_any().downcast_ref::<AggregateExec>().unwrap();
            if self.is_partial {
                if *agg.mode() == AggregateMode::Partial {
                    self.data = Some(node.clone());
                    Ok(TreeNodeRecursion::Stop)
                } else {
                    Ok(TreeNodeRecursion::Continue)
                }
            } else {
                if *agg.mode() == AggregateMode::Final
                    || *agg.mode() == AggregateMode::FinalPartitioned
                {
                    self.data = Some(node.clone());
                    Ok(TreeNodeRecursion::Stop)
                } else {
                    Ok(TreeNodeRecursion::Continue)
                }
            }
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    }
}

fn create_memtable() -> Result<MemTable> {
    MemTable::try_new(get_schema(), vec![create_record_batch()?])
}

fn create_record_batch() -> Result<Vec<RecordBatch>> {
    let id_array = StringArray::from(vec!["127.0.0.1"]);
    let name_array = StringArray::from(vec!["zhangsan"]);

    Ok(vec![
        RecordBatch::try_new(
            get_schema(),
            vec![Arc::new(id_array.clone()), Arc::new(name_array.clone())],
        )
        .unwrap(),
        RecordBatch::try_new(
            get_schema(),
            vec![Arc::new(id_array), Arc::new(name_array)],
        )
        .unwrap(),
    ])
}

fn get_schema() -> SchemaRef {
    SchemaRef::new(Schema::new(vec![
        Field::new("clientip", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

struct ArrowRecordBatchStream {
    schema: SchemaRef,
    reader: arrow::ipc::reader::FileReader<File>,
    current_batch: Option<RecordBatch>,
}

impl ArrowRecordBatchStream {
    pub fn new(
        schema: SchemaRef,
        file_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = arrow::ipc::reader::FileReader::try_new(file, None)?;
        Ok(Self {
            schema,
            reader,
            current_batch: None,
        })
    }
}

impl Stream for ArrowRecordBatchStream {
    type Item = Result<RecordBatch, DataFusionError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.current_batch.is_none() {
            if let Some(batch) = this.reader.next() {
                this.current_batch = Some(batch.unwrap());
            } else {
                return Poll::Ready(None);
            }
        }

        let batch = this.current_batch.take();
        Poll::Ready(Some(Ok(batch.unwrap())))
    }
}

impl RecordBatchStream for ArrowRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
