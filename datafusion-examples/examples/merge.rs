use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion, TreeNodeVisitor};
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::physical_plan::aggregates::merge_phase::GroupedHashAggregateStream;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use std::fs::File;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = SessionConfig::from_env()?;
    let ctx = SessionContext::new_with_config(config);

    let table = create_memtable()?;
    ctx.register_table("cdn_production", Arc::new(table))?;

    let sql = "select clientrequestpath, count(_timestamp) as cnt from cdn_production group by clientrequestpath order by cnt desc limit 3";
    let plan = ctx.state().create_logical_plan(&sql).await?;
    let physical_plan = ctx.state().create_physical_plan(&plan).await?;

    let final_plan = get_final_aggregate_plan(physical_plan.clone());

    let mut group_hash_aggregate_stream =
        GroupedHashAggregateStream::new(&final_plan, ctx.task_ctx())?;

    let start = std::time::Instant::now();
    let file = File::open(
        "/Users/huaijinhao/Downloads/big/1750809600000000_1750831200000000.arrow",
    )?;
    let reader = arrow::ipc::reader::FileReader::try_new(file, None)?;
    let mut record_batchs = Vec::new();
    for batch in reader {
        record_batchs.push(batch?);
    }

    println!("record_batchs.len: {}", record_batchs.len()); // 6286

    for batch in record_batchs.into_iter().skip(1500) {
        group_hash_aggregate_stream.group_aggregate_batch(batch)?;
    }

    let result = group_hash_aggregate_stream.get_final_result()?;

    let mut result_vec = Vec::new();
    for i in 0..result.num_rows() / 8192 {
        result_vec.push(result.slice(i * 8192, 8192));
    }

    // write to disk
    let file = File::create("/Users/huaijinhao/Downloads/big/result.arrow")?;
    let mut writer = arrow::ipc::writer::FileWriter::try_new(file, &result.schema())?;
    for batch in result_vec {
        writer.write(&batch)?;
    }
    writer.finish()?;

    println!("Frist Time: {:?}", start.elapsed());

    Ok(())
}

fn get_final_aggregate_plan(plan: Arc<dyn ExecutionPlan>) -> AggregateExec {
    let mut visitor = AggregateVisitor::new();
    let _ = plan.visit(&mut visitor);
    let data = visitor.get_data();
    data.unwrap()
        .as_any()
        .downcast_ref::<AggregateExec>()
        .unwrap()
        .clone()
}

pub struct AggregateVisitor {
    data: Option<Arc<dyn ExecutionPlan>>,
}

impl AggregateVisitor {
    pub fn new() -> Self {
        Self { data: None }
    }

    pub fn get_data(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        self.data.as_ref()
    }
}

impl Default for AggregateVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl<'n> TreeNodeVisitor<'n> for AggregateVisitor {
    type Node = Arc<dyn ExecutionPlan>;

    fn f_up(&mut self, node: &'n Self::Node) -> Result<TreeNodeRecursion> {
        if node.name() == "AggregateExec" {
            let agg = node.as_any().downcast_ref::<AggregateExec>().unwrap();
            if *agg.mode() == AggregateMode::Final
                || *agg.mode() == AggregateMode::FinalPartitioned
            {
                self.data = Some(node.clone());
                Ok(TreeNodeRecursion::Stop)
            } else {
                Ok(TreeNodeRecursion::Continue)
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
    let timestamp_array = Int64Array::from(vec![1750330800000000]);

    Ok(vec![
        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(id_array.clone()),
                Arc::new(timestamp_array.clone()),
                Arc::new(name_array.clone()),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            get_schema(),
            vec![
                Arc::new(id_array),
                Arc::new(timestamp_array),
                Arc::new(name_array),
            ],
        )
        .unwrap(),
    ])
}

fn get_schema() -> SchemaRef {
    SchemaRef::new(Schema::new(vec![
        Field::new("clientrequestpath", DataType::Utf8, false),
        Field::new("_timestamp", DataType::Int64, false),
        Field::new("clientip", DataType::Utf8, false),
    ]))
}
