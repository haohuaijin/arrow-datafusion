use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::{
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    prelude::{SessionConfig, SessionContext},
};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let mut config = SessionConfig::from_env()?;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config =
        config.set_bool("datafusion.optimizer.enable_dynamic_filter_pushdown", false);
    config.options_mut().optimizer.enable_window_topk_pushdown = false;
    let ctx = SessionContext::new_with_config(config);

    // let path = "/Users/huaijinhao/gitrepo/example/datafusion-example/clickbench/parquet";
    let path = "/Users/huaijinhao/gitrepo/example/datafusion-example/logs/default_flat";
    let file_options = ListingOptions::new(Arc::new(ParquetFormat::default()))
        .with_session_config_options(ctx.state().config());
    let prefix = ListingTableUrl::parse(path).unwrap();
    let config = ListingTableConfig::new(prefix)
        .with_listing_options(file_options)
        .with_schema(Arc::new(get_schema()));
    let table = ListingTable::try_new(config)?;
    ctx.register_table("default", Arc::new(table))?;

    // let sql = "SELECT kubernetes_container_name, count(*) as cnt FROM default where log like '%datafusion%' and _timestamp >= 17498156644757 AND _timestamp <= 174981566564084400 group by kubernetes_container_name order by cnt desc limit 10";
    // let sql = "SELECT kubernetes_container_name, count(*) as cnt FROM default where log like '%datafusion%' group by kubernetes_container_name order by cnt desc limit 10";
    // let sql = "SELECT kubernetes_container_name, count(*) as cnt FROM default where _timestamp >= 17498156644757 AND _timestamp <= 174981566564084400 group by kubernetes_container_name order by cnt desc limit 10";
    // let sql = "SELECT kubernetes_container_name, count(*) as cnt FROM default group by kubernetes_container_name order by cnt desc limit 10";
    let sql = "SELECT
                        kubernetes_namespace_name,
                        kubernetes_container_name,
                        COUNT(*) AS cnt,
                        ROW_NUMBER() OVER (
                          PARTITION BY kubernetes_container_name
                          ORDER BY COUNT(*) DESC
                        ) AS rn
                        FROM
                          default
                        GROUP BY
                          kubernetes_namespace_name,
                          kubernetes_container_name
                        QUALIFY
                          rn <= 2 
                        ORDER BY kubernetes_namespace_name DESC, cnt DESC;";
    // let sql = "SELECT * FROM default order by _timestamp desc limit 10";
    let _ = ctx.sql(sql).await?.explain(false, true)?.show().await?;
    println!("SQL: {}", sql);
    println!("Time: {:?}", start.elapsed());

    Ok(())
}

fn get_schema() -> Schema {
    Schema::new(vec![
        Field::new("_timestamp", DataType::Int64, true),
        Field::new("log", DataType::Utf8View, true),
        Field::new("kubernetes_namespace_name", DataType::Utf8View, true),
        Field::new("kubernetes_container_name", DataType::Utf8View, true),
        Field::new("URL", DataType::Utf8View, true),
    ])
}
