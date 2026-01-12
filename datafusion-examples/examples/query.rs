use std::sync::Arc;

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
    config.options_mut().execution.parquet.pushdown_filters = true;
    let ctx = SessionContext::new_with_config(config);

    let path = "/Users/huaijinhao/gitrepo/example/datafusion-example/clickbench/parquet/hits_0.parquet";
    let file_options = ListingOptions::new(Arc::new(ParquetFormat::default()))
        .with_session_config_options(ctx.state().config());
    let prefix = ListingTableUrl::parse(path).unwrap();
    let config = ListingTableConfig::new(prefix)
        .with_listing_options(file_options)
        .infer_schema(&ctx.state())
        .await?;
    let infer_schema_took = start.elapsed();
    let table = ListingTable::try_new(config)?;
    ctx.register_table("hits", Arc::new(table))?;

    let query_sql =
        "SELECT COUNT(*) AS ShareCount FROM hits WHERE \"URL\" = 'http://kinopoisk.ru';";
    let _ = ctx
        .sql(&query_sql)
        .await?
        .explain(false, true)?
        .show()
        .await?;
    println!("Query: {}", query_sql);
    let total_time = start.elapsed();
    let query_time = total_time - infer_schema_took;
    println!(
        "First Time: {total_time:?}, Infer Schema Time: {infer_schema_took:?}, Query Time: {query_time:?}",
    );

    let start = std::time::Instant::now();
    let _ = ctx.sql(&query_sql).await?.show().await?;
    println!("Second Time: {:?}", start.elapsed());
    Ok(())
}
