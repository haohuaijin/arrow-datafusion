use std::sync::Arc;

use datafusion::{
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    prelude::{SessionConfig, SessionContext},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let query_id: usize = 0;

    let start = std::time::Instant::now();
    let mut config = SessionConfig::from_env()?;
    config.options_mut().execution.parquet.binary_as_string = true;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config = config.set_bool("datafusion.optimizer.enable_dynamic_filter_pushdown", true);
    let ctx = SessionContext::new_with_config(config);

    let path = "/Users/huaijinhao/gitrepo/example/datafusion-example/clickbench/parquet";
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

    let _ = ctx.sql(QUERIES[query_id].1).await?.show().await?;
    println!("Query: {}", QUERIES[query_id].1);
    let total_time = start.elapsed();
    let query_time = total_time - infer_schema_took;
    println!(
        "First Time: {total_time:?}, Infer Schema Time: {infer_schema_took:?}, Query Time: {query_time:?}",
    );
    Ok(())
}

#[rustfmt::skip]
pub const QUERIES: &[(&str, &str)] = &[
    ("q23", "SELECT \"URL\" FROM hits WHERE \"URL\" LIKE '%google%' ORDER BY \"EventTime\" LIMIT 10;"),
    // ("q43", "SELECT COUNT(*) AS ShareCount FROM hits WHERE \"IsMobile\" = 1 AND \"MobilePhoneModel\" LIKE 'iPhone%' AND \"SocialAction\" = 'share' AND \"SocialSourceNetworkID\" IN (5, 12) AND \"ClientTimeZone\" BETWEEN -5 AND 5 AND regexp_match(\"Referer\", '\\/campaign\\/(spring|summer)_promo') IS NOT NULL AND CASE WHEN split_part(split_part(\"URL\", 'resolution=', 2), '&', 1) ~ '^\\d+$' THEN split_part(split_part(\"URL\", 'resolution=', 2), '&', 1)::INT ELSE 0 END > 1920 AND levenshtein(CAST(\"UTMSource\" AS STRING), CAST(\"UTMCampaign\" AS STRING)) < 3;") 
];
