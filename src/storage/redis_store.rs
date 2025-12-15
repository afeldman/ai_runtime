use anyhow::Result;
use redis::AsyncCommands;
use serde::Serialize;

#[derive(Clone)]
pub struct RedisStorage {
    client: redis::Client,
    out_prefix: String,
}

impl RedisStorage {
    pub fn new(url: &str, out_prefix: String) -> Result<Self> {
        Ok(Self { client: redis::Client::open(url)?, out_prefix })
    }

    pub async fn store_json<T: Serialize>(&self, job_id: &str, value: &T) -> Result<()> {
        let mut con = self.client.get_multiplexed_async_connection().await?;
        let key = format!("{}:{}", self.out_prefix, job_id);
        let payload = serde_json::to_string(value)?;
        con.set::<_, _, ()>(key, payload).await?;
        Ok(())
    }
}
