use omniengine::start_runtime;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    start_runtime().await
}
