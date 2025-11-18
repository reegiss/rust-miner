use super::protocol::{StratumError, StratumRequest, StratumResponse, StratumJob, StratumResult, StratumMethod};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Stratum client configuration
#[derive(Debug, Clone)]
pub struct StratumConfig {
    /// Pool URL (e.g., "pool.example.com:3333")
    pub url: String,
    
    /// Worker username (wallet address + worker name)
    pub username: String,
    
    /// Worker password (usually "x")
    pub password: String,
    
    /// User agent string
    pub user_agent: String,
    
    /// Connection timeout in seconds
    pub _timeout_secs: u64,
}

impl StratumConfig {
    pub fn new(url: String, username: String, password: String) -> Self {
        Self {
            url,
            username,
            password,
            user_agent: format!("rust-miner/{}", env!("CARGO_PKG_VERSION")),
            _timeout_secs: 30,
        }
    }
}

/// Stratum client for mining pool communication
pub struct StratumClient {
    config: StratumConfig,
    writer: Arc<Mutex<Option<WriteHalf<TcpStream>>>>,
    request_id: Arc<Mutex<u64>>,
    job_receiver: Arc<Mutex<mpsc::UnboundedReceiver<StratumJob>>>,
    job_sender: mpsc::UnboundedSender<StratumJob>,
    response_receiver: Arc<Mutex<mpsc::UnboundedReceiver<StratumResponse>>>,
    response_sender: mpsc::UnboundedSender<StratumResponse>,
    extranonce1: Arc<Mutex<Option<String>>>,
    extranonce2_size: Arc<Mutex<Option<usize>>>,
}

impl StratumClient {
    pub fn new(config: StratumConfig) -> Self {
        let (job_sender, job_receiver) = mpsc::unbounded_channel();
        let (response_sender, response_receiver) = mpsc::unbounded_channel();
        
        Self {
            config,
            writer: Arc::new(Mutex::new(None)),
            request_id: Arc::new(Mutex::new(0)),
            job_receiver: Arc::new(Mutex::new(job_receiver)),
            job_sender,
            response_receiver: Arc::new(Mutex::new(response_receiver)),
            response_sender,
            extranonce1: Arc::new(Mutex::new(None)),
            extranonce2_size: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Get next request ID
    async fn next_id(&self) -> u64 {
        let mut id = self.request_id.lock().await;
        *id += 1;
        *id
    }
    
    /// Connect to the pool
    pub async fn connect(&self) -> StratumResult<()> {
        tracing::info!("Connecting to pool: {}", self.config.url);
        
        let stream = TcpStream::connect(&self.config.url).await.map_err(|e| {
            StratumError::ConnectionError(format!("Failed to connect to {}: {}", self.config.url, e))
        })?;
        
        tracing::info!("Connected to pool successfully");
        
        // Split stream into reader and writer
        let (reader, writer) = tokio::io::split(stream);
        
        *self.writer.lock().await = Some(writer);
        
        // Start reader task
        self.start_reader(reader).await;
        
        Ok(())
    }
    
    /// Start reader task to handle incoming messages
    async fn start_reader(&self, reader: ReadHalf<TcpStream>) {
        let job_sender = self.job_sender.clone();
        let response_sender = self.response_sender.clone();
        let pool_url = self.config.url.clone();
        
        tokio::spawn(async move {
            let mut reader = BufReader::new(reader);
            let mut line = String::new();
            
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        tracing::error!("Connection closed by server");
                        break;
                    }
                    Ok(_) => {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        
                        tracing::debug!("Received: {}", line);
                        
                        match serde_json::from_str::<StratumResponse>(line) {
                            Ok(response) => {
                                // Check if it's a notification or response
                                if response.id.is_some() {
                                    // This is a response to our request
                                    if response_sender.send(response).is_err() {
                                        tracing::error!("Failed to send response to handler");
                                    }
                                } else {
                                    // This is a notification
                                    if let Some(method) = response.get_method() {
                                        if method == StratumMethod::Notify.as_str() {
                                            if let Some(params) = &response.params {
                                                match StratumJob::from_params(params) {
                                                    Ok(job) => {
                                                        // WildRig-style job notification
                                                        println!("[{}] new job from {} diff {:.2}G/{:.2}P",
                                                            chrono::Local::now().format("%H:%M:%S"),
                                                            pool_url,
                                                            job.difficulty / 1_000_000_000.0, // Convert to G
                                                            job.difficulty / 1_000_000_000.0  // Same as network diff for now
                                                        );
                                                        
                                                        // Block information
                                                        if let Ok(block_height) = u32::from_str_radix(&job.ntime, 16) {
                                                            println!("[{}] block: {}        job target: 0x{} 0x{}",
                                                                chrono::Local::now().format("%H:%M:%S"),
                                                                block_height,
                                                                &job.nbits[..8], // First 8 chars of nbits
                                                                &job.nbits[8..]  // Last 8 chars of nbits
                                                            );
                                                        }
                                                        
                                                        tracing::info!(
                                                            "New job received: {} (clean={})",
                                                            job.job_id,
                                                            job.clean_jobs
                                                        );
                                                        
                                                        if job_sender.send(job).is_err() {
                                                            tracing::error!("Failed to send job to receiver");
                                                        }
                                                    }
                                                    Err(e) => {
                                                        tracing::error!("Failed to parse job: {}", e);
                                                    }
                                                }
                                            }
                                        } else if method == StratumMethod::SetDifficulty.as_str() {
                                            if let Some(params) = &response.params {
                                                if let Some(diff) = params.get(0).and_then(|v| v.as_f64()) {
                                                    tracing::info!("Difficulty set to: {}", diff);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("Failed to parse response: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to read from stream: {}", e);
                        break;
                    }
                }
            }
        });
    }
    
    /// Send a request and wait for response
    async fn send_request(&self, request: StratumRequest) -> StratumResult<StratumResponse> {
        let json_line = request.to_json_line()?;
        
        tracing::debug!("Sending: {}", json_line.trim());
        
        // Write request
        {
            let mut writer_lock = self.writer.lock().await;
            let writer = writer_lock.as_mut().ok_or_else(|| {
                StratumError::ConnectionError("Not connected".to_string())
            })?;
            
            writer.write_all(json_line.as_bytes()).await?;
            writer.flush().await?;
        }
        
        // Wait for response from reader task
        let response = self.response_receiver.lock().await.recv().await
            .ok_or_else(|| StratumError::ConnectionError("Connection closed".to_string()))?;
        
        if response.is_error() {
            return Err(StratumError::ProtocolError(
                format!("Server returned error: {:?}", response.error)
            ));
        }
        
        Ok(response)
    }
    
    /// Subscribe to the pool
    pub async fn subscribe(&self) -> StratumResult<()> {
        tracing::info!("Subscribing to pool...");
        
        let id = self.next_id().await;
        let request = StratumRequest::subscribe(id, &self.config.user_agent);
        
        let response = self.send_request(request).await?;
        
        if let Some(result) = response.result {
            // Parse extranonce1 and extranonce2_size from result
            // Result format: [[["mining.notify", "subscription_id"], "extranonce1", extranonce2_size]]
            if let Some(arr) = result.as_array() {
                if arr.len() >= 2 {
                    if let Some(en1) = arr.get(1).and_then(|v| v.as_str()) {
                        *self.extranonce1.lock().await = Some(en1.to_string());
                        tracing::info!("Extranonce1: {}", en1);
                    }
                    if let Some(en2_size) = arr.get(2).and_then(|v| v.as_u64()) {
                        *self.extranonce2_size.lock().await = Some(en2_size as usize);
                        tracing::info!("Extranonce2 size: {}", en2_size);
                    }
                }
            }
            
            tracing::info!("Subscription successful");
            Ok(())
        } else {
            Err(StratumError::SubscriptionFailed)
        }
    }
    
    /// Authorize worker
    pub async fn authorize(&self) -> StratumResult<()> {
        tracing::info!("Authorizing worker: {}", self.config.username);
        
        let id = self.next_id().await;
        let request = StratumRequest::authorize(
            id,
            &self.config.username,
            &self.config.password,
        );
        
        let response = self.send_request(request).await?;
        
        let authorized = response.result
            .as_ref()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        if !authorized {
            return Err(StratumError::AuthorizationFailed);
        }
        
        tracing::info!("Authorization successful");
        
        Ok(())
    }
    
    /// Submit a share
    pub async fn submit_share(
        &self,
        job_id: &str,
        extranonce2: &str,
        ntime: &str,
        nonce: &str,
    ) -> StratumResult<bool> {
        let id = self.next_id().await;
        let request = StratumRequest::submit(
            id,
            &self.config.username,
            job_id,
            extranonce2,
            ntime,
            nonce,
        );
        
        let response = self.send_request(request).await?;
        
        let accepted = response.result
            .as_ref()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        if accepted {
            tracing::info!("Share accepted!");
        } else {
            tracing::warn!("Share rejected");
        }
        
        Ok(accepted)
    }
    
    /// Start listening for server notifications (mining.notify, etc.)
    
    /// Get next job from the queue
    pub async fn get_job(&self) -> Option<StratumJob> {
        self.job_receiver.lock().await.recv().await
    }
    
    /// Full connection flow: connect, subscribe, authorize, start listener
    pub async fn connect_and_login(&self) -> StratumResult<()> {
        self.connect().await?;
        self.subscribe().await?;
        self.authorize().await?;
        
        tracing::info!("Successfully connected and authorized to pool");
        
        Ok(())
    }
    
    /// Get extranonce1
    pub async fn get_extranonce1(&self) -> Option<String> {
        self.extranonce1.lock().await.clone()
    }
    
    /// Get extranonce2 size
    pub async fn get_extranonce2_size(&self) -> Option<usize> {
        *self.extranonce2_size.lock().await
    }
    
    /// Create extranonce2 with given value
    pub async fn create_extranonce2(&self, value: u32) -> Vec<u8> {
        let size = self.get_extranonce2_size().await.unwrap_or(4);
        let mut extranonce2 = vec![0u8; size];
        
        // Fill with value (little-endian)
        for (i, byte) in value.to_le_bytes().iter().enumerate() {
            if i < size {
                extranonce2[i] = *byte;
            }
        }
        
        extranonce2
    }
    
    /// Check if there's a pending job (non-blocking)
    pub async fn has_pending_job(&self) -> bool {
        let receiver = self.job_receiver.lock().await;
        !receiver.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratum_config() {
        let config = StratumConfig::new(
            "pool.example.com:3333".to_string(),
            "wallet.worker".to_string(),
            "x".to_string(),
        );
        
        assert_eq!(config.url, "pool.example.com:3333");
        assert_eq!(config.username, "wallet.worker");
        assert!(config.user_agent.contains("rust-miner"));
    }
}
