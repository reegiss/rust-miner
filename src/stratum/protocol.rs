use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stratum protocol errors
#[derive(Debug, thiserror::Error)]
pub enum StratumError {
    #[error("Connection error: {0}")]
    ConnectionError(String),
    
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Authorization failed")]
    AuthorizationFailed,
    
    #[error("Subscription failed")]
    SubscriptionFailed,
    
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

pub type StratumResult<T> = Result<T, StratumError>;

/// Stratum method names
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StratumMethod {
    /// mining.subscribe
    Subscribe,
    /// mining.authorize
    Authorize,
    /// mining.submit
    Submit,
    /// mining.notify (from server)
    Notify,
    /// mining.set_difficulty (from server)
    SetDifficulty,
    /// mining.set_extranonce (from server)
    _SetExtranonce,
}

impl StratumMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Subscribe => "mining.subscribe",
            Self::Authorize => "mining.authorize",
            Self::Submit => "mining.submit",
            Self::Notify => "mining.notify",
            Self::SetDifficulty => "mining.set_difficulty",
            Self::_SetExtranonce => "mining.set_extranonce",
        }
    }
}

impl std::fmt::Display for StratumMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Stratum JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumRequest {
    pub id: u64,
    pub method: String,
    pub params: Vec<Value>,
}

impl StratumRequest {
    pub fn new(id: u64, method: StratumMethod, params: Vec<Value>) -> Self {
        Self {
            id,
            method: method.as_str().to_string(),
            params,
        }
    }
    
    /// Create mining.subscribe request
    pub fn subscribe(id: u64, user_agent: &str) -> Self {
        Self::new(
            id,
            StratumMethod::Subscribe,
            vec![Value::String(user_agent.to_string())],
        )
    }
    
    /// Create mining.authorize request
    pub fn authorize(id: u64, username: &str, password: &str) -> Self {
        Self::new(
            id,
            StratumMethod::Authorize,
            vec![
                Value::String(username.to_string()),
                Value::String(password.to_string()),
            ],
        )
    }
    
    /// Create mining.submit request
    pub fn submit(
        id: u64,
        worker_name: &str,
        job_id: &str,
        extranonce2: &str,
        ntime: &str,
        nonce: &str,
    ) -> Self {
        Self::new(
            id,
            StratumMethod::Submit,
            vec![
                Value::String(worker_name.to_string()),
                Value::String(job_id.to_string()),
                Value::String(extranonce2.to_string()),
                Value::String(ntime.to_string()),
                Value::String(nonce.to_string()),
            ],
        )
    }
    
    /// Serialize to JSON string with newline
    pub fn to_json_line(&self) -> StratumResult<String> {
        let json = serde_json::to_string(self)?;
        Ok(format!("{}\n", json))
    }
}

/// Stratum JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumResponse {
    pub id: Option<u64>,
    pub result: Option<Value>,
    pub error: Option<Value>,
    
    /// For mining.notify (no id)
    pub method: Option<String>,
    pub params: Option<Vec<Value>>,

    /// Some pools send extra top-level fields with notifications (e.g., height, algo)
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

impl StratumResponse {
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
    
    pub fn _is_notification(&self) -> bool {
        self.method.is_some() && self.id.is_none()
    }
    
    pub fn get_method(&self) -> Option<String> {
        self.method.clone()
    }
}

/// Mining job received from pool
#[derive(Debug, Clone)]
pub struct StratumJob {
    /// Job ID
    pub job_id: String,
    
    /// Previous block hash (Bitcoin/Qubitcoin)
    pub prevhash: String,
    
    /// Coinbase part 1 (Bitcoin/Qubitcoin)
    pub coinb1: String,
    
    /// Coinbase part 2 (Bitcoin/Qubitcoin)
    pub coinb2: String,
    
    /// Merkle branches (Bitcoin/Qubitcoin)
    pub merkle_branch: Vec<String>,
    
    /// Block version (Bitcoin/Qubitcoin)
    pub version: String,
    
    /// Network difficulty bits (Bitcoin/Qubitcoin)
    pub nbits: String,
    
    /// Network time (Bitcoin/Qubitcoin)
    pub ntime: String,
    
    /// Clean jobs flag (true = discard old jobs)
    pub clean_jobs: bool,
    
    /// Calculated difficulty
    pub difficulty: f64,

    /// Seed Hash (Ethash)
    pub seed_hash: Option<String>,

    /// Header Hash (Ethash)
    pub header_hash: Option<String>,

    /// Optional extra fields from pool
    pub height: Option<u64>,
    pub algo: Option<String>,
}

impl StratumJob {
    /// Parse from mining.notify params
    pub fn from_params(params: &[Value]) -> StratumResult<Self> {
        // Check for Ethash format (Stratum V1 for Ethereum/ETC)
        // Format 1: [job_id, seed_hash, header_hash, clean_jobs]
        // Format 2: [job_id, seed_hash, header_hash, target, clean_jobs] (2Miners ETC)
        if params.len() == 4 || params.len() == 5 {
             let job_id = params[0]
                .as_str()
                .ok_or_else(|| StratumError::InvalidResponse("job_id not a string".to_string()))?
                .to_string();
            
            let seed_hash = params[1]
                .as_str()
                .ok_or_else(|| StratumError::InvalidResponse("seed_hash not a string".to_string()))?
                .to_string();
            
            let header_hash = params[2]
                .as_str()
                .ok_or_else(|| StratumError::InvalidResponse("header_hash not a string".to_string()))?
                .to_string();
            
            // Handle optional target (index 3) and clean_jobs (last index)
            let clean_jobs_idx = params.len() - 1;
            let clean_jobs = params[clean_jobs_idx]
                .as_bool()
                .ok_or_else(|| StratumError::InvalidResponse("clean_jobs not a boolean".to_string()))?;

            // If we have 5 params, the 4th one (index 3) is the target
            let difficulty = 1.0;
            if params.len() == 5 {
                // Optional target (params[3]) present; for now we ignore and keep difficulty as-is.
            }

            return Ok(Self {
                job_id,
                prevhash: String::new(),
                coinb1: String::new(),
                coinb2: String::new(),
                merkle_branch: Vec::new(),
                version: String::new(),
                nbits: String::new(),
                ntime: String::new(),
                clean_jobs,
                difficulty, 
                seed_hash: Some(seed_hash),
                header_hash: Some(header_hash),
                height: None,
                algo: None,
            });
        }

        if params.len() < 9 {
            return Err(StratumError::InvalidResponse(
                format!("Invalid mining.notify params length: {}", params.len())
            ));
        }
        
        let job_id = params[0]
            .as_str()
            .ok_or_else(|| StratumError::InvalidResponse("job_id not a string".to_string()))?
            .to_string();
        
        let prevhash = params[1]
            .as_str()
            .ok_or_else(|| StratumError::InvalidResponse("prevhash not a string".to_string()))?
            .to_string();
        
        let coinb1 = params[2]
            .as_str()
            .ok_or_else(|| StratumError::InvalidResponse("coinb1 not a string".to_string()))?
            .to_string();
        
        let coinb2 = params[3]
            .as_str()
            .ok_or_else(|| StratumError::InvalidResponse("coinb2 not a string".to_string()))?
            .to_string();
        
        let merkle_branch = params[4]
            .as_array()
            .ok_or_else(|| StratumError::InvalidResponse("merkle_branch not an array".to_string()))?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        
        let version = params[5]
            .as_str()
            .ok_or_else(|| StratumError::InvalidResponse("version not a string".to_string()))?
            .to_string();
        
        let nbits = params[6]
            .as_str()
            .ok_or_else(|| StratumError::InvalidResponse("nbits not a string".to_string()))?
            .to_string();
        
        let ntime = params[7]
            .as_str()
            .ok_or_else(|| StratumError::InvalidResponse("ntime not a string".to_string()))?
            .to_string();
        
        let clean_jobs = params[8]
            .as_bool()
            .ok_or_else(|| StratumError::InvalidResponse("clean_jobs not a boolean".to_string()))?;
        
        // Calculate difficulty from nbits (simplified - for Qubitcoin this is usually around 20-30)
        let difficulty = if let Ok(nbits_val) = u32::from_str_radix(&nbits, 16) {
            // Very rough approximation: higher nbits = lower difficulty
            // This is not accurate but gives reasonable values for display
            let exponent = (nbits_val >> 24) as f64;
            let mantissa = (nbits_val & 0x00FFFFFF) as f64;
            // Simplified calculation - actual difficulty would be much more complex
            (0xFFFF as f64 / mantissa) * 2.0f64.powf(32.0 - exponent)
        } else {
            0.0
        };

        Ok(Self {
            job_id,
            prevhash,
            coinb1,
            coinb2,
            merkle_branch,
            version,
            nbits,
            ntime,
            clean_jobs,
            difficulty,
            seed_hash: None,
            header_hash: None,
            height: None,
            algo: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratum_request_serialize() {
        let req = StratumRequest::subscribe(1, "rust-miner/0.1.0");
        let json = req.to_json_line().unwrap();
        assert!(json.contains("mining.subscribe"));
        assert!(json.ends_with('\n'));
    }

    #[test]
    fn test_stratum_authorize() {
        let req = StratumRequest::authorize(2, "wallet.worker", "x");
        assert_eq!(req.method, "mining.authorize");
        assert_eq!(req.params.len(), 2);
    }

    #[test]
    fn test_qhash_job_parsing() {
        // Standard Bitcoin/QHash mining.notify params (9 items)
        let params = vec![
            Value::String("job_id".to_string()),
            Value::String("prevhash".to_string()),
            Value::String("coinb1".to_string()),
            Value::String("coinb2".to_string()),
            Value::Array(vec![]), // merkle_branch
            Value::String("version".to_string()),
            Value::String("nbits".to_string()),
            Value::String("ntime".to_string()),
            Value::Bool(true), // clean_jobs
        ];
        
        let job = StratumJob::from_params(&params).unwrap();
        assert_eq!(job.job_id, "job_id");
        assert!(job.seed_hash.is_none());
        assert!(job.header_hash.is_none());
    }

    #[test]
    fn test_ethash_job_parsing() {
        // Ethash mining.notify params (4 items)
        let params = vec![
            Value::String("job_id".to_string()),
            Value::String("seed_hash".to_string()),
            Value::String("header_hash".to_string()),
            Value::Bool(true),
        ];
        
        let job = StratumJob::from_params(&params).unwrap();
        assert_eq!(job.job_id, "job_id");
        assert_eq!(job.seed_hash.unwrap(), "seed_hash");
        assert_eq!(job.header_hash.unwrap(), "header_hash");
    }
}
