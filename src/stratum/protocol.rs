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
    SetExtranonce,
}

impl StratumMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Subscribe => "mining.subscribe",
            Self::Authorize => "mining.authorize",
            Self::Submit => "mining.submit",
            Self::Notify => "mining.notify",
            Self::SetDifficulty => "mining.set_difficulty",
            Self::SetExtranonce => "mining.set_extranonce",
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
}

impl StratumResponse {
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
    
    pub fn is_notification(&self) -> bool {
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
    
    /// Previous block hash
    pub prevhash: String,
    
    /// Coinbase part 1
    pub coinb1: String,
    
    /// Coinbase part 2
    pub coinb2: String,
    
    /// Merkle branches
    pub merkle_branch: Vec<String>,
    
    /// Block version
    pub version: String,
    
    /// Network difficulty bits
    pub nbits: String,
    
    /// Network time
    pub ntime: String,
    
    /// Clean jobs flag (true = discard old jobs)
    pub clean_jobs: bool,
}

impl StratumJob {
    /// Parse from mining.notify params
    pub fn from_params(params: &[Value]) -> StratumResult<Self> {
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
}
