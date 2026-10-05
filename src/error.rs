use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid preprocessor config: {0}")]
    Config(String),
    #[error("unsupported processor type `{0}`")]
    UnsupportedProcessor(String),
    #[error("invalid image: {0}")]
    Image(String),
    #[error("invalid audio input: {0}")]
    Audio(String),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("hub error: {0}")]
    Hub(String),
}

pub type Result<T> = std::result::Result<T, Error>;
