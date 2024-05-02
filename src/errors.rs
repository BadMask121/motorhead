use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum MemoryError<'a> {
  #[error("MemoryError: An unknown error occured {0}")]
  UnknownError(&'a str),
  #[error("MemoryError: Redis error occured {0}")]
  MemoryRedisError(String),
}
