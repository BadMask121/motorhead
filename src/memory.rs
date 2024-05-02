use anyhow::{bail, Context, Result};
use redis::RedisError;
use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

use crate::{
  errors::MemoryError,
  long_term_memory::{index_messages, search_messages},
  models::{
    AckResponse, GetSessionsQuery, MemoryMessage, MemoryMessagesAndContext, MemoryResponse,
    NamespaceQuery, OpenAIClientManager, RedisearchResult, SearchPayload,
  },
  redis_utils::ensure_redisearch_index,
  reducer::handle_compaction,
};

#[derive(Clone)]
pub struct MotorheadBuilder {
  pub(crate) long_term_memory: bool,
  pub(crate) window_size: i64,
  pub(crate) model: String,
  pub(crate) session_cleanup: Arc<Mutex<HashMap<String, bool>>>,
  pub(crate) openai_pool: deadpool::managed::Pool<OpenAIClientManager>,
  pub(crate) redis: redis::aio::ConnectionManager,
}

#[derive(Clone, Debug)]
pub struct MotorheadBuilderParam {
  pub long_term_memory: Option<bool>,
  pub window_size: Option<i64>,
  pub model: Option<String>,
}

impl MotorheadBuilder {
  pub async fn build(builder: &MotorheadBuilderParam) -> Result<Self> {
    let manager = OpenAIClientManager {};
    let max_size = 8;
    let openai_pool: deadpool::managed::Pool<OpenAIClientManager> =
      deadpool::managed::Pool::builder(manager)
        .max_size(max_size)
        .build()
        .unwrap();

    let redis_url = dotenvy::var("REDIS_URL").expect("REDIS_URL is not set");
    let redis = redis::Client::open(redis_url).unwrap();

    let long_term_memory = builder.long_term_memory.unwrap_or(false);

    if long_term_memory {
      // TODO: Make these configurable - for now just ADA support
      let vector_dimensions = 1536;
      let distance_metric = "COSINE";

      ensure_redisearch_index(&redis, vector_dimensions, distance_metric).unwrap_or_else(|err| {
        eprintln!("RediSearch index error: {}", err);
        std::process::exit(1);
      });
    }

    let window_size = builder.window_size.unwrap_or(12);

    let model = builder
      .model
      .clone()
      .unwrap_or_else(|| "gpt-3.5-turbo".to_string());

    let session_cleanup: Arc<Mutex<HashMap<String, bool>>> = Arc::new(Mutex::new(HashMap::new()));

    let conn: redis::aio::ConnectionManager = redis
      .get_tokio_connection_manager()
      .await
      .context(MemoryError::UnknownError("An unknown error happened"))?;

    Ok(Self {
      long_term_memory,
      window_size,
      model,
      openai_pool,
      session_cleanup,
      redis: conn,
    })
  }

  pub async fn get_sessions(&self, pagination: GetSessionsQuery) -> Result<Vec<String>> {
    let GetSessionsQuery {
      page,
      size,
      namespace,
    } = pagination;

    if page > 100 {
      bail!(MemoryError::UnknownError("Page size must not exceed 100"));
    }

    let start: isize = ((page - 1) * size) as isize; // 0-indexed
    let end: isize = (page * size - 1) as isize; // inclusive

    let mut conn = self.redis.clone();

    let sessions_key = match &namespace {
      Some(namespace) => format!("sessions:{}", namespace),
      None => String::from("sessions"),
    };

    let session_ids: Vec<String> = redis::cmd("ZRANGE")
      .arg(sessions_key)
      .arg(start)
      .arg(end)
      .query_async(&mut conn)
      .await
      .map_err::<Result<RedisError, anyhow::Error>, _>(|e| {
        bail!(MemoryError::MemoryRedisError(e.to_string()))
      })
      .unwrap();

    Ok(session_ids)
  }

  pub async fn get_memory(&self, session_id: String) -> Result<MemoryResponse> {
    let mut conn = self.redis.clone();

    let lrange_key = format!("session:{}", &*session_id);
    let context_key = format!("context:{}", &*session_id);
    let token_count_key = format!("tokens:{}", &*session_id);
    let keys = vec![context_key, token_count_key];

    let (messages, values): (Vec<String>, Vec<Option<String>>) = redis::pipe()
      .cmd("LRANGE")
      .arg(lrange_key)
      .arg(0)
      .arg(self.window_size as isize)
      .cmd("MGET")
      .arg(keys)
      .query_async(&mut conn)
      .await
      .map_err(|e| MemoryError::MemoryRedisError(e.to_string()))?;

    let context = values.get(0).cloned().flatten();
    let tokens = values
      .get(1)
      .cloned()
      .flatten()
      .and_then(|tokens_string| tokens_string.parse::<i64>().ok())
      .unwrap_or(0);

    let mut messages: Vec<MemoryMessage> = messages
      .into_iter()
      .filter_map(|message| {
        let mut parts = message.splitn(2, ": ");

        match (parts.next(), parts.next()) {
          (Some(id_role), Some(content)) => {
            let mut id_role_parts = id_role.splitn(2, ":");

            match (id_role_parts.next(), id_role_parts.next()) {
              (Some(id), Some(role)) => Some(MemoryMessage {
                id: id.to_string(),
                role: role.to_string(),
                content: content.to_string(),
              }),
              _ => None,
            }
          }
          _ => None,
        }
      })
      .collect();

    messages.reverse();

    let response = MemoryResponse {
      messages,
      context,
      tokens: Some(tokens),
    };

    Ok(response)
  }

  pub async fn add_memory(
    &self,
    session_id: String,
    memory_messages: MemoryMessagesAndContext,
    namespace_query: NamespaceQuery,
  ) -> Result<AckResponse> {
    let mut conn = self.redis.clone();

    let memory_messages_clone: Vec<MemoryMessage> = memory_messages.messages.to_vec();

    let messages: Vec<String> = memory_messages
      .messages
      .into_iter()
      .map(|memory_message| {
        format!(
          "{}:{}: {}",
          memory_message.id, memory_message.role, memory_message.content
        )
      })
      .collect();

    // If new context is passed in we overwrite the existing one
    if let Some(context) = memory_messages.context {
      redis::Cmd::set(format!("context:{}", &*session_id), context)
        .query_async::<_, ()>(&mut conn)
        .await
        .map_err(|e| MemoryError::MemoryRedisError(e.to_string()))?;
    }

    let sessions_key = match namespace_query.namespace {
      Some(namespace) => format!("sessions:{}", namespace),
      None => String::from("sessions"),
    };

    // add to sorted set of sessions
    redis::cmd("ZADD")
      .arg(sessions_key)
      .arg(chrono::Utc::now().timestamp())
      .arg(&*session_id)
      .query_async(&mut conn)
      .await
      .map_err(|e| MemoryError::MemoryRedisError(e.to_string()))?;

    let res: i64 = redis::Cmd::lpush(format!("session:{}", &*session_id), messages.clone())
      .query_async::<_, i64>(&mut conn)
      .await
      .map_err(|e| MemoryError::MemoryRedisError(e.to_string()))?;

    if self.long_term_memory {
      let session = session_id.clone();
      let conn_clone = conn.clone();
      let pool = self.openai_pool.clone();

      tokio::spawn(async move {
        let client_wrapper = pool.get().await.unwrap();
        if let Err(e) =
          index_messages(memory_messages_clone, session, &client_wrapper, conn_clone).await
        {
          log::error!("Error in index_messages: {:?}", e);
        }
      });
    }

    if res > self.window_size {
      let mut session_cleanup = self.session_cleanup.lock().await;

      if !session_cleanup.get(&*session_id).unwrap_or(&false) {
        session_cleanup.insert((&*session_id.to_string()).into(), true);
        let session_cleanup = Arc::clone(&self.session_cleanup);
        let session_id = session_id.clone();
        let window_size = self.window_size;
        let model = self.model.to_string();
        let pool = self.openai_pool.clone();

        tokio::spawn(async move {
          log::info!("running compact");
          let client_wrapper = pool.get().await.unwrap();

          let _compaction_result = handle_compaction(
            session_id.to_string(),
            model,
            window_size,
            &client_wrapper,
            conn,
          )
          .await;

          let mut lock = session_cleanup.lock().await;
          lock.remove(&session_id);
        });
      }
    }

    let response = AckResponse { status: "Ok" };
    Ok(response)
  }

  pub async fn delete_memory(
    &self,
    session_id: String,
    namespace_query: NamespaceQuery,
  ) -> Result<AckResponse> {
    let mut conn = self.redis.clone();

    let context_key = format!("context:{}", &*session_id);
    let token_count_key = format!("tokens:{}", &*session_id);
    let session_key = format!("session:{}", &*session_id);
    let keys = vec![context_key, session_key, token_count_key];

    let sessions_key = match namespace_query.namespace {
      Some(namespace) => format!("sessions:{}", namespace),
      None => String::from("sessions"),
    };

    redis::cmd("ZREM")
      .arg(sessions_key)
      .arg(&*session_id)
      .query_async(&mut conn)
      .await
      .map_err(|e| MemoryError::MemoryRedisError(e.to_string()))?;

    redis::Cmd::del(keys)
      .query_async(&mut conn)
      .await
      .map_err(|e| MemoryError::MemoryRedisError(e.to_string()))?;

    let response = AckResponse { status: "Ok" };
    Ok(response)
  }

  pub async fn run_retrieval(
    &self,
    session_id: String,
    payload: SearchPayload,
  ) -> Result<Vec<RedisearchResult>> {
    if !self.long_term_memory {
      bail!(MemoryError::UnknownError("long term memory disabled"));
    }

    let conn = self.redis.clone();

    let client_wrapper = self.openai_pool.get().await.unwrap();

    match search_messages(payload.text, session_id.clone(), &client_wrapper, conn).await {
      Ok(results) => Ok(results),
      Err(e) => {
        log::error!("Error Retrieval API: {:?}", e);
        println!("{:?}", e);
        bail!(MemoryError::UnknownError("unable to retrieve messages"))
      }
    }
  }
}
