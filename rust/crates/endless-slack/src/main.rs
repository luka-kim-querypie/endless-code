use std::collections::{BTreeSet, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use api::{
    max_tokens_for_model, resolve_model_alias, InputContentBlock, InputMessage, MessageRequest,
    OutputContentBlock, ProviderClient, ToolChoice, ToolDefinition, ToolResultContentBlock,
};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use chrono::Utc;
use hmac::{Hmac, Mac};
use runtime::{
    load_system_prompt, ApiClient, ApiRequest, AssistantEvent, ContentBlock, ConversationMessage,
    ConversationRuntime, MessageRole, PermissionMode, PermissionPolicy, RuntimeError, Session,
    ToolError, ToolExecutor,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use tools::{mvp_tool_specs, GlobalToolRegistry};

type HmacSha256 = Hmac<Sha256>;

const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8787";
const DEFAULT_MODEL: &str = "claude-sonnet-4-6";
const MAX_SLACK_TEXT_LEN: usize = 3500;

#[derive(Debug, Clone)]
struct Config {
    bind_addr: String,
    workdir: PathBuf,
    state_dir: PathBuf,
    slack_bot_token: String,
    slack_signing_secret: String,
    default_model: String,
    permission_mode: PermissionMode,
    allowed_tools: Option<BTreeSet<String>>,
    allowed_channels: Option<HashSet<String>>,
    system_prompt_append: Option<String>,
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let workdir = env::var("ENDLESS_WORKDIR")
            .map(PathBuf::from)
            .unwrap_or(env::current_dir().map_err(|error| error.to_string())?);
        let state_dir = env::var("ENDLESS_STATE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| workdir.join(".endless/slack"));

        Ok(Self {
            bind_addr: env::var("ENDLESS_SLACK_BIND")
                .unwrap_or_else(|_| DEFAULT_BIND_ADDR.to_string()),
            workdir,
            state_dir,
            slack_bot_token: read_required_env("SLACK_BOT_TOKEN")?,
            slack_signing_secret: read_required_env("SLACK_SIGNING_SECRET")?,
            default_model: env::var("ENDLESS_DEFAULT_MODEL")
                .unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            permission_mode: parse_permission_mode(
                &env::var("ENDLESS_PERMISSION_MODE")
                    .unwrap_or_else(|_| "read-only".to_string()),
            )?,
            allowed_tools: parse_csv_set(env::var("ENDLESS_ALLOWED_TOOLS").ok()),
            allowed_channels: parse_hash_set(env::var("ENDLESS_ALLOWED_CHANNELS").ok()),
            system_prompt_append: env::var("ENDLESS_SYSTEM_PROMPT_APPEND").ok(),
        })
    }
}

#[derive(Clone)]
struct AppState {
    config: Config,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum SlackEnvelope {
    #[serde(rename = "url_verification")]
    UrlVerification { challenge: String },
    #[serde(rename = "event_callback")]
    EventCallback { event: SlackEvent },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
struct SlackEvent {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
    user: Option<String>,
    bot_id: Option<String>,
    subtype: Option<String>,
    channel: Option<String>,
    thread_ts: Option<String>,
    ts: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ThreadState {
    model: String,
    turn_count: u64,
    created_at_ms: i64,
    updated_at_ms: i64,
}

impl ThreadState {
    fn new(default_model: &str) -> Self {
        let now = now_millis();
        Self {
            model: default_model.to_string(),
            turn_count: 0,
            created_at_ms: now,
            updated_at_ms: now,
        }
    }

    fn touch(&mut self) {
        self.updated_at_ms = now_millis();
    }
}

#[derive(Debug, Clone)]
struct ThreadPaths {
    session: PathBuf,
    state: PathBuf,
}

#[derive(Debug, Clone)]
struct ThreadContext {
    channel: String,
    thread_ts: String,
    prompt: String,
    paths: ThreadPaths,
}

#[derive(Debug)]
struct EndlessRuntimeClient {
    runtime: tokio::runtime::Runtime,
    client: ProviderClient,
    model: String,
    allowed_tools: Option<BTreeSet<String>>,
}

impl EndlessRuntimeClient {
    fn new(model: String, allowed_tools: Option<BTreeSet<String>>) -> Result<Self, String> {
        let model = resolve_model_alias(&model);
        let client = ProviderClient::from_model(&model).map_err(|error| error.to_string())?;
        Ok(Self {
            runtime: tokio::runtime::Runtime::new().map_err(|error| error.to_string())?,
            client,
            model,
            allowed_tools,
        })
    }
}

impl ApiClient for EndlessRuntimeClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let tools = allowed_tool_definitions(self.allowed_tools.as_ref());
        let message_request = MessageRequest {
            model: self.model.clone(),
            max_tokens: max_tokens_for_model(&self.model),
            messages: convert_messages(&request.messages),
            system: (!request.system_prompt.is_empty()).then(|| request.system_prompt.join("\n\n")),
            tools: (!tools.is_empty()).then_some(tools),
            tool_choice: (!tools.is_empty()).then_some(ToolChoice::Auto),
            stream: false,
        };

        let response = self
            .runtime
            .block_on(async { self.client.send_message(&message_request).await })
            .map_err(|error| RuntimeError::new(error.to_string()))?;

        Ok(response_to_events(response))
    }
}

#[derive(Debug, Clone)]
struct EndlessToolExecutor {
    registry: GlobalToolRegistry,
    allowed_tools: Option<BTreeSet<String>>,
}

impl EndlessToolExecutor {
    fn new(allowed_tools: Option<BTreeSet<String>>) -> Self {
        Self {
            registry: GlobalToolRegistry::builtin(),
            allowed_tools,
        }
    }
}

impl ToolExecutor for EndlessToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError> {
        if self
            .allowed_tools
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(tool_name))
        {
            return Err(ToolError::new(format!(
                "tool `{tool_name}` is not enabled for the Slack bridge"
            )));
        }

        let value: Value = serde_json::from_str(input)
            .map_err(|error| ToolError::new(format!("invalid tool input JSON: {error}")))?;
        self.registry
            .execute(tool_name, &value)
            .map_err(ToolError::new)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env().map_err(io_error)?;
    fs::create_dir_all(config.state_dir.join("sessions"))?;
    fs::create_dir_all(config.state_dir.join("threads"))?;

    let state = Arc::new(AppState { config });
    let app = Router::new()
        .route("/slack/events", post(slack_events))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(&state.config.bind_addr).await?;

    println!(
        "endless-slack listening on http://{}/slack/events",
        state.config.bind_addr
    );

    axum::serve(listener, app).await?;
    Ok(())
}

async fn slack_events(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    if let Err(error) = verify_slack_signature(&state.config.slack_signing_secret, &headers, &body)
    {
        return (StatusCode::UNAUTHORIZED, error).into_response();
    }

    let envelope: SlackEnvelope = match serde_json::from_str(&body) {
        Ok(envelope) => envelope,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid Slack payload: {error}"),
            )
                .into_response();
        }
    };

    match envelope {
        SlackEnvelope::UrlVerification { challenge } => {
            Json(json!({ "challenge": challenge })).into_response()
        }
        SlackEnvelope::EventCallback { event } => {
            let app_state = state.clone();
            tokio::task::spawn_blocking(move || {
                if let Err(error) = process_event(&app_state.config, event) {
                    eprintln!("slack event processing failed: {error}");
                }
            });
            StatusCode::OK.into_response()
        }
        SlackEnvelope::Other => StatusCode::OK.into_response(),
    }
}

fn process_event(config: &Config, event: SlackEvent) -> Result<(), String> {
    if event.bot_id.is_some() || event.subtype.as_deref() == Some("bot_message") {
        return Ok(());
    }

    let channel = match event.channel {
        Some(channel) => channel,
        None => return Ok(()),
    };
    if config
        .allowed_channels
        .as_ref()
        .is_some_and(|allowed| !allowed.contains(&channel))
    {
        return Ok(());
    }

    let ts = event.ts.clone().unwrap_or_default();
    if ts.is_empty() {
        return Ok(());
    }
    let thread_ts = event.thread_ts.clone().unwrap_or_else(|| ts.clone());
    let text = event.text.unwrap_or_default();
    let paths = thread_paths(&config.state_dir, &channel, &thread_ts);
    let conversation_exists = paths.state.exists() || paths.session.exists();
    let is_app_mention = event.kind == "app_mention";

    if !is_app_mention && !(event.kind == "message" && conversation_exists) {
        return Ok(());
    }

    let prompt = sanitize_slack_prompt(&text, is_app_mention);
    if prompt.is_empty() {
        return Ok(());
    }

    let context = ThreadContext {
        channel: channel.clone(),
        thread_ts,
        prompt,
        paths,
    };

    handle_thread_command_or_prompt(config, context)
}

fn handle_thread_command_or_prompt(config: &Config, context: ThreadContext) -> Result<(), String> {
    let mut thread_state = load_thread_state(&context.paths.state, &config.default_model)?;

    if let Some(model) = context.prompt.strip_prefix("/model ").map(str::trim) {
        if model.is_empty() {
            return post_message(
                config,
                &context.channel,
                &context.thread_ts,
                "usage: /model <model-name>",
            );
        }
        thread_state.model = model.to_string();
        thread_state.touch();
        save_thread_state(&context.paths.state, &thread_state)?;
        return post_message(
            config,
            &context.channel,
            &context.thread_ts,
            &format!("model switched to `{}`", thread_state.model),
        );
    }

    if context.prompt == "/reset" {
        if context.paths.session.exists() {
            fs::remove_file(&context.paths.session).map_err(|error| error.to_string())?;
        }
        thread_state = ThreadState::new(&config.default_model);
        save_thread_state(&context.paths.state, &thread_state)?;
        return post_message(
            config,
            &context.channel,
            &context.thread_ts,
            "thread session reset",
        );
    }

    if context.prompt == "/status" {
        let status = format!(
            "model: `{}`\npermission: `{}`\nsession: `{}`",
            thread_state.model,
            config.permission_mode.as_str(),
            context.paths.session.display()
        );
        return post_message(config, &context.channel, &context.thread_ts, &status);
    }

    if context.prompt == "/help" {
        return post_message(
            config,
            &context.channel,
            &context.thread_ts,
            "commands: `/model <name>`, `/status`, `/reset`",
        );
    }

    let reply = run_slack_turn(config, &context, &mut thread_state)?;
    save_thread_state(&context.paths.state, &thread_state)?;
    post_message(config, &context.channel, &context.thread_ts, &reply)
}

fn run_slack_turn(
    config: &Config,
    context: &ThreadContext,
    thread_state: &mut ThreadState,
) -> Result<String, String> {
    if let Some(parent) = context.paths.session.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let session = if context.paths.session.exists() {
        Session::load_from_path(&context.paths.session).map_err(|error| error.to_string())?
    } else {
        Session::new().with_persistence_path(context.paths.session.clone())
    };

    let model = thread_state.model.clone();
    let mut system_prompt = load_system_prompt(
        config.workdir.clone(),
        current_date_string(),
        env::consts::OS.to_string(),
        "unknown".to_string(),
    )
    .unwrap_or_else(|_| {
        vec!["You are Endless Code, a software engineering assistant.".to_string()]
    });
    system_prompt.push(
        "You are replying inside a Slack thread. Keep responses concise, practical, and easy to scan in chat."
            .to_string(),
    );
    if let Some(extra) = &config.system_prompt_append {
        system_prompt.push(extra.clone());
    }

    let api_client = EndlessRuntimeClient::new(model, config.allowed_tools.clone())?;
    let tool_executor = EndlessToolExecutor::new(config.allowed_tools.clone());
    let permission_policy = permission_policy(config.permission_mode);
    let mut runtime = ConversationRuntime::new(
        session,
        api_client,
        tool_executor,
        permission_policy,
        system_prompt,
    )
    .with_max_iterations(10);

    let summary = runtime
        .run_turn(context.prompt.clone(), None)
        .map_err(|error| error.to_string())?;
    let session = runtime.into_session();
    if let Some(path) = session.persistence_path().map(Path::to_path_buf) {
        session
            .save_to_path(&path)
            .map_err(|error| error.to_string())?;
    }

    thread_state.turn_count += 1;
    thread_state.touch();

    Ok(limit_for_slack(&render_assistant_reply(&summary.assistant_messages)))
}

fn render_assistant_reply(messages: &[ConversationMessage]) -> String {
    let text = messages
        .iter()
        .flat_map(|message| &message.blocks)
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    if text.trim().is_empty() {
        "completed, but no assistant text was produced".to_string()
    } else {
        text
    }
}

fn permission_policy(mode: PermissionMode) -> PermissionPolicy {
    mvp_tool_specs()
        .into_iter()
        .fold(PermissionPolicy::new(mode), |policy, spec| {
            policy.with_tool_requirement(spec.name, spec.required_permission)
        })
}

fn allowed_tool_definitions(allowed_tools: Option<&BTreeSet<String>>) -> Vec<ToolDefinition> {
    mvp_tool_specs()
        .into_iter()
        .filter(|spec| {
            allowed_tools.is_none_or(|allowed| allowed.contains(spec.name))
        })
        .map(|spec| ToolDefinition {
            name: spec.name.to_string(),
            description: Some(spec.description.to_string()),
            input_schema: spec.input_schema,
        })
        .collect()
}

fn convert_messages(messages: &[ConversationMessage]) -> Vec<InputMessage> {
    messages
        .iter()
        .filter_map(|message| {
            let role = match message.role {
                MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
                MessageRole::Assistant => "assistant",
            };
            let content = message
                .blocks
                .iter()
                .map(|block| match block {
                    ContentBlock::Text { text } => InputContentBlock::Text { text: text.clone() },
                    ContentBlock::ToolUse { id, name, input } => InputContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: serde_json::from_str(input)
                            .unwrap_or_else(|_| serde_json::json!({ "raw": input })),
                    },
                    ContentBlock::ToolResult {
                        tool_use_id,
                        output,
                        is_error,
                        ..
                    } => InputContentBlock::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: vec![ToolResultContentBlock::Text {
                            text: output.clone(),
                        }],
                        is_error: *is_error,
                    },
                })
                .collect::<Vec<_>>();
            (!content.is_empty()).then(|| InputMessage {
                role: role.to_string(),
                content,
            })
        })
        .collect()
}

fn response_to_events(response: api::MessageResponse) -> Vec<AssistantEvent> {
    let mut events = Vec::new();
    let mut pending_tools = std::collections::BTreeMap::new();

    for (index, block) in response.content.into_iter().enumerate() {
        let index = u32::try_from(index).expect("response block index overflow");
        push_output_block(block, index, &mut events, &mut pending_tools, false);
        if let Some((id, name, input)) = pending_tools.remove(&index) {
            events.push(AssistantEvent::ToolUse { id, name, input });
        }
    }

    events.push(AssistantEvent::Usage(response.usage.token_usage()));
    events.push(AssistantEvent::MessageStop);
    events
}

fn push_output_block(
    block: OutputContentBlock,
    block_index: u32,
    events: &mut Vec<AssistantEvent>,
    pending_tools: &mut std::collections::BTreeMap<u32, (String, String, String)>,
    streaming_tool_input: bool,
) {
    match block {
        OutputContentBlock::Text { text } => {
            if !text.is_empty() {
                events.push(AssistantEvent::TextDelta(text));
            }
        }
        OutputContentBlock::ToolUse { id, name, input } => {
            let initial_input = if streaming_tool_input
                && input.is_object()
                && input.as_object().is_some_and(serde_json::Map::is_empty)
            {
                String::new()
            } else {
                input.to_string()
            };
            pending_tools.insert(block_index, (id, name, initial_input));
        }
        OutputContentBlock::Thinking { .. } | OutputContentBlock::RedactedThinking { .. } => {}
    }
}

fn verify_slack_signature(
    signing_secret: &str,
    headers: &HeaderMap,
    body: &str,
) -> Result<(), String> {
    let timestamp = header_value(headers, "x-slack-request-timestamp")?;
    let signature = header_value(headers, "x-slack-signature")?;
    let payload = format!("v0:{timestamp}:{body}");

    let mut mac =
        HmacSha256::new_from_slice(signing_secret.as_bytes()).map_err(|error| error.to_string())?;
    mac.update(payload.as_bytes());
    let expected = format!("v0={}", hex::encode(mac.finalize().into_bytes()));

    if expected == signature {
        Ok(())
    } else {
        Err("invalid Slack signature".to_string())
    }
}

fn header_value(headers: &HeaderMap, key: &str) -> Result<String, String> {
    headers
        .get(key)
        .ok_or_else(|| format!("missing header: {key}"))?
        .to_str()
        .map(|value| value.to_string())
        .map_err(|error| error.to_string())
}

fn thread_paths(base: &Path, channel: &str, thread_ts: &str) -> ThreadPaths {
    let key = format!("{}_{}", sanitize_path_component(channel), sanitize_path_component(thread_ts));
    ThreadPaths {
        session: base.join("sessions").join(format!("{key}.jsonl")),
        state: base.join("threads").join(format!("{key}.json")),
    }
}

fn sanitize_path_component(input: &str) -> String {
    input.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn load_thread_state(path: &Path, default_model: &str) -> Result<ThreadState, String> {
    if !path.exists() {
        return Ok(ThreadState::new(default_model));
    }
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&contents).map_err(|error| error.to_string())
}

fn save_thread_state(path: &Path, state: &ThreadState) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let payload = serde_json::to_string_pretty(state).map_err(|error| error.to_string())?;
    fs::write(path, payload).map_err(|error| error.to_string())
}

fn sanitize_slack_prompt(text: &str, remove_mentions: bool) -> String {
    text.split_whitespace()
        .filter(|token| !(remove_mentions && token.starts_with("<@") && token.ends_with('>')))
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn post_message(
    config: &Config,
    channel: &str,
    thread_ts: &str,
    text: &str,
) -> Result<(), String> {
    let payload = json!({
        "channel": channel,
        "thread_ts": thread_ts,
        "text": limit_for_slack(text),
        "unfurl_links": false,
        "unfurl_media": false
    });

    let response = reqwest::blocking::Client::new()
        .post("https://slack.com/api/chat.postMessage")
        .bearer_auth(&config.slack_bot_token)
        .json(&payload)
        .send()
        .map_err(|error| error.to_string())?;

    let status = response.status();
    let value: Value = response.json().map_err(|error| error.to_string())?;
    if status.is_success() && value.get("ok").and_then(Value::as_bool) == Some(true) {
        Ok(())
    } else {
        Err(format!("Slack post failed: {value}"))
    }
}

fn limit_for_slack(text: &str) -> String {
    if text.chars().count() <= MAX_SLACK_TEXT_LEN {
        return text.to_string();
    }

    text.chars()
        .take(MAX_SLACK_TEXT_LEN.saturating_sub(24))
        .collect::<String>()
        + "\n\n[truncated for Slack]"
}

fn parse_permission_mode(value: &str) -> Result<PermissionMode, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "read-only" => Ok(PermissionMode::ReadOnly),
        "workspace-write" => Ok(PermissionMode::WorkspaceWrite),
        "danger-full-access" => Ok(PermissionMode::DangerFullAccess),
        other => Err(format!(
            "unsupported ENDLESS_PERMISSION_MODE `{other}`; expected read-only, workspace-write, or danger-full-access"
        )),
    }
}

fn parse_csv_set(value: Option<String>) -> Option<BTreeSet<String>> {
    value.and_then(|raw| {
        let items = raw
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .filter(|item| !item.is_empty())
            .map(ToOwned::to_owned)
            .collect::<BTreeSet<_>>();
        (!items.is_empty()).then_some(items)
    })
}

fn parse_hash_set(value: Option<String>) -> Option<HashSet<String>> {
    value.and_then(|raw| {
        let items = raw
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .filter(|item| !item.is_empty())
            .map(ToOwned::to_owned)
            .collect::<HashSet<_>>();
        (!items.is_empty()).then_some(items)
    })
}

fn read_required_env(key: &str) -> Result<String, String> {
    match env::var(key) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) | Err(_) => Err(format!("missing required environment variable {key}")),
    }
}

fn current_date_string() -> String {
    Utc::now().format("%Y-%m-%d").to_string()
}

fn now_millis() -> i64 {
    Utc::now().timestamp_millis()
}

fn io_error(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::{
        limit_for_slack, parse_csv_set, parse_permission_mode, sanitize_path_component,
        sanitize_slack_prompt,
    };
    use runtime::PermissionMode;

    #[test]
    fn sanitizes_mentions_for_app_mentions() {
        assert_eq!(
            sanitize_slack_prompt("<@U123> review this diff", true),
            "review this diff"
        );
    }

    #[test]
    fn keeps_text_when_not_removing_mentions() {
        assert_eq!(
            sanitize_slack_prompt("follow up in thread", false),
            "follow up in thread"
        );
    }

    #[test]
    fn parses_allowed_tools() {
        let parsed = parse_csv_set(Some("read_file, grep_search glob_search".to_string()))
            .expect("tools should parse");
        assert!(parsed.contains("read_file"));
        assert!(parsed.contains("grep_search"));
        assert!(parsed.contains("glob_search"));
    }

    #[test]
    fn parses_permission_mode_values() {
        assert_eq!(
            parse_permission_mode("workspace-write").expect("mode should parse"),
            PermissionMode::WorkspaceWrite
        );
    }

    #[test]
    fn truncates_long_messages() {
        let text = "a".repeat(5000);
        let limited = limit_for_slack(&text);
        assert!(limited.len() < text.len());
        assert!(limited.contains("[truncated for Slack]"));
    }

    #[test]
    fn sanitizes_path_components() {
        assert_eq!(sanitize_path_component("C01.234"), "C01_234");
    }
}
