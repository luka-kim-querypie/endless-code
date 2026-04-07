# Endless Code Slack Bridge

`endless-slack` exposes the Endless Code runtime through Slack thread conversations.

## What it does

- Receives Slack Events API payloads
- Starts one persistent agent session per Slack thread
- Reuses the existing Rust provider/runtime/tool loop
- Supports thread-local commands:
  - `/model <model>` — switch the thread to another model
  - `/status` — show current model and session path
  - `/reset` — delete the thread session and start fresh

## Supported model routing

The Slack bridge uses the same model routing rules as the Rust CLI:

- Anthropic via `ANTHROPIC_API_KEY`
- OpenAI via `OPENAI_API_KEY`
- xAI via `XAI_API_KEY`
- Groq via `GROQ_API_KEY`
- OpenAI-compatible gateways via `OPENAI_API_KEY` + `OPENAI_BASE_URL`

Examples:

- `claude-sonnet-4-6`
- `claude-opus-4-6`
- an OpenAI model your endpoint accepts, such as `gpt-4.1`
- `grok-3`

## Environment

Required:

```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_SIGNING_SECRET="..."
```

Agent configuration:

```bash
export ENDLESS_SLACK_BIND="127.0.0.1:8787"
export ENDLESS_WORKDIR="/absolute/path/to/your/repo"
export ENDLESS_DEFAULT_MODEL="claude-sonnet-4-6"
export ENDLESS_PERMISSION_MODE="read-only"
```

Optional:

```bash
export ENDLESS_ALLOWED_CHANNELS="C01234567,C07654321"
export ENDLESS_ALLOWED_TOOLS="read_file,grep_search,glob_search"
export ENDLESS_SYSTEM_PROMPT_APPEND="Prefer concise Slack-friendly replies."
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-openai-compatible-gateway.example/v1"
export XAI_API_KEY="..."
export GROQ_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Running

```bash
cd rust
cargo run -p endless-slack
```

Expose the bind address to Slack with your preferred tunnel or ingress, then configure Slack Events API to send events to:

```text
https://your-host.example/slack/events
```

Subscribe to:

- `app_mention`
- `message.channels`
- `message.groups`

For the first turn in a thread, mention the app. Once a thread session exists, replies in that thread continue the same conversation without another mention.
