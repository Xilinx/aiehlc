## Team Chat (Mattermost)

Self-bootstrapping Mattermost deployment for team communication and Claude Code integration.

### Quick Start

```bash
# One command to set up everything and launch Claude Code with chat
.claude/skills/chatsupport/claude-chat.sh

# Or manage Mattermost independently
.claude/skills/chatsupport/mattermost-setup.sh start    # Bootstrap or start
.claude/skills/chatsupport/mattermost-setup.sh stop     # Stop containers
.claude/skills/chatsupport/mattermost-setup.sh status   # Check health
.claude/skills/chatsupport/mattermost-setup.sh backup   # Database backup
.claude/skills/chatsupport/mattermost-setup.sh destroy  # Tear down everything
.claude/skills/chatsupport/mattermost-setup.sh token    # Show bot token
```

### Chat in Claude Code

Use `/chat` slash command (requires Mattermost running + MCP configured):
- `/chat` -- read recent messages from `aiehlc-dev`
- `/chat read build-status` -- read specific channel
- `/chat post <message>` -- post to `claude-assistant` channel
- `/chat status` -- post build/test status to `build-status`

### Files

| File | Purpose |
|------|---------|
| `.claude/skills/chatsupport/mattermost-setup.sh` | Bootstrap & lifecycle management |
| `.claude/skills/chatsupport/docker-compose.yml` | Mattermost + PostgreSQL containers |
| `.claude/skills/chatsupport/claude-chat.sh` | One-command launcher |
| `.claude/skills/chatsupport/chat.md` | `/chat` slash command definition |

### Infrastructure

- **Deploy location**: `<deploy-dir>/` (persistent, outside repo)
- **Web UI**: `http://<hostname>:8065`
- **PostgreSQL**: `localhost:5433` (avoids conflict with standard 5432)
- **Bot token**: `~/.mattermost-claude-token`
- **MCP**: Auto-configured in `.mcp.json` via `@nicobailon/mattermost-mcp-server`