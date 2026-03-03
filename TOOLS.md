# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.

### API Keys (in ~/.bashrc)

- **EXA_API_KEY** — Exa neural search API (https://api.exa.ai). Use for literature search, finding papers by semantic query. Working as of 2026-03-03.
- **GEMINI_API_KEY** — Google Gemini API (gemini-2.0-flash). Use for fast synthesis, critique, hypothesis sharpening. Working as of 2026-03-03.

### Usage Pattern

```bash
EXA_KEY=$(grep EXA_API_KEY ~/.bashrc | cut -d= -f2-)
GEMINI_KEY=$(grep GEMINI_API_KEY ~/.bashrc | cut -d= -f2-)

# EXA search
curl -s "https://api.exa.ai/search" \
  -H "x-api-key: $EXA_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"...","numResults":8,"useAutoprompt":true}'

# Gemini generate
curl -s "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$GEMINI_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"..."}]}]}'
```

### Notes
- Keys are NOT in env by default (source ~/.bashrc or grep directly)
- EXA costs ~$0.005/search — be conservative in cron jobs
- Gemini 2.0 Flash is fast and cheap — good for synthesis/critique loops
