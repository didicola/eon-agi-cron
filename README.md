# eon-agi-cron

24/7 GitHub-Actions heartbeat driving `eon-agi` (Cloudflare Worker) through
self-improvement cycles every 10 minutes — even when all local machines are OFF.

- Worker: https://eon-agi.eon-sovereign.workers.dev
- State: KV-persistent (survives worker restarts)
- Schedule: `*/10 * * * *` + hourly heartbeat commit (keeps cron alive)
