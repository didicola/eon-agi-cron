# eon-agi-cron

Cloud watchdog for the `eon-agi` Cloudflare Worker (cloud AGI loop) — runs on
GitHub Actions, **0 locals, 24/7**.

- Worker: https://eon-agi.eon-sovereign.workers.dev
- State: KV-persistent (survives worker restarts)
- DO alarm (every 10 min, CF-native) owns the learning cycles; this repo is the
  **independent redundancy layer**:

## Jobs

| Job | When | What |
|---|---|---|
| `tick` | every schedule (~6x/h) + dispatch | watchdog: health probe (3x retry) → staleness check (`last_run` > 25min = alarm broken → self-heal `POST /run-cron`) → `POST /horus/drain` (cloud 24/7 execution) → ledger summary |
| `goal` | hourly `:17` + dispatch | author goal via cloud-brain, pin to ghost, read-back verify |
| `heartbeat` | hourly `:17` + dispatch | commit `hb/last_beat.txt` — GH disables cron after 60d with no commit |

## Reverse-engineering notes (2026-10-13)

- **Bug fixed**: original `if:` compared `github.event.schedule == '*/10 * * * *'`
  but the cron entry is `'4-59/10 * * * *'` → never matched → `tick` NEVER ran on
  schedule (only on manual dispatch). Gate is now `github.event_name`.
- `tick` used to blindly `POST /run-cron` — would blow CF KV ~1000 puts/day budget
  (DO alarm already owns cycles). Now a watchdog: kicks `/run-cron` only when stale.
- GitHub schedule events can arrive hours late or be skipped under load; CF DO
  alarm is the primary cadence, this repo is the safety net.
- `horus-tasks.json` mirrors the cloud ledger (GitHub fallback for the phone bridge).
