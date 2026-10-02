# Subrosa backlog

Ideas worth doing, not yet scheduled. Move an item into its own plan in
`plans/` when work starts.

## Hourly scan

- **Deterministic prefetch, LLM only for judgment** — Python fetches Slack
  mentions/DMs, unread + already-replied filtering, Jira mention comments,
  Brock posts and channel activity; the model only classifies and summarizes.
  Skip the LLM call entirely when nothing survives the filters. *(in progress
  2026-10-01)*
- **Remember what was already reported** — store ids of messages/comments
  that were alerted on so a slow thread never re-alerts; optionally surface
  "still unanswered after N hours" instead of repeating or going silent.

## Telegram

- **Reply buttons / reply-to-act** — inline buttons on alerts (Reply,
  Snooze 2h, Done), or reply to a Telegram message to post into the Slack
  thread / Jira comment. Note: Atilio usually acts on his work machine
  directly, so weigh this against real use before building.

## Awareness

- **Follow-up tracker** — detect Atilio's own commitments in Slack ("I'll
  look into it", "will get back to you") and nudge if there's no follow-through
  after a day.
- **Error-channel analysis** — cluster #eng-scout_errors alerts by signature
  over time and classify each cluster: noise/flapping alert, has a Jira
  ticket or not, looks like a genuine new problem, recurring known issue.
  Link existing tickets; call out genuine problems with no ticket.
