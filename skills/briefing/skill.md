# Scout Engineering Briefing

Generate a comprehensive briefing on Scout engineering activity covering Slack, Jira, and GitHub.

## Instructions

**CRITICAL: Slack Channel Reading**
1. First, check if Slack channels are cached and ready:
   - Use `slack_list_channels` with `name_contains: "scout"`
   - If status is "loading", wait 5-10 seconds and retry
   - Do NOT proceed until channels are ready
2. For each Scout channel below, read the last 50 messages using the channel name
3. Check for activity in the last 24-48 hours
4. Include thread replies — important discussions happen in threads

**Scout Channels to Scan:**

Core Subsystems:
- scout-product-compass
- scout-product-missions
- scout-product-evals
- scout-product-storage
- scout-product-connectors
- scout-product-platform
- scout-product-skills

Operations:
- scout-guild
- scout-errors
- scout-coders
- scout-releases
- scout-red_alert
- scout-aws-finance

Cross-functional:
- support-general
- rd-scout-all

**Jira Analysis:**
- Query: `project = SCOUT AND updated >= -2d ORDER BY updated DESC`
- Focus on: status changes, new issues, blockers, items moving to Code Review
- Note assignees and which team they're on (use team.json)

**GitHub Activity:**
- Check open PRs via `gh pr list --repo sitetracker/scout --limit 20`
- Note: who opened, which team, status, how long open
- Flag: PRs waiting >3 days for review

**Output Format:**

Start with **TOP PRIORITY** section if there are:
- DMs requiring attention
- Red alerts or errors
- Blockers flagged in Jira
- PRs waiting >5 days for review

Then organize by:
1. **Slack Highlights** — channel by channel, only if activity found
2. **Jira Activity** — tickets updated in last 2 days
3. **GitHub PRs** — open PRs by team
4. **Risks/Blockers** — anything flagged

**Formatting:**
- Lead with most important information
- Use bullet points
- Include ticket numbers (SCOUT-1234), PR numbers (#123), team names
- Distinguish facts (what you observed) from analysis (what it means)
- Skip channels with no activity — only report signal

## Context

**Organization:**
- VP Engineering: Atilio (Colorado, CT timezone)
- 30-person Scout team across 6 subsystems
- 6 band-named teams: Incubus (Geoff), Pink Floyd (Jared), The Killers (Robi), B-52s (Sean), RHCP (Jake), Queen (Yuval)

**Active Priorities:**
1. Scout platform development and stabilization
2. Customer pilots (RWE, Enfinity, Evyve, Powerhouse)
3. Subsystem integration and reliability
4. PR review backlog (major pain point)

**Risk Register:**
- 20 zombie tickets from 2021-2022 in 'In Progress'
- No monitoring for Scout nightly eval runs
- Workflow conflict: local-work vs daily-push (Yuval/Brock)

**Key Stakeholders:**
- Brock (CTO, Atilio's boss)
- Walter (Head of Scout product)
