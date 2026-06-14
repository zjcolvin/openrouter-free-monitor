# OpenRouter Free Models Monitor

An automated monitor for OpenRouter's free models. It tracks active free models, detects new additions, removals, upgrades, and ranking changes, and posts daily summaries to Discord using native Markdown Embeds.

## Features
- **Daily Discord Notifications**: Beautiful dark-mode embeds showing performance tiers (T1-T4).
- **Automated Alerts**: Mentions like `@here` are sent only when actual changes (additions, removals, upgrades) occur.
- **Smart Tracking**: Detects version upgrades (e.g. Llama 3 -> 3.1) and handles model renames.
- **Clickable Links**: Direct URLs to OpenRouter playground for every model in the daily summary.
- **Serverless Automation**: Fully powered by GitHub Actions with automatic state preservation.

## Setup

### Local Run
1. Clone the repository.
2. Create a `.env` file:
   ```env
   DISCORD_WEBHOOK_URL=your_discord_webhook_url
   OPENROUTER_API_KEY=your_openrouter_api_key
   ALERT_MENTION=@here
   ```
3. Run the script:
   ```bash
   pip install -r requirements.txt
   python openrouter_free_monitor.py
   ```

### GitHub Actions Automation
1. Push this repository to GitHub.
2. Under **Settings** -> **Secrets and variables** -> **Actions**, add:
   - `DISCORD_WEBHOOK_URL` (Required)
   - `OPENROUTER_API_KEY` (Required)
   - `ALERT_MENTION` (Optional, e.g., `@here`)
3. Go to **Settings** -> **Actions** -> **General** -> **Workflow permissions**, and enable **Read and write permissions**.
