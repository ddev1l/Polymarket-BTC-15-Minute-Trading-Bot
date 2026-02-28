# Running the Full Stack: Bot, Prometheus, and Grafana

Step-by-step guide to run the Polymarket BTC 15-minute trading bot with metrics and the Grafana dashboard. Use this when starting fresh or returning to the project.

---

## What Runs Where

| Component | Port | Role |
|-----------|------|------|
| Trading bot | 8000 | Runs the strategy and exposes Prometheus metrics at `/metrics` |
| Prometheus | 9090 | Scrapes the bot's metrics and stores them; Grafana queries Prometheus |
| Grafana | 3000 | Web UI and dashboards; you point it at Prometheus as a data source |

The bot runs in a **Python virtual environment**. Prometheus and Grafana are **separate programs** (not in the venv), installed via your OS or downloaded binaries.

---

## Prerequisites

- Python 3.11+ (for the bot)
- Prometheus installed (package manager or binary from [prometheus.io/download](https://prometheus.io/download))
- Grafana installed (package manager or from [grafana.com](https://grafana.com/grafana/download))
- Optional: Redis if you use it for bot control

---

## Step 1: Bot (virtual environment)

From the project root:

```bash
# Create and activate the virtual environment (only needed once)
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# On Windows: venv\Scripts\activate

# Fix requirements.txt encoding if it was saved as UTF-16 (run once if needed)
python scripts/fix_requirements_encoding.py

# Install dependencies (only when requirements change)
pip install -r requirements.txt

# Optional: set env vars (keys, API URLs, etc.) in .env or export them

# Run the bot in test mode (simulation, fast 1-minute clock) with Grafana metrics on
python bot.py --test-mode
```

Leave this terminal open. The bot will expose metrics at **http://localhost:8000/metrics**. For live trading you would use `python bot.py --live` (and accept the risk); omit `--test-mode` for normal simulation.

---

## Step 2: Prometheus

Prometheus must scrape the bot and use a **writable data directory** (to avoid permission errors when not running as root).

From the project root:

```bash
# Create a local data directory (only needed once)
mkdir -p prometheus_data

# Start Prometheus with the project config and local storage
prometheus --config.file=grafana/prometheus.yml --storage.tsdb.path=./prometheus_data
```

Leave this terminal open. Prometheus will listen on **http://localhost:9090**.

- If port 9090 is already in use, stop the other process (e.g. `sudo systemctl stop prometheus`) or use a different port:
  ```bash
  prometheus --config.file=grafana/prometheus.yml --storage.tsdb.path=./prometheus_data --web.listen-address=:9091
  ```
  If you use 9091, point Grafana's data source at **http://localhost:9091** instead of 9090.

**Check:** Open http://localhost:9090 → **Status → Targets**. The target `localhost:8000` (job `polymarket-bot`) should be **UP**. In **Graph**, run the query `trades_total` to confirm data.

---

## Step 3: Grafana

Start Grafana using your normal method, for example:

```bash
# Linux (system service)
sudo systemctl start grafana-server

# Or run the Grafana binary directly from its install directory
```

Grafana usually listens on **http://localhost:3000**. Log in (default admin/admin; change when prompted).

### Add Prometheus as a data source

1. Go to **Connections → Data sources** (or the gear icon).
2. **Add data source** → choose **Prometheus**.
3. Set **URL** to `http://localhost:9090` (or `http://localhost:9091` if you used the alternate port).
4. **Save & test**. It should report "Data source is working".

### Load the dashboard

1. **Dashboards → New → Import**.
2. Upload `grafana/dashboard.json` (or paste its contents), or use **Upload dashboard file** and select the file from the project's `grafana/` folder.
3. When asked **"Select a Prometheus data source"**, choose the Prometheus source you just added.
4. Click **Import**.

Open the **Polymarket BTC 15 mins Trading Bot** dashboard. Set the time range (e.g. **Last 15 minutes**) and refresh; you should see metrics (Total Trades, P&amp;L, Win Rate, etc.).

---

## Start Order (quick reference)

When you come back to the project, start in this order:

1. **Prometheus** (so it can scrape the bot as soon as the bot is up):
   ```bash
   cd /path/to/Polymarket-BTC-15-Minute-Trading-Bot
   prometheus --config.file=grafana/prometheus.yml --storage.tsdb.path=./prometheus_data
   ```

2. **Bot** (in a separate terminal, with venv activated):
   ```bash
   cd /path/to/Polymarket-BTC-15-Minute-Trading-Bot
   source venv/bin/activate
   python bot.py --test-mode
   ```

3. **Grafana** (if not already running as a service):
   ```bash
   sudo systemctl start grafana-server
   # or run the Grafana binary
   ```

Then open **http://localhost:3000**, open the dashboard, and ensure the data source is the Prometheus you configured.

---

## Troubleshooting

- **Bot: "No module named 'nautilus_trader'"**  
  Activate the venv and run `pip install -r requirements.txt` (after fixing encoding with `scripts/fix_requirements_encoding.py` if needed).

- **Prometheus: "permission denied" or "address already in use"**  
  Use `--storage.tsdb.path=./prometheus_data` from the project root. If 9090 is in use, stop the other process or use `--web.listen-address=:9091` and point Grafana at 9091.

- **Grafana: "No data"**  
  Confirm Prometheus has data: http://localhost:9090 → Graph → `trades_total`. Then in Grafana, set the dashboard (and each panel if needed) to use the Prometheus data source and set a sensible time range (e.g. Last 15 minutes).

- **Dashboard shows wrong metrics**  
  Re-import `grafana/dashboard.json` and select the correct Prometheus data source on import.
