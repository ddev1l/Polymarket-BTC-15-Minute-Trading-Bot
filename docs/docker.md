# Running the Bot with Docker

This project can run as a Docker stack for easier startup and sharing. You get the bot, Redis, Prometheus, and Grafana in one place.

## Prerequisites

- Docker and Docker Compose (v2+)
- A `.env` file with your configuration (see below)

## Quick start

1. **Create environment file**

   Copy the example and edit with your values. If you skip this, create an empty `.env` (e.g. `touch .env`) so Docker Compose does not fail when loading the bot service.

   ```bash
   cp env.example .env
   ```

   For **simulation/test mode** you can leave Polymarket credentials empty or use placeholders; the bot will not place real orders. For **live trading** you must set:

   - `POLYMARKET_PK` – private key
   - `POLYMARKET_API_KEY`
   - `POLYMARKET_API_SECRET`
   - `POLYMARKET_PASSPHRASE`

   Redis is configured automatically in Docker (`REDIS_HOST=redis`). Override in `.env` if needed.

2. **Start the stack**

   ```bash
   docker compose up -d
   ```

   This starts:

   - **Bot** (simulation, test mode) on port 8000
   - **Redis** on port 6379
   - **Prometheus** on port 9090
   - **Grafana** on port 3000

3. **View logs**

   ```bash
   docker compose logs -f bot
   ```

## Running only the bot and Redis

If you do not need Prometheus or Grafana:

```bash
docker compose up -d bot redis
```

## Live trading

**Use with caution. Real money is at risk.**

To run the bot in live mode instead of simulation:

```bash
docker compose run --rm bot python bot.py --live
```

Ensure your `.env` is in the project directory so the bot container receives your Polymarket credentials. Or change the default command in `docker-compose.yml` for the `bot` service to `["python", "bot.py", "--live"]` and use `docker compose up -d`.

## Grafana dashboard

1. Open http://localhost:3000 and log in (default: admin / admin).
2. Add Prometheus as a data source: **Connections → Data sources → Add data source → Prometheus**.
   - URL: `http://prometheus:9090` (from inside Docker) or `http://localhost:9090` (if Grafana runs on the host).
   If Grafana is running in Docker, use `http://prometheus:9090`.
3. **Dashboards → New → Import** and upload `grafana/dashboard.json` from the repo (or paste its contents). Select the Prometheus data source when prompted.

## Building the image only

```bash
docker build -t polymarket-btc-15m-bot:latest .
```

Run the container manually (e.g. with Redis on the host):

```bash
docker run --rm -it --env-file .env -e REDIS_HOST=host.docker.internal -p 8000:8000 polymarket-btc-15m-bot:latest
```

On Linux, use the host’s IP instead of `host.docker.internal` or run Redis in a container and attach to the same network.

## Stopping and cleaning up

```bash
docker compose down
```

To remove volumes (Redis data, Prometheus data, Grafana data):

```bash
docker compose down -v
```

## File reference

| File | Purpose |
|------|--------|
| `Dockerfile` | Builds the bot image (Python 3.12, fixes requirements encoding, runs `bot.py`) |
| `docker-compose.yml` | Defines bot, Redis, Prometheus, and Grafana services |
| `grafana/prometheus.docker.yml` | Prometheus config that scrapes the bot at `bot:8000` |
| `env.example` | Template for `.env` (copy to `.env` and fill in) |
| `.dockerignore` | Excludes venv, secrets, and cache from the build context |
