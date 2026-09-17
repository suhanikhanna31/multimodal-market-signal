# Local deployment

This project ships as a FastAPI app (`serving/app.py`) that serves the
trained model and, alongside each prediction, a retrieval-based contextual
signal (see `retrieval/README.md`). Three ways to run it:

## Option A — Hugging Face Spaces (recommended: free, no credit card, fits TensorFlow)

Hugging Face Spaces' free CPU tier (2 vCPU / 16 GB RAM) comfortably fits
`tensorflow-cpu`, unlike most free-tier PaaS plans. The Dockerfile and
`start.sh` in this repo are already set up for it: `start.sh` trains the
model and builds the retrieval index automatically on first boot if
they're not already present, since Spaces gives the container a fresh
filesystem on every restart (no assumption of a persisted volume).

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **SDK**: Docker, template **Blank**. **Hardware**: CPU basic (free).
   Visibility: Public (Private Docker Spaces need a paid plan).
3. Push this repo to the Space (it's a git remote, same idea as GitHub):
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
   git push hf main
   ```
   When prompted for a password, use a Hugging Face access token
   (Settings → Access Tokens on huggingface.co), not your account password.
4. The Space will build the image and boot it. The first boot is slower
   than normal (it's training + indexing, not just starting a server —
   roughly a minute on this project's small synthetic dataset, per the
   `[start.sh]` log lines visible in the Space's build/run logs).
5. Once it's up, the Space's URL serves the same API —
   `https://<your-username>-<space-name>.hf.space/health`, `/info`,
   `/predict`.

Notes specific to Spaces:
- `README.md` at the repo root carries the YAML front matter
  (`sdk: docker`, `app_port: 7860`) that tells Spaces how to build and
  which port to route to. Don't remove it.
- Free-tier Spaces sleep after a period of inactivity and cold-start
  (retraining from scratch, per the note above) on the next visit.
  That's expected given there's no persisted disk on the free tier — an
  intentional trade-off for zero cost, not a bug.
- To skip the on-boot retrain (e.g. if you later add a paid Space with
  persistent storage, or want faster restarts), pre-train locally, commit
  `saved_model.keras` and `retrieval/local_index/` (temporarily removing
  them from `.gitignore`), and `start.sh` will skip straight to serving.

## Option B — Docker Compose (local development)

```bash
# 1. (Optional) configure Pinecone. Leave .env absent/blank to use the
#    local vector-store fallback instead — no account needed.
cp .env.example .env

# 2. Train the model and build the retrieval index once, inside the
#    container, so the artifacts land in the mounted project directory
#    on your host (and are picked up by `app` without a rebuild):
docker compose run --rm app python data/generate_synthetic_data.py
docker compose run --rm app python -m src.train_distributed
docker compose run --rm app python -m retrieval.build_index --limit 2000

# 3. Start the API
docker compose up --build
```

Then:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "SYN00 shares surges after quarterly results",
    "lookback_returns": [0.001, 0.002, -0.001, 0.003, 0.0, 0.001, -0.002, 0.004, 0.001, 0.002],
    "volume_z": 0.5,
    "volatility": 0.02,
    "rsi_like": 0.6,
    "top_k_similar": 3
  }'
```

`docker compose down` stops the container; the trained model and retrieval
index persist on the host (they're bind-mounted, not baked into the
image — see `docker-compose.yml`), so `docker compose up` again picks up
right where you left off without retraining.

## Option C — Bare metal (no Docker)

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py
python -m src.train_distributed
python -m retrieval.build_index --limit 2000

uvicorn serving.app:app --reload --port 8000
```

Same `curl` calls as above work against `localhost:8000` either way.

## What `/predict` returns

```json
{
  "direction": "up",
  "probabilities": {"down": 0.28, "flat": 0.15, "up": 0.57},
  "contextual_signal": {
    "query_headline": "...",
    "matches": [{"headline": "...", "score": 0.98, "label_name": "up", "ticker": "SYN04", "day": 464}, ...],
    "label_distribution": {"up": 2, "flat": 1},
    "majority_label": "up"
  }
}
```

`contextual_signal` is the retrieval workflow's output: the model's own
prediction plus what happened after similar historical headlines. Set
`"top_k_similar": 0` in the request to skip retrieval and get only the
model's prediction.

## Switching to real Pinecone

Set `PINECONE_API_KEY` (in `.env` for Compose, or exported in your shell
for bare metal) before running `build_index` and starting the app — see
`.env.example` and `retrieval/README.md`. No code changes needed; the
factory in `retrieval/vector_store.py` picks the backend based on that
one environment variable.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. **test** job — installs `requirements.txt`, runs a syntax/undefined-name
   lint pass, then `pytest tests/ -v` on Python 3.10 and 3.11. All tests
   run against the local vector-store fallback, so no secrets are needed
   for CI to pass.
2. **docker** job — builds the image on every push/PR; on pushes to `main`,
   also pushes `ghcr.io/<owner>/<repo>:latest` and `:<commit-sha>` to
   GitHub Container Registry, using the repo's built-in `GITHUB_TOKEN`
   (no extra secrets to configure).

To pull and run a published image instead of building locally:

```bash
docker pull ghcr.io/<owner>/<repo>:latest
docker run -p 8000:7860 \
  -v $(pwd)/saved_model.keras:/app/saved_model.keras \
  -v $(pwd)/retrieval/local_index:/app/retrieval/local_index \
  ghcr.io/<owner>/<repo>:latest
```

(You still need a trained model + built index on the host to mount in —
the published image doesn't bundle them, since they're generated from
data, not source code. Skip the `-v` flags entirely to let `start.sh`
train and index from scratch inside the container instead — same
behavior as the Hugging Face Spaces path above, just without persistence
between `docker run`s.)
