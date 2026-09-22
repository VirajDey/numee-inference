# Numee Inference Service

Embeddings, vector retrieval and document extraction for the Numee agents API.

Split out of the API so the CPU-bound work scales on its own hardware. This
service holds the only copy of `sentence-transformers`, `torch`, `pymupdf` and
`pandas` in the system; the API is a thin HTTP client.

Runs as a standalone deployment — it shares no code with the API repo and talks
to it only over HTTP.

```
┌──────────────┐   HTTPS + X-Internal-Token   ┌───────────────────┐
│  numee-api   │ ───────────────────────────► │ numee-inference   │ ──► Qdrant
│  (thin)      │ ◄─────────────────────────── │ (this repo)       │ ──► OpenAI
└──────────────┘        ranked results        └───────────────────┘     embeddings
```

## API

Everything under `/v1` requires the `X-Internal-Token` header. The probes do
not, so an orchestrator can reach them without holding the secret.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/retrieve` | Embed a query, search both vector spaces, return the fused ranking |
| `POST` | `/v1/documents/extract` | Download and parse a PDF into pages |
| `GET` | `/v1/knowledge_sources` | List Qdrant collections |
| `POST` | `/v1/knowledge_sources` | Build a collection from a spreadsheet (parse, embed, upload) |
| `POST` | `/v1/knowledge_sources/delete` | Drop collections |
| `GET` | `/health` | Qdrant reachability and model state |
| `GET` | `/ready` | 503 until the embedding model is resident |

Interactive docs at `/docs` once running.

### Retrieval

```http
POST /v1/retrieve
X-Internal-Token: <secret>

{"query": "I led a migration", "collection": "numee_competency", "top_k": 3}
```

```json
{"results": [{"id": 7, "score": 0.91, "text": "Description : ...",
              "metadata": {"Competency": "Change Management", "Catagory": "Social"}}]}
```

Both vector spaces are queried separately, min-max normalised within each, then
blended (`alpha` weights the OpenAI space, default 0.6). The fusion runs here so
one ranked list crosses the network instead of two full result sets.

## Configuration

Copy `.env.example` to `.env` and fill it in. Config is validated at import, so
a missing or malformed value fails the process on startup rather than on the
first request.

| Variable | Required | Meaning |
| --- | --- | --- |
| `INFERENCE_SERVICE_TOKEN` | yes | Shared secret, min 16 chars. Must match the API's copy. `openssl rand -hex 32` |
| `QDRANT_URL` | yes | Qdrant endpoint |
| `QDRANT_API_KEY` | no | Qdrant auth, if enabled |
| `OPENAI_API_KEY` | yes | Used for embeddings only, not chat |
| `OPENAI_EMBEDDING_MODEL` | no | Default `text-embedding-3-small` |
| `HUGGINGFACE_EMBEDDING_MODEL` | no | Default `sentence-transformers/all-MiniLM-L6-v2` |
| `OPENAI_VECTOR_SIZE` | no | 1536. Must match how the collections were created |
| `SENTENCE_TRANSFORMER_VECTOR_SIZE` | no | 384. Must match the HF model's output |
| `INFERENCE_MAX_CONCURRENCY` | no | 4. Concurrent embedding/parsing operations per replica |
| `LOG_LEVEL` | no | `INFO` |

Changing the embedding model means rebuilding every collection: existing vectors
were written by the old model and the two are not comparable.

## Running

```bash
# local
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements-dev.txt
pytest
python main.py                                       # http://localhost:8100

# container
docker compose up --build
```

The image installs CPU-only torch and bakes the embedding model in at build
time, so replicas start without reaching the Hugging Face hub. If you change
`HUGGINGFACE_EMBEDDING_MODEL`, rebuild with a matching build arg:

```bash
docker build --build-arg HUGGINGFACE_EMBEDDING_MODEL=<model> -t numee-inference .
```

## Deployment

- **Never expose this publicly.** It fetches URLs it is given and returns their
  parsed contents — those URLs are candidate CVs. Keep it on a private network,
  a VPC peer or a VPN, reachable only by the API.
- Point the load balancer's readiness probe at `/ready`, not `/health`. A
  replica whose model is still loading answers `/health` but must not receive
  traffic.
- Scale on CPU. Each replica holds one copy of the model; raise
  `INFERENCE_MAX_CONCURRENCY` with the CPU allocation, not with request volume.
- Allow 60-120s of start period — the model load dominates boot.
- Data residency: this service handles personal data. Deploy it in the same
  jurisdiction as the API and the Qdrant cluster.

## Client side

The Numee API calls this through `src/clients/inference_client.py` in the
`numee` repo. Its `INFERENCE_URL` and `INFERENCE_SERVICE_TOKEN` must point here
and match. That client retries transport failures twice with backoff but never
retries an error response, and never retries knowledge-source ingest — a read
timeout there usually means the ingest is still running.

## Tests

`pytest` — the Qdrant/OpenAI backed service is stubbed, so the suite needs no
credentials and no network. It covers auth, request validation, response
shapes, the readiness gate, the fusion maths and the rollback behaviour when a
knowledge-source build fails halfway.
