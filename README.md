# park-concierge

A conceptual demo that wires six things together in one small project: a **LangGraph** agent loop,
tools served over **MCP**, an internal **Postgres** data source, **RAG** over an external data
source in a local vector DB, and an **eval suite** that judges the agent's answers inside real
containers, with every run traced in **LangSmith**.

The domain is a National Park trip concierge. It was picked because it gives every capability a
non-contrived job: customer records live in SQL, park knowledge lives in a vector store and groups
naturally by likeness (desert / alpine / coastal / volcanic), and good answers have checkable
constraints - respect this visitor's elevation limit, surface that closure, don't invent a park.

## System

```mermaid
flowchart LR
    U[CLI / eval runner] --> G

    subgraph agent["park_agent - LangGraph"]
        G[llm node<br/>gpt-4o] <--> T[tools node<br/>+ ToolTrace]
    end

    T -->|MCP: stdio or in-process| M

    subgraph mcp["park_mcp - MCPServer"]
        C[customers toolset]
        P[parks toolset]
    end

    C -->|psycopg3, read-only| DB[(Postgres 17<br/>customer, visit)]
    P --> VS[(Chroma<br/>collection: parks)]
    P --> SN[(parks_snapshot.json<br/>alerts, elevations)]

    subgraph evals["park_evals"]
        J[judge<br/>gpt-4o-mini]
    end
    G -.answer + trace + retrieved.-> J
```

One turn, end to end:

```mermaid
sequenceDiagram
    participant E as pytest (eval)
    participant TC as testcontainers
    participant A as park_agent
    participant M as park_mcp
    participant PG as Postgres
    participant CH as Chroma
    participant J as judge model

    E->>TC: start postgres:17, apply db/init.sql
    E->>M: build_server(toolsets, ServerContext(dsn=container_dsn))
    E->>A: run_scenario(prompt, server=in-process)
    A->>M: customer_lookup(1042)
    M->>PG: SELECT ... WHERE id = %s
    PG-->>A: tier, state, accessibility, max_elevation_ft
    A->>M: parks_search("warm low-elevation accessible")
    M->>CH: similarity_search(k)
    CH-->>A: park chunks + metadata
    A->>M: parks_alerts("bibe")
    M-->>A: Chisos Basin closure
    A-->>E: answer + ToolTrace + retrieved context
    E->>E: assert must_call / must_not_call (cheap, runs first)
    E->>J: prompt + answer + context + scoring rubric
    J-->>E: {passed, score, reason}
```

## Modules

| Module | Role |
|---|---|
| `park_core` | Settings, the Chroma access contract, and the NPS ingest. Shared so the server and agent cannot disagree about where vectors live or which model wrote them. |
| `park_mcp` | The MCP server. Two toolsets, every tool is read-only and decorator-registered. |
| `park_agent` | The LangGraph loop, plus the hand-rolled MCP to LangChain tool bridge that records the **tool trace**. |
| `park_evals` | Scenario loading, deterministic trace checks, and the LLM judge. |
| `park_core.tracing` | LangSmith setup and flush; see [docs/langsmith.md](docs/langsmith.md). |

## Tools

| Tool | Source | Returns |
|---|---|---|
| `customer_lookup` | Postgres | One customer: pass tier, home state, party size, accessibility needs, elevation limit |
| `customer_find_by_email` | Postgres | Same, by email (case-insensitive) |
| `customer_visits` | Postgres | Past visits, newest first, with ratings |
| `parks_search` | Chroma | Semantic search over park descriptions, filterable by `state` and `max_elevation_ft` |
| `parks_similar_to` | Chroma | Nearest neighbours of a park's own embedding - the likeness showcase; takes the same filters |
| `parks_get` | snapshot | Full structured record for one park |
| `parks_alerts` | snapshot | Closures, road status, hazards, reservation requirements |

Alerts and elevations are read from the committed snapshot rather than the vector store: they are
structured facts, not things worth retrieving by cosine similarity. The vector store stays
responsible for what it is actually good at.

## Setup

```bash
uv sync
cp .env.example .env        # add OPENAI_API_KEY; NPS_API_KEY is optional
docker compose up -d        # Postgres on host port 5433
uv run park-ingest          # embed the parks into ./data/chroma
```

`park-ingest` fetches live from the National Park Service API when `NPS_API_KEY` is set, and
rewrites `data/parks_snapshot.json` as it goes. Without a key it falls back to the committed
snapshot, so the demo and the eval suite are reproducible with no key and no network.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | - | Embeddings, the agent model, and the judge |
| `NPS_API_KEY` | empty | Live NPS ingest; falls back to the snapshot when unset.  You can get a Parks Service developer api key here :  https://www.nps.gov/subjects/developer/get-started.htm|
| `PG_DSN` | `postgresql://postgres:postgres@localhost:5433/parks` | Customer database |
| `CHROMA_DIR` | `./data/chroma` | Vector store location |
| `AGENT_MODEL` | `gpt-4o` | The agent |
| `JUDGE_MODEL` | `gpt-4o-mini` | The eval judge - deliberately not the agent's model |

## Run

```bash
# one-shot, over stdio - the agent spawns park-mcp as a real subprocess
uv run park-agent "Customer 1042 is planning a trip in March. Where should they go?"

# REPL
uv run park-agent

# skip the subprocess when debugging a tool
uv run park-agent --inprocess "What is similar to Zion?"
```

The server also runs standalone, and `.mcp.json` registers it for Claude Code so you can drive the
tools by hand:

```bash
uv run park-mcp                  # stdio
uv run park-mcp streamable-http  # http on :8050
```

## Observability: LangSmith tracing

Every agent turn is recorded in LangSmith - graph steps, model prompts and tokens, tool calls and
results, latency - with no change to how you run the app. Set `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` (shell or `.env`), run `park-agent`, and open the `park-concierge` project in
the [portal](https://smith.langchain.com/o/5eb087f6-34d7-4bf9-8857-af42ef22e08c). The project is
created on the first trace; nothing to set up in the portal.

| Variable | Default | Purpose |
|---|---|---|
| `LANGSMITH_TRACING` | unset (off) | Master switch |
| `LANGSMITH_API_KEY` | - | Credentials |
| `LANGSMITH_PROJECT` | `park-concierge` | Where traces land |

The wiring is small: `park_core/tracing.py` (setup and flush), one `@traceable` root span on
`run_agent`, and one on the judge. LangGraph and LangChain emit the rest themselves. Full detail,
connection points and portal walkthrough: [docs/langsmith.md](docs/langsmith.md).

## Run the tests

```bash
uv run pytest -v          # fast suite: free, needs no OPENAI_API_KEY
uv run pytest -m eval -v  # eval suite: spends tokens on two models
```

The fast suite starts a throwaway `postgres:17` with **testcontainers** and builds the vector store
with deterministic offline embeddings, so it asserts real wiring against real infrastructure without
spending anything. The eval suite uses real embeddings and real models.

| Scenario | What it catches |
|---|---|
| `warm_march_accessible` | Happy path - both data sources, and a hard accessibility/elevation constraint |
| `similar_to_zion` | Likeness search, and whether explanations stay grounded in tool output |
| `unknown_customer` | A null lookup must not become an invented profile |
| `alert_blocks_recommendation` | A seeded closure must be surfaced, not quietly ignored |
| `off_topic_guard` | An unrelated question must be declined, not answered from general knowledge |

Each scenario runs two kinds of check. `must_call` / `must_call_any` / `must_not_call` assert on the
recorded tool trace - deterministic, and they run **before** the judge, so a wiring regression fails
loudly and for free. The judge then grades the answer against the scenario's rubric, returning one
ruling per criterion:

```
[warm_march_accessible] score=5/5 passed=True
  all 4 criteria met
```

Grading criterion-by-criterion rather than holistically was not a stylistic choice. Asked to weigh
three or four criteria in one pass, the judge reliably lost track of quantifiers - reading "for at
least two of its suggestions" as "for all of them", and "either the park is under the limit or the
answer flags it" as "both" - and failed answers that were correct. One ruling per criterion made the
verdicts stable, and the score becomes arithmetic instead of a vibe.

## Things worth knowing

A few traps this project hit, documented in the code where they bite:

- **Tools are closures over an injected `ServerContext`, not readers of module-level settings.**
  Under testcontainers the DSN is only known after a fixture starts the container, long after
  `park_mcp` was imported. A tool body reading frozen settings would make the evals silently test
  your own database and still pass.
- **`run_agent(server=...)` has no default.** Same reason: defaulting to stdio would spawn a server
  that reads its own `.env` and bypasses the test container.
- **The stdio client does not forward your environment.** It passes a six-variable allowlist, so the
  agent explicitly forwards `OPENAI_API_KEY` and friends to the subprocess.
- **The MCP envelope wraps non-dict returns as `{"result": ...}`.** Unwrap it, or the model - and the
  judge scoring groundedness - sees the envelope instead of the parks.
- **`langchain_openai` memoises its async httpx client process-wide.** With a fresh event loop per
  test that yields `RuntimeError: Event loop is closed` from inside the llm node; passing an
  `httpx.Timeout` sidesteps the cache.
- **Negative rubric criteria grade badly.** "Does not claim X is available" got read as a positive
  requirement to state the opposite. Phrase criteria as the thing you want said.
- **Hard constraints belong in the metadata filter, not the query text.** An embedding has no idea
  that 5000 is less than 7860, so "a park under 5000 ft" as prose returns scenic alpine parks. The
  eval suite caught this twice: first that the constraint was being described rather than filtered,
  then that only `parks_search` could enforce it while the agent was reaching the knowledge base
  through `parks_similar_to`. A constraint one retrieval path cannot enforce is a coin flip over
  which tool the model picks, so every park-returning tool now takes the same filter.

## Simplifications and next steps

- Every tool is read-only. There is no booking, no auth, no writes from the agent.
- Alerts and elevations live in a JSON snapshot; in a real system they would be tables, and the
  vectors would plausibly move into pgvector alongside them.
- Elevation data is curated, not from the NPS API, which does not expose it.
- The eval suite has no scorecard history - results are per-run. Persisting verdicts to compare runs
  over time is the obvious next step.
- Retrieval is pure dense similarity. No hybrid search, no reranking, no chunk-level citations.
- The agent is single-turn. There is no conversation memory between questions in the REPL.
