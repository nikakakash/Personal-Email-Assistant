# Retrieval over past email (RAG)

When the assistant drafts a reply it can recall earlier conversations: the price you agreed,
the date someone moved, the booking code from last month. It cites each recalled email, so
you can check every claim before you approve. Retrieval is off by default (`RAG_ENABLED`).

```
ingest_message ──(owner + ordinary inbound mail)──► job index_message
                                                     │ worker: clean → chunk → [contextualize] → embed → upsert
                                                     ▼
                        Postgres: message_chunks (text, generated tsvector, GIN)
                                  chunk_embeddings (vector, one partial HNSW index per model)
                                                     ▲
response subgraph ─► retrieve_history ──search()─────┘  account scope · not the current thread
        │                                               · every current participant was party to it
        │                                               keyword ∥ vector → RRF → [cross-encoder rerank]
        ▼
plan_next_step / draft_reply see "Related past emails (untrusted data)" as <past_message ref="history:<id>">
        ▼
validate_proposal: history:<id> is valid only if it was retrieved for this run · recipients ⊆ thread
        ▼
review screen: "Sources used" shows each cited email
```

## Pieces

| Piece | Where | Notes |
|---|---|---|
| Chunking | `app/retrieval/chunking.py` | Drops quoted replies (`>` lines, wrapped "On … wrote:", Outlook headers) and signatures, and keeps forwarded content. Packs paragraphs into 260-word windows with a 40-word overlap. Each chunk starts with a `From / To / Date / Subject` line. |
| Embeddings | `app/retrieval/embeddings.py` | `FastEmbedClient` (local `bge-small-en-v1.5`, 384 dimensions, free) is the default. `VoyageEmbeddingsClient` (`voyage-3.5-lite`, 512 dimensions, budgeted) and a deterministic hash fake for tests are the others. Queries and documents are embedded asymmetrically. |
| Storage | migration `0004`, `app/db/models.py` | The embedding column has no fixed dimension. Each model has a partial HNSW index over `embedding::vector(dim)`, so models coexist and switching one means a reindex, not a migration. Deleting an email cascades to its chunks and vectors. |
| Indexing | `app/retrieval/indexer.py`, job `index_message` | Incremental: only new or changed chunks are embedded. The embedding runs outside any transaction, and the write takes a row lock on the message, so a delete in the middle writes nothing. `python -m app reindex [--force]` rebuilds the index. |
| Search | `app/retrieval/search.py` | Keyword search is Postgres full text (`ts_rank_cd`, the question's words OR-ed; not BM25). Vector search is cosine distance with `hnsw.iterative_scan`. Hybrid fuses the two with reciprocal rank fusion (k=60). `hybrid_rerank` adds a local cross-encoder (`ms-marco-MiniLM-L-6-v2`) over the top 20 candidates. Results are one hit per message. |
| Contextual retrieval | `app/retrieval/contextualize.py` | Optional (`RAG_CONTEXTUALIZE`). Haiku writes one situating sentence per chunk, and the sentence is indexed only, never returned. |
| Graph | `app/graphs/nodes/retrieve.py` | The first node of the response subgraph (`GRAPH_VERSION` `wf2`). A failure becomes a `history` finding and never fails the run. |

## Security boundaries (enforced in code, not by the prompt)

1. **No cross-correspondent leakage.** With `RETRIEVAL_SCOPE=participants` (the default), a
   past message is in scope only if every external participant of the current thread was a
   party to it. Replies can only go to those participants, so a recalled email never reaches
   someone who was not on it. This also stops an outsider from CC-ing your contact into a new
   thread to fish out their history; a regression test covers that case.
   `RETRIEVAL_SCOPE=mailbox` searches everything and is an explicit opt-in.
2. **Past email is data, not authority.** Retrieved text is wrapped in `<past_emails>`, and
   delimiter look-alikes inside it are defused (the same treatment as the current thread). The
   prompts (`plan_v2`, `draft_reply_v2`, `draft_meeting_v2`) say it must never be obeyed.
   Validation still limits recipients to the current thread, whatever a past email asks for.
3. **No invented citations.** `history:<id>` passes validation only if that id was retrieved for
   this run.
4. **Model-written context stays out of prompts.** Contextual-retrieval prefixes are embedded
   and keyword-indexed, but search returns only the chunk text.
5. **Mandatory scope.** Every query requires an `account_id` and excludes the current thread.
6. **Spend.** Voyage and contextualize calls reserve and settle through the daily budget guard.

## Evaluation

`evals/retrieval/corpus.py` is a hand-written mailbox: 151 messages from about 30
correspondents, and 60 queries with known answers. It contains deliberate traps:
- superseded facts;
- two Alexes and two Jordans;
- near-duplicate threads;
- identifiers;
- paraphrased questions;
- hard distractors (the same person and topic, but a wrong or outdated answer).

`evals/run_retrieval_eval.py` runs each query through the app's `search()` and reports recall@5
and recall@10, MRR, nDCG@10 and latency, per scope and mode. CI runs it with the local model
and fails below `evals/retrieval/baseline.json`.

The participant scope is nearly saturated (each person has only a few messages), so
**mailbox scope** is the real benchmark. `--enron` adds the public Enron corpus as noise.

See the README for the current numbers. The paid end-to-end eval
(`evals/run_rag_draft_eval.py`) measures what retrieval does to the drafts themselves: fact
accuracy, citation precision and faithfulness under an LLM judge, with RAG on and off.

## Operating it

| Setting | Default | Meaning |
|---|---|---|
| `RAG_ENABLED` | `false` | Index new mail and retrieve history when drafting. |
| `EMBEDDINGS_PROVIDER` / `EMBED_MODEL` | `fastembed` / `bge-small-en-v1.5` | `voyage` with `voyage-3.5-lite` needs `VOYAGE_API_KEY`. |
| `RETRIEVAL_MODE` | `hybrid` | `keyword`, `vector`, `hybrid` or `hybrid_rerank`. |
| `RETRIEVAL_TOP_K` | `5` | The number of past emails shown to the model. |
| `RETRIEVAL_SCOPE` | `participants` | `mailbox` lets every correspondent's mail compete. |
| `RAG_CONTEXTUALIZE` | `false` | Contextual retrieval with `MODEL_CONTEXTUALIZE` (Haiku). |

- **Mail stored before you enabled RAG:** run `docker compose run --rm worker python -m app reindex`.
- **After changing the model, the chunking or contextualization:** add `--force`.
- **Model files:** they download once into the `modelcache` volume.
