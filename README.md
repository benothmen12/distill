# Distill

**Goal: a small open model that calls an agent's tools as well as a model 30 times its size, and runs on a factory's own hardware.**

> 🚧 Work in progress. Results will be published here as each stage lands.

## Why

AI agents depend on one skill above all: picking the right tool and filling its arguments correctly. Today that usually means sending every request to a very large model. For an industrial plant, that is a problem:

- **Cost and latency:** an operations copilot makes thousands of tool calls a day.
- **Privacy:** production data often cannot leave the site.
- **Reliability:** the plant should keep working when the internet does not.

Distill tests whether a 3 to 4B parameter model, fine-tuned on one domain's tools, can match a large general model on that domain while running locally.

## The use case

A copilot for plant operations teams, with 12 tools:

| Area | Tools |
|---|---|
| Machines | `get_machine_status`, `get_sensor_readings`, `get_failure_risk` |
| Maintenance | `create_maintenance_ticket`, `update_ticket_status`, `schedule_downtime`, `search_maintenance_history` |
| Production | `get_production_kpis`, `get_shift_report` |
| Spare parts | `get_spare_part_stock`, `order_spare_part` |
| People | `notify_team` |

Every tool is a strict Pydantic schema ([`schemas.py`](src/distill/tools/schemas.py)). The same schema validates the training data and grades the model's outputs.

## How it works

```mermaid
flowchart LR
    T[Tool schemas] --> G[Synthetic data<br/>teacher model]
    G --> V[Validation<br/>and dedup]
    V --> S[QLoRA fine-tuning]
    S --> P[DPO on<br/>failure cases]
    P --> Q[Quantization]
    Q --> R[Serving<br/>vLLM / llama.cpp]
    V --> H[Held-out<br/>test set]
    H --> E[Evaluation harness]
    R --> E
    B[Teacher as<br/>baseline] --> E
```

1. **Data:** expected tool calls are sampled from the schemas, and an open-weight teacher model (Apache 2.0, so its outputs can be used for training) writes the operator messages that lead to them. Every example is validated, deduplicated and split without leakage. Details below.
2. **Training:** supervised fine-tuning with QLoRA, then DPO on the cases the model still gets wrong.
3. **Serving:** the model is quantized and served through an OpenAI-compatible API, so any agent framework can use it unchanged.
4. **Evaluation:** a held-out test set, scored the same way for the student and the teacher.

## The data

### Labels are correct by construction

The teacher does not invent tool calls. For each example, the pipeline first samples the expected calls inside the schema constraints, then asks the teacher to write a realistic operator message that leads to exactly those calls. This gives full control over the distribution, and labels that never contain a hallucinated argument.

| Category | Share | Expected behavior |
|---|---|---|
| Single call | 62% | One tool call |
| Parallel calls | 14% | Two independent calls from one message |
| Clarification | 12% | A required detail is missing: ask one short question, call nothing |
| Out of scope | 12% | Not about plant operations: decline briefly, call nothing |

Messages are 60% English and 40% French, in nine styles, from polite full sentences to stressed operators during an incident, voice dictation, and bare fragments like "M-104 vibre ticket méca". Each example has its own current date and time, so the model has to resolve relative dates like "tomorrow at 2 pm". Optional arguments are left out half of the time, so the model learns to rely on defaults.

### Every example is checked

| Check | Rejection reason |
|---|---|
| Arguments pass the tool schema | `invalid_call` |
| Every identifier in the label is in the message, exactly or written loosely ("machine 104") | `identifier_missing` |
| Every number, time and date can be read from the message ("2 hours" = 120 minutes) | `number_missing`, `time_missing`, `date_missing` |
| Free-text arguments restate the message and add no new fact | `ungrounded_free_text` |
| Free-text arguments are real values, not the field name | `placeholder_free_text` |
| Enums are written as words, never as code ("in_progress") | `raw_enum_value` |
| A clarification message really lacks the missing identifier | `missing_value_leaked` |
| The message never names a tool | `leaks_tool_name` |
| Clarifications are questions | `bad_reply` |

The grounding checks ([`checks.py`](src/distill/data/checks.py)) came from reviewing the first real run: in 8 examples read by hand, 3 had a label value the operator never said (a ticket note, a search keyword, a result limit). Such labels teach a model to invent arguments, the failure that matters most for an agent. Each of those cases is now a regression test.

Rejected examples are kept in `rejected.jsonl` with the reason, to tune the prompts.

### Robust to messy input

Operators type fast, on phones, sometimes with gloves. On top of the teacher's styles, a noise layer ([`noise.py`](src/distill/data/noise.py)) makes 35% of the training examples messier, without changing their answer:

| Noise | Example |
|---|---|
| Keyboard typos (AZERTY in French, QWERTY in English) | vibration → vibartion |
| Missing accents, lowercase, no punctuation | priorité élevée → priorite elevee |
| SMS abbreviations | s'il te plaît → stp, beaucoup → bcp, please → pls |
| Loose identifiers | M-104 → M104, machine 104; L3 → ligne 3; TCK-00412 → ticket 412 |

Digits are never corrupted, so every value stays recoverable. The test set also gets a fully noisy twin, `test_noisy.jsonl`, to measure how much accuracy the model loses on messy input.

### Embeddings

Multilingual sentence embeddings ([`embeddings.py`](src/distill/embeddings.py), `paraphrase-multilingual-MiniLM-L12-v2`) compare messages by meaning, across French and English:

- **Semantic deduplication:** two messages that mean the same thing *and* expect the same answer are merged. Messages about M-104 and M-105 are both kept: telling them apart is exactly what the model must learn.
- **Semantic leakage check:** a test message that paraphrases a training message with the same answer moves to train.
- **Free-text scoring:** in evaluation, "Vibrations anormales" and "Abnormal vibration" count as the same issue.

### Splits without leakage

Exact and near duplicates are removed (word 3-gram Jaccard similarity). The test set is stratified by category and tool, and any test example too similar to a training example is moved to train, so the test score measures generalization rather than memorization.

**Why the test set is not filtered by the teacher:** a common shortcut is to keep only the examples the teacher solves correctly. That would make the teacher score 100% on the test set by design, and the comparison with the student would be meaningless. Test quality comes from the construction and the checks above, plus a manual review.

### Generating it

Generation runs on Groq's free tier and is resumable: if the daily quota runs out, it stops cleanly and the next run continues where it stopped.

**Automated:** the [`Generate data`](.github/workflows/generate-data.yml) workflow runs this every 6 hours on GitHub Actions, saves progress to the `dataset` branch, and builds the splits once every example is generated. It only needs a `GROQ_API_KEY` repository secret.

**By hand:**

```bash
cp .env.example .env            # then add your GROQ_API_KEY
python -m distill.data.generate --n 3000
python -m distill.data.build --embeddings   # dedup, splits, noise, stats
python -m distill.data.show data/processed/test.jsonl --n 5
```

## What is measured

| Metric | Meaning |
|---|---|
| Valid call rate | Output parses and passes the schema |
| Tool accuracy | Right tool chosen |
| Argument accuracy | Every argument correct |
| Abstention accuracy | No tool called when none fits |
| Robustness | Accuracy on the noisy test set, compared to the clean one |
| Latency (p50, p95) | Time to a complete tool call |
| Cost per 1,000 calls | Hardware or API cost |

Results table: *coming with stage 3.*

## Roadmap

- [x] Stage 1: tool schemas, validation, CI
- [x] Stage 2: data pipeline: sampling, generation, validation, deduplication, splits
- [ ] Stage 2b: generate the full dataset and publish it on the Hugging Face Hub
- [ ] Stage 3: evaluation harness and teacher baseline
- [ ] Stage 4: QLoRA fine-tuning
- [ ] Stage 5: DPO on failure cases
- [ ] Stage 6: quantization, serving and benchmarks
- [ ] Stage 7: public demo and evaluation gate in CI

## Quickstart

```bash
git clone https://github.com/benothmen12/distill.git
cd distill
python -m venv .venv && source .venv/bin/activate
make install                     # add the embeddings extra: pip install -e ".[dev,embeddings]"
make test
make tools   # print the tool definitions sent to the model
```

## Project structure

```
distill/
├── src/distill/
│   ├── tools/          # tool schemas, OpenAI export, call validation
│   ├── data/           # scenario sampling, teacher client, grounding checks, noise, splits
│   ├── embeddings.py   # multilingual embeddings: semantic dedup, leakage, similarity
│   └── prompts.py      # the copilot system prompt, shared by training and evaluation
├── tests/              # runs offline with a fake teacher
└── .github/workflows/  # CI on every push, scheduled data generation
```

## License

MIT
