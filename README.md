---
title: RCTD Environment
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
tags:
  - openenv
  - epistemic-reasoning
short_description: Epistemic reasoning under uncertainty
---

# 🔬 RCTD Environment — Research Coordination & Truth Discovery

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/hrshkr/rctd-env)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)

**This is the only OpenEnv environment that tests whether an LLM can distinguish reliable information from noise — the core unsolved challenge of autonomous agents.**

An environment for evaluating AI agents on **epistemic reasoning under uncertainty**: active information gathering, belief updating with noisy evidence, and hypothesis elimination under budget constraints.

---

## Why This Matters

Standard benchmarks give the agent all context upfront. Real-world autonomous agents must *decide what to investigate*, handle unreliable information, and reason under resource constraints. RCTD tests exactly this:

| Standard Benchmarks | RCTD Environment |
|---|---|
| Context given upfront | Agent must **actively gather** information |
| Information is reliable | Evidence may be **noisy or misleading** |
| Single-step reasoning | **Multi-step** investigation with budget |
| Binary success/failure | **4-component scoring** + failure mode analysis |

---

## Example Episode Walkthrough

> **Scenario**: Medical Diagnosis (seed=0, easy mode)
> **True answer**: H0 — Bacterial infection

```
Step 1:  read_evidence(E0)    → "Patient shows elevated WBC" — Supports H0, H2 (conf: 85%)
Step 2:  read_evidence(E1)    → "Blood culture positive"     — Supports H0 (conf: 92%)
Step 3:  read_evidence(E2)    → "No viral markers detected"  — Contradicts H1 (conf: 78%)
Step 4:  read_evidence(E3)    → "Toxin panel negative"       — Contradicts H2 (conf: 88%)
Step 5:  read_evidence(E4)    → "Rapid onset symptoms"       — Supports H0, H1 (conf: 55%) ⚠️ suspicious
Step 6:  run_experiment(E4)   → VERIFIED: Actually supports only H0 (noise removed!)
Step 7:  consult_expert(H0)   → "Very likely correct" (prob: 82%)
Step 8:  discard_hypothesis(H1) → ✓ Correct elimination
Step 9:  discard_hypothesis(H2) → ✓ Correct elimination
Step 10: submit_answer(H0)    → ✅ CORRECT! Score: 0.95
```

The agent earned a high score by: reading broadly, verifying suspicious evidence, consulting when uncertain, eliminating confidently, and submitting with budget remaining.

---

## The Investigation

Each episode presents:

1. **Multiple competing hypotheses** (3–5) — only one is correct
2. **A pool of evidence** (6–10 items) — some noisy, some reliable
3. **A limited budget** of action points

### Scenario Domains

| Domain | Example |
|---|---|
| 🏥 Medical Diagnosis | Bacteria vs. virus vs. toxin vs. autoimmune vs. genetic |
| 📈 Market Analysis | Supply shock vs. demand shift vs. regulation vs. tech breakthrough |
| 🔒 Cybersecurity | Insider threat vs. APT vs. zero-day vs. phishing vs. misconfig |
| 🌍 Climate Attribution | El Niño vs. greenhouse gas vs. volcanic vs. urban heat vs. deforestation |
| 🏺 Artifact Authentication | Authentic vs. forgery vs. misdated vs. replica vs. composite |

---

## Action Space

| Action | Cost | Effect |
|---|---|---|
| `read_evidence` | 1 | Read evidence. Reveals text and *apparent* support (may be noisy) |
| `run_experiment` | 3 | Deep-verify evidence. Strips noise, reveals TRUE support |
| `consult_expert` | 2 | Probabilistic hint about a hypothesis |
| `discard_hypothesis` | 0 | Remove a hypothesis from consideration |
| `submit_answer` | 0 | Final answer (terminal) |

```json
{"type": "read_evidence", "evidence_id": 0}
{"type": "run_experiment", "evidence_id": 2}
{"type": "consult_expert", "hypothesis_id": 1}
{"type": "discard_hypothesis", "hypothesis_id": 3}
{"type": "submit_answer", "hypothesis_id": 2}
```

---

## Tasks & Difficulty

| Task | Hypotheses | Evidence | Noise | Budget | Description |
|---|---|---|---|---|---|
| **easy** | 3 | 6 | 10% | 20 | Low noise, generous budget |
| **medium** | 4 | 8 | 30% | 15 | Moderate noise, requires verification |
| **hard** | 5 | 10 | 45% | 12 | High noise, tight budget, deep reasoning |

### Grading (0.0 – 1.0)

| Component | Weight | Measures |
|---|---|---|
| Accuracy | 60% | Correct hypothesis identified? |
| Efficiency | 20% | Budget remaining / total budget |
| Utilization | 10% | Optimal evidence gathering (40-70% sweet spot) |
| Process | 10% | Quality of hypothesis elimination |

---

## Baseline Scores (20 episodes × 3 tasks, seed=42)

These are **actual measured scores**, not estimates:

| Agent | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| Random | 0.313 (15%) | 0.271 (10%) | 0.406 (35%) | **0.330** |
| Heuristic | 0.505 (75%) | 0.445 (65%) | 0.545 (65%) | **0.498** |
| Qwen2.5-72B (2 episodes) | 0.951 (1/1) | — | 0.970 (1/1) | **0.961** |

*LLM results from 2 complete episodes before HF API credits depleted. The LLM successfully used all 5 action types including `run_experiment` and `consult_expert`.*

### Failure Mode Distribution (Heuristic, n=60)

| Mode | Count | Description |
|---|---|---|
| `misled_by_noise` | 9 | Read noisy evidence without verifying |
| `budget_exhausted` | 8 | Ran out of action points |
| `discarded_correct` | 5 | Eliminated the true answer |
| `reasoning_error` | 3 | Had enough evidence, wrong conclusion |

---

## Setup & Installation

```bash
git clone https://github.com/hrsh-kr/rctd_env.git
cd rctd_env
pip install -e .
```

### Docker

```bash
docker build -t rctd-env .
docker run -p 7860:7860 rctd-env
```

---

## Quick Start

### Local Usage

```python
from rctd_env.server.environment import RCTDEnvironment
from rctd_env.models import RCTDAction

env = RCTDEnvironment()
obs = env.reset(task_id="medium", seed=42)

obs = env.step(RCTDAction(type="read_evidence", evidence_id=0))
obs = env.step(RCTDAction(type="run_experiment", evidence_id=0))  # Verify!
obs = env.step(RCTDAction(type="consult_expert", hypothesis_id=1))
obs = env.step(RCTDAction(type="discard_hypothesis", hypothesis_id=2))
obs = env.step(RCTDAction(type="submit_answer", hypothesis_id=1))
print(f"Score: {obs.reward}, Success: {obs.metrics['success']}")
```

### Inference Script

```bash
# Heuristic baseline (no API key needed)
python inference.py

# With LLM
export HF_TOKEN=hf_...
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Live API

```bash
curl https://hrshkr-rctd-env.hf.space/health
curl https://hrshkr-rctd-env.hf.space/tasks
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks and action schemas |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Current episode state |
| `/metadata` | GET | Environment metadata |
| `/docs` | GET | Interactive Swagger docs |

---

## Training with GRPO (TRL Integration)

```bash
python -m rctd_env.training_example --quick    # CPU, ~5 min
python -m rctd_env.training_example            # A100, ~90 min
```

**5 reward signals** for rich gradient:

| Reward | Measures | Range |
|---|---|---|
| `reward_correct` | Right hypothesis? | 0/1 |
| `reward_efficiency` | Budget remaining | 0–1 |
| `reward_evidence_quality` | Smart use of verification | 0–1 |
| `reward_process` | Correct eliminations | 0–1 |
| `reward_format` | Valid JSON output | 0/1 |

*Validated end-to-end on M4 Mac (TRL v1.0.0, 6 training steps in 13s).*

---

## Architecture

```
├── inference.py           ← Baseline inference (OpenAI client, [START]/[STEP]/[END])
├── Dockerfile             ← Container (HF Spaces, port 7860)
├── server/app.py          ← Server entry point
├── openenv.yaml           ← OpenEnv manifest
├── rctd_env/
│   ├── models.py          ← Pydantic types (Action, Observation, State)
│   ├── client.py          ← WebSocket client (EnvClient)
│   ├── server/
│   │   ├── environment.py ← Core epistemic engine (1131 lines)
│   │   ├── app.py         ← FastAPI (create_fastapi_app + custom endpoints)
│   │   ├── graders.py     ← 4-component grading + failure modes
│   │   └── tasks.py       ← Task definitions (easy/medium/hard)
│   └── training_example.py ← TRL/GRPO pipeline
├── trajectories.jsonl     ← Baseline + LLM trajectories
└── uv.lock
```

---

## License

MIT
