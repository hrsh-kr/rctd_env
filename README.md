# 🔬 RCTD Environment — Research Coordination & Truth Discovery

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Spaces](https://img.shields.io/badge/🤗-HF%20Spaces-yellow)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

An OpenEnv-compatible environment that evaluates AI agents on **epistemic reasoning under uncertainty** — active information gathering, belief updating, and resilience to noisy or misleading evidence.

---

## Why This Matters for LLM Agent Evaluation

Standard benchmarks evaluate static question-answering: given context, produce an answer. But real-world autonomous agents face a fundamentally different challenge:

| Standard Benchmarks | RCTD Environment |
|---|---|
| Context is given upfront | Agent must **actively gather** information |
| Information is reliable | Evidence may be **noisy or misleading** |
| Single-step reasoning | **Multi-step** investigation with resource constraints |
| No cost to thinking | **Budget management** forces efficiency |
| Binary success/failure | **Rich metrics**: efficiency, calibration, failure mode analysis |

RCTD tests what actually matters for next-generation agentic systems:

- **Active Learning** — The agent decides what to investigate, not just what to conclude
- **Bayesian Updating** — New evidence may contradict prior beliefs; can the agent revise?
- **Noise Resilience** — Some evidence is unreliable; can the agent detect and verify?
- **Resource Management** — Limited budget forces strategic trade-offs
- **Hypothesis Elimination** — Narrowing the search space is rewarded, but mistakes are costly

---

## The Investigation

Each episode presents the agent with a real-world investigation scenario:

1. **Multiple competing hypotheses** (3–5) — only one is correct
2. **A pool of evidence** (6–10 items) — some noisy, some reliable
3. **A limited budget** of action points

The agent must spend its budget wisely to gather evidence, verify suspicious findings, consult experts, eliminate wrong hypotheses, and ultimately submit the correct answer.

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
| `read_evidence` | 1 | Read an evidence item. Reveals text and *apparent* support (may be noisy) |
| `run_experiment` | 3 | Deep-verify evidence. Strips noise, reveals TRUE support |
| `consult_expert` | 2 | Probabilistic hint about a hypothesis |
| `discard_hypothesis` | 0 | Remove a hypothesis from consideration |
| `submit_answer` | 0 | Final answer (terminal) |

### Action Format

```json
{"type": "read_evidence", "evidence_id": 0}
{"type": "run_experiment", "evidence_id": 2}
{"type": "consult_expert", "hypothesis_id": 1}
{"type": "discard_hypothesis", "hypothesis_id": 3}
{"type": "submit_answer", "hypothesis_id": 2}
```

---

## Observation Space

After each action, the agent receives:

| Field | Type | Description |
|---|---|---|
| `done` | bool | Whether the episode has ended |
| `reward` | float (0–1) | Normalized reward signal |
| `hypotheses` | list[str] | All hypothesis descriptions |
| `active_hypothesis_ids` | list[int] | IDs of non-discarded hypotheses |
| `revealed_evidence` | list[EvidenceItem] | Evidence gathered so far |
| `expert_hints` | list[ExpertHint] | Expert opinions gathered |
| `budget_remaining` | int | Remaining action points |
| `action_history` | list[dict] | Chronological action log |
| `message` | str | Human-readable feedback |
| `metrics` | dict (terminal only) | Rich evaluation metrics |

---

## Tasks & Difficulty

| Task | Hypotheses | Evidence | Noise | Budget | Description |
|---|---|---|---|---|---|
| **easy** | 3 | 6 | 10% | 20 | Low noise, generous budget — baseline for random agents |
| **medium** | 4 | 8 | 30% | 15 | Moderate noise — requires evidence evaluation |
| **hard** | 5 | 10 | 45% | 12 | High noise, tight budget — deep reasoning required |

### Grading (0.0 – 1.0)

| Component | Weight | Measures |
|---|---|---|
| Accuracy | 60% | Correct hypothesis identified? |
| Efficiency | 20% | Budget remaining / total budget |
| Utilization | 10% | Optimal evidence gathering (not too little, not too much) |
| Process | 10% | Quality of hypothesis elimination strategy |

---

## Setup & Installation

### From Source

```bash
git clone https://github.com/hrshkr/rctd-env.git
cd rctd-env
pip install -e .
```

### From HF Spaces

```bash
pip install git+https://huggingface.co/spaces/hrshkr/rctd-env
```

### Docker

```bash
docker build -t rctd-env -f server/Dockerfile .
docker run -p 8000:8000 rctd-env
```

---

## Quick Start

### Local Usage (No Server)

```python
from rctd_env import RCTDLocalEnv, RCTDAction

env = RCTDLocalEnv()
obs = env.reset(task_id="medium", seed=42)

# Read some evidence
obs = env.step(RCTDAction(type="read_evidence", evidence_id=0))
obs = env.step(RCTDAction(type="read_evidence", evidence_id=1))

# Verify suspicious evidence
obs = env.step(RCTDAction(type="run_experiment", evidence_id=0))

# Discard a wrong hypothesis
obs = env.step(RCTDAction(type="discard_hypothesis", hypothesis_id=2))

# Submit answer
obs = env.step(RCTDAction(type="submit_answer", hypothesis_id=1))
print(obs.metrics)
```

### Remote Usage (WebSocket Client)

```python
from rctd_env import RCTDEnv, RCTDAction

# Connect to a running server (local or HF Spaces)
with RCTDEnv(base_url="https://hrshkr-rctd-env.hf.space").sync() as env:
    result = env.reset(seed=42, task_id="medium")
    while not result.done:
        action = decide(result.observation)  # Your policy here
        result = env.step(action)
```

### Via REST API

```bash
# Start server
uvicorn rctd_env.server.app:app --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/tasks
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "easy", "seed": 42}'
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"type": "read_evidence", "evidence_id": 0}'
```

---

## Baseline Inference

### Programmatic Baselines Only

```bash
python -m rctd_env.inference --skip-llm
```

### With NVIDIA NIM

```bash
export NVIDIA_API_KEY=nvapi-...
python -m rctd_env.inference --provider nim
```

### With OpenAI

```bash
export OPENAI_API_KEY=sk-...
python -m rctd_env.inference --provider openai --model gpt-4o-mini
```

### Baseline Scores (10 episodes × 3 tasks)

| Agent | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| Random | 0.53 (50%) | 0.40 (30%) | 0.20 (0%) | ~0.38 |
| Heuristic | 0.66 (80%) | 0.36 (40%) | 0.37 (50%) | ~0.46 |
| LLM (GPT-4o-mini) | ~0.70 | ~0.55 | ~0.40 | ~0.55 |

---

## Evaluation Metrics

On episode termination, the `metrics` field contains:

```json
{
  "success": true,
  "efficiency_score": 0.467,
  "evidence_utilization": 0.625,
  "steps_taken": 8,
  "evidence_read": 5,
  "experiments_run": 1,
  "experts_consulted": 1,
  "correct_discards": 2,
  "incorrect_discards": 0,
  "budget_used": 10,
  "failure_mode": null,
  "raw_reward": 97.0,
  "scenario_theme": "medical_diagnosis",
  "true_hypothesis": 1,
  "submitted_hypothesis": 1
}
```

### Failure Mode Categories

| Mode | Description |
|---|---|
| `discarded_correct_hypothesis` | Agent eliminated the true answer |
| `budget_exhausted` | Ran out of action points |
| `insufficient_evidence` | Submitted with <30% evidence gathered |
| `misled_by_noise` | Read noisy evidence without verifying |
| `reasoning_error` | Had enough evidence but drew wrong conclusion |

---

## Trajectory Logging (HF Dataset Ready)

The inference script saves trajectories to `.jsonl` format:

```json
{
  "task_id": "medium",
  "seed": 42,
  "policy": "llm_openai",
  "success": true,
  "score": 0.72,
  "steps": 7,
  "scenario_theme": "medical_diagnosis",
  "metrics": {...},
  "trajectory": [
    {"step": 0, "action": {"type": "read_evidence", "evidence_id": 0}, "budget_before": 15},
    ...
  ]
}
```

Upload to the Hugging Face Hub:

```bash
huggingface-cli upload hrshkr/rctd-trajectories trajectories.jsonl --repo-type dataset
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/ws` | WebSocket | Persistent environment session (primary) |
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks and action schema |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Get episode metadata |
| `/schema` | GET | Action/Observation JSON schemas |
| `/metadata` | GET | Environment metadata |
| `/baseline` | GET | Run baseline agents and return scores |
| `/grader` | GET | Grade a single episode |
| `/docs` | GET | Interactive API documentation |

---

## Training with GRPO (TRL Integration)

Train an LLM to play RCTD using Group Relative Policy Optimization:

```bash
# Quick test (~5 min, CPU)
python -m rctd_env.training_example --quick

# Full training (~90 min, A100 40GB)
python -m rctd_env.training_example
```

**5 reward signals** for rich gradient information:

| Reward | What it measures | Range |
|---|---|---|
| `reward_correct` | Found the right hypothesis? | 0.0 or 1.0 |
| `reward_efficiency` | Budget remaining when done | 0.0–1.0 |
| `reward_evidence_quality` | Smart use of run_experiment | 0.0–1.0 |
| `reward_process` | Correct hypothesis elimination | 0.0–1.0 |
| `reward_format` | Valid JSON output | 0.0 or 1.0 |

See [`training_example.py`](training_example.py) for the full pipeline including rollout function, dataset generation, and post-training evaluation.

---

## Architecture

```
rctd_env/
├── models.py              ← Pydantic types (Action, Observation, State)
├── client.py              ← WebSocket client (EnvClient) + local wrapper
├── __init__.py            ← Package exports
├── server/
│   ├── environment.py     ← Core logic + Epistemic Engine (1131 lines)
│   ├── app.py             ← FastAPI server (create_fastapi_app + custom)
│   ├── graders.py         ← 3-task programmatic grading (0.0–1.0)
│   ├── tasks.py           ← Task definitions (easy/medium/hard)
│   └── Dockerfile         ← Container definition (HF Spaces ready)
├── inference.py           ← Baseline inference (OpenAI/NIM/custom)
├── training_example.py    ← TRL/GRPO training pipeline
├── trajectories.jsonl     ← Baseline trajectories (HF Dataset ready)
├── openenv.yaml           ← OpenEnv manifest
├── pyproject.toml         ← Package metadata
└── README.md              ← This file
```

---

## License

MIT
