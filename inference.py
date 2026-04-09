#!/usr/bin/env python3
"""
RCTD Environment — Inference Script
===================================
MANDATORY
- Environment variables:
    HF_TOKEN         Your Hugging Face API token (primary). Also accepts OPENAI_API_KEY as fallback.
    API_BASE_URL     The API endpoint for the LLM (default: https://router.huggingface.co/v1).
    MODEL_NAME       The model identifier to use for inference.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment.

STDOUT FORMAT
    [START] task=<task_name> env=rctd_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

USAGE
    # Heuristic + random baselines only (no API key needed):
    python inference.py --skip-llm

    # LLM inference (requires HF_TOKEN):
    HF_TOKEN=hf_... python inference.py
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from rctd_env import RCTDAction, RCTDObservation
from rctd_env.server.environment import RCTDEnvironment
from rctd_env.server.graders import grade_all_tasks, heuristic_policy, random_policy

# ═══════════════════════════════════════════════════════════════════════════
# Environment Variables (read lazily — only validated when LLM mode is used)
# ═══════════════════════════════════════════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "rctd_env"
MAX_STEPS = 50
TEMPERATURE = 0.1
MAX_TOKENS = 100


def _get_api_key() -> str:
    """Get the API key from environment variables.

    HF_TOKEN is the primary credential per spec.
    OPENAI_API_KEY is accepted as a fallback.
    """
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        return hf_token
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("[DEBUG] Using OPENAI_API_KEY as fallback", file=sys.stderr, flush=True)
        return openai_key
    raise ValueError(
        "HF_TOKEN environment variable is required for LLM inference. "
        "Set HF_TOKEN=hf_... or use --skip-llm for baseline-only mode."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Structured Logging — [START] / [STEP] / [END]
# ═══════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LLM Agent
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert investigator AI. You are presented with competing hypotheses
about a real-world scenario and must determine which one is TRUE by gathering
and analyzing evidence under a limited budget.

Actions (respond with ONLY a JSON object):
- {"type": "read_evidence", "evidence_id": N}     — cost 1, read evidence
- {"type": "run_experiment", "evidence_id": N}     — cost 3, verify evidence
- {"type": "consult_expert", "hypothesis_id": N}   — cost 2, expert opinion
- {"type": "discard_hypothesis", "hypothesis_id": N} — cost 0, eliminate
- {"type": "submit_answer", "hypothesis_id": N}    — cost 0, final answer

Strategy: Read evidence → verify suspicious items → discard weak hypotheses → submit.
Unused budget improves your score. Respond with ONLY a JSON object.""")


def format_observation(obs: RCTDObservation) -> str:
    """Format observation as a text prompt for the LLM."""
    parts = [f"Step {obs.step_count} | Budget: {obs.budget_remaining}"]

    parts.append("HYPOTHESES:")
    for i, h in enumerate(obs.hypotheses):
        tag = "ACTIVE" if i in obs.active_hypothesis_ids else "DISCARDED"
        parts.append(f"  H{i} [{tag}]: {h}")

    if obs.revealed_evidence:
        parts.append("EVIDENCE:")
        for ev in obs.revealed_evidence:
            v = " [VERIFIED]" if ev.verified else ""
            sup = ", ".join(f"H{s}" for s in ev.apparent_support) or "none"
            con = ", ".join(f"H{c}" for c in ev.apparent_contradiction)
            con_str = f" | Contradicts: {con}" if con else ""
            parts.append(f"  E{ev.evidence_id}{v}: {ev.text}")
            parts.append(f"    Supports: {sup}{con_str} (conf: {ev.confidence:.0%})")

    if obs.expert_hints:
        parts.append("EXPERTS:")
        for h in obs.expert_hints:
            parts.append(f"  H{h.hypothesis_id}: {h.hint_text} (prob: {h.estimated_probability:.0%})")

    if obs.message:
        parts.append(f"MSG: {obs.message}")

    return "\n".join(parts)


def parse_llm_action(text: str) -> Optional[RCTDAction]:
    """Parse LLM response into an RCTDAction."""
    text = text.strip()
    if "```" in text:
        import re
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        data = json.loads(text[start:end])
        return RCTDAction(
            type=data.get("type", ""),
            evidence_id=data.get("evidence_id"),
            hypothesis_id=data.get("hypothesis_id"),
        )
    except (json.JSONDecodeError, Exception):
        return None


def find_best_hypothesis(obs: RCTDObservation) -> int:
    """Find the most-supported active hypothesis from evidence."""
    if not obs.active_hypothesis_ids:
        return 0
    support: Dict[int, float] = {h: 0.0 for h in obs.active_hypothesis_ids}
    for ev in obs.revealed_evidence:
        w = 2.0 if ev.verified else 1.0
        for h in ev.apparent_support:
            if h in support:
                support[h] += w * ev.confidence
    for hint in obs.expert_hints:
        if hint.hypothesis_id in support:
            support[hint.hypothesis_id] += hint.estimated_probability
    return max(support, key=support.get)


def get_llm_action(
    client: OpenAI,
    obs: RCTDObservation,
    history: List[Dict[str, str]],
) -> RCTDAction:
    """Get action from LLM with retry logic."""
    user_msg = format_observation(obs)
    history.append({"role": "user", "content": user_msg})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-10:],
    ]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            reply = response.choices[0].message.content or ""
            history.append({"role": "assistant", "content": reply})

            action = parse_llm_action(reply)
            if action is not None:
                return action

            if attempt < 2:
                retry_msg = 'Respond with ONLY JSON: {"type": "read_evidence", "evidence_id": 0}'
                history.append({"role": "user", "content": retry_msg})
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history[-10:]]
        except Exception as e:
            print(f"[DEBUG] LLM error attempt {attempt+1}: {e}", file=sys.stderr, flush=True)
            import time
            time.sleep(0.5 * (attempt + 1))

    # Fallback: submit best hypothesis
    return RCTDAction(type="submit_answer", hypothesis_id=find_best_hypothesis(obs))


# ═══════════════════════════════════════════════════════════════════════════
# Run Episode
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(
    env: RCTDEnvironment,
    client: Optional[OpenAI],
    task_id: str,
    seed: int,
    policy_name: str,
) -> Dict[str, Any]:
    """Run one episode with [START]/[STEP]/[END] logging."""
    log_start(task=task_id, env=BENCHMARK, model=policy_name)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.001

    try:
        obs = env.reset(seed=seed, task_id=task_id)
        history: List[Dict[str, str]] = []

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # Get action
            if client is not None and policy_name not in ("random", "heuristic"):
                action = get_llm_action(client, obs, history)
            elif policy_name == "heuristic":
                action = heuristic_policy(obs)
            else:
                action = random_policy(obs)

            # Execute step
            obs = env.step(action)
            reward = obs.reward
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step

            # Format action string for logging
            action_str = action.type
            if action.evidence_id is not None:
                action_str += f"(e={action.evidence_id})"
            elif action.hypothesis_id is not None:
                action_str += f"(h={action.hypothesis_id})"

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Force submit if not done
        if not obs.done:
            best = find_best_hypothesis(obs)
            action = RCTDAction(type="submit_answer", hypothesis_id=best)
            obs = env.step(action)
            steps_taken += 1
            rewards.append(obs.reward)
            log_step(
                step=steps_taken,
                action=f"submit_answer(h={best})",
                reward=obs.reward,
                done=True,
                error=None,
            )

        score = obs.reward  # Already in (0, 1)
        success = obs.metrics.get("success", False) if obs.metrics else False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "seed": seed, "score": score, "success": success}


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Runner (--skip-llm mode)
# ═══════════════════════════════════════════════════════════════════════════

def run_baselines(num_episodes: int = 10, base_seed: int = 42) -> Dict[str, Any]:
    """Run heuristic and random baselines across all tasks.

    Returns results dict and writes to baseline.json.
    """
    env = RCTDEnvironment()
    results = {}

    for agent_name, policy in [("random", random_policy), ("heuristic", heuristic_policy)]:
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"Running {agent_name} baseline ({num_episodes} episodes × 3 tasks)...",
              file=sys.stderr, flush=True)

        agent_results = grade_all_tasks(
            env=env,
            policy=policy,
            num_episodes=num_episodes,
            base_seed=base_seed,
        )
        results[agent_name] = {
            "overall_score": agent_results["overall_score"],
            "tasks": {
                task_id: {
                    "average_score": task_data["average_score"],
                    "success_rate": task_data["success_rate"],
                    "min_score": task_data["min_score"],
                    "max_score": task_data["max_score"],
                    "failure_modes": task_data["failure_modes"],
                }
                for task_id, task_data in agent_results["tasks"].items()
            },
        }
        print(f"  {agent_name} overall: {agent_results['overall_score']:.4f}",
              file=sys.stderr, flush=True)
        for tid, td in agent_results["tasks"].items():
            print(f"    {tid}: score={td['average_score']:.4f}, "
                  f"success={td['success_rate']:.0%}",
                  file=sys.stderr, flush=True)

    baseline_output = {
        "baseline_scores": results,
        "num_episodes_per_task": num_episodes,
        "seed": base_seed,
    }

    # Write baseline.json
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")
    with open(output_path, "w") as f:
        json.dump(baseline_output, f, indent=2)
    print(f"\n✅ Baseline results written to {output_path}", file=sys.stderr, flush=True)

    return baseline_output


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RCTD Environment — Inference & Baseline Script",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run only heuristic and random baselines (no API key needed)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes per task for baseline mode (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.skip_llm:
        # Baseline-only mode — no API key needed
        print("Running baseline agents (--skip-llm mode)...", file=sys.stderr, flush=True)
        run_baselines(num_episodes=args.num_episodes, base_seed=args.seed)
        return

    # LLM mode — requires HF_TOKEN
    api_key = _get_api_key()
    env = RCTDEnvironment()
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    policy_name = MODEL_NAME
    print(f"[DEBUG] Using LLM: {MODEL_NAME} via {API_BASE_URL}", file=sys.stderr, flush=True)

    # Run all 3 tasks
    all_results = []
    for task_id in ["easy", "medium", "hard"]:
        for seed in range(3):  # 3 episodes per task
            result = run_episode(env, client, task_id, seed=seed, policy_name=policy_name)
            all_results.append(result)

    # Write baseline.json with LLM results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "episodes": all_results,
        }, f, indent=2)
    print(f"\n✅ Results written to {output_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
