#!/usr/bin/env python3
"""
RCTD Environment — TRL/GRPO Training Example.

Trains a small LLM (Qwen3-1.7B) to play the RCTD environment using
Group Relative Policy Optimization (GRPO). This demonstrates how the
RCTD environment integrates with the TRL training pipeline.

Requirements:
    pip install trl transformers torch accelerate
    # For vLLM acceleration (recommended on A100):
    pip install vllm

Usage:
    # Quick test (CPU/small GPU, ~5 min):
    python -m rctd_env.training_example --quick

    # Full training (A100 40GB, ~90 min):
    python -m rctd_env.training_example

    # Custom model:
    python -m rctd_env.training_example --model Qwen/Qwen3-1.7B

Hardware:
    - Quick mode: Any machine with 8GB+ RAM (CPU or GPU)
    - Full mode: A100 40GB (Colab Pro, Lambda, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

from .models import RCTDAction, RCTDObservation
from .server.environment import RCTDEnvironment


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert investigator. You must determine which hypothesis is TRUE \
by gathering evidence under a limited budget.

ACTIONS (respond with JSON only):
- {"type": "read_evidence", "evidence_id": N}     — cost 1, read evidence item
- {"type": "run_experiment", "evidence_id": N}     — cost 3, verify evidence (removes noise)
- {"type": "consult_expert", "hypothesis_id": N}   — cost 2, get expert opinion
- {"type": "discard_hypothesis", "hypothesis_id": N} — cost 0, eliminate hypothesis
- {"type": "submit_answer", "hypothesis_id": N}    — cost 0, final answer

STRATEGY: Read evidence → identify contradictions → verify suspicious items → \
discard weak hypotheses → submit when confident. Unused budget improves your score.

Respond with ONLY a JSON object. No explanation."""


# ═══════════════════════════════════════════════════════════════════════════
# Observation Formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_obs_for_training(obs: RCTDObservation) -> str:
    """Format observation into a compact string for the model."""
    parts = [f"Step {obs.step_count} | Budget: {obs.budget_remaining}"]

    # Hypotheses
    parts.append("HYPOTHESES:")
    for i, h in enumerate(obs.hypotheses):
        tag = "ACTIVE" if i in obs.active_hypothesis_ids else "DISCARDED"
        parts.append(f"  H{i} [{tag}]: {h}")

    # Evidence
    if obs.revealed_evidence:
        parts.append("EVIDENCE:")
        for ev in obs.revealed_evidence:
            v = " [VERIFIED]" if ev.verified else ""
            sup = ", ".join(f"H{s}" for s in ev.apparent_support) or "none"
            con = ", ".join(f"H{c}" for c in ev.apparent_contradiction)
            con_str = f" | Contradicts: {con}" if con else ""
            parts.append(f"  E{ev.evidence_id}{v}: {ev.text}")
            parts.append(f"    → Supports: {sup}{con_str} (conf: {ev.confidence:.0%})")

    # Expert hints
    if obs.expert_hints:
        parts.append("EXPERT OPINIONS:")
        for h in obs.expert_hints:
            parts.append(f"  H{h.hypothesis_id}: {h.hint_text} (prob: {h.estimated_probability:.0%})")

    if obs.message:
        parts.append(f"RESULT: {obs.message}")

    return "\n".join(parts)


def parse_action(text: str) -> Optional[RCTDAction]:
    """Parse model output into an RCTDAction."""
    text = text.strip()
    if "```" in text:
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


# ═══════════════════════════════════════════════════════════════════════════
# Reward Functions (for GRPO multi-reward training)
# ═══════════════════════════════════════════════════════════════════════════

def reward_correct(obs: RCTDObservation, env_state) -> float:
    """Binary reward: did the agent find the correct hypothesis?"""
    if not obs.done or not obs.metrics:
        return 0.0
    return 1.0 if obs.metrics.get("success", False) else 0.0


def reward_efficiency(obs: RCTDObservation, env_state) -> float:
    """Reward for budget efficiency — higher is better."""
    if not obs.done or not obs.metrics:
        return 0.0
    return obs.metrics.get("efficiency_score", 0.0)


def reward_evidence_quality(obs: RCTDObservation, env_state) -> float:
    """Reward for using run_experiment on noisy evidence (smart verification)."""
    if not obs.done or not obs.metrics:
        return 0.0
    experiments = obs.metrics.get("experiments_run", 0)
    evidence_read = obs.metrics.get("evidence_read", 0)
    if evidence_read == 0:
        return 0.0
    # Reward a moderate verification ratio (not too few, not too many)
    ratio = experiments / max(1, evidence_read)
    if 0.2 <= ratio <= 0.5:
        return 1.0
    elif ratio < 0.2:
        return ratio / 0.2
    else:
        return max(0, 1.0 - (ratio - 0.5) / 0.5)


def reward_process(obs: RCTDObservation, env_state) -> float:
    """Reward for correct hypothesis elimination."""
    if not obs.done or not obs.metrics:
        return 0.0
    correct_discards = obs.metrics.get("correct_discards", 0)
    incorrect_discards = obs.metrics.get("incorrect_discards", 0)
    if correct_discards == 0 and incorrect_discards == 0:
        return 0.5  # Neutral — didn't try to eliminate
    # Reward correct eliminations, heavily penalize incorrect
    return max(0.0, min(1.0,
        correct_discards * 0.3 - incorrect_discards * 0.8
    ))


def reward_format(response_text: str) -> float:
    """Reward for producing valid JSON output."""
    action = parse_action(response_text)
    return 1.0 if action is not None else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Rollout Function (RCTD-specific)
# ═══════════════════════════════════════════════════════════════════════════

def rctd_rollout(
    trainer,
    env: RCTDEnvironment,
    tokenizer,
    prompts: List[str],
    max_turns: int = 15,
    task_id: str = "easy",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Play one RCTD episode and return rollout data for GRPO.

    This is the bridge between TRL's GRPOTrainer and the RCTD environment.
    For each prompt, it plays a full episode and collects:
      - prompt_ids / completion_ids / logprobs (for GRPO policy update)
      - Multiple reward signals (for rich gradient information)

    Args:
        trainer: The GRPOTrainer instance (for generate()).
        env: RCTD environment instance.
        tokenizer: Model tokenizer.
        prompts: List of initial prompts (one per group member).
        max_turns: Maximum turns per episode.
        task_id: Difficulty level.
        seed: Random seed for reproducibility.

    Returns:
        List of rollout dicts, one per prompt.
    """
    from trl.trainer.utils import generate_rollout_completions

    results = []

    for i, prompt in enumerate(prompts):
        obs = env.reset(seed=seed + i, task_id=task_id)
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        full_response = ""

        for turn in range(max_turns):
            if obs.done:
                break

            # Build the message for this turn
            obs_text = format_obs_for_training(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ]

            # Format as model input
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            # Generate with the model
            rollout = generate_rollout_completions(
                trainer,
                [formatted],
                max_new_tokens=80,
                temperature=0.7,
            )

            response_text = rollout["text"][0]
            full_response += response_text + "\n"

            # Collect token IDs and logprobs
            all_prompt_ids.extend(rollout["prompt_ids"][0])
            all_completion_ids.extend(rollout["completion_ids"][0])
            if "logprobs" in rollout:
                all_logprobs.extend(rollout["logprobs"][0])

            # Parse and execute action
            action = parse_action(response_text)
            if action is None:
                # Format failure — give the agent one more chance
                action = RCTDAction(type="submit_answer", hypothesis_id=0)

            obs = env.step(action)

        # If still not done, force submission
        if not obs.done:
            best_h = 0
            if obs.active_hypothesis_ids:
                support = {h: 0.0 for h in obs.active_hypothesis_ids}
                for ev in obs.revealed_evidence:
                    w = 2.0 if ev.verified else 1.0
                    for h in ev.apparent_support:
                        if h in support:
                            support[h] += w * ev.confidence
                best_h = max(support, key=support.get)
            obs = env.step(RCTDAction(type="submit_answer", hypothesis_id=best_h))

        # Compute rewards
        env_state = env.state
        results.append({
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "reward_correct": reward_correct(obs, env_state),
            "reward_efficiency": reward_efficiency(obs, env_state),
            "reward_evidence_quality": reward_evidence_quality(obs, env_state),
            "reward_process": reward_process(obs, env_state),
            "reward_format": reward_format(full_response),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Standalone Training Script
# ═══════════════════════════════════════════════════════════════════════════

def create_training_dataset(
    num_episodes: int = 100,
    task_ids: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Create a dataset of RCTD prompts for GRPO training.

    Each entry is an initial observation from a fresh episode, formatted
    as a prompt the model can respond to.
    """
    if task_ids is None:
        task_ids = ["easy", "medium", "hard"]

    env = RCTDEnvironment()
    dataset = []

    for i in range(num_episodes):
        task_id = task_ids[i % len(task_ids)]
        obs = env.reset(seed=i, task_id=task_id)

        prompt = format_obs_for_training(obs)
        dataset.append({
            "prompt": prompt,
            "task_id": task_id,
            "seed": str(i),
        })

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train an LLM to play RCTD using GRPO",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="Base model to fine-tune (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (fewer episodes, smaller batch)",
    )
    parser.add_argument(
        "--output-dir",
        default="rctd_grpo_checkpoint",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of training episodes (default: 50 quick, 500 full)",
    )

    args = parser.parse_args()

    # ── Check dependencies ────────────────────────────────────────────
    try:
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError:
        print("❌ Required packages not installed. Run:")
        print("   pip install trl transformers datasets accelerate")
        print("   pip install vllm  # Optional, for GPU acceleration")
        sys.exit(1)

    # ── Configuration ─────────────────────────────────────────────────
    num_episodes = args.num_episodes or (50 if args.quick else 500)

    if args.quick:
        grpo_config = GRPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=1,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=2,
            max_completion_length=80,
            gradient_checkpointing=True,
            logging_steps=1,
            report_to="none",
        )
    else:
        grpo_config = GRPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=1,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,
            num_generations=4,
            max_completion_length=80,
            use_vllm=True,
            vllm_gpu_memory_utilization=0.3,
            gradient_checkpointing=True,
            logging_steps=5,
            report_to="none",
        )

    # ── Create dataset ────────────────────────────────────────────────
    print(f"\n📊 Creating training dataset ({num_episodes} episodes)...")
    raw_dataset = create_training_dataset(num_episodes=num_episodes)
    dataset = Dataset.from_list(raw_dataset)
    print(f"   Dataset: {len(dataset)} episodes across easy/medium/hard")

    # ── Define reward functions ───────────────────────────────────────
    # GRPO uses multiple reward signals for richer gradients
    env = RCTDEnvironment()

    def combined_reward_func(completions: List[str], **kwargs) -> List[float]:
        """Evaluate completions by playing them in the environment.

        This is called by GRPOTrainer during training. Each completion
        is a model response that should be a valid JSON action.
        """
        rewards = []
        for i, completion in enumerate(completions):
            # Parse the action
            action = parse_action(completion)
            format_score = 1.0 if action is not None else 0.0

            # Play a quick episode to evaluate
            seed = hash(completion) % (2**31)
            obs = env.reset(seed=seed, task_id="easy")

            if action is not None:
                try:
                    obs = env.step(action)
                except Exception:
                    pass

            # Weighted reward combining format + correctness signals
            reward = format_score * 0.3  # 30% for valid JSON
            if obs.done and obs.metrics:
                reward += 0.5 if obs.metrics.get("success") else 0.0  # 50% for correct
                reward += obs.metrics.get("efficiency_score", 0.0) * 0.2  # 20% for efficiency

            rewards.append(reward)

        return rewards

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n🚀 Starting GRPO training...")
    print(f"   Model: {args.model}")
    print(f"   Mode: {'quick (CPU/small GPU)' if args.quick else 'full (A100)'}")
    print(f"   Config: {grpo_config.num_generations} generations, "
          f"lr={grpo_config.learning_rate}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[combined_reward_func],
        train_dataset=dataset,
        args=grpo_config,
    )

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    print(f"\n✅ Model saved to {args.output_dir}/")
    print(f"   To evaluate: python -m rctd_env.inference --model {args.output_dir}")

    # ── Quick evaluation ──────────────────────────────────────────────
    print("\n📈 Post-training evaluation (5 episodes × 3 tasks)...")
    from .server.graders import grade_all_tasks

    # Create a policy from the trained model
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        def trained_policy(obs: RCTDObservation) -> RCTDAction:
            obs_text = format_obs_for_training(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, temperature=0.1,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            action = parse_action(response)
            if action is not None:
                return action
            # Fallback
            if obs.active_hypothesis_ids:
                return RCTDAction(
                    type="submit_answer",
                    hypothesis_id=obs.active_hypothesis_ids[0],
                )
            return RCTDAction(type="submit_answer", hypothesis_id=0)

        eval_result = grade_all_tasks(
            env=env,
            policy=trained_policy,
            num_episodes=5,
            base_seed=100,
        )
        print(f"   Trained model overall score: {eval_result['overall_score']:.3f}")
        for tid, td in eval_result["tasks"].items():
            print(f"     {tid}: score={td['average_score']:.3f}, "
                  f"success={td['success_rate']:.0%}")

    except Exception as e:
        print(f"   Post-training eval skipped: {e}")


if __name__ == "__main__":
    main()
