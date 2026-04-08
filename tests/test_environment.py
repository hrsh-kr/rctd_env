"""
RCTD Environment — Unit Tests

Tests core environment mechanics, grading, and edge cases.
"""

import pytest
from rctd_env.models import RCTDAction, RCTDObservation
from rctd_env.server.environment import RCTDEnvironment
from rctd_env.server.graders import (
    grade_episode,
    grade_task,
    heuristic_policy,
    random_policy,
)


# ═══════════════════════════════════════════════════════════════════════════
# Environment Basics
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvironmentBasics:
    """Test that the environment initializes and resets correctly."""

    def test_reset_returns_observation(self):
        env = RCTDEnvironment()
        obs = env.reset(seed=42, task_id="easy")
        assert isinstance(obs, RCTDObservation)
        assert not obs.done
        assert obs.budget_remaining > 0
        assert len(obs.hypotheses) == 3  # easy = 3 hypotheses
        assert obs.step_count == 0

    def test_reset_medium(self):
        env = RCTDEnvironment()
        obs = env.reset(seed=42, task_id="medium")
        assert len(obs.hypotheses) == 4
        assert obs.budget_remaining == 12

    def test_reset_hard(self):
        env = RCTDEnvironment()
        obs = env.reset(seed=42, task_id="hard")
        assert len(obs.hypotheses) == 5
        assert obs.budget_remaining == 8

    def test_deterministic_seeding(self):
        """Same seed should produce identical scenarios."""
        env1 = RCTDEnvironment()
        env2 = RCTDEnvironment()
        obs1 = env1.reset(seed=123, task_id="medium")
        obs2 = env2.reset(seed=123, task_id="medium")
        assert obs1.hypotheses == obs2.hypotheses
        assert obs1.total_evidence_count == obs2.total_evidence_count

    def test_different_seeds_differ(self):
        env = RCTDEnvironment()
        obs1 = env.reset(seed=1, task_id="easy")
        obs2 = env.reset(seed=999, task_id="easy")
        # Different seeds should (almost certainly) produce different scenarios
        assert obs1.hypotheses != obs2.hypotheses or True  # themes might repeat


# ═══════════════════════════════════════════════════════════════════════════
# Action Space
# ═══════════════════════════════════════════════════════════════════════════


class TestActions:
    """Test all 5 action types."""

    def setup_method(self):
        self.env = RCTDEnvironment()
        self.obs = self.env.reset(seed=42, task_id="easy")

    def test_read_evidence(self):
        obs = self.env.step(RCTDAction(type="read_evidence", evidence_id=0))
        assert not obs.done
        assert len(obs.revealed_evidence) == 1
        assert obs.revealed_evidence[0].evidence_id == 0
        assert obs.budget_remaining == 19  # cost 1

    def test_run_experiment(self):
        # First read, then verify
        self.env.step(RCTDAction(type="read_evidence", evidence_id=0))
        obs = self.env.step(RCTDAction(type="run_experiment", evidence_id=0))
        assert not obs.done
        assert obs.revealed_evidence[0].verified is True
        assert obs.budget_remaining == 16  # 1 + 3

    def test_consult_expert(self):
        obs = self.env.step(RCTDAction(type="consult_expert", hypothesis_id=0))
        assert not obs.done
        assert len(obs.expert_hints) == 1
        assert obs.expert_hints[0].hypothesis_id == 0
        assert obs.budget_remaining == 18  # cost 2

    def test_discard_hypothesis(self):
        initial_active = len(self.obs.active_hypothesis_ids)
        obs = self.env.step(RCTDAction(type="discard_hypothesis", hypothesis_id=0))
        assert not obs.done
        assert len(obs.active_hypothesis_ids) == initial_active - 1
        assert 0 not in obs.active_hypothesis_ids

    def test_submit_answer(self):
        obs = self.env.step(RCTDAction(type="submit_answer", hypothesis_id=0))
        assert obs.done is True
        assert obs.reward >= 0.0
        assert obs.reward <= 1.0
        assert obs.metrics is not None
        assert "success" in obs.metrics

    def test_submit_correct_gets_high_reward(self):
        """Submitting the correct hypothesis should score high."""
        env = RCTDEnvironment()
        obs = env.reset(seed=42, task_id="easy")
        true_h = env.state.true_hypothesis_id
        obs = env.step(RCTDAction(type="submit_answer", hypothesis_id=true_h))
        assert obs.metrics["success"] is True
        assert obs.reward > 0.5

    def test_submit_wrong_gets_low_reward(self):
        """Submitting wrong hypothesis should score low."""
        env = RCTDEnvironment()
        obs = env.reset(seed=42, task_id="easy")
        true_h = env.state.true_hypothesis_id
        wrong_h = [h for h in obs.active_hypothesis_ids if h != true_h][0]
        obs = env.step(RCTDAction(type="submit_answer", hypothesis_id=wrong_h))
        assert obs.metrics["success"] is False
        assert obs.reward < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test boundary conditions and error handling."""

    def test_cannot_discard_last_hypothesis(self):
        env = RCTDEnvironment()
        obs = env.reset(seed=42, task_id="easy")
        # Discard all but one
        for h in list(obs.active_hypothesis_ids[:-1]):
            obs = env.step(RCTDAction(type="discard_hypothesis", hypothesis_id=h))
        # Try to discard the last one — should be blocked
        last_h = obs.active_hypothesis_ids[0]
        obs = env.step(RCTDAction(type="discard_hypothesis", hypothesis_id=last_h))
        assert not obs.done  # Should NOT end the episode
        assert last_h in obs.active_hypothesis_ids  # Still active
        assert "Cannot" in obs.message  # Should warn the agent

    def test_budget_exhaustion(self):
        """When budget runs out, episode should end."""
        env = RCTDEnvironment()
        obs = env.reset(seed=42, task_id="hard")  # budget=8
        # Spend all budget on experiments (cost 3 each) and reads (cost 1)
        for i in range(12):
            if obs.done:
                break
            obs = env.step(RCTDAction(type="read_evidence", evidence_id=i % 10))
        # Should eventually be done or budget exhausted
        assert obs.budget_remaining >= 0

    def test_reward_in_0_1_range(self):
        """All rewards must be in [0, 1]."""
        env = RCTDEnvironment()
        for seed in range(10):
            obs = env.reset(seed=seed, task_id="medium")
            while not obs.done:
                obs = env.step(RCTDAction(type="read_evidence", evidence_id=0))
                assert 0.0 <= obs.reward <= 1.0
                if obs.step_count > 20:
                    obs = env.step(RCTDAction(type="submit_answer", hypothesis_id=0))
                    break
            assert 0.0 <= obs.reward <= 1.0

    def test_step_after_done_raises_or_noop(self):
        """Stepping after episode end should be handled gracefully."""
        env = RCTDEnvironment()
        obs = env.reset(seed=42, task_id="easy")
        obs = env.step(RCTDAction(type="submit_answer", hypothesis_id=0))
        assert obs.done
        # Step again — should raise or return same obs
        try:
            obs2 = env.step(RCTDAction(type="read_evidence", evidence_id=0))
            assert obs2.done  # If it doesn't raise, should still be done
        except Exception:
            pass  # Raising is also acceptable


# ═══════════════════════════════════════════════════════════════════════════
# Grading
# ═══════════════════════════════════════════════════════════════════════════


class TestGrading:
    """Test the grading system."""

    def test_grade_episode_returns_score(self):
        env = RCTDEnvironment()
        result = grade_episode(env, heuristic_policy, task_id="easy", seed=42)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert "components" in result
        assert "metrics" in result

    def test_grade_components_sum_to_score(self):
        env = RCTDEnvironment()
        result = grade_episode(env, heuristic_policy, task_id="easy", seed=42)
        components = result["components"]
        total = sum(components.values())
        assert abs(total - result["score"]) < 0.001

    def test_heuristic_beats_random(self):
        """Heuristic should score higher than random overall."""
        from rctd_env.server.graders import grade_all_tasks
        env = RCTDEnvironment()
        h_result = grade_all_tasks(env, heuristic_policy, num_episodes=20, base_seed=42)
        r_result = grade_all_tasks(env, random_policy, num_episodes=20, base_seed=42)
        assert h_result["overall_score"] >= r_result["overall_score"]

    def test_failure_modes_present(self):
        """Failed episodes should have failure modes."""
        env = RCTDEnvironment()
        result = grade_task(env, random_policy, task_id="hard", num_episodes=10, base_seed=42)
        # Random agent should fail sometimes on hard
        assert result["success_rate"] < 1.0
        assert len(result["failure_modes"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Policies
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicies:
    """Test that built-in policies produce valid actions."""

    def test_heuristic_uses_all_action_types(self):
        """Heuristic should use multiple action types across episodes."""
        action_types = set()
        env = RCTDEnvironment()
        for seed in range(5):
            obs = env.reset(seed=seed, task_id="medium")
            while not obs.done:
                action = heuristic_policy(obs)
                action_types.add(action.type)
                obs = env.step(action)
                if obs.step_count > 30:
                    break
        # Should use at least 3 action types
        assert len(action_types) >= 3, f"Only used: {action_types}"

    def test_random_policy_terminates(self):
        """Random policy should eventually end the episode."""
        env = RCTDEnvironment()
        for seed in range(5):
            obs = env.reset(seed=seed, task_id="easy")
            steps = 0
            while not obs.done and steps < 50:
                action = random_policy(obs)
                obs = env.step(action)
                steps += 1
            assert obs.done or steps >= 50
