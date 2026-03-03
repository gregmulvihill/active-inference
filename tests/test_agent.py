"""Tests for the active inference agent."""

import numpy as np
import pytest
from active_inference.generative_model import GenerativeModel
from active_inference.agent import ActiveInferenceAgent
from active_inference.environments import GridWorldEnv, TMazeEnv


def make_simple_model():
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)
    B[:, :, 1] = np.array([[0, 1], [1, 0]])
    C = np.array([0.0, 2.0])
    D = np.array([0.5, 0.5])
    return GenerativeModel(A=A, B=B, C=C, D=D)


class TestActiveInferenceAgent:
    def test_should_initialize_with_prior_beliefs(self):
        model = make_simple_model()
        agent = ActiveInferenceAgent(model=model)
        np.testing.assert_allclose(agent.beliefs, model.D)

    def test_should_update_beliefs_on_observe(self):
        model = make_simple_model()
        agent = ActiveInferenceAgent(model=model)
        agent.observe(obs_idx=0)
        assert agent.beliefs[0] > 0.5

    def test_should_select_valid_action(self):
        model = make_simple_model()
        agent = ActiveInferenceAgent(model=model, precision=4.0)
        action = agent.step(obs_idx=0)
        assert action in range(model.num_actions)

    def test_should_record_history(self):
        model = make_simple_model()
        agent = ActiveInferenceAgent(model=model)
        agent.step(obs_idx=0)
        agent.step(obs_idx=1, prev_action=0)
        assert len(agent.history) == 2
        assert "vfe" in agent.history[0]
        assert "efe" in agent.history[0]

    def test_should_reset_cleanly(self):
        model = make_simple_model()
        agent = ActiveInferenceAgent(model=model)
        agent.step(obs_idx=0)
        agent.reset()
        assert agent.step_count == 0
        assert len(agent.history) == 0
        np.testing.assert_allclose(agent.beliefs, model.D)

    def test_should_adapt_precision_when_enabled(self):
        model = make_simple_model()
        agent = ActiveInferenceAgent(
            model=model,
            precision=4.0,
            precision_learning_rate=0.5,
        )
        initial_precision = agent.precision
        agent.step(obs_idx=0)
        # Precision should have changed
        assert agent.precision != initial_precision

    def test_report_should_contain_required_fields(self):
        model = make_simple_model()
        agent = ActiveInferenceAgent(model=model)
        agent.step(obs_idx=0)
        report = agent.report()
        assert "step" in report
        assert "beliefs" in report
        assert "belief_entropy" in report
        assert "precision" in report
        assert "last_vfe" in report


class TestAgentGridWorld:
    def test_should_reach_target_in_grid_world(self):
        """Agent should find the target in a simple grid within 30 steps."""
        np.random.seed(42)
        env = GridWorldEnv(grid_size=5, target=4, obs_noise=0.1)
        A, B, C, D = env.build_generative_model()
        model = GenerativeModel(A=A, B=B, C=C, D=D)
        agent = ActiveInferenceAgent(model=model, precision=4.0)

        obs = env.reset()
        action = None
        for _ in range(30):
            action = agent.step(obs, prev_action=action)
            result = env.step(action)
            obs = result.obs
            if env.state == env.target:
                break

        assert env.state == env.target


class TestAgentTMaze:
    def test_should_beat_random_baseline(self):
        """Agent should solve T-maze better than 50% random chance."""
        np.random.seed(42)
        correct = 0
        n_trials = 50

        for trial in range(n_trials):
            env = TMazeEnv(cue_reliability=0.9)
            A, B, C, D = env.build_generative_model()
            model = GenerativeModel(A=A, B=B, C=C, D=D)
            agent = ActiveInferenceAgent(model=model, precision=4.0)

            obs = env.reset()
            action = None
            for _ in range(10):
                action = agent.step(obs, prev_action=action)
                result = env.step(action)
                obs = result.obs
                if result.done:
                    break

            if result.reward > 0:
                correct += 1

        accuracy = correct / n_trials
        assert accuracy > 0.6, f"Accuracy {accuracy:.0%} should beat 50% baseline"
