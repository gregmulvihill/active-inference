"""Demo: Active inference agent solving a T-maze.

The T-maze is the classic active inference benchmark. It demonstrates
the agent's ability to seek information (epistemic action) before
committing to a decision.

The agent starts at the bottom of a T and receives a cue about which
arm has reward. It must:
1. Read the cue (epistemic action — reduce uncertainty)
2. Go to the junction
3. Choose the correct arm

An agent that only exploits (no curiosity) would guess randomly.
An active inference agent reads the cue first because reducing
uncertainty lowers expected free energy.

Run: python -m examples.tmaze_demo
"""

import numpy as np
from active_inference.generative_model import GenerativeModel
from active_inference.agent import ActiveInferenceAgent
from active_inference.environments import TMazeEnv


def run_tmaze(
    cue_reliability: float = 0.9,
    precision: float = 4.0,
    n_trials: int = 100,
    seed: int = 42,
    verbose: bool = True,
):
    np.random.seed(seed)

    env = TMazeEnv(cue_reliability=cue_reliability)
    A, B, C, D = env.build_generative_model()
    model = GenerativeModel(A=A, B=B, C=C, D=D)

    obs_names = TMazeEnv.OBS_NAMES
    action_names = ["stay", "go_up", "go_left", "go_right"]

    correct = 0
    total_reward = 0.0

    if verbose:
        print(f"T-Maze: cue_reliability={cue_reliability}, precision={precision}")
        print(f"Running {n_trials} trials...")
        print(f"{'='*60}")

    for trial in range(n_trials):
        agent = ActiveInferenceAgent(model=model, precision=precision)
        obs = env.reset()

        action = None
        trial_log = []

        for step in range(10):
            action = agent.step(obs, prev_action=action)

            trial_log.append({
                "step": step,
                "obs": obs_names[obs],
                "action": action_names[action],
                "belief_entropy": agent.history[-1]["belief_entropy"],
            })

            result = env.step(action)
            obs = result.obs

            if result.done:
                break

        got_reward = result.reward > 0
        if got_reward:
            correct += 1
        total_reward += result.reward

        if verbose and trial < 5:
            ctx = env.REWARD_CONTEXTS[env.reward_context]
            outcome = "CORRECT" if got_reward else "WRONG"
            print(f"\nTrial {trial+1} (reward={ctx}): {outcome}")
            for entry in trial_log:
                print(f"  Step {entry['step']}: "
                      f"obs={entry['obs']:>12} → "
                      f"action={entry['action']:>10} "
                      f"(H={entry['belief_entropy']:.3f})")

    accuracy = correct / n_trials
    avg_reward = total_reward / n_trials

    print(f"\n{'='*60}")
    print(f"Results over {n_trials} trials:")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{n_trials})")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Random baseline: 50%")
    print(f"  Cue-following ceiling: {cue_reliability:.0%}")

    return accuracy, avg_reward


def run_cue_comparison():
    """Test how agent performance varies with cue reliability."""
    print("\n" + "="*60)
    print("COMPARISON: Cue reliability impact")
    print("="*60)

    for reliability in [0.5, 0.7, 0.8, 0.9, 1.0]:
        acc, _ = run_tmaze(
            cue_reliability=reliability,
            n_trials=200,
            verbose=False,
        )
        print(f"Cue={reliability:.0%} → Accuracy={acc:.1%} "
              f"(baseline=50%, ceiling={reliability:.0%})")


if __name__ == "__main__":
    run_tmaze(n_trials=50, verbose=True)
    run_cue_comparison()
