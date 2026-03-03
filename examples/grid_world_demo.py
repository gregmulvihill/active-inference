"""Demo: Active inference agent navigating a 1D grid world.

Shows:
- Perception as inference (noisy observations → belief updating)
- Action selection via expected free energy
- Exploration (reducing uncertainty) vs exploitation (reaching target)
- Precision weighting (adaptive confidence)

Run: python -m examples.grid_world_demo
"""

import numpy as np
from active_inference.generative_model import GenerativeModel
from active_inference.agent import ActiveInferenceAgent
from active_inference.environments import GridWorldEnv


def run_grid_world(
    grid_size: int = 7,
    target: int = 6,
    obs_noise: float = 0.15,
    precision: float = 4.0,
    adaptive_precision: bool = True,
    max_steps: int = 30,
    seed: int = 42,
):
    np.random.seed(seed)

    # Build environment
    env = GridWorldEnv(grid_size=grid_size, target=target, obs_noise=obs_noise)

    # Build generative model from environment ground truth
    A, B, C, D = env.build_generative_model()
    model = GenerativeModel(A=A, B=B, C=C, D=D)

    # Build agent
    agent = ActiveInferenceAgent(
        model=model,
        precision=precision,
        precision_learning_rate=0.1 if adaptive_precision else 0.0,
    )

    print(f"Grid World: size={grid_size}, target={target}, noise={obs_noise}")
    print(f"Agent: precision={precision}, adaptive={adaptive_precision}")
    print(f"{'='*70}")
    print(f"{'Step':>4} | {'True':>4} | {'Obs':>3} | {'Action':>8} | {'VFE':>6} | "
          f"{'Prec':>5} | {'H(belief)':>9} | Belief Distribution")
    print(f"{'-'*70}")

    # Initial observation
    obs = env.reset(start=0)
    action = None
    total_reward = 0.0

    for t in range(max_steps):
        # Agent step: perceive + decide
        action = agent.step(obs, prev_action=action)
        rec = agent.history[-1]

        # Format belief distribution as bar
        belief_bar = "".join(
            f"{b:.2f} " for b in rec["beliefs"]
        )

        action_names = ["stay", "left", "right"]
        print(
            f"{t:4d} | "
            f"{env.state:4d} | "
            f"{obs:3d} | "
            f"{action_names[action]:>8} | "
            f"{rec['vfe']:6.2f} | "
            f"{rec['precision']:5.2f} | "
            f"{rec['belief_entropy']:9.4f} | "
            f"[{belief_bar}]"
        )

        # Environment step
        result = env.step(action)
        obs = result.obs
        total_reward += result.reward

        if result.done:
            break

    print(f"{'='*70}")
    print(f"Final: total_reward={total_reward:.1f}, "
          f"steps={agent.step_count}, "
          f"reached_target={env.state == target}")

    return agent, env


def run_comparison():
    """Compare agent performance with different precision settings."""
    print("\n" + "="*70)
    print("COMPARISON: Precision impact on exploration/exploitation")
    print("="*70)

    results = []
    for prec in [0.5, 2.0, 8.0, 16.0]:
        successes = 0
        total_steps = 0
        n_trials = 50

        for trial in range(n_trials):
            np.random.seed(trial * 100)

            env = GridWorldEnv(grid_size=7, target=6, obs_noise=0.15)
            A, B, C, D = env.build_generative_model()
            model = GenerativeModel(A=A, B=B, C=C, D=D)
            agent = ActiveInferenceAgent(model=model, precision=prec)

            obs = env.reset()
            action = None
            for _ in range(30):
                action = agent.step(obs, prev_action=action)
                result = env.step(action)
                obs = result.obs
                if result.done:
                    break

            if env.state == env.target:
                successes += 1
            total_steps += agent.step_count

        avg_steps = total_steps / n_trials
        success_rate = successes / n_trials
        results.append((prec, success_rate, avg_steps))
        print(f"Precision={prec:5.1f} | "
              f"Success={success_rate:5.1%} | "
              f"Avg steps={avg_steps:5.1f}")

    return results


if __name__ == "__main__":
    # Single detailed run
    agent, env = run_grid_world()

    # Batch comparison
    run_comparison()
