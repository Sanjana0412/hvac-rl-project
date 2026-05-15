"""
train.py — Train, Tune & Evaluate HVAC RL Agents
=================================================
Single entry point for all training workflows.

Usage:
  python train.py                          # train Q-Learning (defaults)
  python train.py --agent sarsa            # train SARSA
  python train.py --episodes 500           # custom episode count
  python train.py --tune                   # hyperparameter grid search
  python train.py --evaluate               # evaluate saved models
  python train.py --agent both --tune      # tune both agents
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid

import numpy as np
import yaml

from env import ACTION_LABELS, HVACEnv, QLearningAgent, SARSAAgent, build_agent

# ── MLflow (optional) ────────────────────────────────────────
try:
    import mlflow

    MLFLOW = True
except ImportError:
    MLFLOW = False

CONFIG_PATH = "configs/config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ── Plotting ─────────────────────────────────────────────────
def plot_training(rewards, energies, carbons, plot_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def smooth(v, w=30):
            return [np.mean(v[max(0, i - w + 1) : i + 1]) for i in range(len(v))]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        eps = range(1, len(rewards) + 1)
        for ax, data, label, color in zip(
            axes,
            [rewards, energies, carbons],
            ["Reward", "Energy (kWh)", "Carbon (kg CO₂)"],
            ["#1f77b4", "#2ca02c", "#d62728"],
        ):
            ax.plot(eps, data, alpha=0.25, color=color)
            ax.plot(eps, smooth(data), color=color, linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, "training.png"), dpi=140)
        plt.close(fig)
        print(f"  [Plot] {plot_dir}/training.png")
    except ImportError:
        pass


def plot_comparison(results, plot_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["label"] for r in results]
        metrics = [
            ("avg_reward", "Reward"),
            ("avg_energy", "Energy kWh"),
            ("avg_carbon", "Carbon kg"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
        for ax, (key, title) in zip(axes, metrics):
            vals = [r[key] for r in results]
            bars = ax.bar(labels, vals, color=colors[: len(labels)])
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, "comparison.png"), dpi=140)
        plt.close(fig)
        print(f"  [Plot] {plot_dir}/comparison.png")
    except ImportError:
        pass


# ── Core Training Loop ───────────────────────────────────────
def train_agent(agent_type, cfg, mlflow_parent_id=None):
    ag = cfg["agent"]
    tr = cfg["training"]
    env = HVACEnv(
        seed=cfg["simulator"]["seed"],
        w_energy=cfg["environment"]["w_energy"],
        w_comfort=cfg["environment"]["w_comfort"],
        w_carbon=cfg["environment"]["w_carbon"],
    )

    agent = build_agent(
        agent_type,
        lr=ag["learning_rate"],
        discount=ag["discount"],
        eps_start=ag["epsilon_start"],
        eps_min=ag["epsilon_min"],
        eps_decay=ag["epsilon_decay"],
    )
    is_sarsa = agent_type == "sarsa"
    run_id = str(uuid.uuid4())[:8]
    n_eps = tr["episodes"]
    model_path = f"models/best_{agent_type}.pkl"
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # MLflow
    mlrun = None
    if MLFLOW:
        mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment_name", "hvac_rl"))
        mlrun = mlflow.start_run(
            run_name=f"{agent_type}_{run_id}", nested=mlflow_parent_id is not None
        )
        mlflow.log_params({**ag, "agent_type": agent_type, "episodes": n_eps})

    print(
        f"\n{'='*55}\n Training [{agent_type.upper()}] | {n_eps} episodes | run={run_id}\n{'='*55}"
    )

    rewards, energies, carbons = [], [], []
    best_reward = float("-inf")
    t0 = time.time()

    for ep in range(1, n_eps + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        if is_sarsa:
            action = agent.choose_action(state)

        while not done:
            if not is_sarsa:
                action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            if is_sarsa:
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action
            else:
                agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            steps += 1

        agent.decay_epsilon()
        # FIX: removed the arbitrary ×10 scalar that distorted checkpoint selection.
        # Raw mean reward per step keeps comparison honest across episode lengths.
        ep_r = float(np.clip(ep_reward / max(steps, 1), -100, 100))
        rewards.append(ep_r)
        energies.append(env.sim.energy_used)
        carbons.append(env.sim.carbon_emitted)

        if ep_r > best_reward:
            best_reward = ep_r
            agent.save(model_path)

        if MLFLOW and mlrun:
            mlflow.log_metrics(
                {
                    "reward": ep_r,
                    "energy": env.sim.energy_used,
                    "carbon": env.sim.carbon_emitted,
                    "epsilon": agent.epsilon,
                },
                step=ep,
            )

        if ep % max(1, n_eps // 10) == 0:
            print(
                f"  Ep {ep:>4}/{n_eps} | reward={np.mean(rewards[-20:]):.2f} "
                f"| energy={np.mean(energies[-20:]):.2f} | ε={agent.epsilon:.3f}"
            )

    elapsed = time.time() - t0
    summary = {
        "run_id": run_id,
        "agent_type": agent_type,
        "episodes": n_eps,
        "avg_reward": round(float(np.mean(rewards)), 4),
        "best_reward": round(best_reward, 4),
        "avg_energy": round(float(np.mean(energies)), 4),
        "avg_carbon": round(float(np.mean(carbons)), 4),
        "elapsed_s": round(elapsed, 1),
    }

    with open(f"logs/{run_id}_{agent_type}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Append to experiment CSV
    csv_path = "experiments/runs.csv"
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary.keys())
        if not exists:
            w.writeheader()
        w.writerow(summary)

    plot_training(rewards, energies, carbons, "plots")

    if MLFLOW and mlrun:
        mlflow.log_metrics(
            {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        )
        mlflow.log_artifact(f"logs/{run_id}_{agent_type}.json")
        mlflow.end_run()

    print(f"\n  Best reward : {best_reward:.3f}")
    print(f"  Model saved : {model_path}")
    print(f"  Time        : {elapsed:.1f}s")
    return agent, summary


# ── Evaluation ───────────────────────────────────────────────
def evaluate_agent(agent, label, n_episodes=100, seeds=None):
    """Evaluate agent across multiple seeds (RL cross-validation)."""
    seeds = seeds or [42, 142, 242, 342, 442]
    all_rewards, all_energy, all_carbon, all_viol = [], [], [], []

    for seed in seeds:
        env = HVACEnv(seed=seed)
        ep_rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            total = 0.0
            while not done:
                action = agent.choose_action(state, training=False)
                state, r, done, _ = env.step(action)
                total += r
            ep_rewards.append(total)
            all_energy.append(env.sim.energy_used)
            all_carbon.append(env.sim.carbon_emitted)
            all_viol.append(env.sim.comfort_violations)
        all_rewards.extend(ep_rewards)

    return {
        "label": label,
        "avg_reward": round(float(np.mean(all_rewards)), 3),
        "std_reward": round(float(np.std(all_rewards)), 3),
        "avg_energy": round(float(np.mean(all_energy)), 3),
        "avg_carbon": round(float(np.mean(all_carbon)), 3),
        "avg_violations": round(float(np.mean(all_viol)), 2),
    }


def run_evaluation(cfg):
    """Load saved models and compare them including a fixed baseline."""
    print(f"\n{'='*55}\n Evaluation (5-seed cross-validation)\n{'='*55}")
    n_eps = cfg["evaluation"]["episodes"]

    class FixedBaseline:
        def choose_action(self, state, training=False):
            return 2  # always COOL

    results = [evaluate_agent(FixedBaseline(), "Fixed Baseline", n_eps)]

    for agent_type in ("qlearning", "sarsa"):
        path = f"models/best_{agent_type}.pkl"
        if not os.path.isfile(path):
            print(f"  [Skip] {path} not found — run training first")
            continue
        cls = SARSAAgent if agent_type == "sarsa" else QLearningAgent
        agent = cls.load(path)
        results.append(evaluate_agent(agent, agent_type.upper(), n_eps))

    # Print table
    print(
        f"\n{'Label':<20} {'Reward':>10} {'±Std':>8} {'Energy':>10} {'Carbon':>10} {'Violations':>12}"
    )
    print("-" * 72)
    for r in results:
        print(
            f"{r['label']:<20} {r['avg_reward']:>10.3f} {r['std_reward']:>8.3f} "
            f"{r['avg_energy']:>10.3f} {r['avg_carbon']:>10.3f} {r['avg_violations']:>12.2f}"
        )

    # Save CSV
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  [Saved] experiments/evaluation.csv")

    plot_comparison(results, "plots")
    return results


# ── Hyperparameter Tuning ────────────────────────────────────
def run_tuning(agent_type, cfg):
    grid = cfg.get("tuning", {}).get(
        "param_grid",
        {
            "learning_rate": [0.05, 0.1, 0.2],
            "discount": [0.90, 0.95, 0.99],
            "epsilon_decay": [0.990, 0.995, 0.999],
        },
    )
    n_eps = cfg.get("tuning", {}).get("tune_episodes", 100)
    agents = ["qlearning", "sarsa"] if agent_type == "both" else [agent_type]

    print(f"\n{'='*55}\n Hyperparameter Tuning | {n_eps} episodes per combo\n{'='*55}")

    overall_best = {"score": float("-inf"), "agent": None, "params": None}

    for atype in agents:
        best = {"score": float("-inf"), "params": None}
        import itertools

        keys = list(grid.keys())
        combos = list(itertools.product(*[grid[k] for k in keys]))
        print(f"\n  [{atype.upper()}] testing {len(combos)} combinations...")

        for combo in combos:
            params = dict(zip(keys, combo))
            tune_cfg = {
                **cfg,
                "agent": {
                    **cfg["agent"],
                    **params,
                    "epsilon_start": 1.0,
                    "epsilon_min": 0.05,
                },
            }
            tune_cfg["training"] = {**cfg["training"], "episodes": n_eps}

            env = HVACEnv(seed=cfg["simulator"]["seed"])
            agent = build_agent(
                atype,
                lr=params["learning_rate"],
                discount=params["discount"],
                eps_decay=params["epsilon_decay"],
            )
            is_sarsa = atype == "sarsa"
            rewards = []

            for _ in range(n_eps):
                state = env.reset()
                done = False
                ep_r = 0.0
                steps = 0
                if is_sarsa:
                    action = agent.choose_action(state)
                while not done:
                    if not is_sarsa:
                        action = agent.choose_action(state)
                    ns, r, done, _ = env.step(action)
                    if is_sarsa:
                        na = agent.choose_action(ns)
                        agent.update(state, action, r, ns, na, done)
                        action = na
                    else:
                        agent.update(state, action, r, ns, done)
                    state = ns
                    ep_r += r
                    steps += 1
                agent.decay_epsilon()
                rewards.append(ep_r / max(steps, 1))

            score = float(np.mean(rewards[-max(1, n_eps // 5) :]))
            print(f"    {params} → score={score:.4f}")
            if score > best["score"]:
                best = {"score": score, "params": params}

        print(f"  Best [{atype}]: score={best['score']:.4f} params={best['params']}")

        # Save best config
        best_cfg = {**cfg, "agent": {**cfg["agent"], **best["params"]}}
        save_config(best_cfg, f"configs/best_{atype}_config.yaml")
        print(f"  [Saved] configs/best_{atype}_config.yaml")

        if best["score"] > overall_best["score"]:
            overall_best = {
                "score": best["score"],
                "agent": atype,
                "params": best["params"],
            }

    print(
        f"\n  Overall best: agent={overall_best['agent']} score={overall_best['score']:.4f}"
    )
    return overall_best


# ── MLflow Model Registry ────────────────────────────────────
def register_models(cfg):
    if not MLFLOW:
        print("[Skip] mlflow not installed")
        return
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "mlruns"))
    client = MlflowClient()

    for agent_type, name in [("qlearning", "hvac-qlearning"), ("sarsa", "hvac-sarsa")]:
        path = f"models/best_{agent_type}.pkl"
        if not os.path.isfile(path):
            print(f"  [Skip] {path} not found")
            continue
        mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment_name", "hvac_rl"))
        with mlflow.start_run(run_name=f"register_{agent_type}"):
            mlflow.log_artifact(path, artifact_path="model")
            mlflow.set_tag("agent_type", agent_type)
            uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        try:
            client.create_registered_model(name)
        except Exception:
            pass
        mv = mlflow.register_model(uri, name)
        client.transition_model_version_stage(
            name, mv.version, "Staging", archive_existing_versions=False
        )
        print(f"  [Registry] {name} v{mv.version} → Staging")


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVAC RL — Train / Tune / Evaluate")
    parser.add_argument(
        "--agent", choices=["qlearning", "sarsa", "both"], default="qlearning"
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--tune", action="store_true", help="Run hyperparameter grid search"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved models")
    parser.add_argument(
        "--register", action="store_true", help="Register models in MLflow registry"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.episodes:
        cfg["training"]["episodes"] = args.episodes

    if args.tune:
        run_tuning(args.agent, cfg)

    elif args.evaluate:
        run_evaluation(cfg)

    elif args.register:
        register_models(cfg)

    else:
        agents = ["qlearning", "sarsa"] if args.agent == "both" else [args.agent]
        for atype in agents:
            train_agent(atype, cfg)
