"""
env.py — HVAC Simulator, Environment, and RL Agents
=====================================================
Combines simulator, environment discretisation, Q-Learning and SARSA
into one compact module so the rest of the project stays readable.
"""

import os
import pickle

import numpy as np

# ── Constants ────────────────────────────────────────────────
COMFORT_LOW = 20.0
COMFORT_HIGH = 26.0
CARBON_INTENSITY = 0.82
TARGET_TEMP = 23.0

# action → (cooling_effect_°C, power_kW)
ACTION_MAP = {0: (0.0, 0.00), 1: (0.6, 0.50), 2: (1.4, 1.10), 3: (2.3, 1.90)}
ACTION_LABELS = {0: "OFF", 1: "FAN", 2: "COOL", 3: "HEAT"}

# Discretisation bins
TEMP_IN_BINS = [20, 22, 24, 26, 28, 30]
TEMP_OUT_BINS = [20, 25, 30, 35]
HOUR_BINS = [6, 12, 18]

N_ACTIONS = 4
STATE_SHAPE = (7, 5, 2, 4)  # t_in, t_out, occupancy, hour buckets


# ── Simulator ────────────────────────────────────────────────
class HVACSimulator:
    def __init__(self, seed=42, dt_min=15):
        self.rng = np.random.default_rng(seed)
        self.dt = dt_min / 60.0
        self.reset()

    def reset(self):
        self.indoor_temp = float(self.rng.uniform(22.0, 30.0))
        self.hour = int(self.rng.integers(0, 24))
        self.outdoor_temp = self._outdoor(self.hour)
        self.occupancy = self._occupancy(self.hour)
        self.step_count = 0
        self.energy_used = 0.0
        self.carbon_emitted = 0.0
        self.comfort_violations = 0
        return self._state()

    def step(self, action):
        cool, power = ACTION_MAP[action]
        efficiency = 0.85 if self.outdoor_temp > 36 else 1.0
        delta = (
            0.12 * (self.outdoor_temp - self.indoor_temp)
            + (0.35 if 8 <= self.hour <= 17 else 0.0)
            + 0.6 * self.occupancy
            - cool * efficiency
            + float(self.rng.normal(0, 0.08))
        )
        self.indoor_temp = float(np.clip(self.indoor_temp + delta, 10.0, 45.0))
        energy = power * self.dt
        carbon = energy * CARBON_INTENSITY
        self.energy_used += energy
        self.carbon_emitted += carbon
        if not (COMFORT_LOW <= self.indoor_temp <= COMFORT_HIGH):
            self.comfort_violations += 1
        self.step_count += 1
        self.hour = (self.hour + 1) % 24
        self.outdoor_temp = self._outdoor(self.hour)
        self.occupancy = self._occupancy(self.hour)
        return self._state(), energy, carbon, self.step_count >= 96

    def _state(self):
        return {
            "indoor_temp": round(self.indoor_temp, 2),
            "outdoor_temp": round(self.outdoor_temp, 2),
            "occupancy": self.occupancy,
            "hour_of_day": self.hour,
        }

    def _outdoor(self, h):
        # FIX: Wider amplitude (10.0 vs 6.5) and noise (1.5 vs 0.6) to cover the
        # 18–45°C range seen in real/simulated traffic.  The clipped result maps
        # to the same TEMP_OUT_BINS used at inference, so no discretisation mismatch.
        raw = 30.0 + 10.0 * np.sin(2 * np.pi * (h - 6) / 24) + self.rng.normal(0, 1.5)
        return float(np.clip(raw, 15.0, 45.0))

    def _occupancy(self, h):
        if 9 <= h <= 18:
            return int(self.rng.random() > 0.3) if 13 <= h <= 14 else 1
        return int(self.rng.random() > 0.75) if 19 <= h <= 21 else 0


# ── Environment (discretises simulator state + computes reward) ──
class HVACEnv:
    def __init__(self, seed=42, w_energy=1.0, w_comfort=2.0, w_carbon=0.5):
        self.sim = HVACSimulator(seed=seed)
        self.w_energy = w_energy
        self.w_comfort = w_comfort
        self.w_carbon = w_carbon

    def reset(self):
        raw = self.sim.reset()
        return self._disc(raw)

    def step(self, action):
        raw, energy, carbon, done = self.sim.step(action)
        state = self._disc(raw)
        reward = self._reward(raw["indoor_temp"], raw["occupancy"], energy, carbon)
        info = {
            **raw,
            "energy_step": energy,
            "carbon_step": carbon,
            "action": action,
            "reward": reward,
            "total_energy": self.sim.energy_used,
            "total_carbon": self.sim.carbon_emitted,
            "comfort_violations": self.sim.comfort_violations,
        }
        return state, reward, done, info

    def _disc(self, raw):
        return (
            int(np.digitize(raw["indoor_temp"], TEMP_IN_BINS)),
            int(np.digitize(raw["outdoor_temp"], TEMP_OUT_BINS)),
            int(raw["occupancy"]),
            int(np.digitize(raw["hour_of_day"], HOUR_BINS)),
        )

    def _reward(self, t_in, occ, energy, carbon):
        # FIX: Two-tier comfort penalty.
        # When occupied and outside comfort band → full penalty (original behaviour).
        # When unoccupied but outside comfort band → reduced penalty (0.3x) so
        # the agent doesn't learn to always choose OFF just because nobody is home.
        # This prevents ~50% action-0 dominance caused by the former zero-penalty gap.
        if occ and not (COMFORT_LOW <= t_in <= COMFORT_HIGH):
            comfort_pen = abs(t_in - TARGET_TEMP)  # occupied, out of band
        elif not occ and not (COMFORT_LOW <= t_in <= COMFORT_HIGH):
            comfort_pen = 0.3 * abs(t_in - TARGET_TEMP)  # unoccupied, out of band
        else:
            comfort_pen = 0.0  # within band: no penalty

        cost = (
            self.w_energy * energy
            + self.w_comfort * comfort_pen
            + self.w_carbon * carbon
        )
        return round(float(np.clip(np.tanh(-cost / 5.0) * 100.0, -100.0, 100.0)), 4)


# ── Base Agent ───────────────────────────────────────────────
class _BaseAgent:
    def __init__(
        self, lr=0.1, discount=0.95, eps_start=1.0, eps_min=0.05, eps_decay=0.995
    ):
        self.lr = lr
        self.gamma = discount
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.q_table = np.zeros((*STATE_SHAPE, N_ACTIONS), dtype=np.float32)

    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "agent_type": self.agent_type,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "eps_min": self.eps_min,
                    "eps_decay": self.eps_decay,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        agent = cls(d["lr"], d["gamma"], d["epsilon"], d["eps_min"], d["eps_decay"])
        agent.q_table = d["q_table"]
        return agent


# ── Q-Learning Agent (off-policy) ───────────────────────────
class QLearningAgent(_BaseAgent):
    """Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)]"""

    agent_type = "qlearning"

    def update(self, state, action, reward, next_state, done):
        target = reward + (0 if done else self.gamma * np.max(self.q_table[next_state]))
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])


# ── SARSA Agent (on-policy) ──────────────────────────────────
class SARSAAgent(_BaseAgent):
    """Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]"""

    agent_type = "sarsa"

    def update(self, state, action, reward, next_state, next_action, done):
        target = reward + (
            0 if done else self.gamma * self.q_table[next_state][next_action]
        )
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])


def build_agent(agent_type, **kwargs):
    cls = SARSAAgent if agent_type == "sarsa" else QLearningAgent
    return cls(**kwargs)
