"""
tests/test_all.py — Full test suite
=====================================
Covers: environment, agents, and API endpoints.

Run:
  pytest tests/ -v
"""

import sys, os, pickle, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from env import HVACEnv, QLearningAgent, SARSAAgent, N_ACTIONS, STATE_SHAPE


# ── Environment Tests ────────────────────────────────────────
class TestEnv:
    def setup_method(self):
        self.env = HVACEnv()

    def test_reset_returns_4tuple(self):
        assert len(self.env.reset()) == 4

    def test_state_nonnegative_ints(self):
        for v in self.env.reset():
            assert isinstance(v, (int, np.integer)) and v >= 0

    def test_step_returns_4values(self):
        self.env.reset()
        assert len(self.env.step(0)) == 4

    def test_all_actions_valid(self):
        for a in range(N_ACTIONS):
            self.env.reset()
            s, r, d, info = self.env.step(a)
            assert isinstance(r, float) and isinstance(d, bool)

    def test_reward_bounded(self):
        self.env.reset()
        for a in range(N_ACTIONS):
            _, r, _, _ = self.env.step(a)
            assert -100.0 <= r <= 100.0

    def test_episode_terminates(self):
        self.env.reset()
        done, steps = False, 0
        while not done and steps < 200:
            _, _, done, _ = self.env.step(np.random.randint(N_ACTIONS))
            steps += 1
        assert done

    def test_info_keys(self):
        self.env.reset()
        _, _, _, info = self.env.step(1)
        for k in ["indoor_temp", "energy_step", "carbon_step", "comfort_violations"]:
            assert k in info


# ── Agent Tests ──────────────────────────────────────────────
class TestQLearning:
    def setup_method(self):
        self.agent = QLearningAgent()
        self.env   = HVACEnv()

    def test_q_table_shape(self):
        assert self.agent.q_table.shape == (*STATE_SHAPE, N_ACTIONS)

    def test_q_table_zeros(self):
        assert np.all(self.agent.q_table == 0.0)

    def test_choose_action_valid(self):
        s = self.env.reset()
        assert 0 <= self.agent.choose_action(s, training=False) < N_ACTIONS

    def test_greedy_deterministic(self):
        self.agent.epsilon = 0.0
        s = self.env.reset()
        actions = {self.agent.choose_action(s, training=True) for _ in range(20)}
        assert len(actions) == 1

    def test_update_no_crash(self):
        s = self.env.reset()
        ns, r, done, _ = self.env.step(0)
        self.agent.update(s, 0, r, ns, done)  # must not raise

    def test_epsilon_decays(self):
        self.agent.epsilon = 1.0
        self.agent.decay_epsilon()
        assert self.agent.epsilon < 1.0

    def test_epsilon_floor(self):
        self.agent.epsilon = self.agent.eps_min
        self.agent.decay_epsilon()
        assert self.agent.epsilon == self.agent.eps_min

    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            self.agent.save(path)
            loaded = QLearningAgent.load(path)
            np.testing.assert_array_equal(self.agent.q_table, loaded.q_table)
        finally:
            os.unlink(path)


class TestSARSA:
    def setup_method(self):
        self.agent = SARSAAgent()
        self.env   = HVACEnv()

    def test_q_table_shape(self):
        assert self.agent.q_table.shape == (*STATE_SHAPE, N_ACTIONS)

    def test_choose_action_valid(self):
        s = self.env.reset()
        assert 0 <= self.agent.choose_action(s, training=False) < N_ACTIONS

    def test_update_with_next_action(self):
        s = self.env.reset()
        ns, r, done, _ = self.env.step(0)
        na = self.agent.choose_action(ns)
        self.agent.update(s, 0, r, ns, na, done)  # must not raise

    def test_epsilon_decays(self):
        self.agent.epsilon = 1.0
        self.agent.decay_epsilon()
        assert self.agent.epsilon < 1.0


# ── API Tests ────────────────────────────────────────────────
try:
    from fastapi.testclient import TestClient
    from app import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.mark.skipif(not API_AVAILABLE, reason="fastapi not installed")
class TestAPI:
    PAYLOAD = {"indoor_temp": 25.0, "outdoor_temp": 32.0,
               "occupancy": 1, "hour_of_day": 14.0, "model": "qlearning"}

    def test_health_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_models_200(self, client):
        assert client.get("/models").status_code == 200

    def test_predict_200(self, client):
        assert client.post("/predict", json=self.PAYLOAD).status_code == 200

    def test_predict_action_range(self, client):
        data = client.post("/predict", json=self.PAYLOAD).json()
        assert 0 <= data["action"] <= 3

    def test_predict_has_label(self, client):
        data = client.post("/predict", json=self.PAYLOAD).json()
        assert isinstance(data["action_label"], str)

    def test_predict_invalid_temp_422(self, client):
        bad = {**self.PAYLOAD, "indoor_temp": 999.0}
        assert client.post("/predict", json=bad).status_code == 422

    def test_predict_unknown_model_404(self, client):
        bad = {**self.PAYLOAD, "model": "unknown"}
        assert client.post("/predict", json=bad).status_code == 404

    def test_metrics_200(self, client):
        assert client.get("/metrics").status_code == 200
