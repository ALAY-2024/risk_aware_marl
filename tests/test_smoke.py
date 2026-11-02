"""Small CPU-only end-to-end smoke tests for the LEO routing pipeline."""

import numpy as np
import torch

from sat_net.routing_env import RoutingEnvAsync
from sat_net.solver.primal_cvar import PrimalCVaRAgent
from sat_net.solver.spf import SPF
from sat_net.util import NamedDict


def test_event_driven_spf_small_constellation():
    config = NamedDict.load("configs/starlink_dvbs2_train.json")
    config.network.num_orbits = 8
    config.network.num_sats_per_orbit = 8
    config.network.phasing = 0
    config.traffic.packet_rate_per_ms = 2.0
    config.traffic.mean_packets_per_flowlet = 4
    config.time_limit_seconds = 0.15
    config.update_interval_ms = 50
    config.verbose = False

    env = RoutingEnvAsync(config)
    env.reset(seed=12345)
    assert len(env.network.satellites) == 64
    env.run(SPF())
    metrics = env.calc_metrics()
    assert len(env.generated_packets) > 0
    assert len(env.delivered_packets) + len(env.dropped_packets) <= len(env.generated_packets)
    assert np.isfinite(metrics.throughput)


def test_primal_cvar_cpu_training_step():
    """Validate the full reward critic / IQN cost critic / actor update path."""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = NamedDict.load("configs/primal_cvar.json")
    config.device = "cpu"
    config.hidden_dim = 32
    config.feature_dim = 16
    config.embedding_dim = 8
    config.num_quantiles = 8
    config.num_hidden_layers = 1
    config.buffer_size = 64
    config.batch_size = 8
    config.train_start_size = 8
    config.actor_update_freq = 1
    config.cost_multiplier_update_freq = 1
    config.consider_risk_after_step = 0
    config.update_lambda_after_step = 0
    agent = PrimalCVaRAgent(obs_dim=94, action_dim=4, config=config)

    for i in range(8):
        state = np.full(94, i / 100.0, dtype=np.float32)
        next_state = state + 0.01
        mask = np.ones(4, dtype=np.bool_)
        agent.replay_buffer.add(
            state=state,
            action=i % 4,
            action_mask=mask,
            reward=1.0,
            cost=0.1,
            done=False,
            truncated=False,
            next_state=next_state,
            next_action_mask=mask,
            target_cost=0.2,
        )

    agent.learn()
    assert agent.training_steps == 1
    assert torch.isfinite(agent.alpha()).all()
    assert torch.isfinite(agent.lambdar()).all()
