"""Bounded CPU evaluation on the repository's original 1584-satellite topology.

This is a compatibility/sanity test, NOT a reproduction of published QoS results.
The horizon is intentionally short and many generated flowlets remain in flight.
"""

import argparse
import json

from sat_net.routing_env import RoutingEnvAsync
from sat_net.solver import SPF, create_solver
from sat_net.util import NamedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    env_cfg = NamedDict.load("configs/starlink_dvbs2_test.json")
    env_cfg.time_limit_seconds = args.seconds
    env_cfg.verbose = False
    env = RoutingEnvAsync(env_cfg)
    assert len(env.network.satellites) == 1584

    primal_cfg = NamedDict.load("saved_models/primal_cvar/solver_config.json")
    primal_cfg.device = "cpu"
    primal = create_solver(env.obs_dim, env.action_dim, primal_cfg)
    primal.load_models("saved_models/primal_cvar/models/best_model")

    for solver in (SPF(), primal):
        solver.set_eval()
        env.reset(seed=args.seed)
        env.run(solver)
        metrics = env.calc_metrics()
        assert len(env.generated_packets) > 0
        record = {
            "solver": solver.name,
            "satellites": len(env.network.satellites),
            "seconds": args.seconds,
            "seed": args.seed,
            "generated_flowlets": len(env.generated_packets),
            "delivered_flowlets": len(env.delivered_packets),
            "dropped_flowlets": len(env.dropped_packets),
            "throughput": metrics.throughput,
            "mean_e2e_delay_ms": metrics.e2e_delay_mean,
        }
        print("FULL_SCALE_SMOKE " + json.dumps(record))


if __name__ == "__main__":
    main()
