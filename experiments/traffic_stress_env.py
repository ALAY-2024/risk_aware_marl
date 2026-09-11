from __future__ import annotations

import numpy as np

from sat_net.datablock import DataBlock
from sat_net.event import EventType
from sat_net.routing_env import RoutingEnvAsync
from sat_net.traffic_stress import TrafficStressScenario


class TrafficStressRoutingEnv(RoutingEnvAsync):
    """Routing environment with regional source-traffic surge injection.

    This PoC intentionally subclasses the upstream environment instead of
    changing its core behavior.  The only changed mechanism is flowlet source
    generation: during the configured stress window, selected source regions
    receive a multiplicative demand surge and the total offered load increases
    accordingly.
    """

    def __init__(self, config, tf_writer=None, stress_config=None):
        super().__init__(config=config, tf_writer=tf_writer)
        if stress_config is None:
            stress_config = self.traffic_config.get("stress", None)
        self.traffic_stress = TrafficStressScenario.from_config(stress_config)

    def _schedule_region_flowlet_traffic(self, interval_ms: float):
        num_flowlets_generated = 0
        num_bins = int(np.ceil(interval_ms / self.flowlet_interval_ms))
        baseline_expected_flowlets_per_bin = (
            self.packet_rate_per_ms
            * self.flowlet_interval_ms
            / max(self.mean_packets_per_flowlet, 1.0)
        )

        for bin_idx in range(num_bins):
            bin_start = self.current_time + bin_idx * self.flowlet_interval_ms
            bin_end = min(
                self.current_time + interval_ms,
                bin_start + self.flowlet_interval_ms,
            )
            if bin_end <= bin_start:
                continue

            source_prob, rate_scale = self.traffic_stress.source_distribution(
                regions=self.traffic_model.regions,
                base_weights=self.traffic_model.weights,
                timestamp_ms=bin_start,
            )
            expected_flowlets_per_bin = baseline_expected_flowlets_per_bin * rate_scale
            num_flowlets = int(self.np_random.poisson(lam=expected_flowlets_per_bin))
            if num_flowlets <= 0:
                continue

            source_region_ids = self.np_random.choice(
                len(self.traffic_model.regions),
                size=num_flowlets,
                p=source_prob,
            )
            target_region_ids = self.traffic_model.sample_target_ids(
                self.np_random, source_region_ids
            )
            is_normal_array = (
                self.np_random.uniform(size=num_flowlets) < self.prob_normal_packet
            )
            creation_times = self.np_random.uniform(
                low=bin_start, high=bin_end, size=num_flowlets
            )
            packet_counts = np.maximum(
                1,
                self.np_random.poisson(
                    lam=self.mean_packets_per_flowlet,
                    size=num_flowlets,
                ),
            )

            for i in range(num_flowlets):
                is_normal = bool(is_normal_array[i])
                packet_size = (
                    self.normal_packet_size if is_normal else self.small_packet_size
                )
                delay_tolerance = (
                    self.normal_packet_delay_limit
                    if is_normal
                    else self.small_packet_delay_limit
                )
                packet_count = int(packet_counts[i])

                packet = DataBlock(
                    block_id=self.next_packet_id,
                    source=None,
                    source_region_id=int(source_region_ids[i]),
                    target_region_id=int(target_region_ids[i]),
                    packet_count=packet_count,
                    packet_size=packet_size,
                    is_normal=is_normal,
                    size=packet_size * packet_count,
                    delay_limit=delay_tolerance,
                    creation_time=float(creation_times[i]),
                    ttl=self.default_ttl,
                )

                packet.last_event = self.scheduler.push_event(
                    event_type=EventType.DATA_GENERATED,
                    time=packet.creation_time,
                    data=packet,
                )
                self.generated_packets.append(packet)
                self.next_packet_id += 1
                self.stats.on_packet_generated(packet)
                num_flowlets_generated += 1

        return num_flowlets_generated
