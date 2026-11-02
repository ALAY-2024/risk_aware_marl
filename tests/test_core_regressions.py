"""Regression tests for routing action validity and queue-cost accounting.

These tests express intended invariants, not assumptions about implementation details.
"""

import numpy as np
import pytest
import torch

from sat_net.datablock import DataBlock
from sat_net.event import EventScheduler
from sat_net.link import Link
from sat_net.nn import DiscretePolicy
from sat_net.node import Node
from sat_net.routing_env import RoutingEnvAsync
from sat_net.util import NamedDict


def test_invalid_actions_have_zero_selection_probability():
    """An action mask must exclude invalid next hops after softmax."""
    torch.manual_seed(0)
    policy = DiscretePolicy(
        state_dim=4, action_dim=4, hidden_dim=8, num_hidden_layers=1,
        use_layer_norm=False,
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        logits = policy(
            torch.zeros((1, 4)),
            torch.tensor([[True, False, False, False]], dtype=torch.bool),
        )
        probabilities = torch.softmax(logits, dim=-1)
    assert probabilities[0, 0].item() == pytest.approx(1.0, abs=1e-7)
    assert probabilities[0, 1:].max().item() < 1e-7


def test_queue_wait_is_recorded_once_in_transition_cost():
    """One queuing interval must not be charged twice to the same action."""
    source = Node(name="source", buffer_size=8.0, node_id=1)
    sink = Node(name="sink", buffer_size=8.0, node_id=2)
    link = Link(source, sink, capacity=8.0, propagation_delay=2.0, data_rate=0.05)
    source.add_outgoing_link(link)

    env = object.__new__(RoutingEnvAsync)
    env.scheduler = EventScheduler()
    env.current_time = 0.0

    packets = []
    for packet_id in (1, 2):
        packet = DataBlock(
            block_id=packet_id, source=source.id, is_normal=True,
            size=1.0, delay_limit=100.0, creation_time=0.0,
        )
        packet.last_action = NamedDict({"queue_delay": 0.0})
        assert source.receive(packet)[0]
        assert env._forward_data_block(source, packet, next_hop=sink.id)
        packets.append(packet)

    assert packets[1].total_queue_cost == pytest.approx(20.0)
    assert packets[1].last_action.queue_delay == pytest.approx(20.0)

    while len(env.scheduler):
        event = env.scheduler.pop_event()
        env.current_time = event.time
        env._handle_transmit_end(event)

    assert packets[1].queue_delay == pytest.approx(20.0)
    assert packets[1].last_action.queue_delay == pytest.approx(
        packets[1].queue_delay
    ), "The cost used by PRIMAL must equal the simulated queue wait, not twice the wait."
