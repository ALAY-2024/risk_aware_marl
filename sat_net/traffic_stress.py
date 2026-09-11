from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TrafficStressScenario:
    """A lightweight regional traffic-surge model for LEO routing experiments.

    The scenario multiplies the offered source traffic of selected terrestrial
    regions during a bounded time window.  The unnormalised increase is kept as
    an aggregate rate scale, so a regional surge increases both the source
    probability of the hotspot and the total offered load.
    """

    name: str = "normal"
    hotspot_regions: tuple[str, ...] = ()
    multiplier: float = 1.0
    start_ms: float = 0.0
    duration_ms: float = float("inf")

    @classmethod
    def from_config(cls, config=None) -> "TrafficStressScenario":
        if config is None:
            return cls()
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        if not isinstance(config, Mapping):
            raise TypeError("traffic stress config must be a mapping or NamedDict")

        regions = config.get("hotspot_regions", ())
        if isinstance(regions, str):
            regions = (regions,)
        else:
            regions = tuple(str(x) for x in regions)

        scenario = cls(
            name=str(config.get("name", "traffic_stress")),
            hotspot_regions=regions,
            multiplier=float(config.get("multiplier", 1.0)),
            start_ms=float(config.get("start_ms", 0.0)),
            duration_ms=float(config.get("duration_ms", float("inf"))),
        )
        scenario.validate()
        return scenario

    def validate(self) -> None:
        if self.multiplier < 1.0:
            raise ValueError("traffic stress multiplier must be >= 1.0")
        if self.start_ms < 0:
            raise ValueError("traffic stress start_ms must be >= 0")
        if self.duration_ms <= 0:
            raise ValueError("traffic stress duration_ms must be > 0")

    def is_active(self, timestamp_ms: float) -> bool:
        if self.multiplier <= 1.0 or not self.hotspot_regions:
            return False
        return self.start_ms <= timestamp_ms < self.start_ms + self.duration_ms

    def source_distribution(
        self,
        regions: Sequence,
        base_weights: Iterable[float],
        timestamp_ms: float,
    ) -> tuple[np.ndarray, float]:
        """Return source probabilities and aggregate offered-load scaling.

        Args:
            regions: sequence of objects exposing a ``name`` attribute.
            base_weights: positive baseline region weights.  They may already be
                normalised; only relative values matter.
            timestamp_ms: traffic generation time.

        Returns:
            (probabilities, rate_scale).  ``rate_scale`` multiplies the baseline
            flowlet arrival rate and is 1.0 outside the stress window.
        """
        weights = np.asarray(list(base_weights), dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(regions):
            raise ValueError("base_weights must be one-dimensional and match regions")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("base_weights must be non-negative with positive sum")

        baseline_total = float(weights.sum())
        stressed = weights.copy()

        if self.is_active(timestamp_ms):
            name_to_index = {str(region.name): idx for idx, region in enumerate(regions)}
            missing = [name for name in self.hotspot_regions if name not in name_to_index]
            if missing:
                raise ValueError(f"unknown hotspot region(s): {missing}")
            for name in self.hotspot_regions:
                stressed[name_to_index[name]] *= self.multiplier

        stressed_total = float(stressed.sum())
        probabilities = stressed / stressed_total
        rate_scale = stressed_total / baseline_total
        return probabilities, rate_scale
