"""Shared agent class definitions for heterogeneous cooperative localization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class AgentClass(str, Enum):
    """Discrete hardware classes used for class-conditional calibration."""

    CLASS_A_UGV = "CLASS_A_UGV"
    CLASS_B_UAV = "CLASS_B_UAV"


@dataclass(frozen=True)
class AgentClassProfile:
    """Class-specific motion and sensing scales."""

    label: str
    max_v_scale: float
    max_omega_scale: float
    process_var_scale: float
    unobserved_process_var_scale: float
    range_var_scale: float
    bearing_var_scale: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


DEFAULT_AGENT_CLASS_PROFILES: dict[AgentClass, AgentClassProfile] = {
    AgentClass.CLASS_A_UGV: AgentClassProfile(
        label="Differential-drive UGV",
        max_v_scale=1.0,
        max_omega_scale=1.0,
        process_var_scale=1.0,
        unobserved_process_var_scale=1.0,
        range_var_scale=1.0,
        bearing_var_scale=1.0,
    ),
    AgentClass.CLASS_B_UAV: AgentClassProfile(
        label="Aerial robot with sharper maneuvers",
        max_v_scale=1.6,
        max_omega_scale=1.8,
        process_var_scale=1.8,
        unobserved_process_var_scale=1.4,
        range_var_scale=0.65,
        bearing_var_scale=0.55,
    ),
}


def normalize_agent_class(agent_class: AgentClass | str | None) -> AgentClass:
    """Normalize external labels to the enum used throughout the module."""

    if agent_class is None:
        return AgentClass.CLASS_A_UGV
    if isinstance(agent_class, AgentClass):
        return agent_class
    return AgentClass(str(agent_class))
