"""Config loading and agent construction, shared by the training and evaluation scripts.

Both entry points have to turn the same YAML into the same objects.  Keeping
that logic here rather than duplicating it in ``scripts/`` guarantees that a
model evaluated by ``evaluate.py`` is rebuilt with exactly the architecture
``train.py`` saved - a mismatch would surface as an inscrutable
``load_state_dict`` error, or worse, silently wrong results.
"""

from __future__ import annotations

import yaml

from agents.hierarchical import HierarchicalMAPPO
from agents.mappo import MAPPO, MAPPOConfig
from envs.uav_env import UAVEnv


def load_config(path: str) -> dict:
    """Read the sectioned YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_dataclass(cls, section: dict | None):
    """Instantiate a dataclass from a config section, ignoring unknown keys.

    Numeric fields are coerced from strings.  This is not cosmetic: YAML 1.1
    only treats an exponent as a number when its sign is explicit, so a
    perfectly reasonable-looking ``carrier_hz: 2.0e9`` arrives here as the
    *string* ``"2.0e9"`` and would otherwise blow up deep inside the channel
    model with an unrecognisable dtype error.
    """
    section = section or {}
    fields = cls.__dataclass_fields__
    kwargs = {}

    for key, value in section.items():
        if key not in fields:
            continue
        declared = fields[key].type
        if isinstance(value, str) and declared in ("float", "int", float, int):
            caster = float if declared in ("float", float) else int
            try:
                value = caster(value)
            except ValueError as exc:
                raise ValueError(
                    f"config key {key!r} = {value!r} is not a valid {caster.__name__}"
                ) from exc
        kwargs[key] = value
    return cls(**kwargs)


def build_agent(
    env: UAVEnv, config: dict, hierarchical: bool | None = None
) -> MAPPO | HierarchicalMAPPO:
    """Construct the flat or hierarchical agent described by the config.

    Parameters
    ----------
    env : UAVEnv
        Supplies the observation, state and scenario dimensions.
    config : dict
        The full parsed YAML.
    hierarchical : bool, optional
        Overrides ``config["hierarchical"]["enabled"]``.  ``evaluate.py`` uses
        this to build both agent variants from one config.

    Notes
    -----
    The high level reads the ``high_level`` section (falling back to
    ``mappo``) and overrides ``gamma``, because one of its transitions spans
    several environment steps.  It sees ``interval`` times fewer transitions
    than a flat agent, so it usually wants smaller minibatches.
    """
    mappo_cfg = build_dataclass(MAPPOConfig, config.get("mappo"))
    hier = config.get("hierarchical", {}) or {}
    use_hierarchy = hier.get("enabled", True) if hierarchical is None else hierarchical

    if not use_hierarchy:
        return MAPPO(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            n_uav=env.n_uav,
            n_users=env.n_users,
            config=mappo_cfg,
        )

    interval = int(hier.get("interval", 5))
    high_cfg = build_dataclass(MAPPOConfig, config.get("high_level") or config.get("mappo"))
    high_cfg.gamma = float(hier.get("high_gamma", mappo_cfg.gamma**interval))
    low_cfg = build_dataclass(MAPPOConfig, config.get("low_level") or config.get("mappo"))

    return HierarchicalMAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_uav=env.n_uav,
        n_users=env.n_users,
        interval=interval,
        high_config=high_cfg,
        low_config=low_cfg,
    )
