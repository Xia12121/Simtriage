"""Environment adapters with a unified clone/rollout interface.

Catan (Catanatron) is sim-native and the cleanest clone+rollout; implement & test
it first. StS / Minecraft are scaffolded with ``supports_cheap_clone=False``.
"""

from .base_env import GameEnv, State, make_env  # noqa: F401
