"""llm_efficiency_attack: a general-purpose, white-box efficiency (sponge)
attack toolbox for Hugging Face language models.

Public API
----------
>>> from llm_efficiency_attack import Attacker
>>> attack = Attacker(model, tokenizer=tokenizer)
>>> adv_x, logs = attack.run(x, config)
"""

from .attacker import Attacker
from .config import AttackConfig
from .metrics import EfficiencyDamage, measure_efficiency_damage

__all__ = ["Attacker", "AttackConfig", "EfficiencyDamage", "measure_efficiency_damage"]
__version__ = "0.1.0"
