from dataclasses import dataclass

from pysc2.lib.named_array import NamedNumpyArray


@dataclass
class Cost:
    minerals: int = 0
    vespene: int = 0
    supply: int = 0

    def can_pay(self, player: NamedNumpyArray) -> bool:
        return (
            (player.minerals >= self.minerals) and
            (player.vespene >= self.vespene) and
            (player.food_cap - player.food_used >= self.supply)
        )

class GameConstraints:
    MAX_SUPPLY_DEPOTS = 10

class SC2Costs:
    REFINERY     = Cost(minerals=50)
    SCV          = Cost(minerals=50, supply=1)
    SUPPLY_DEPOT = Cost(minerals=100)
    COMMAND_CENTER = Cost(minerals=400)
