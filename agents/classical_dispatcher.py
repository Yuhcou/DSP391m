# agents/classical_dispatcher.py
import numpy as np
from collections import defaultdict

class ClassicalDispatcher:
    """
    Two simple heuristics:
    - nearest_car: pick elevator with minimum expected distance to the hall call floor
    - zoning: each elevator owns a contiguous zone of floors
    """

    def __init__(self, num_elevators=6, num_floors=15, mode='nearest_car'):
        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.mode = mode
        if mode == 'zoning':
            self.zones = self._assign_zones()
        else:
            self.zones = None

    def _assign_zones(self):
        floors_per_zone = max(1, self.num_floors // self.num_elevators)
        zones = defaultdict(list)
        for e in range(self.num_elevators):
            start = e * floors_per_zone
            end = start + floors_per_zone if e < self.num_elevators - 1 else self.num_floors
            zones[e] = list(range(start, end))
        return zones

    def select_action(self, state_dict, hall_call_floor):
        """
        state_dict: {'elevator_positions': np.array, 'elevator_directions': np.array}
        Returns: list of elevator indices (choose one for simplicity)
        """
        elevator_positions = state_dict['elevator_positions']
        elevator_directions = state_dict['elevator_directions']

        if self.mode == 'nearest_car':
            # prefer elevators moving toward call, penalize those moving away
            scores = []
            for e, pos in enumerate(elevator_positions):
                pos = int(pos)
                dist = abs(hall_call_floor - pos)
                dirc = int(elevator_directions[e])
                # if elevator currently moving away from the call, penalize a bit
                moving_away = (hall_call_floor - pos) * dirc < 0 and dirc != 0
                penalty = 3 if moving_away else 0
                scores.append(dist + penalty)
            chosen = int(np.argmin(scores))
            return [chosen]

        elif self.mode == 'zoning':
            for e, floors in self.zones.items():
                if hall_call_floor in floors:
                    return [e]
            return [int(np.random.randint(0, self.num_elevators))]

        else:
            raise ValueError("Unknown dispatcher mode")
