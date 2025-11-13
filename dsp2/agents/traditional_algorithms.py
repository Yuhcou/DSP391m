"""
Traditional Elevator Group Control System (EGCS) Algorithms

This module implements classic elevator dispatching algorithms:
- Collective Control (Simple Up/Down)
- Nearest Car (NC) Algorithm
- Sectoring/Zoning
- Fixed Priority Rules
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from enum import IntEnum


class Direction(IntEnum):
    """Elevator direction states"""
    DOWN = -1
    IDLE = 0
    UP = 1


class CollectiveControlDispatcher:
    """
    Collective Control (Simple Up/Down Control)
    
    Rule: Each elevator answers calls in the direction it is already traveling 
    until no more calls remain in that direction, then reverses.
    
    Pros: Simple, low computation.
    Cons: Poor performance under heavy or unbalanced traffic.
    """
    
    def __init__(self, n_floors: int, m_elevators: int, capacity: int):
        self.N = n_floors
        self.M = m_elevators
        self.capacity = capacity
        self.elevator_targets = [None] * m_elevators  # Target floor for each elevator
        
    def select_action(self, state: np.ndarray, positions: np.ndarray, 
                     directions: np.ndarray, hall_up: np.ndarray, 
                     hall_down: np.ndarray, car_calls: np.ndarray,
                     onboard_counts: List[int]) -> np.ndarray:
        """
        Select actions for all elevators based on collective control logic.
        
        Actions: 0=Stay, 1=Up, 2=Down, 3=Open
        """
        actions = np.zeros(self.M, dtype=int)
        
        for i in range(self.M):
            pos = int(positions[i])
            direction = int(directions[i])
            car_call = car_calls[i]
            onboard = onboard_counts[i]
            
            # Check if we need to open doors at current floor
            should_open = False
            
            # 1. Check car calls at current floor
            if car_call[pos] > 0:
                should_open = True
            
            # 2. Check hall calls in current direction
            if direction > 0 and hall_up[pos] > 0:
                should_open = True
            elif direction < 0 and hall_down[pos] > 0:
                should_open = True
            elif direction == 0:  # Idle - answer any hall call
                if hall_up[pos] > 0 or hall_down[pos] > 0:
                    should_open = True
            
            if should_open:
                actions[i] = 3  # Open
                continue
            
            # 3. Determine next move based on direction and remaining calls
            if direction > 0 or direction == 0:  # Moving up or idle
                # Check if there are calls above
                has_calls_above = False
                for f in range(pos + 1, self.N):
                    if car_call[f] > 0 or hall_up[f] > 0 or hall_down[f] > 0:
                        has_calls_above = True
                        break
                
                if has_calls_above and pos < self.N - 1:
                    actions[i] = 1  # Up
                else:
                    # Check if there are calls below
                    has_calls_below = False
                    for f in range(pos):
                        if car_call[f] > 0 or hall_up[f] > 0 or hall_down[f] > 0:
                            has_calls_below = True
                            break
                    
                    if has_calls_below and pos > 0:
                        actions[i] = 2  # Down
                    else:
                        actions[i] = 0  # Stay
            
            else:  # Moving down
                # Check if there are calls below
                has_calls_below = False
                for f in range(pos):
                    if car_call[f] > 0 or hall_up[f] > 0 or hall_down[f] > 0:
                        has_calls_below = True
                        break
                
                if has_calls_below and pos > 0:
                    actions[i] = 2  # Down
                else:
                    # Check if there are calls above
                    has_calls_above = False
                    for f in range(pos + 1, self.N):
                        if car_call[f] > 0 or hall_up[f] > 0 or hall_down[f] > 0:
                            has_calls_above = True
                            break
                    
                    if has_calls_above and pos < self.N - 1:
                        actions[i] = 1  # Up
                    else:
                        actions[i] = 0  # Stay
        
        return actions


class NearestCarDispatcher:
    """
    Nearest Car (NC) Algorithm
    
    Rule: Assign the hall call to the elevator that can serve it soonest.
    
    Scoring: Each elevator gets a score based on:
    - Distance from the call floor
    - Direction match (bonus if already moving toward the call)
    - Idle status (bonus if idle)
    
    Example Scoring (Otis-style):
    - Same direction & ahead: 2 × (N + 1 − distance)
    - Idle: N + 1
    - Opposite direction: 1
    where N = total floors.
    """
    
    def __init__(self, n_floors: int, m_elevators: int, capacity: int):
        self.N = n_floors
        self.M = m_elevators
        self.capacity = capacity
        self.assigned_calls = {}  # {(floor, direction): elevator_index}
        
    def _calculate_score(self, elevator_pos: int, elevator_dir: int, 
                        call_floor: int, call_direction: int) -> float:
        """
        Calculate priority score for assigning a call to an elevator.
        Higher score = higher priority.
        """
        distance = abs(elevator_pos - call_floor)
        
        # Idle elevator
        if elevator_dir == 0:
            return self.N + 1 - distance * 0.5
        
        # Same direction and elevator is ahead of call
        if elevator_dir == call_direction:
            if (elevator_dir > 0 and elevator_pos <= call_floor) or \
               (elevator_dir < 0 and elevator_pos >= call_floor):
                # Elevator is moving toward the call
                return 2 * (self.N + 1 - distance)
        
        # Opposite direction or wrong position
        return 1
    
    def _find_best_elevator(self, call_floor: int, call_direction: int,
                           positions: np.ndarray, directions: np.ndarray,
                           onboard_counts: List[int]) -> int:
        """Find the best elevator to serve a hall call."""
        best_elevator = 0
        best_score = -float('inf')
        
        for i in range(self.M):
            # Skip if elevator is at capacity
            if onboard_counts[i] >= self.capacity:
                continue
            
            score = self._calculate_score(
                int(positions[i]), 
                int(directions[i]),
                call_floor, 
                call_direction
            )
            
            if score > best_score:
                best_score = score
                best_elevator = i
        
        return best_elevator
    
    def select_action(self, state: np.ndarray, positions: np.ndarray, 
                     directions: np.ndarray, hall_up: np.ndarray, 
                     hall_down: np.ndarray, car_calls: np.ndarray,
                     onboard_counts: List[int]) -> np.ndarray:
        """Select actions based on nearest car algorithm."""
        actions = np.zeros(self.M, dtype=int)
        
        # Clean up completed assignments
        self.assigned_calls = {k: v for k, v in self.assigned_calls.items() 
                              if (k[1] > 0 and hall_up[k[0]] > 0) or 
                                 (k[1] < 0 and hall_down[k[0]] > 0)}
        
        # Assign unassigned hall calls
        for floor in range(self.N):
            if hall_up[floor] > 0 and (floor, 1) not in self.assigned_calls:
                best_elev = self._find_best_elevator(floor, 1, positions, 
                                                     directions, onboard_counts)
                self.assigned_calls[(floor, 1)] = best_elev
            
            if hall_down[floor] > 0 and (floor, -1) not in self.assigned_calls:
                best_elev = self._find_best_elevator(floor, -1, positions, 
                                                     directions, onboard_counts)
                self.assigned_calls[(floor, -1)] = best_elev
        
        # Determine actions for each elevator
        for i in range(self.M):
            pos = int(positions[i])
            direction = int(directions[i])
            car_call = car_calls[i]
            
            # Check if we should open doors
            should_open = False
            
            # 1. Check car calls at current floor
            if car_call[pos] > 0:
                should_open = True
            
            # 2. Check assigned hall calls at current floor
            if (pos, 1) in self.assigned_calls and self.assigned_calls[(pos, 1)] == i:
                if hall_up[pos] > 0:
                    should_open = True
            if (pos, -1) in self.assigned_calls and self.assigned_calls[(pos, -1)] == i:
                if hall_down[pos] > 0:
                    should_open = True
            
            if should_open:
                actions[i] = 3  # Open
                continue
            
            # 3. Find nearest assigned call or car call
            targets = []
            
            # Add car calls
            for f in range(self.N):
                if car_call[f] > 0:
                    targets.append(f)
            
            # Add assigned hall calls
            for (floor, dir_call), elev in self.assigned_calls.items():
                if elev == i:
                    targets.append(floor)
            
            if targets:
                # Move toward nearest target
                nearest = min(targets, key=lambda f: abs(f - pos))
                if nearest > pos and pos < self.N - 1:
                    actions[i] = 1  # Up
                elif nearest < pos and pos > 0:
                    actions[i] = 2  # Down
                else:
                    actions[i] = 0  # Stay
            else:
                actions[i] = 0  # Stay
        
        return actions


class SectoringDispatcher:
    """
    Sectoring/Zoning Algorithm
    
    Rule: Divide the building into zones; each elevator primarily serves 
    calls in its zone.
    
    Pros: Reduces unnecessary travel.
    Cons: Can be inefficient if traffic is uneven.
    """
    
    def __init__(self, n_floors: int, m_elevators: int, capacity: int):
        self.N = n_floors
        self.M = m_elevators
        self.capacity = capacity
        
        # Divide floors into zones
        floors_per_zone = self.N / self.M
        self.zones = []
        for i in range(self.M):
            start = int(i * floors_per_zone)
            end = int((i + 1) * floors_per_zone) if i < self.M - 1 else self.N
            self.zones.append((start, end))
    
    def _in_zone(self, floor: int, elevator: int) -> bool:
        """Check if floor is in elevator's zone."""
        start, end = self.zones[elevator]
        return start <= floor < end
    
    def _has_calls_in_zone(self, elevator: int, hall_up: np.ndarray, 
                          hall_down: np.ndarray, car_call: np.ndarray) -> bool:
        """Check if there are calls in the elevator's zone."""
        start, end = self.zones[elevator]
        for f in range(start, end):
            if hall_up[f] > 0 or hall_down[f] > 0 or car_call[f] > 0:
                return True
        return False
    
    def select_action(self, state: np.ndarray, positions: np.ndarray, 
                     directions: np.ndarray, hall_up: np.ndarray, 
                     hall_down: np.ndarray, car_calls: np.ndarray,
                     onboard_counts: List[int]) -> np.ndarray:
        """Select actions based on sectoring/zoning."""
        actions = np.zeros(self.M, dtype=int)
        
        for i in range(self.M):
            pos = int(positions[i])
            direction = int(directions[i])
            car_call = car_calls[i]
            zone_start, zone_end = self.zones[i]
            
            # Check if we should open doors
            should_open = False
            
            # 1. Check car calls at current floor
            if car_call[pos] > 0:
                should_open = True
            
            # 2. Check hall calls at current floor in our zone
            if self._in_zone(pos, i):
                if hall_up[pos] > 0 or hall_down[pos] > 0:
                    should_open = True
            
            if should_open:
                actions[i] = 3  # Open
                continue
            
            # 3. Find calls in zone
            targets_in_zone = []
            for f in range(zone_start, zone_end):
                if car_call[f] > 0 or hall_up[f] > 0 or hall_down[f] > 0:
                    targets_in_zone.append(f)
            
            # 4. If no calls in zone, help other zones
            if not targets_in_zone:
                for f in range(self.N):
                    if car_call[f] > 0 or hall_up[f] > 0 or hall_down[f] > 0:
                        targets_in_zone.append(f)
            
            # 5. Move toward nearest target
            if targets_in_zone:
                if direction != 0:
                    # Continue in current direction if there are targets ahead
                    targets_ahead = [f for f in targets_in_zone 
                                   if (direction > 0 and f > pos) or 
                                      (direction < 0 and f < pos)]
                    if targets_ahead:
                        if direction > 0 and pos < self.N - 1:
                            actions[i] = 1  # Up
                        elif direction < 0 and pos > 0:
                            actions[i] = 2  # Down
                        else:
                            actions[i] = 0  # Stay
                    else:
                        # Change direction
                        nearest = min(targets_in_zone, key=lambda f: abs(f - pos))
                        if nearest > pos and pos < self.N - 1:
                            actions[i] = 1  # Up
                        elif nearest < pos and pos > 0:
                            actions[i] = 2  # Down
                        else:
                            actions[i] = 0  # Stay
                else:
                    # Idle - move to nearest target
                    nearest = min(targets_in_zone, key=lambda f: abs(f - pos))
                    if nearest > pos and pos < self.N - 1:
                        actions[i] = 1  # Up
                    elif nearest < pos and pos > 0:
                        actions[i] = 2  # Down
                    else:
                        actions[i] = 0  # Stay
            else:
                # Return to zone center if idle
                zone_center = (zone_start + zone_end) // 2
                if pos < zone_center and pos < self.N - 1:
                    actions[i] = 1  # Up
                elif pos > zone_center and pos > 0:
                    actions[i] = 2  # Down
                else:
                    actions[i] = 0  # Stay
        
        return actions


class FixedPriorityDispatcher:
    """
    Fixed Priority Rules
    
    Rule: Assign calls based on a fixed priority list 
    (e.g., elevator 1 gets lower floors, elevator 2 gets upper floors).
    
    Pros: Predictable.
    Cons: Not adaptive to traffic changes.
    """
    
    def __init__(self, n_floors: int, m_elevators: int, capacity: int):
        self.N = n_floors
        self.M = m_elevators
        self.capacity = capacity
        
        # Assign priority zones (similar to sectoring but strict)
        floors_per_zone = self.N / self.M
        self.priority_zones = []
        for i in range(self.M):
            start = int(i * floors_per_zone)
            end = int((i + 1) * floors_per_zone) if i < self.M - 1 else self.N
            self.priority_zones.append((start, end))
    
    def _get_priority_elevator(self, floor: int) -> int:
        """Get the elevator with priority for this floor."""
        for i, (start, end) in enumerate(self.priority_zones):
            if start <= floor < end:
                return i
        return 0
    
    def select_action(self, state: np.ndarray, positions: np.ndarray, 
                     directions: np.ndarray, hall_up: np.ndarray, 
                     hall_down: np.ndarray, car_calls: np.ndarray,
                     onboard_counts: List[int]) -> np.ndarray:
        """Select actions based on fixed priority rules."""
        actions = np.zeros(self.M, dtype=int)
        
        for i in range(self.M):
            pos = int(positions[i])
            car_call = car_calls[i]
            zone_start, zone_end = self.priority_zones[i]
            
            # Check if we should open doors
            should_open = False
            
            # 1. Always serve car calls
            if car_call[pos] > 0:
                should_open = True
            
            # 2. Serve hall calls only in priority zone
            priority_elev = self._get_priority_elevator(pos)
            if priority_elev == i:
                if hall_up[pos] > 0 or hall_down[pos] > 0:
                    should_open = True
            
            if should_open:
                actions[i] = 3  # Open
                continue
            
            # 3. Find targets (car calls + priority hall calls)
            targets = []
            
            # Add all car calls
            for f in range(self.N):
                if car_call[f] > 0:
                    targets.append(f)
            
            # Add hall calls in priority zone
            for f in range(zone_start, zone_end):
                if hall_up[f] > 0 or hall_down[f] > 0:
                    targets.append(f)
            
            # 4. Move toward nearest target
            if targets:
                nearest = min(targets, key=lambda f: abs(f - pos))
                if nearest > pos and pos < self.N - 1:
                    actions[i] = 1  # Up
                elif nearest < pos and pos > 0:
                    actions[i] = 2  # Down
                else:
                    actions[i] = 0  # Stay
            else:
                # Return to zone center
                zone_center = (zone_start + zone_end) // 2
                if pos < zone_center and pos < self.N - 1:
                    actions[i] = 1  # Up
                elif pos > zone_center and pos > 0:
                    actions[i] = 2  # Down
                else:
                    actions[i] = 0  # Stay
        
        return actions


class TraditionalAlgorithmAdapter:
    """
    Adapter to use traditional algorithms with the EGCSEnv interface.
    """
    
    def __init__(self, algorithm_type: str, n_floors: int, m_elevators: int, capacity: int):
        self.algorithm_type = algorithm_type.lower()
        
        if self.algorithm_type == 'collective':
            self.dispatcher = CollectiveControlDispatcher(n_floors, m_elevators, capacity)
        elif self.algorithm_type == 'nearest_car':
            self.dispatcher = NearestCarDispatcher(n_floors, m_elevators, capacity)
        elif self.algorithm_type == 'sectoring':
            self.dispatcher = SectoringDispatcher(n_floors, m_elevators, capacity)
        elif self.algorithm_type == 'fixed_priority':
            self.dispatcher = FixedPriorityDispatcher(n_floors, m_elevators, capacity)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        self.N = n_floors
        self.M = m_elevators
    
    def select_action(self, state: np.ndarray, env) -> np.ndarray:
        """
        Select actions given state and environment.
        
        Args:
            state: State vector from environment
            env: EGCSEnv instance (to access internal state)
        
        Returns:
            Action array of shape (M,)
        """
        # Parse state vector
        # State order: positions (M,), directions (M,), hall_up (N,), hall_down (N,), car_calls (M*N,)
        positions = state[:self.M]
        directions = state[self.M:2*self.M]
        hall_up = state[2*self.M:2*self.M+self.N]
        hall_down = state[2*self.M+self.N:2*self.M+2*self.N]
        car_calls_flat = state[2*self.M+2*self.N:]
        car_calls = car_calls_flat.reshape(self.M, self.N)
        
        # Get onboard counts from environment
        onboard_counts = [len(ob) for ob in env.onboard]
        
        # Call dispatcher
        return self.dispatcher.select_action(
            state, positions, directions, hall_up, hall_down, 
            car_calls, onboard_counts
        )
