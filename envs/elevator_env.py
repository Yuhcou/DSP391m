# envs/elevator_env.py
import numpy as np

class ElevatorEnv:
    """
    Simple elevator environment.
    - Discrete time steps.
    - At each step a new hall call is generated.
    - Action: choose one elevator (int index) or choose a list of elevator indices.
    - State vector: concatenation of elevator_positions (int) and elevator_directions (-1/0/1).
    """
    def __init__(self, num_elevators=6, num_floors=15, episode_length=200, seed=None):
        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.episode_length = episode_length
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self):
        # Initialize elevators at random floors and idle
        self.elevator_positions = np.random.randint(0, self.num_floors, size=self.num_elevators)
        self.elevator_directions = np.zeros(self.num_elevators, dtype=int)  # -1,0,1
        self.time = 0
        self.total_reward = 0.0
        self.current_call = None  # (origin, destination)
        return self._get_state_vector()

    def _generate_call(self):
        origin = np.random.randint(0, self.num_floors)
        dest = np.random.randint(0, self.num_floors)
        while dest == origin:
            dest = np.random.randint(0, self.num_floors)
        return (int(origin), int(dest))

    def step(self, action):
        """
        action: integer elevator index OR list/tuple of indices OR None (no-op)
        Returns: next_state_vector, reward (float), done (bool), info (dict)
        """
        # Generate a hall call each step (event-driven)
        origin, dest = self._generate_call()
        self.current_call = (origin, dest)

        # Normalize action to list
        if action is None:
            assigned = []
        elif isinstance(action, (list, tuple, np.ndarray)):
            assigned = list(action)
        else:
            # single elevator index
            assigned = [int(action)]

        reward = 0.0
        # Simple ETD approximation: distance from elevator to origin + travel distance origin->dest
        travel_distance = abs(dest - origin)
        etd_list = []
        for e in assigned:
            # clamp e
            if e < 0 or e >= self.num_elevators:
                continue
            dist_to_origin = abs(int(self.elevator_positions[e]) - origin)
            etd = dist_to_origin + travel_distance
            etd_list.append(etd)
            # Move elevator: in this simple simulator, elevator teleports to dest (abstracted)
            self.elevator_positions[e] = dest
            # Update direction based on final movement (for next steps)
            if dest > origin:
                self.elevator_directions[e] = 1
            elif dest < origin:
                self.elevator_directions[e] = -1
            else:
                self.elevator_directions[e] = 0

        if len(etd_list) > 0:
            reward = -float(np.mean(etd_list))  # lower ETD -> higher reward (less negative)
        else:
            # No elevator assigned -> penalize by expected waiting (distance to closest elevator + travel)
            distances = np.abs(self.elevator_positions - origin)
            reward = -float(np.min(distances) + travel_distance)

        self.time += 1
        self.total_reward += reward
        done = (self.time >= self.episode_length)

        info = {
            'call': (origin, dest),
            'assigned': assigned,
            'etd': etd_list
        }
        return self._get_state_vector(), reward, done, info

    def _get_state_vector(self):
        # Flat vector: [pos_0 ... pos_n-1, dir_0 ... dir_n-1] normalized to floats
        pos = self.elevator_positions.astype(np.float32)
        dirc = self.elevator_directions.astype(np.float32)
        vec = np.concatenate([pos, dirc]).astype(np.float32)
        return vec

    # convenience accessors for dispatcher
    def get_state_dict(self):
        return {
            'elevator_positions': self.elevator_positions.copy(),
            'elevator_directions': self.elevator_directions.copy()
        }

    def sample_state_vector(self):
        return self._get_state_vector()
