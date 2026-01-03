import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CarbonCityEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size=40,
        k_plants=5, #number of plants
        reduction_m=0.9, #plants reduction intensity
        emission_radius=6, #carbon blob
    ):
        super().__init__()

        self.grid_size = grid_size
        self.k_plants = k_plants
        self.reduction_m = reduction_m
        self.emission_radius = emission_radius

        self.action_space = spaces.MultiDiscrete([grid_size, grid_size])

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, grid_size, grid_size),
            dtype=np.uint8, 
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plants_placed = 0

        self.map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # --- SCENARIO GENERATION ---
        # Roll a die (0-100) to decide what kind of map this is
        roll = np.random.randint(0, 100)

        if roll < 10:
            # 10% Chance: EMPTY MAP
            #"If no targets, just spread out to avoid penalties."
            num_hotspots = 0
            
        elif roll < 30:
            # 20% Chance: SPARSE MAP (Very easy)
            # "Precision targeting on small dots."
            num_hotspots = np.random.randint(1, 4)
            
        else:
            # 70% Chance: STANDARD BUSY MAP
            # "Massive carbon reduction."
            min_spots = int(self.grid_size / 2.5)
            max_spots = int(self.grid_size / 2)
            if max_spots <= min_spots: max_spots = min_spots + 1  
            num_hotspots = np.random.randint(min_spots, max_spots)
        # ---------------------------

        for _ in range(num_hotspots):
            cx, cy = np.random.randint(0, self.grid_size, size=2)
            min_r = max(6, int(self.grid_size * 0.10))
            max_r = max(12, int(self.grid_size * 0.20))
            r = np.random.randint(min_r, max_r)
            intensity = np.random.uniform(0.6, 1.0)

            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if 0 <= cx + dx < self.grid_size and 0 <= cy + dy < self.grid_size:
                        if dx * dx + dy * dy <= r * r:
                            self.map[cx + dx, cy + dy] += intensity

        max_val = np.max(self.map)
        if max_val > 0: self.map /= max_val
        self.coverage_map = np.zeros_like(self.map)

        return self._get_obs(), {}

    def _get_obs(self):
        plants_left = (self.k_plants - self.plants_placed) / self.k_plants
        plants_layer = np.full_like(self.map, plants_left, dtype=np.float32)
        
        obs_float = np.stack([self.map, self.coverage_map, plants_layer], axis=0)
        return (obs_float * 255).astype(np.uint8)

    def step(self, action):
        row, col = action
        r = self.emission_radius

        # --- ANTI-STACKING LOGIC ---
        # Check if the specific center point is already covered
        if self.coverage_map[row, col] > 0:
            # PENALTY: Wasted turn.
            reward = -1.0
            
            self.plants_placed += 1
            terminated = self.plants_placed >= self.k_plants
            
            return self._get_obs(), reward, terminated, False, {}

        # Area of Effect (AOE)
        x_min, x_max = max(0, row - r), min(self.grid_size, row + r + 1)
        y_min, y_max = max(0, col - r), min(self.grid_size, col + r + 1)

        area_carbon = self.map[x_min:x_max, y_min:y_max]
        area_coverage = self.coverage_map[x_min:x_max, y_min:y_max]

        # --- EFFICIENCY (OVERLAP) PENALTY ---
        # 1. Count how many pixels are ALREADY covered
        overlap_count = np.sum(area_coverage > 0)
        
        # 2. Penalty factor: Scaled down significantly
        overlap_penalty = overlap_count * 0.05
        
        # 3. Calculate Carbon Reward
        effective_carbon = area_carbon * (1.0 - area_coverage)
        carbon_removed = np.sum(effective_carbon) * self.reduction_m
        
        carbon_reward = carbon_removed 

        # 4. Final Reward
        reward = carbon_reward - overlap_penalty

        # Small punishment for placing in empty space
        if carbon_reward < 1e-6 and overlap_penalty == 0:
            reward -= 0.1

        # Update State
        self.map[x_min:x_max, y_min:y_max] *= (1.0 - self.reduction_m)
        self.coverage_map[x_min:x_max, y_min:y_max] = 1.0

        self.plants_placed += 1
        terminated = self.plants_placed >= self.k_plants

        return self._get_obs(), reward, terminated, False, {}