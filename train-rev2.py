import gymnasium as gym
from stable_baselines3 import PPO
from carbon_env_v2 import CarbonCityEnv  
import os

# --- CONFIGURATION ---
GRID_SIZE = 40
MODEL_NAME = "carbon_brain_v6"
TOTAL_TIMESTEPS = 400_000   
SAVE_INTERVAL = 100_000       # Save a copy of trained brain, for safety, and error checking.
LOG_DIR = "logs"
MODELS_DIR = "models"
# ---------------------


if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Instantiate the Environment
env = CarbonCityEnv(grid_size=GRID_SIZE, k_plants=8)

# 3. Define the Model
# 'ent_coef=0.01' to keep the AI curious and prevent "Zombie Brain" (repeating actions).
# 'tensorboard_log' so visualizing graphs later if you want.
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003, 
    ent_coef=0.01,  
    tensorboard_log=LOG_DIR
)

print(f"--- STARTING TRAINING: {MODEL_NAME} ---")
print(f"Grid Size: {GRID_SIZE} | Radius: {env.emission_radius}")

# 4. The Checkpoint Loop
# Instead of one big model.learn(), we loop.
for step in range(0, TOTAL_TIMESTEPS, SAVE_INTERVAL):
    
    # Train for one interval )
    model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False)
    
    # Calculate current progress
    current_step_count = step + SAVE_INTERVAL
    
    # Save a checkpoint file (e.g., "models/carbon_brain_v4_200000.zip")
    save_path = f"{MODELS_DIR}/{MODEL_NAME}_{current_step_count}"
    model.save(save_path)
    
    print(f"Saved Checkpoint: {save_path}.zip")

print("--- TRAINING FINISHED ---")