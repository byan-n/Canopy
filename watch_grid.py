import gymnasium as gym
from stable_baselines3 import PPO
from carbon_env_v2 import CarbonCityEnv
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
import math

# --- CONFIGURATION ---
GRID_SIZE = 40      
MODEL_NAME = "models/carbon_brain_v6_2000000"
NUM_INSTANCES = 9   
K_PLANTS_PER_ENV = 5 
# ---------------------

# 1. Load Model
if not os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"Error: {MODEL_NAME}.zip not found.")
    exit()

print(f"Loading {MODEL_NAME}...")
model = PPO.load(MODEL_NAME)

# 2. Setup Figure
ncols = int(math.ceil(math.sqrt(NUM_INSTANCES)))
nrows = int(math.ceil(NUM_INSTANCES / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
fig.suptitle(f"AI Multi-Watch: {MODEL_NAME}", fontsize=16)
plt.subplots_adjust(bottom=0.2, hspace=0.4, wspace=0.3) # Make room at bottom for button

if NUM_INSTANCES > 1:
    ax_flat = axes.flatten()
else:
    ax_flat = [axes]

# Global Storage for State
envs = []
obs_list = []
done_list = []
total_rewards = []
plant_coords_list = []
imgs = []
scatters = []

# --- INITIALIZATION FUNCTION ---
def init_simulation():
    global envs, obs_list, done_list, total_rewards, plant_coords_list, imgs, scatters
    
    # Clear old lists if restarting
    envs.clear()
    obs_list.clear()
    done_list.clear()
    total_rewards.clear()
    plant_coords_list.clear()
    
    # If this is a restart, we need to clear the actual plot objects too
    # But we want to reuse the axes, so we just update data later.
    
    print(f"Initializing {NUM_INSTANCES} environments...")
    
    for i in range(NUM_INSTANCES):
        env = CarbonCityEnv(grid_size=GRID_SIZE, k_plants=K_PLANTS_PER_ENV)
        envs.append(env)
        obs, _ = env.reset()
        obs_list.append(obs)
        done_list.append(False)
        total_rewards.append(0)
        plant_coords_list.append([])

    # Setup Plots (Only runs once for creation, or updates data on reset)
    if len(imgs) == 0: # First time setup
        for i, ax in enumerate(ax_flat):
            if i < NUM_INSTANCES:
                img = ax.imshow(envs[i].map, cmap='hot', vmin=0, vmax=1)
                imgs.append(img)
                scatter, = ax.plot([], [], 'go', markersize=10, markeredgecolor='white')
                scatters.append(scatter)
                ax.axis('off')
            else:
                ax.axis('off')
    else: # Reset existing plots
        for i in range(NUM_INSTANCES):
            imgs[i].set_data(envs[i].map)
            scatters[i].set_data([], [])
            ax_flat[i].set_title(f"Inst {i+1}: Resetting...")
            
    fig.canvas.draw_idle()

# --- THE LOOP FUNCTION ---
# We use a recursive loop with 'after' or a generator approach for buttons to work.
# But for Matplotlib, a simple generator loop is tricky with buttons.
# The best way in Matplotlib is to use a Timer or manually control the loop.

is_running = False

def run_step(event=None):
    global is_running
    if not is_running: return

    # Check if all done
    if all(done_list):
        is_running = False
        print("All Finished.")
        return

    # Run one step for all envs
    for i in range(NUM_INSTANCES):
        if not done_list[i]:
            action, _ = model.predict(obs_list[i])
            row, col = action
            
            obs, reward, terminated, truncated, _ = envs[i].step(action)
            done = terminated or truncated
            
            obs_list[i] = obs
            done_list[i] = done
            total_rewards[i] += reward
            plant_coords_list[i].append((row, col))

            # Visual Updates
            imgs[i].set_data(envs[i].map)
            px = [c[1] for c in plant_coords_list[i]]
            py = [c[0] for c in plant_coords_list[i]]
            scatters[i].set_data(px, py)
            
            status = "Done" if done else "Run"
            ax_flat[i].set_title(f"Inst {i+1} | Score: {total_rewards[i]:.0f} | {status}", fontsize=9)

    fig.canvas.draw_idle()
    # Schedule next frame (ms)
    fig.canvas.start_event_loop(0.1) 

# --- BUTTON CALLBACK ---
def restart(event):
    global is_running
    print("--- RESTARTING ---")
    is_running = False # Stop current loop
    init_simulation()  # Reset Envs
    is_running = True  # Start Flag
    
    # We need a new loop driver
    while is_running and not all(done_list):
        run_step()
        plt.pause(0.2) # Controls speed

# --- SETUP BUTTON ---
# [left, bottom, width, height]
ax_btn = plt.axes([0.4, 0.05, 0.2, 0.075]) 
btn = Button(ax_btn, 'Restart Simulation', color='lightblue', hovercolor='0.975')
btn.on_clicked(restart)

# Start first run
init_simulation()
is_running = True

# Main Loop
while True:
    if is_running and not all(done_list):
        run_step()
        plt.pause(0.2) # Speed control
    else:
        plt.pause(0.1) # Idle wait for button press