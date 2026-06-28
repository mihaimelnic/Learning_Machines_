import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from robobo_interface import SimulationRobobo, HardwareRobobo
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class DuelingDQN(nn.Module):
    def __init__(self, state_size=8, action_size=3):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.feature(x)
        adv = self.advantage(features)
        val = self.value(features)
        return val + adv - adv.mean(dim=-1, keepdim=True)

class ContinuousRoboboEnv:
    def __init__(self, rob):
        self.rob = rob
        self.front = [2, 3, 4, 5, 7] 
        self.max_ir = 100.0
        self.stuck_counter = 0
        self.spin_counter = 0  
        self.is_hardware = isinstance(rob, HardwareRobobo)

    def get_state(self):
        irs = np.array(self.rob.read_irs(), dtype=np.float32) / self.max_ir
        noise = np.random.normal(0, 0.05, size=irs.shape)
        return np.clip(irs + noise, 0.0, 1.0)

    def reset(self):
        self.stuck_counter = 0
        self.spin_counter = 0
        return self.get_state()

    def step(self, action):
        move_time = 100 if self.is_hardware else 0.05

        if action == 0:   
            self.rob.move_blocking(8, 8, move_time)      
        elif action == 1: 
            self.rob.move_blocking(5, 8, move_time)      
        elif action == 2: 
            self.rob.move_blocking(8, 5, move_time)      

        raw_irs = self.rob.read_irs()
        front_max_raw = max(raw_irs[i] for i in self.front)
        next_state = self.get_state()
        reward = 0.0
        done = False

        reward -= 0.01

        if action == 0:
            if front_max_raw < 40:
                reward += 1.0
            else:
                reward -= 1.0
            self.stuck_counter = 0
            self.spin_counter = 0     
        else:         
            if front_max_raw < 40:
                reward += 0.2   
            else:
                reward += 0.5    
            if front_max_raw > 50:
                self.stuck_counter += 1
            self.spin_counter += 1 

        proximity_penalty = max(0.0, ((front_max_raw - 50.0) / 50.0) ** 2)
        reward -= proximity_penalty

        if self.spin_counter > 5:
            reward -= 0.1 * (self.spin_counter - 5)

        if self.stuck_counter > 20:
            reward -= 2.0
            done = True
            self.stuck_counter = 0

        if front_max_raw > 90:
            reward -= 10.0
            done = True
            self.stuck_counter = 0

        return next_state, reward, done

def validate(rob, policy_net, num_episodes=5, max_steps=1000):
    """
    Run greedy (epsilon=0) episodes and return:
      - avg_reward: average cumulative reward per episode
      - avg_collisions: average number of episodes that ended with a collision.
    """
    env = ContinuousRoboboEnv(rob)
    total_rewards = []
    collision_counts = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        collisions = 0
        step = 0
        while step < max_steps:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(state_t).argmax().item()
            next_state, reward, done = env.step(action)
            total_reward += reward
            if done:
                collisions += 1    
                break
            state = next_state
            step += 1
        total_rewards.append(total_reward)
        collision_counts.append(collisions)
    avg_reward = np.mean(total_rewards)
    avg_collisions = np.mean(collision_counts)
    return avg_reward, avg_collisions

def train(rob, episodes=2500):
    model_path = "/root/results/d3qn_robobo11.pth"
    try:
        rob.set_simulation_speed(0)
    except AttributeError:
        pass

    env = ContinuousRoboboEnv(rob)
    state_size = 8
    action_size = 3

    policy_net = DuelingDQN(state_size, action_size).to(device)
    target_net = DuelingDQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    replay = deque(maxlen=10000)
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.98
    batch_size = 128
    target_update = 50
    learn_every = 4
    train_rewards = []               
    val_epochs = []                  
    val_rewards = []                
    val_collisions = []            
    validate_every = 50            

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        step = 0

        while step < 1000:
            # Epsilon-greedy action selection.
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = torch.argmax(policy_net(state_t)).item()

            next_state, reward, done = env.step(action)
            replay.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            step += 1

            if step % 200 == 0:
                print(f"Ep {episode} | Step {step} | Reward {total_reward:.1f} | ε {epsilon:.3f} | Replay {len(replay)}")

            if step % learn_every == 0 and len(replay) > batch_size:
                batch = random.sample(replay, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                actions = torch.tensor(actions, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)

                with torch.no_grad():
                    best_next_actions = policy_net(next_states).argmax(1, keepdim=True)
                    next_q_values = target_net(next_states).gather(1, best_next_actions)
                    targets = rewards + gamma * next_q_values * (1 - dones)

                loss = loss_fn(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        # Decay epsilon after each episode.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network.
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Store training episode reward.
        train_rewards.append(total_reward)

        if (episode + 1) % validate_every == 0:
            avg_r, avg_c = validate(rob, policy_net, num_episodes=5)
            val_epochs.append(episode + 1)
            val_rewards.append(avg_r)
            val_collisions.append(avg_c)
            print(f"Validation Ep {episode+1}: Avg Reward {avg_r:.2f}, Avg Collisions {avg_c:.2f}")

        print(f"Episode {episode} finished | Reward: {total_reward:.2f} | ε: {epsilon:.3f} | Steps: {step}")

        torch.save(policy_net.state_dict(), model_path)

    print(f"Training finished. Model saved to: {model_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, episodes+1), train_rewards, label="Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("DQL Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_reward.png")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(val_epochs, val_rewards, 'b-o', label="Validation Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg Validation Reward", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(val_epochs, val_collisions, 'r-s', label="Avg Collision Count")
    ax2.set_ylabel("Avg Collision Count", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("Validation Performance: Reward & Collisions")
    fig.tight_layout()
    plt.savefig("validation_performance.png")
    plt.close()

    print("Plots saved: training_reward.png, validation_performance.png")

def run_all_actions(rob: SimulationRobobo):
    print("Starting D3QN training in simulation...")
    rob.play_simulation()
    train(rob)
    rob.stop_simulation()