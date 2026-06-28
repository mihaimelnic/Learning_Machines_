import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from robobo_interface import SimulationRobobo

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

    def get_state(self):
        irs = np.array(self.rob.read_irs(), dtype=np.float32) / self.max_ir
        noise = np.random.normal(0, 0.05, size=irs.shape)
        return np.clip(irs + noise, 0.0, 1.0)

    def reset(self):
        self.stuck_counter = 0
        return self.get_state()

    def step(self, action):
        state = self.get_state()
        front_max = max([state[i] * self.max_ir for i in self.front])
        
        reward = 0.0
        done = False
        move_time = 0.05  

        if action == 0:  
            self.rob.move_blocking(80, 80, move_time)
            reward = 1.0 if front_max < 40 else -0.5
            self.stuck_counter = 0
        elif action == 1:  
            self.rob.move_blocking(30, 80, move_time)
            reward = 0.6 if front_max < 40 else 0.8
            self.stuck_counter += 1 if front_max > 50 else 0
        elif action == 2:  
            self.rob.move_blocking(80, 30, move_time)
            reward = 0.6 if front_max < 40 else 0.8
            self.stuck_counter += 1 if front_max > 50 else 0

        proximity_penalty = max(0.0, ((front_max - 50.0) / 50.0) ** 2)
        reward -= proximity_penalty

        if self.stuck_counter > 10:
            reward -= 2.0
            self.stuck_counter = 0 

        return self.get_state(), reward, done

def train(rob, episodes=300):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "d3qn_robobo.pth")

    env = ContinuousRoboboEnv(rob)
    state_size = 8
    action_size = 3

    policy_net = DuelingDQN(state_size, action_size)
    target_net = DuelingDQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
    loss_fn = nn.SmoothL1Loss()

    replay = deque(maxlen=10000)
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    target_update = 10

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(1000):
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = torch.argmax(policy_net(state_t)).item()

            next_state, reward, done = env.step(action)
            replay.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward

            if len(replay) > batch_size:
                batch = random.sample(replay, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

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

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
        torch.save(policy_net.state_dict(), model_path)
        
    print(f"Training finished. Model saved to: {model_path}")

def run_all_actions(rob: SimulationRobobo):
    print("Starting D3QN training in simulation...")
    rob.play_simulation()
    train(rob)
    rob.stop_simulation()