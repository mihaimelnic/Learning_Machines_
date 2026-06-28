"""
dqn_task2.py
DQN training for the package‑collecting task.
Continuous state: [merged_IR_left, merged_IR_center, merged_IR_right,
                   green_left, green_center, green_right]
Actions: "forward_full", "right", "left"
"""

import os
import cv2  
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

from robobo_interface import SimulationRobobo
from .ai_utils_simulation_task2 import (
    actions_set, do_move, get_state_key, merge_irs,
    calculate_reward, QTABLE_DIR
)

STATE_DIM = 6              
ACTION_DIM = len(actions_set)  # 3
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
GAMMA = 0.9
BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10     
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995            
EPOCHS = 100
MAX_EPISODE_TIME = 10       
FOOD_GOAL = 7
REWARD_COLLECT = 50     

IR_MAX = 100.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)).to(device),
                torch.LongTensor(actions).to(device),
                torch.FloatTensor(rewards).to(device),
                torch.FloatTensor(np.array(next_states)).to(device),
                torch.FloatTensor(dones).to(device))

    def __len__(self):
        return len(self.buffer)

def preprocess_state(merged_irs, green_counts, image_shape=None):
    ir_norm = np.array(merged_irs) / IR_MAX
    if image_shape is None:
        height, width = 480, 640
    else:
        height, width = image_shape[:2]
    section_width = width // 7
    area_left = height * (3 * section_width)
    area_center = height * section_width
    area_right = height * (3 * section_width)

    green_norm = np.array([
        green_counts[0] / area_left,
        green_counts[1] / area_center,  
        green_counts[2] / area_right
    ])
    return np.concatenate([ir_norm, green_norm]).astype(np.float32)

def merge_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([45, 70, 70])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    height, width = image.shape[:2]
    section_width = width // 7
    count_left = np.sum(mask[:, 0:section_width*3] > 0)
    count_center = np.sum(mask[:, section_width*3:4*section_width] > 0)
    count_right = np.sum(mask[:, 4*section_width:] > 0)
    return [count_left, count_center*4, count_right]

def select_action(policy_net, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(ACTION_DIM)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        return policy_net(state_tensor).argmax(dim=1).item()

def optimize_model(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target_q = rewards + GAMMA * next_q * (1 - dones)
    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def validate_policy(policy_net, rob_val):
    total_reward = 0
    rob_val.play_simulation()
    pos, rot = rob_val.get_position(), rob_val.get_orientation()
    rot.pitch = random.uniform(-1.5, 1.5)
    rob_val.set_position(pos, rot)

    while True:
        try:
            rob_val.set_phone_tilt_blocking(90, 100)
            break
        except:
            time.sleep(0.01)

    start = time.time()
    prev_food = 0
    prev_move = None
    while (time.time() - start < MAX_EPISODE_TIME and
           rob_val.is_running() and not rob_val.is_stopped()):
        try:
            food = rob_val.get_nr_food_collected()
            irs = rob_val.read_irs()
            image = rob_val.read_image_front()
        except:
            continue
        if food >= FOOD_GOAL:
            break

        merged = merge_irs(irs)
        green = merge_image(image)
        state = preprocess_state(merged, green, image.shape)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_idx = policy_net(state_tensor).argmax().item()
        action = actions_set[action_idx]
        do_move(rob_val, action)

        new_irs = rob_val.read_irs()
        new_image = rob_val.read_image_front()
        new_merged = merge_irs(new_irs)
        new_green = merge_image(new_image)
        new_state_key = get_state_key(new_irs, new_image)

        reward = calculate_reward(
            new_state_key, new_merged,
            moved_forward=(prev_move == action == "forward_full"),
            moved_forward_now=(action == "forward_full")
        )
        if food > prev_food:
            reward += REWARD_COLLECT * (food - prev_food)
            prev_food = food
        total_reward += reward
        prev_move = action

    rob_val.stop_simulation()
    return total_reward

def reinforcement_learning(_):
    policy_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    target_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    reward_trials = []
    validation_rewards = []

    rob = SimulationRobobo(api_port=20000)
    rob_val = SimulationRobobo(api_port=23000)

    epsilon = EPS_START

    for episode in range(EPOCHS):
        cumulative_reward = 0
        start_time = time.time()
        prev_move = None
        prev_food = 0
        rob.play_simulation()
        pos, rot = rob.get_position(), rob.get_orientation()
        rot.pitch = random.uniform(-1.5, 1.5)
        rob.set_position(pos, rot)

        while True:
            try:
                rob.set_phone_tilt_blocking(90, 100)
                break
            except:
                time.sleep(0.01)

        try:
            irs = rob.read_irs()
            image = rob.read_image_front()
        except:
            continue
        merged = merge_irs(irs)
        green = merge_image(image)
        state = preprocess_state(merged, green, image.shape)

        while (time.time() - start_time < MAX_EPISODE_TIME and
               rob.is_running() and not rob.is_stopped()):
            try:
                food = rob.get_nr_food_collected()
            except:
                continue
            if food >= FOOD_GOAL:
                break

            action_idx = select_action(policy_net, state, epsilon)
            action = actions_set[action_idx]
            do_move(rob, action)

            try:
                new_irs = rob.read_irs()
                new_image = rob.read_image_front()
            except:
                continue
            new_merged = merge_irs(new_irs)
            new_green = merge_image(new_image)
            next_state = preprocess_state(new_merged, new_green, new_image.shape)

            new_state_key = get_state_key(new_irs, new_image)
            reward = calculate_reward(
                new_state_key, new_merged,
                moved_forward=(prev_move == action == "forward_full"),
                moved_forward_now=(action == "forward_full")
            )
            
            if food > prev_food:
                reward += REWARD_COLLECT * (food - prev_food)
                prev_food = food

            cumulative_reward += reward

            done = (food >= FOOD_GOAL)
            replay_buffer.push(state, action_idx, reward, next_state, done)
            optimize_model(policy_net, target_net, optimizer, replay_buffer)

            prev_move = action
            state = next_state

        rob.stop_simulation()
        reward_trials.append(cumulative_reward)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if episode % 10 == 0:
            val_reward = validate_policy(policy_net, rob_val)
            validation_rewards.append((episode, val_reward))
            print(f"Validation @ ep {episode}: {val_reward}")

        model_path = os.path.join(QTABLE_DIR, "dqn_model.pth")
        torch.save(policy_net.state_dict(), model_path)
        print(f"Ep {episode}, train reward: {cumulative_reward:.2f}, "
              f"eps: {epsilon:.3f}, buffer: {len(replay_buffer)}")

    plt.figure()
    plt.plot(reward_trials, label="Training Reward")
    if validation_rewards:
        x, y = zip(*validation_rewards)
        plt.plot(x, y, marker='o', linestyle='--', label="Validation Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(QTABLE_DIR, "dqn_reward_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Reward plot saved to {plot_path}")
    return policy_net, reward_trials

def run_all_actions(rob):
    """
    Called by learning_robobo_controller.py.
    We ignore the passed rob instance and start our own simulation loop.
    """
    reinforcement_learning(rob)