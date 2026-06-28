import os
import cv2
import sys
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from torch.distributions import Categorical
from robobo_interface import SimulationRobobo, HardwareRobobo

__all__ = ("run_all_actions",)


HIDDEN_SIZE = 64
LEARNING_RATE = 2.5e-4
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2

PPO_EPOCHS = 8
MINIBATCH_SIZE = 64
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.025
MAX_GRAD_NORM = 0.5

STEPS_PER_EP = 450
LOG_EVERY = 5
OBS_DIM = 10  
N_ACTIONS = 6

MOVE_TIME = 0.22
FWD_SPEED = 28
BACK_SPEED = 22
TURN_OUTER = 20
TURN_INNER = 13
SPIN_OUTER = 13
SPIN_INNER = 9

MODEL_FILE = "/root/results/ppo_task90.pth"
RESULTS_DIR = "/root/results/figures"

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        return self.policy_head(features), self.value_head(features)


@dataclass
class RolloutBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def add(self, obs, action, logprob, reward, done, value):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def __len__(self):
        return len(self.obs)

    def clear(self):
        for lst in (self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.values):
            lst.clear()


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.n_actions = n_actions

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value[0].item()

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.net(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + DISCOUNT_FACTOR * next_value * mask - values[t]
            gae = delta + DISCOUNT_FACTOR * GAE_LAMBDA * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        obs = torch.as_tensor(np.stack(buffer.obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buffer.actions, dtype=torch.int64, device=self.device)
        old_logprobs = torch.as_tensor(buffer.logprobs, dtype=torch.float32, device=self.device)

        advantages, returns = self._compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(buffer)
        idx = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, n, MINIBATCH_SIZE):
                mb_idx = idx[start:start + MINIBATCH_SIZE]
                if len(mb_idx) == 0: continue
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.int64, device=self.device)

                logits, values = self.net(obs[mb_idx_t])
                values = values.squeeze(-1)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions[mb_idx_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs[mb_idx_t])
                adv = advantages[mb_idx_t]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns[mb_idx_t])
                loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["updates"] += 1

        for k in ("policy_loss", "value_loss", "entropy"):
            stats[k] /= max(stats["updates"], 1)
        return stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


# 5-Region Detection
def get_blob_features(image, color="red"):
    if color == "red":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 60, 60]); upper1 = np.array([15, 255, 255])
        lower2 = np.array([160, 60, 60]); upper2 = np.array([180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    else:  # green
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([40, 60, 60])
        upper = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    h, w = image.shape[:2]
    sections = 5
    width = w // sections
    counts = [int(np.sum(mask[:, i*width:(i+1)*width] > 0)) for i in range(sections)]
    total = sum(counts)
    if total == 0:
        return 0, 0.0 
    direction = np.argmax(counts) 
    size = 2.0 if total > 80 else 1.0
    return direction, size


def _sense(rob, prev_obs: Optional[np.ndarray] = None):
    red_dir = red_size = green_dir = green_size = 0.0
    try:
        img = rob.get_image_front()
        red_dir, red_size = get_blob_features(img, "red")
        green_dir, green_size = get_blob_features(img, "green")
    except Exception:
        pass

    front_ir = back_ir = 0.0
    try:
        raw = np.nan_to_num(np.array(rob.read_irs(), dtype=np.float32), nan=0.0)
        front_idx = [0, 1, 7]
        back_idx = [2, 3, 4, 5, 6]
        front_max = float(np.max(raw[front_idx])) if len(raw) > 6 else 0.0
        back_max = float(np.max(raw[back_idx])) if len(raw) > 6 else 0.0

        front_ir = 0.0 if front_max < 55 else (1.0 if front_max < 90 else 2.0)
        back_ir = 0.0 if back_max < 55 else (1.0 if back_max < 90 else 2.0)
    except Exception:
        pass

    obs = np.array([
        red_dir / 4.0, red_size / 2.0,
        green_dir / 4.0, green_size / 2.0,
        front_ir / 2.0, back_ir / 2.0,
        0.0, 0.0
    ], dtype=np.float32)

    if random.random() < 0.12:
        print(f"[DEBUG] Red(dir{red_dir} size{red_size}) Green(dir{green_dir} size{green_size}) | F:{front_ir:.1f} B:{back_ir:.1f}")

    return obs, red_size, green_size


def execute_action(rob, action):
    speeds = [
        (FWD_SPEED,   FWD_SPEED),  
        (TURN_INNER,  TURN_OUTER), 
        (TURN_OUTER,  TURN_INNER), 
        (SPIN_INNER,  SPIN_OUTER), 
        (SPIN_OUTER,  SPIN_INNER), 
        (-BACK_SPEED, -BACK_SPEED),
    ]
    l, r = speeds[action]
    rob.move_blocking(l, r, MOVE_TIME)


def get_reward(prev_food, curr_food, obs):
    rdir_s, rsize_s, gdir_s, gsize_s, f_ir_s, b_ir_s, _, _ = obs
    rdir = rdir_s * 4.0
    rsize = rsize_s * 2.0
    gdir = gdir_s * 4.0
    gsize = gsize_s * 2.0
    f_ir = f_ir_s * 2.0
    b_ir = b_ir_s * 2.0

    reward = float(curr_food - prev_food) * 300.0

    if rsize > 0:
        reward += 22 if abs(rdir - 2.0) < 0.8 else 8
        if rsize >= 2.0:
            reward += 25
    else:
        reward -= 2.5

    if gsize > 0:
        reward += 12 if abs(gdir - 2.0) < 0.8 else 4

    if rsize > 0 and gsize > 0 and abs(rdir - 2.0) < 0.8:
        reward += 30

    if f_ir == 2.0:
        reward -= 20
    if b_ir == 2.0:
        reward -= 8

    return reward


def _reset_episode(rob):
    """Clean start - no randomization"""
    try:
        rob.stop_simulation()
        time.sleep(0.4)
    except Exception:
        pass

    rob.play_simulation()
    time.sleep(0.5)

    while True:
        try:
            rob.set_phone_tilt_blocking(105, 100)
            break
        except Exception:
            time.sleep(0.01)


def make_plots(ep_rewards, ep_deliveries, ep_green_max, ep_red_max, outdir):
    os.makedirs(outdir, exist_ok=True)
    x = list(range(len(ep_rewards)))

    plt.figure(figsize=(10, 5))
    plt.plot(ep_rewards, color="#9bbcd6", label="Raw")
    plt.plot([sum(ep_rewards[i:i+5])/5 for i in range(len(ep_rewards)-4)], 
             color="#1f4e79", lw=2, label="Avg")
    plt.title("PPO Training Reward"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(outdir, "ppo_reward.png"), dpi=140); plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, ep_deliveries, color="#2e9e5b")
    plt.title("Deliveries per Episode"); plt.grid()
    plt.savefig(os.path.join(outdir, "ppo_deliveries.png"), dpi=140); plt.close()

    print(f"Plots saved to {outdir}")


def train(rob, episodes):
    agent = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    buffer = RolloutBuffer()

    if os.path.exists(MODEL_FILE):
        try:
            agent.load(MODEL_FILE)
            print(f"[PPO] Loaded model from {MODEL_FILE}")
        except Exception:
            print("[PPO] Starting fresh.")

    global_food = 0
    ep_rewards = []
    ep_deliveries = []
    ep_green_max = []
    ep_red_max = []
    prev_obs = None

    for ep in range(episodes):
        _reset_episode(rob)
        obs, red_s, green_s = _sense(rob)
        ep_food_prev = 0
        total_r = 0.0
        deliveries = 0
        max_g = green_s
        max_r = red_s

        for step in range(STEPS_PER_EP):
            action, logprob, value = agent.act(obs)
            execute_action(rob, action)
            next_obs, nr, ng = _sense(rob, prev_obs=obs)

            try:
                ep_food_now = rob.get_nr_food_collected()
            except:
                ep_food_now = ep_food_prev

            reward = get_reward(ep_food_prev, ep_food_now, next_obs)
            done = (step == STEPS_PER_EP - 1)

            buffer.add(obs, action, logprob, reward, done, value)

            global_food += max(0, ep_food_now - ep_food_prev)
            if ep_food_now > ep_food_prev:
                deliveries += (ep_food_now - ep_food_prev)

            ep_food_prev = ep_food_now
            prev_obs = obs
            obs = next_obs
            total_r += reward
            max_g = max(max_g, ng)
            max_r = max(max_r, nr)

        _, _, last_value = agent.act(obs)
        stats = agent.update(buffer, last_value)
        buffer.clear()

        ep_rewards.append(total_r)
        ep_deliveries.append(deliveries)
        ep_green_max.append(max_g)
        ep_red_max.append(max_r)

        if (ep + 1) % LOG_EVERY == 0:
            mean_r = float(np.mean(ep_rewards[-LOG_EVERY:]))
            print(f"[PPO] Ep {(ep+1):3d} | AvgR: {mean_r:7.2f} | Food: {global_food} | Del: {deliveries}")

        agent.save(MODEL_FILE)
        rob.stop_simulation()
        time.sleep(0.8)

    make_plots(ep_rewards, ep_deliveries, ep_green_max, ep_red_max, RESULTS_DIR)
    print(f"\n[PPO] Training finished!")
    return agent


def run_greedy(rob, duration_seconds=40):
    agent = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    if os.path.exists(MODEL_FILE):
        try:
            agent.load(MODEL_FILE)
        except Exception:
            pass

    print(f"[PPO] Running greedy for {duration_seconds}s...")
    try:
        rob.set_phone_tilt_blocking(105, 100)
    except:
        pass

    start = time.time()
    obs, _, _ = _sense(rob)
    food = 0
    while time.time() - start < duration_seconds:
        action = agent.act_greedy(obs)
        execute_action(rob, action)
        obs, _, _ = _sense(rob)
        try:
            food = rob.get_nr_food_collected()
        except:
            pass
    print(f"[PPO] Total Food: {food}")
    return food


def run_all_actions(rob):
    episodes = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--simulation":
            try:
                episodes = int(sys.argv[1:][i + 1])
            except:
                pass
            break

    is_sim = isinstance(rob, SimulationRobobo)
    if is_sim:
        if episodes is None:
            episodes = 0 if os.path.exists(MODEL_FILE) else 30
            print(f"[PPO] Defaulting to {episodes} episodes.")

        if episodes > 0:
            train(rob, episodes)
            rob.play_simulation()
            run_greedy(rob, 40)
            rob.stop_simulation()
        else:
            rob.play_simulation()
            run_greedy(rob, 40)
            rob.stop_simulation()
    else:
        run_greedy(rob, 40)


if __name__ == "__main__":
    rob = SimulationRobobo(api_port=23000, ip_adress="host.docker.internal")
    run_all_actions(rob)