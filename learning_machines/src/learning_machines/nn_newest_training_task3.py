import os
import csv
import time
import math
import json
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robobo_interface import SimulationRobobo, HardwareRobobo
from .ai_utils_soft import (
    QTABLE_DIR, QTABLE_PATH, actions_set, do_move, get_state_key,
    merge_irs, calculate_reward, save_q_table, q_learning, obtain_irs,
    load_q_table,
)

EPOCHS = 200
EPISODE_SECONDS = 60
VALIDATE_EVERY = 10
HARDWARE_SECONDS = 60

def run_all_actions(rob):
    if isinstance(rob, SimulationRobobo):
        reinforcement_learning(rob)
    else:
        q = load_q_table(QTABLE_PATH)
        print(f"[hardware] loaded Q-table with {len(q)} states")
        total = validate_policy(q, rob, duration=HARDWARE_SECONDS)
        print(f"[hardware] run finished. total shaped reward: {total:.0f}")


def choose_greedy(state, q_table):
    if state in q_table and q_table[state]:
        max_q = max(q_table[state].values())
        return random.choice(
            [a for a, qv in q_table[state].items() if qv == max_q]
        )
    return random.choice(actions_set)

def _reset_robot(rob, randomise_heading=True):
    """Play simulation, optionally randomise start heading, tilt camera."""
    rob.play_simulation()
    if randomise_heading:
        pos, rot = rob.get_position(), rob.get_orientation()
        rot.pitch = random.uniform(-1.5, 1.5)
        rob.set_position(pos, rot)
    # Tilt phone camera to point at the floor/cube level.
    while True:
        try:
            rob.set_phone_tilt_blocking(150, 140)
            break
        except Exception:
            time.sleep(0.01)


def _get_food(rob):
    """Safe wrapper around get_nr_food_collected."""
    try:
        return rob.get_nr_food_collected()
    except Exception:
        return 0

def _base_has_food(rob):
    """Safe wrapper around base_detects_food."""
    try:
        return rob.base_detects_food()
    except Exception:
        return False

def validate_policy(q_table, rob, duration=EPISODE_SECONDS):
    total_reward = 0
    is_sim = isinstance(rob, SimulationRobobo)
    if is_sim:
        _reset_robot(rob, randomise_heading=True)
    else:
        while True:
            try:
                rob.set_phone_tilt_blocking(150, 140)
                break
            except Exception:
                time.sleep(0.01)
    start = time.time()
    img = rob.read_image_front()
    food_now = _get_food(rob)
    # Pass the same image for both green and red extraction;
    # the new helpers separate channels internally.
    current_state = get_state_key(rob.read_irs(), img, img, food_now)

    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break
        if is_sim and (not rob.is_running() or rob.is_stopped()):
            break
        food = _get_food(rob)
        if food >= 2 or _base_has_food(rob):
            break

        action = choose_greedy(current_state, q_table)
        do_move(rob, action)

        img = rob.read_image_front()
        new_irs = rob.read_irs()
        food_now = _get_food(rob)
        food_delta = max(0, food_now - food)

        new_state = get_state_key(new_irs, img, img, food_now)
        total_reward += calculate_reward(new_state, obtain_irs(new_irs),
                                         action, food_delta=food_delta)
        current_state = new_state

    if is_sim:
        rob.stop_simulation()
    return total_reward

def sigmoid_epsilon(i, epochs, eps_min=0.05, eps_max=1.0, k=0.1):
    midpoint = epochs / 2
    return eps_min + (eps_max - eps_min) / (1 + math.exp(k * (i - midpoint)))

def _moving_avg(a, w=10):
    if len(a) < w:
        return list(range(len(a))), list(a)
    out = [sum(a[i: i + w]) / w for i in range(len(a) - w + 1)]
    return list(range(w - 1, len(a))), out


def make_plots(reward_trials, validation_rewards, size_per_epoch,
               unique_per_epoch, q_table, outdir):
    os.makedirs(outdir, exist_ok=True)

    # 1. Reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(reward_trials, color="#9bbcd6", lw=0.8, label="Training (raw)")
    mx, mv = _moving_avg(reward_trials, 10)
    plt.plot(mx, mv, color="#1f4e79", lw=2, label="Training (10-epoch avg)")
    if validation_rewards:
        vx, vy = zip(*validation_rewards)
        plt.plot(vx, vy, "o--", color="#c55a11", ms=5, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Reward")
    plt.title("Task 3 – reward over training")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reward_curve.png"), dpi=140); plt.close()

    # 2. State-space growth
    plt.figure(figsize=(10, 5))
    plt.plot(size_per_epoch, color="#1f4e79", lw=2, label="Q-table size (total)")
    plt.plot(unique_per_epoch, color="#2e9e5b", lw=1.5, ls="--", label="Unique states / epoch")
    plt.xlabel("Epoch"); plt.ylabel("Number of states")
    plt.title("Task 3 – state-space growth")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "qtable_size.png"), dpi=140); plt.close()

    # 3. Value per state (bar chart)
    items = [(k[0], max(v.values())) for k, v in q_table.items() if v]
    items.sort(key=lambda x: x[1], reverse=True)
    if items:
        labels = [s for s, _ in items]
        vals = [v for _, v in items]
        plt.figure(figsize=(12, max(4, len(labels) * 0.28)))
        colors = ["#1f4e79" if v >= 0 else "#c55a11" for v in vals]
        plt.barh(range(len(labels)), vals, color=colors)
        plt.yticks(range(len(labels)), labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.xlabel("Max Q-value")
        plt.title("Task 3 – learned value per state")
        plt.grid(axis="x", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "state_values.png"), dpi=140); plt.close()

    # 4. Q-value heatmap
    states = [k[0] for k in q_table.keys()]
    if states:
        M = np.array(
            [[q_table[k].get(a, 0.0) for a in actions_set] for k in q_table.keys()],
            dtype=float,
        )
        plt.figure(figsize=(6, max(4, len(states) * 0.28)))
        im = plt.imshow(M, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Q-value")
        plt.yticks(range(len(states)), states, fontsize=8)
        plt.xticks(range(len(actions_set)), actions_set, rotation=20, ha="right")
        plt.title("Task 3 – Q-values (state × action)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "qtable_heatmap.png"), dpi=140); plt.close()

def reinforcement_learning(rob):
    q_table = {}
    reward_trials = []
    validation_rewards = []
    size_per_epoch = []
    unique_per_epoch = []

    # A second sim instance for validation runs (separate port/same IP), for avoiding communication errors.
    rob_val = SimulationRobobo(
        api_port=rob._api_port if hasattr(rob, "_api_port") else 23000,
        ip_adress=rob._ip if hasattr(rob, "_ip") else "host.docker.internal",
    )

    for i in range(EPOCHS):
        cumulative_reward = 0
        seen_states = set()
        _reset_robot(rob, randomise_heading=True)
        epsilon = sigmoid_epsilon(i, EPOCHS)
        start = time.time()
        img = rob.read_image_front()
        food_now = _get_food(rob)
        current_state = get_state_key(rob.read_irs(), img, img, food_now)

        while time.time() - start < EPISODE_SECONDS and rob.is_running() and not rob.is_stopped():
            food = _get_food(rob)
            if food >= 2 or _base_has_food(rob):
                break

            seen_states.add(current_state[0])

            # ε-greedy action selection
            if random.random() < epsilon:
                action = random.choice(actions_set)
            else:
                action = choose_greedy(current_state, q_table)

            do_move(rob, action)

            img = rob.read_image_front()
            new_irs = rob.read_irs()
            food_now = _get_food(rob)
            food_delta = max(0, food_now - food)

            new_state = get_state_key(new_irs, img, img, food_now)
            reward = calculate_reward(new_state, obtain_irs(new_irs),
                                     action, food_delta=food_delta)
            cumulative_reward += reward

            q_learning(current_state, new_state, action, reward, q_table)
            current_state = new_state

        rob.stop_simulation()

        reward_trials.append(cumulative_reward)
        size_per_epoch.append(len(q_table))
        unique_per_epoch.append(len(seen_states))

        # Periodic validation
        if i % VALIDATE_EVERY == 0:
            val_reward = validate_policy(q_table, rob_val)
            validation_rewards.append((i, val_reward))
            print(f" [validation] epoch {i}: {val_reward:.0f}")

        save_q_table(q_table, QTABLE_PATH)
        print(f"Epoch {i:3d} | reward {cumulative_reward:7.0f} | eps {epsilon:.3f} "
              f"| q_table {len(q_table)} states")
    # results.
    os.makedirs(QTABLE_DIR, exist_ok=True)
    val_dict = dict(validation_rewards) if validation_rewards else {}
    with open(os.path.join(QTABLE_DIR, "training_results.json"), "w") as f:
        json.dump({
            "epochs": EPOCHS,
            "training_rewards": reward_trials,
            "validation_rewards": {str(k): v for k, v in val_dict.items()},
            "q_table_size": len(q_table),
            "size_per_epoch": size_per_epoch,
            "unique_states_per_epoch": unique_per_epoch,
        }, f, indent=2)

    with open(os.path.join(QTABLE_DIR, "training_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "training_reward", "validation_reward",
                         "q_table_size", "unique_states"])
        for idx, r in enumerate(reward_trials):
            writer.writerow([idx, r, val_dict.get(idx, ""),
                             size_per_epoch[idx], unique_per_epoch[idx]])

    make_plots(reward_trials, validation_rewards,
               size_per_epoch, unique_per_epoch, q_table, QTABLE_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Epochs: {EPOCHS} | Final Q-table size: {len(q_table)} states")
    print("=" * 60)
    return q_table, reward_trials