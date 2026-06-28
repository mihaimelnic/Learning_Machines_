import random
import time
import math
import os
import matplotlib.pyplot as plt
from robobo_interface import SimulationRobobo, HardwareRobobo
from .ai_utils_simulation_task2 import (
    QTABLE_DIR, QTABLE_PATH, actions_set, do_move, get_state_key,
    merge_irs, calculate_reward, save_q_table, q_learning,
    load_q_table, randomize_packages, validate_policy,
    count_collected_food, count_total_food, reset_food_collected_counter
)

q_table = load_q_table(QTABLE_PATH)
ROUNDS_PER_EPOCH = 3
ROUND_DURATION = 55


def sigmoid_epsilon(i, epochs, eps_min=0.05, eps_max=1.0, k=0.12):
    midpoint = epochs / 2
    return eps_min + (eps_max - eps_min) / (1 + math.exp(k * (i - midpoint)))


def run_one_round(rob, q_table, epsilon, round_duration, is_hardware):
    """
    Runs a single randomize -> act -> count round and returns
    (cumulative_reward, collected_in_round).
    This is the part that used to be the whole epoch; now an epoch
    can consist of several of these, so we can average collection counts.
    """
    cumulative_reward = 0
    prev_move = None

    if not is_hardware:
        randomize_packages(rob)
        try:
            pos, rot = rob.get_position(), rob.get_orientation()
            rot.pitch = random.uniform(-1.5, 1.5)
            rob.set_position(pos, rot)
        except Exception as e:
            print(f"[run_one_round] could not randomize pose: {e}")

    current_state = get_state_key(rob.read_irs(), rob.read_image_front())

    # food_collected (in food.lua) only resets to 0 on a true simulation
    # restart (sysCall_init), NOT when we just randomize package positions.
    # Since one epoch can run several rounds inside a single play_simulation()
    # session, we take a baseline reading here and report deltas, otherwise
    # round 2/3 would include round 1's already-counted food.
    if not is_hardware:
        baseline_collected = count_collected_food(rob)
    else:
        baseline_collected = 0

    last_collected = 0
    total_food = None
    start_time = time.time()

    while time.time() - start_time < round_duration and rob.is_running() and not rob.is_stopped():
        try:
            if random.random() < epsilon:
                action = random.choice(actions_set)
            else:
                if current_state in q_table:
                    max_q = max(q_table[current_state].values())
                    best_actions = [a for a, q in q_table[current_state].items() if q == max_q]
                    action = random.choice(best_actions)
                else:
                    action = random.choice(actions_set)

            do_move(rob, action, is_hardware=is_hardware)

            new_irs = rob.read_irs()
            new_image = rob.read_image_front()
            new_state = get_state_key(new_irs, new_image)

            reward = calculate_reward(
                new_state, merge_irs(new_irs),
                moved_forward=(prev_move == "forward_full" and action == "forward_full"),
                moved_forward_now=action == "forward_full"
            )

            prev_move = action
            cumulative_reward += reward
            q_learning(current_state, new_state, action, reward, q_table)
            current_state = new_state

            if not is_hardware:
                collected_now = count_collected_food(rob) - baseline_collected

                if total_food is None:
                    total_food = count_total_food(rob) or 7

                if collected_now > last_collected:
                    print(f"[PACKAGE] Collected: {collected_now}/{total_food}")
                    last_collected = collected_now

        except Exception as e:
            print(f"Training error: {e}")
            continue

    if not is_hardware:
        collected_in_round = count_collected_food(rob) - baseline_collected
    else:
        collected_in_round = 0

    return cumulative_reward, collected_in_round


def reinforcement_learning(rob=None, is_hardware=False):
    epochs = 100
    reward_trials = []
    validation_rewards = []
    validation_collected = []

    if rob is None:
        rob = SimulationRobobo(api_port=20000)

    rob_val = rob if not is_hardware else None

    for i in range(epochs):
        epsilon = sigmoid_epsilon(i, epochs)

        if not is_hardware:
            rob.play_simulation()
            reset_food_collected_counter(rob)  
        # Here we play with the tilt.
        for _ in range(20):
            try:
                rob.set_phone_tilt_blocking(90, 100)
                break
            except Exception:
                time.sleep(0.01)

        epoch_reward = 0
        round_collected_counts = []

        n_rounds = ROUNDS_PER_EPOCH if not is_hardware else 1
        round_duration = ROUND_DURATION if not is_hardware else 180

        for r in range(n_rounds):
            round_reward, round_collected = run_one_round(
                rob, q_table, epsilon, round_duration, is_hardware
            )
            epoch_reward += round_reward
            round_collected_counts.append(round_collected)
            if not is_hardware:
                print(f"  Round {r + 1}/{n_rounds}: reward={round_reward:.1f}, collected={round_collected}")

        if not is_hardware:
            rob.stop_simulation()

        avg_collected = (sum(round_collected_counts) / len(round_collected_counts)
                          if round_collected_counts else 0)

        reward_trials.append(epoch_reward)

        if i % 10 == 0 and rob_val is not None:
            val_reward, val_collected = validate_policy(q_table, rob_val, duration=180)
            validation_rewards.append((i, val_reward))
            validation_collected.append((i, val_collected))
            print(f"Validation @ epoch {i}: Reward={val_reward:.0f}, Collected={val_collected}")

        v_epochs = [v[0] for v in validation_rewards] if validation_rewards else []
        v_rewards = [v[1] for v in validation_rewards] if validation_rewards else []
        v_collected = [v[1] for v in validation_collected] if validation_collected else []

        save_q_table(
            q_table=q_table,
            epoch=i,
            train_rewards=reward_trials,
            train_collected=[avg_collected],
            val_epochs=v_epochs,
            val_rewards=v_rewards,
            val_collected=v_collected,
            path=QTABLE_PATH)
        print(f"Epoch {i:3d} | Train Reward: {epoch_reward:6.1f} | "
              f"Avg Collected: {avg_collected:.2f} (rounds: {round_collected_counts}) | Eps: {epsilon:.3f}")

    # ---- Plotting ----
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(reward_trials, label="Training Reward")
    if validation_rewards:
        x, y = zip(*validation_rewards)
        plt.plot(x, y, marker="o", linestyle="--", label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Reward"); plt.legend(); plt.grid()

    plt.subplot(1, 2, 2)
    if validation_collected:
        x, y = zip(*validation_collected)
        plt.plot(x, y, marker="s", linestyle="--", color="green", label="Collected (validation)")
    plt.xlabel("Epoch"); plt.ylabel("Packages"); plt.legend(); plt.grid()

    plot_path = os.path.join(QTABLE_DIR, "reward_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training finished. Plot saved to {plot_path}")
    return q_table, reward_trials