import time
import random
import os
from .ai_utils import (
    QTABLE_PATH,
    actions_set,
    do_move,
    get_state_key,
    obtain_irs,
    calculate_reward,
    load_q_table,
)
from robobo_interface import SimulationRobobo, HardwareRobobo

TEST_DURATION_SIM = 68     
TEST_DURATION_HW = 60       


def choose_greedy(state, q_table):
    if state in q_table and q_table[state]:
        max_q = max(q_table[state].values())
        best = [a for a, qv in q_table[state].items() if qv == max_q]
        return random.choice(best)
    return random.choice(actions_set)


def _get_food(rob):
    try:
        return rob.get_nr_food_collected()
    except Exception:
        return 0


def _base_has_food(rob):
    try:
        return rob.base_detects_food()
    except Exception:
        return False


def run_greedy_episode(q_table, rob, duration):
    """Execute a greedy policy using the given Q-table, return total shaped reward."""
    total_reward = 0.0
    is_sim = isinstance(rob, SimulationRobobo)

    if is_sim:
        rob.play_simulation()

    # Tilt the phone camera
    while True:
        try:
            rob.set_phone_tilt_blocking(150, 140)
            break
        except Exception:
            time.sleep(0.01)
    # here is the initial state.
    start = time.time()
    img = rob.read_image_front()
    food_now = _get_food(rob)
    current_state = get_state_key(rob.read_irs(), img, img, food_now)

    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            print("[test] Time limit reached")
            break
        if is_sim and (not rob.is_running() or rob.is_stopped()):
            print("[test] Simulation stopped")
            break

        food = _get_food(rob)
        if food >= 2 or _base_has_food(rob):
            print(f"[test] Completed: {food} food items collected")
            break

        action = choose_greedy(current_state, q_table)
        do_move(rob, action)
        img = rob.read_image_front()
        new_irs = rob.read_irs()
        food_now = _get_food(rob)
        food_delta = max(0, food_now - food)
        new_state = get_state_key(new_irs, img, img, food_now)
        # here we accumulate the reward/calculate.
        reward = calculate_reward(new_state, obtain_irs(new_irs),
                                  action, food_delta=food_delta)
        total_reward += reward

        current_state = new_state

    if is_sim:
        rob.stop_simulation()

    return total_reward

def run_all_actions(rob):
    q_table = load_q_table(QTABLE_PATH)
    print(f"[test] Loaded Q-table with {len(q_table)} states from {QTABLE_PATH}")

    if isinstance(rob, SimulationRobobo):
        print(f"[test] Running simulation test ({TEST_DURATION_SIM}s)")
        total = run_greedy_episode(q_table, rob, duration=TEST_DURATION_SIM)
    else:
        print(f"[test] Running hardware test ({TEST_DURATION_HW}s)")
        total = run_greedy_episode(q_table, rob, duration=TEST_DURATION_HW)

    print(f"[test] Episode finished – total shaped reward: {total:.0f}")
    return total