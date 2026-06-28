import os
import cv2
import time
import random
from datetime import datetime
from .ai_utils_hardware_task3 import (
    QTABLE_PATH,
    actions_set,
    do_move,
    get_state_key,
    obtain_irs,
    calculate_reward,
    load_q_table,
)

HARDWARE_DURATION = 280
SAVE_DIR = "/root/results/hardware_images" 


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


def annotate_image(img, state_str, action, reward, step, food):
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thick = 1
    color = (0, 255, 0)
    y0, dy = 25, 25

    lines = [
        f"Step: {step}",
        f"State: {state_str}",
        f"Action: {action}",
        f"Reward: {reward:.1f}",
        f"Food: {food}",
        datetime.now().strftime("%H:%M:%S"),
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, y0 + i * dy), font, font_scale, color, thick, cv2.LINE_AA)
    return out


def run_hardware_episode(q_table, rob, duration=HARDWARE_DURATION):
    total_reward = 0.0

    # create image save directory.
    os.makedirs(SAVE_DIR, exist_ok=True)

    # limit tilt attempts so we don't hang.
    tilt_attempts = 0
    while tilt_attempts < 3:
        try:
            rob.set_phone_tilt_blocking(120, 115)
            break
        except Exception:
            tilt_attempts += 1
            time.sleep(0.5)

    start = time.time()
    img = rob.read_image_front()
    img = cv2.flip(img, 0)          # flip vertically (up <-> down).
    food_now = _get_food(rob)
    current_state = get_state_key(rob.read_irs(), img, img, food_now)

    step = 0
    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            print("[hardware test] Time limit reached")
            break

        food = _get_food(rob)
        if food >= 2 or _base_has_food(rob):
            print(f"[hardware test] Completed: {food} food items collected")
            break

        step += 1
        if step % 10 == 0:
            print(f"[hardware test] step {step}, state: {current_state}, food: {food}")

        action = choose_greedy(current_state, q_table)

        try:
            do_move(rob, action)
        except Exception as e:
            print(f"[error] during action '{action}': {e}")
            break

        img = rob.read_image_front()
        img = cv2.flip(img, 0)      # flip vertically (up <-> down).
        new_irs = rob.read_irs()
        food_now = _get_food(rob)
        food_delta = max(0, food_now - food)
        new_state = get_state_key(new_irs, img, img, food_now)

        reward = calculate_reward(new_state, obtain_irs(new_irs),
                                  action, food_delta=food_delta)
        total_reward += reward

        annotated = annotate_image(img, new_state[0], action, reward, step, food_now)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] 
        filename = f"step_{step:04d}_{timestamp}.jpg"
        full_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(full_path, annotated)

        current_state = new_state

    return total_reward


def run_all_actions(rob):
    q_table = load_q_table(QTABLE_PATH)
    print(f"[hardware test] Loaded Q-table with {len(q_table)} states")
    total = run_hardware_episode(q_table, rob)
    print(f"[hardware test] Episode finished – total shaped reward: {total:.0f}")