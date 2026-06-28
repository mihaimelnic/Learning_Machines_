import os
import cv2
import time
import pickle
import random
import numpy as np
from std_msgs.msg import Int8, Int16, Int32
from robobo_interface import SimulationRobobo
from robobo_interface.hardware import HardwareRobobo

os.environ["QT_QPA_PLATFORM"] = "offscreen"

if cv2.__version__.startswith('4'):
    cv2.setNumThreads(0)

QTABLE_DIR = "/root/results/figures/results/"
QTABLE_PATH = os.path.join(QTABLE_DIR, "q_table4.pkl")

collision_threshold = 15
collected_threshold = 15
green_threshold = 80
red_threshold = 50

actions_set = ["forward_full", "right", "left", "back"]

learning_rate = 0.1
discount_factor = 0.9


def do_move(rob, action, is_hardware=False):
    if action == "forward_full":
        l_speed, r_speed = (45, 45)
        duration_ms = 800 if is_hardware else 600000
    elif action == "right":
        l_speed, r_speed = (3, -3)
        duration_ms = 600 if is_hardware else 600000
    elif action == "left":
        l_speed, r_speed = (-3, 3)
        duration_ms = 600 if is_hardware else 600000
    elif action == "back":
        l_speed, r_speed = (-45, -45)
        duration_ms = 500 if is_hardware else 500000
    else:
        l_speed, r_speed = (45, 45)
        duration_ms = 800 if is_hardware else 600000

    if is_hardware:
        rob._move_srv(Int8(l_speed), Int8(r_speed), Int32(duration_ms), Int16(1))
        return

    rob._sim.callScriptFunction(
        "moveWheelsByTime",
        rob._wheels_script,
        [r_speed, l_speed],
        [duration_ms / 1000.0],
        [rob._block_string(1)],
        bytearray(),
    )
    rob.block()


def merge_irs(irs):
    left = (irs[7] * 2 + irs[2] + irs[4]) / 4.0
    center = (irs[2] + irs[4] * 1.5 + irs[3]) / 3.5
    right = (irs[4] + irs[3] + irs[5] * 2) / 4.0
    return [left, center, right]


def obtain_irs(irs):
    return [irs[7], irs[4], irs[5]]


def get_image_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_g = np.array([40, 60, 60])
    upper_g = np.array([90, 255, 255])
    mask_g = cv2.inRange(hsv, lower_g, upper_g)

    lower_r1 = np.array([0, 60, 60])
    upper_r1 = np.array([15, 255, 255])
    lower_r2 = np.array([165, 60, 60])
    upper_r2 = np.array([180, 255, 255])
    mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1),
                            cv2.inRange(hsv, lower_r2, upper_r2))

    h, w = image.shape[:2]
    section_w = w // 7
    left_g = np.sum(mask_g[:, 0:section_w*3] > 0)
    left_r = np.sum(mask_r[:, 0:section_w*3] > 0)
    center_g = np.sum(mask_g[:, section_w*3:section_w*4] > 0)
    center_r = np.sum(mask_r[:, section_w*3:section_w*4] > 0)
    right_g = np.sum(mask_g[:, section_w*4:] > 0)
    right_r = np.sum(mask_r[:, section_w*4:] > 0)
    return [left_g, center_g, right_g, left_r, center_r, right_r]


def get_state_key(irs, image):
    ir_left, ir_center, ir_right = obtain_irs(irs)
    g_left, g_center, g_right, r_left, r_center, r_right = get_image_features(image)

    def ir_state(val):
        if val < 12: return 2
        elif val < collision_threshold: return 1
        else: return 0

    def color_state(g_cnt, r_cnt):
        if g_cnt > green_threshold:
            return 3 if r_cnt > red_threshold else 1
        elif r_cnt > red_threshold:
            return 2
        return 0

    ir_l_st = ir_state(ir_left)
    ir_c_st = ir_state(ir_center)
    ir_r_st = ir_state(ir_right)
    col_l = color_state(g_left, r_left)
    col_c = color_state(g_center, r_center)
    col_r = color_state(g_right, r_right)

    state_str = f"{ir_l_st}{ir_c_st}{ir_r_st}_{col_l}{col_c}{col_r}"
    return (state_str,)


def calculate_reward(new_state, irs_merged, moved_forward, moved_forward_now, action=None):
    reward = 0
    state_str = new_state[0]
    parts = state_str.split('_')
    irs_st = parts[0]
    cols_st = parts[1]

    green_present = '1' in cols_st or '3' in cols_st
    red_present = '2' in cols_st
    ir_center = irs_merged[1] if len(irs_merged) > 1 else 999

    is_collected = (ir_center < collected_threshold) and green_present

    if is_collected:
        reward += 200
        print("=== PACKAGE COLLECTED! ===")
    elif green_present:
        reward += 35
        if moved_forward_now and cols_st[1] in ('1', '3'):
            reward += 45
        elif action == "left" and cols_st[0] in ('1', '3'):
            reward += 30
        elif action == "right" and cols_st[2] in ('1', '3'):
            reward += 30
    elif red_present:
        reward -= 40
    else:
        reward -= 6
        if moved_forward_now:
            reward += 5

    if irs_st[1] == '2' and moved_forward_now:
        reward -= 140
        print("Collision risk!")

    return reward


def q_learning(current_state, new_state, action, reward, q_table):
    if current_state not in q_table:
        q_table[current_state] = {a: 0.0 for a in actions_set}
    q_s_a = q_table[current_state].get(action, 0.0)
    best_next = max(q_table[new_state].values()) if new_state in q_table else 0.0
    new_q = q_s_a + learning_rate * (reward + discount_factor * best_next - q_s_a)
    q_table[current_state][action] = new_q
    return new_q


def save_q_table(q_table, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(q_table, f)


def load_q_table(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def randomize_packages(rob):
    try:
        all_handles = rob._sim.getObjectsInTree(rob._sim.handle_world, rob._sim.handle_all, 1)
        food_handles = [h for h in all_handles if rob._sim.getObjectName(h).startswith("Food")]
        for h in food_handles:
            x = random.uniform(-2.5, 2.5)
            y = random.uniform(-2.5, 2.5)
            rob._sim.setObjectPosition(h, [x, y, 0.1])
        print(f"Randomised {len(food_handles)} food packages.")
    except Exception as e:
        print(f"Randomisation failed: {e}")


def count_total_food(rob) -> int:
    try:
        all_handles = rob._sim.getObjectsInTree(
            rob._sim.handle_scene, rob._sim.handle_all, 1)
        total = 0
        for h in all_handles:
            try:
                alias = rob._sim.getObjectAlias(h, 0)
                if alias and 'food' in alias.lower():
                    total += 1
            except Exception:
                continue
        return total
    except Exception as e:
        print(f"[count_total_food] failed: {e}")
        return 0


def reset_food_collected_counter(rob):
    global _food_script_handle
    _food_script_handle = None
    _find_food_script_handle(rob, verbose=False)        


def count_collected_food(rob) -> int:
    script_handle = _find_food_script_handle(rob)
    if script_handle is None:
        return 0

    try:
        result = rob._sim.callScriptFunction(
            "remote_get_collected_food", script_handle, [], [], [], ""
        )
    except Exception as e:
        print(f"[count_collected_food] callScriptFunction failed: {e}")
        return 0

    try:
        out_ints = result[0] if isinstance(result, (list, tuple)) else result
        if out_ints and len(out_ints) > 0:
            return int(out_ints[0])
    except Exception as e:
        print(f"[count_collected_food] could not parse result {result!r}: {e}")

    return 0


def validate_policy(q_table, rob, duration=180, is_hardware=True):
    total_reward = 0
    collected = 0

    is_sim = isinstance(rob, SimulationRobobo)

    if is_sim:
        rob.play_simulation()
        try:
            pos, rot = rob.get_position(), rob.get_orientation()
            rot.pitch = random.uniform(-1.5, 1.5)
            rob.set_position(pos, rot)
        except:
            pass

    # here is for camera tilt
    for _ in range(20):
        try:
            rob.set_phone_tilt_blocking(90, 100)
            break
        except:
            time.sleep(0.01)

    start = time.time()
    current_state = get_state_key(rob.read_irs(), rob.read_image_front())
    prev_move = None

    os.makedirs("/root/results/live_camera", exist_ok=True)
    frame_count = 0

    while time.time() - start < duration:
        try:
            if is_sim and (not rob.is_running() or rob.is_stopped()):
                break

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

            if new_image is not None:
                display_img = new_image.copy()

                cv2.putText(display_img, f"Action: {action}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(display_img, f"Collected: {collected}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(display_img, f"Time: {int(time.time()-start)}s", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                frame_count += 1
                if frame_count % 3 == 0:
                    ts = int(time.time())
                    path = f"/root/results/live_camera/frame_{ts}_{frame_count:04d}.jpg"
                    cv2.imwrite(path, display_img)
                    if frame_count % 15 == 0:
                        print(f"📸 Saved: {path}")

            ir_center = merge_irs(new_irs)[1]
            if ir_center < collected_threshold and ('1' in new_state[0] or '3' in new_state[0]):
                collected += 1

            reward = calculate_reward(
                new_state, merge_irs(new_irs),
                moved_forward=(prev_move == "forward_full" and action == "forward_full"),
                moved_forward_now=action == "forward_full",
                action=action
            )

            prev_move = action
            total_reward += reward
            current_state = new_state

        except Exception as e:
            print(f"Validation error: {e}")
            time.sleep(0.2)
            continue

    if is_sim:
        rob.stop_simulation()

    print(f"\n=== HARDWARE TEST COMPLETE ===\nCollected packages: {collected}\n")
    return total_reward, collected