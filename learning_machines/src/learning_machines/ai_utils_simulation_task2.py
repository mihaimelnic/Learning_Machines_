import os
import cv2
import time
import math
import pickle
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt
from robobo_interface import SimulationRobobo, HardwareRobobo
import matplotlib
from std_msgs.msg import Int8, Int16, Int32
matplotlib.use('Agg'
               )                 


QTABLE_DIR = "/root/results/figures/results/"
QTABLE_PATH = os.path.join(QTABLE_DIR, "q_table4.pkl")

PLOT_DIR = QTABLE_DIR
TRAIN_REWARD_PLOT   = os.path.join(PLOT_DIR, "training_reward.png")
TRAIN_COLL_PLOT     = os.path.join(PLOT_DIR, "training_collected.png")
VAL_REWARD_PLOT     = os.path.join(PLOT_DIR, "validation_reward.png")
VAL_COLL_PLOT       = os.path.join(PLOT_DIR, "validation_collected.png")

collision_threshold = 20
collected_threshold = 10
green_threshold = 80
red_threshold = 50

actions_set = ["forward_full", "right", "left", "back"]

learning_rate = 0.1
discount_factor = 0.9

EPOCHS = 50
EPISODES_PER_EPOCH = 10
STEPS_PER_EPISODE = 200
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

VALIDATE_EVERY = 1
VALIDATION_DURATION = 30

DEBUG_DUMP_SCENE_ONCE = True
_scene_dumped = False

def do_move(rob, action, is_hardware=False):
    if action == "forward_full":
        l_speed, r_speed = (45, 45)
        duration_ms = 600000
    elif action == "right":
        l_speed, r_speed = (3, -3)
        duration_ms = 600000
    elif action == "left":
        l_speed, r_speed = (-3, 3)
        duration_ms = 600000
    elif action == "back":
        l_speed, r_speed = (-50, -50)
        duration_ms = 500000
    else:
        l_speed, r_speed = (45, 45)
        duration_ms = 600000

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
    lower_g = np.array([45, 70, 70])
    upper_g = np.array([85, 255, 255])
    mask_g = cv2.inRange(hsv, lower_g, upper_g)

    lower_r1 = np.array([0, 70, 70])
    upper_r1 = np.array([10, 255, 255])
    lower_r2 = np.array([170, 70, 70])
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
        if val < 8: return 2
        elif val < collision_threshold: return 1
        else: return 0

    def color_state(g_cnt, r_cnt):
        if g_cnt > green_threshold and r_cnt > red_threshold:
            return 3
        elif g_cnt > green_threshold:
            return 1
        elif r_cnt > red_threshold:
            return 2
        else:
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
        reward += 250
        print("=== PACKAGE COLLECTED! ===")
    elif green_present:
        reward += 18
        if moved_forward_now and cols_st[1] in ('1', '3'):
            reward += 45
        elif action == "left" and cols_st[0] in ('1', '3'):
            reward += 30
        elif action == "right" and cols_st[2] in ('1', '3'):
            reward += 30
        elif action == "left" and cols_st[2] in ('1', '3'):
            reward -= 20
        elif action == "right" and cols_st[0] in ('1', '3'):
            reward -= 20
    elif red_present:
        reward -= 40
        if moved_forward_now:
            reward -= 35
    else:
        reward -= 6
        if moved_forward_now:
            reward += 5

    if irs_st[1] == '2' and moved_forward_now and not is_collected:
        reward -= 160
        print("Collision risk!")

    return reward

def q_learning(current_state, new_state, action, reward, q_table):

    if current_state in q_table and not isinstance(q_table[current_state], dict):
        old = q_table[current_state]
        if isinstance(old, (tuple, list)):
            q_table[current_state] = {a: float(v) for a, v in zip(actions_set, old)}
        else:
            q_table[current_state] = {a: 0.0 for a in actions_set}

    if current_state not in q_table:
        q_table[current_state] = {a: 0.0 for a in actions_set}

    if new_state not in q_table:
        q_table[new_state] = {a: 0.0 for a in actions_set}
    elif not isinstance(q_table[new_state], dict):
        old = q_table[new_state]
        if isinstance(old, (tuple, list)):
            q_table[new_state] = {a: float(v) for a, v in zip(actions_set, old)}
        else:
            q_table[new_state] = {a: 0.0 for a in actions_set}

    q_s_a = q_table[current_state].get(action, 0.0)
    best_next = max(q_table[new_state].values()) if new_state in q_table else 0.0

    new_q = q_s_a + learning_rate * (reward + discount_factor * best_next - q_s_a)
    q_table[current_state][action] = new_q
    return new_q

def save_q_table(q_table, epoch, train_rewards, train_collected,
                 val_epochs, val_rewards, val_collected, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        'q_table': q_table,
        'epoch': epoch,
        'train_rewards': train_rewards,
        'train_collected': train_collected,
        'val_epochs': val_epochs,
        'val_rewards': val_rewards,
        'val_collected': val_collected
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[SAVE] Q‑table & history saved (epoch {epoch}) to {path}")

def load_q_table(path):
    """Load Q‑table, epoch and history. Always returns dict Q‑table."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, tuple):
            print("[LOAD] Legacy tuple format detected in root data. Starting fresh.")
            return {}, 0, [], [], [], [], []

        q_table = data.get('q_table', {})
        
        if isinstance(q_table, tuple):
            print("[LOAD] Legacy tuple format detected in Q-table. Starting fresh.")
            return {}, 0, [], [], [], [], []

        epoch = data.get('epoch', 0)
        train_rewards = data.get('train_rewards', [])
        train_collected = data.get('train_collected', [])
        val_epochs = data.get('val_epochs', [])
        val_rewards = data.get('val_rewards', [])
        val_collected = data.get('val_collected', [])
    
        migrated = 0
        for state, actions in q_table.items():
            if not isinstance(actions, dict):
                if isinstance(actions, (tuple, list)):
                    q_table[state] = {a: float(v) for a, v in zip(actions_set, actions)}
                else:
                    q_table[state] = {a: 0.0 for a in actions_set}
                migrated += 1
        if migrated:
            print(f"[MIGRATE] Converted {migrated} states from old tuple format to dicts.")

        print(f"[LOAD] Loaded Q‑table from epoch {epoch}")
        return q_table, epoch, train_rewards, train_collected, val_epochs, val_rewards, val_collected
    else:
        print("[LOAD] No existing Q‑table found, starting fresh.")
    
    return {}, 0, [], [], [], [], []

_food_script_handle = None

def _find_food_script_handle(rob, verbose=True):
    global _food_script_handle
    if _food_script_handle is not None:
        return _food_script_handle

    try:
        all_handles = rob._sim.getObjectsInTree(rob._sim.handle_scene, rob._sim.handle_all, 1)
    except Exception as e:
        print(f"[_find_food_script_handle] getObjectsInTree failed: {e}")
        return None

    for h in all_handles:
        try:
            script_handle = rob._sim.getScript(rob._sim.scripttype_childscript, h)
        except Exception:
            continue
        if not script_handle or script_handle == -1:
            continue
        try:
            result = rob._sim.callScriptFunction("remote_get_collected_food", script_handle, [], [], [], "")
            if verbose:
                alias = None
                try:
                    alias = rob._sim.getObjectAlias(h, 0)
                except Exception:
                    pass
                print(f"[_find_food_script_handle] found working script on object handle={h} (alias={alias!r}), script_handle={script_handle}")
            _food_script_handle = script_handle
            return _food_script_handle
        except Exception:
            continue

    if verbose:
        print("[_find_food_script_handle] WARNING: no object's child script responded to remote_get_collected_food. "
              "Check that food.lua is actually loaded as a child script in this scene.")
    return None

def count_collected_food(rob) -> int:
    script_handle = _find_food_script_handle(rob)
    if script_handle is None:
        return 0
    try:
        result = rob._sim.callScriptFunction("remote_get_collected_food", script_handle, [], [], [], "")
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

def reset_food_collected_counter(rob):
    global _food_script_handle
    _food_script_handle = None
    _find_food_script_handle(rob, verbose=False)

def count_total_food(rob) -> int:
    try:
        all_handles = rob._sim.getObjectsInTree(rob._sim.handle_scene, rob._sim.handle_all, 1)
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

def validate_policy(q_table, rob, duration=VALIDATION_DURATION, is_hardware=False):
    total_reward = 0

    if not is_hardware:
        rob.play_simulation()
        reset_food_collected_counter(rob)
        try:
            pos, rot = rob.get_position(), rob.get_orientation()
            rot.pitch = random.uniform(-1.5, 1.5)
            rob.set_position(pos, rot)
        except Exception as e:
            print(f"[validate_policy] could not randomize pose: {e}")

    for _ in range(20):
        try:
            rob.set_phone_tilt_blocking(90, 100)
            break
        except Exception:
            time.sleep(0.01)

    start = time.time()
    current_state = get_state_key(rob.read_irs(), rob.read_image_front())
    prev_move = None
    last_collected = 0
    total_food = None

    while time.time() - start < duration and rob.is_running() and not rob.is_stopped():
        try:
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
                new_state,
                merge_irs(new_irs),
                moved_forward=(prev_move == "forward_full" and action == "forward_full"),
                moved_forward_now=action == "forward_full",
                action=action
            )
            prev_move = action
            total_reward += reward
            current_state = new_state

            if not is_hardware:
                collected_now = count_collected_food(rob)
                if total_food is None:
                    total_food = count_total_food(rob) or 7
                if collected_now > last_collected:
                    print(f"[VALIDATION] Collected: {collected_now}/{total_food}")
                    last_collected = collected_now
        except Exception:
            traceback.print_exc()
            continue

    if not is_hardware:
        collected = count_collected_food(rob)
        rob.stop_simulation()
    else:
        collected = 0

    return total_reward, collected

def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)

def plot_training_reward(epochs, rewards, save_path):
    ensure_plot_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, rewards, 'b-o', markersize=4, linewidth=1.5, label='Avg Reward per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Reward')
    plt.title('Training Progress – Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Training reward saved to {save_path}")

def plot_training_collected(epochs, collected, save_path):
    ensure_plot_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, collected, 'g-s', markersize=4, linewidth=1.5, label='Avg Collected per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Collected Items')
    plt.title('Training Progress – Collected Items')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Training collected saved to {save_path}")

def plot_validation_reward(val_epochs, val_rewards, save_path):
    ensure_plot_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(val_epochs, val_rewards, 'r-o', markersize=6, linewidth=2, label='Validation Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Reward')
    plt.title('Validation Performance – Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Validation reward saved to {save_path}")

def plot_validation_collected(val_epochs, val_collected, save_path):
    ensure_plot_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(val_epochs, val_collected, 'm-D', markersize=6, linewidth=2, label='Validation Collected')
    plt.xlabel('Epoch')
    plt.ylabel('Collected Items')
    plt.title('Validation Performance – Collected Items')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Validation collected saved to {save_path}")

def choose_action(state, q_table, epsilon):
    if random.random() < epsilon:
        return random.choice(actions_set)
    if state in q_table and isinstance(q_table[state], dict) and q_table[state]:
        max_q = max(q_table[state].values())
        best = [a for a, q in q_table[state].items() if q == max_q]
        return random.choice(best)
    return random.choice(actions_set)

def run_episode(rob, q_table, epsilon, is_hardware=False):
    """Run one episode, returns (total_reward, collected_items)."""
    if not is_hardware:
        rob.play_simulation()
        reset_food_collected_counter(rob)
        try:
            pos, rot = rob.get_position(), rob.get_orientation()
            rot.pitch = random.uniform(-1.5, 1.5)
            rob.set_position(pos, rot)
        except Exception as e:
            print(f"[run_episode] couldn't randomize pose: {e}")

    for _ in range(20):
        try:
            rob.set_phone_tilt_blocking(90, 100)
            break
        except Exception:
            time.sleep(0.01)

    current_state = get_state_key(rob.read_irs(), rob.read_image_front())
    prev_action = None
    total_reward = 0

    for step in range(STEPS_PER_EPISODE):
        if not rob.is_running() or rob.is_stopped():
            break
        try:
            action = choose_action(current_state, q_table, epsilon)
            do_move(rob, action, is_hardware=is_hardware)

            new_irs = rob.read_irs()
            new_image = rob.read_image_front()
            new_state = get_state_key(new_irs, new_image)

            reward = calculate_reward(
                new_state,
                merge_irs(new_irs),
                moved_forward=(prev_action == "forward_full" and action == "forward_full"),
                moved_forward_now=(action == "forward_full"),
                action=action
            )

            q_learning(current_state, new_state, action, reward, q_table)
            total_reward += reward
            current_state = new_state
            prev_action = action
        except Exception:
            traceback.print_exc()
            continue

    if not is_hardware:
        collected = count_collected_food(rob)
        rob.stop_simulation()
    else:
        collected = 0

    return total_reward, collected
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

def train(rob, is_hardware=False):
    (q_table, start_epoch,
     train_rewards, train_collected,
     val_epochs, val_rewards, val_collected) = load_q_table(QTABLE_PATH)

    if len(train_rewards) != start_epoch or len(train_collected) != start_epoch:
        print("[WARN] History length mismatch, resetting history lists.")
        train_rewards = []
        train_collected = []
        val_epochs = []
        val_rewards = []
        val_collected = []

    print(f"Resuming from epoch {start_epoch + 1}")

    epsilon = EPSILON_START
    for _ in range(1, start_epoch + 1):
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} (ε={epsilon:.3f}) ---")
        epoch_reward = 0.0
        epoch_collected = 0

        for ep in range(1, EPISODES_PER_EPOCH + 1):
            ep_reward, ep_collected = run_episode(rob, q_table, epsilon, is_hardware)
            epoch_reward += ep_reward
            epoch_collected += ep_collected
            print(f"  Episode {ep}/{EPISODES_PER_EPOCH} | reward: {ep_reward:.1f} | collected: {ep_collected}")

        avg_reward = epoch_reward / EPISODES_PER_EPOCH
        avg_collected = epoch_collected / EPISODES_PER_EPOCH

        train_rewards.append(avg_reward)
        train_collected.append(avg_collected)

        print(f"Epoch {epoch} | avg reward: {avg_reward:.1f} | avg collected: {avg_collected:.2f}")

        save_q_table(q_table, epoch, train_rewards, train_collected,
                     val_epochs, val_rewards, val_collected, QTABLE_PATH)

        plot_training_reward(range(1, epoch+1), train_rewards, TRAIN_REWARD_PLOT)
        plot_training_collected(range(1, epoch+1), train_collected, TRAIN_COLL_PLOT)

        if epoch % VALIDATE_EVERY == 0:
            print("Running validation...")
            val_reward, val_collected_count = validate_policy(q_table, rob, duration=VALIDATION_DURATION, is_hardware=is_hardware)
            val_epochs.append(epoch)
            val_rewards.append(val_reward)
            val_collected.append(val_collected_count)
            print(f"Validation epoch {epoch} | reward: {val_reward:.1f} | collected: {val_collected_count}")

            if val_epochs:
                plot_validation_reward(val_epochs, val_rewards, VAL_REWARD_PLOT)
                plot_validation_collected(val_epochs, val_collected, VAL_COLL_PLOT)

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    print("Training finished.")

if __name__ == "__main__":
    rob = SimulationRobobo()
    train(rob, is_hardware=False)