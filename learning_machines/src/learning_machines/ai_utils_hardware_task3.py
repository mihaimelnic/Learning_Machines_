import os
import cv2
import time
import pickle
import numpy as np
from robobo_interface import SimulationRobobo, HardwareRobobo

QTABLE_DIR = "/root/results/figures/results/"
QTABLE_PATH = os.path.join(QTABLE_DIR, "q_hhh.pkl")
collision_threshold = 90
green_threshold = 10
red_threshold = 10
ir_near = 200
ir_mid   = 80
actions_set = ["forward_full", "right", "left"]
learning_rate = 0.1
discount_factor = 0.9

def do_move(rob, action):
    """Execute one action step. Works for both sim and hardware."""
    is_hardware = isinstance(rob, HardwareRobobo)
    l_speed, r_speed = 20, 20
    duration_ms = 500 

    if action == "right":
        l_speed, r_speed = 20, -20
        duration_ms = 400
    elif action == "left":
        l_speed, r_speed = -20, 20
        duration_ms = 400

    if is_hardware:
        try:
            rob.move(l_speed, r_speed, duration_ms)
        except AttributeError:
            from std_msgs.msg import Int8, Int16, Int32
            rob._move_srv(Int8(l_speed), Int8(r_speed),
                          Int32(duration_ms), Int16(1))
        time.sleep(duration_ms / 1000.0)
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

# This is IR processing.
def merge_irs(irs):
    """Blend raw IR readings into three directional zones."""
    def safe(i):
        v = irs[i]
        return float(v) if v else 0.0

    front_left   = (safe(7) * 3 + safe(2) + safe(4)) / 3.0
    front_center = (safe(2) + safe(4) * 1.5 + safe(3)) / 3.0
    front_right  = (safe(4) + safe(3) + safe(5) * 3) / 3.0
    return [front_left, front_center, front_right]


def obtain_irs(irs):
    """Return the single front-centre IR value."""
    return float(irs[4]) if irs[4] else 0.0


def _ir_bin(v):
    if v > ir_near:
        return 2    # very close -> about to collide.
    elif v > ir_mid:
        return 1    # mid-range.
    return 0        # nothing nearby.

# camera strategy for 5 zones
ZONES_5 = [0.15, 0.15, 0.40, 0.15, 0.15]   # max_left, left, center, right, and max_right.

def _zone_of_blob(image, lower_hsv, upper_hsv, zones=ZONES_5):
    """
    Find the largest blob of the given colour and return:
        (found: bool, zone_index: int)
    Zone index 0..4 corresponding to the 5 horizontal slices.
    If no blob is found → (False, -1).
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Build mask from possibly two HSV ranges (handles red wrap‑around).
    if isinstance(lower_hsv, list):
        mask = cv2.inRange(hsv, lower_hsv[0], upper_hsv[0])
        for lo, hi in zip(lower_hsv[1:], upper_hsv[1:]):
            mask |= cv2.inRange(hsv, lo, hi)
    else:
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Find contours of the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, -1

    # Largest contour by area.
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return False, -1
    cx = int(M["m10"] / M["m00"])

    # Map centroid x to zone index.
    cum = 0.0
    for idx, frac in enumerate(zones):
        cum += frac * w
        if cx < cum:
            return True, idx
    return True, len(zones) - 1   # last bin (max_right).


def get_green_zone(image):
    """Return (found, zone_index) for green objects."""
    lower_green = np.array([45, 70, 70])
    upper_green = np.array([85, 255, 255])
    return _zone_of_blob(image, lower_green, upper_green)


def get_red_zone(image):
    """Return (found, zone_index) for red objects (handles hue wrap)."""
    lower_red = [np.array([0, 70, 70]), np.array([170, 70, 70])]
    upper_red = [np.array([10, 255, 255]), np.array([180, 255, 255])]
    return _zone_of_blob(image, lower_red, upper_red)


# Reward shaping, where staged: red first → red+green.
def calculate_reward(state_tuple, center_ir, action, food_delta=0):
    """
    state_tuple : output of get_state_key(), e.g. ("012_RG_RZ2_GZ2",)
    center_ir   : float from obtain_irs()
    action      : string from actions_set
    food_delta  : int, how many food items were just delivered
    """
    state = state_tuple[0]
    parts = state.split("_")
    ir_str    = parts[0]         
    colour    = parts[1] if len(parts) > 1 else "F"

    # Safe extraction of zone indices.
    red_zone = None
    green_zone = None
    if len(parts) > 2 and parts[2].startswith("RZ"):
        zone_str = parts[2][2:]   # e.g. "2" or "N"
        red_zone = int(zone_str) if zone_str.isdigit() else None
    if len(parts) > 3 and parts[3].startswith("GZ"):
        zone_str = parts[3][2:]
        green_zone = int(zone_str) if zone_str.isdigit() else None

    ir_c = ir_str[1] if len(ir_str) >= 2 else "0"
    moved_forward = (action == "forward_full")
    reward = 0

    # Food delivery this is setted as highest priority.
    if food_delta > 0:
        reward += 100 * food_delta
        print(f"[reward] DELIVERED {food_delta} food item(s)!")

    # Colour-based shaping with staged priorities.
    if colour == "CG":               # carrying AND green zone visible -> go!
        reward += 20
        if moved_forward:
            reward += 30
        if green_zone == 2:          # centre zone.
            reward += 15

    elif colour == "C":              # carrying, no green visible.
        reward += 5
        if moved_forward:
            reward += 5

    elif colour == "R":              # red only (no green).
        reward += 10
        if red_zone == 2:
            reward += 15             # centre bonus.
            if ir_c == "2" and moved_forward:
                reward += 25         # push when centred and close.
        elif red_zone in (1, 3):     # near centre.
            reward += 5
        elif red_zone in (0, 4):     # extreme edges – steer back.
            reward -= 5
        if moved_forward and red_zone not in (2, 1, 3):
            reward -= 10

    elif colour == "RG":             # BOTH red and green visible!
        # High priority: we want the robot to push red into the green zone.
        reward += 20                 # base for seeing both.
        if red_zone == 2:
            reward += 20             # red centred is very good.
            if green_zone == 2:
                reward += 20         # both centred = perfect alignment.
            # Push when red is close and green is also somewhere visible.
            if ir_c == "2" and moved_forward:
                reward += 35         # massive push reward.
        elif red_zone in (1, 3):
            reward += 10
        elif red_zone in (0, 4):
            reward -= 5
        # Penalise moving forward when not well aligned.
        if moved_forward and red_zone not in (2, 1, 3):
            reward -= 15

    elif colour == "G":              # green only; shouldn't happen if red exists.
        reward -= 5                  # discourage chasing green without red.
        if moved_forward:
            reward -= 10

    elif colour == "F":              # nothing relevant visible
        reward -= 2
        if moved_forward:
            reward -= 2

    # collision avoidance: penalise driving into obstacles.
    if ir_c == "2" and moved_forward and colour not in ("CG", "RG"):
        reward -= 15

    return reward


# State builder -> colour flag now shows "RG" when both are visible.
def get_state_key(irs, green_img, red_img, food_collected=0):
    """
    Build a discrete state tuple from sensor readings.
    State format: "irLirCirR_colour_RZrz_GZgz"
      colour: "R", "G", "RG", "CG", "C", "F"
      rz, gz: zone indices (0-4) or "N" if not found.
    """
    m = merge_irs(irs)
    ir_l = _ir_bin(m[0])
    ir_c = _ir_bin(m[1])
    ir_r = _ir_bin(m[2])

    # Multi‑zone colour detection.
    red_found, red_z = get_red_zone(red_img)
    green_found, green_z = get_green_zone(green_img)

    # Convert zone indices to strings -> use "N" for none.
    rz_str = str(red_z) if red_found else "N"
    gz_str = str(green_z) if green_found else "N"

    # Decide global colour flag -> now distinguishes "RG" when both visible.
    if food_collected > 0:
        colour = "CG" if green_found else "C"
    else:
        if red_found and green_found:
            colour = "RG"
        elif red_found:
            colour = "R"
        elif green_found:
            colour = "G"
        else:
            colour = "F"

    state_str = f"{ir_l}{ir_c}{ir_r}_{colour}_RZ{rz_str}_GZ{gz_str}"
    return (state_str,)


# Q‑learning core
def q_learning(current_state, new_state, action, reward, q_table):
    """Standard Q-learning update (Bellman equation)."""
    if current_state not in q_table:
        q_table[current_state] = {a: 0.0 for a in actions_set}
    q_s_a = q_table[current_state].get(action, 0.0)
    best_next = 0.0
    if new_state in q_table and q_table[new_state]:
        best_next = max(q_table[new_state].values())
    q_table[current_state][action] = (
        q_s_a + learning_rate * (reward + discount_factor * best_next - q_s_a)
    )
    return q_table[current_state][action]


def save_q_table(q_table, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(q_table, f)


def load_q_table(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print(f"[load_q_table] no file at {path}; starting fresh")
    return {}