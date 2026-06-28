import torch
import numpy as np

from robobo_interface import SimulationRobobo, HardwareRobobo, IRobobo
from learning_machines.dqn_train_robobo import DuelingDQN, ContinuousRoboboEnv

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def run_trained(rob: IRobobo,
                sim_speed: float = 0.3,   
                debug: bool = True):      
    """
    We run the trained D3QN model continuously.

    here are our parameters:
    rob : IRobobo
        The robot (simulation or hardware).
    sim_speed : float
        Simulation speed factor. Only used for SimulationRobobo.
        1.0 = real-time, <1 slower, 0 = max speed.
        Hardware robots ignore this.
    debug : bool
        If True, prints IR readings, action, reward, and termination info.
    """
    model_path = "/root/results/d3qn_robobo.pth"

    if isinstance(rob, SimulationRobobo):
        try:
            rob.set_simulation_speed(sim_speed)
            print(f"Simulation speed set to {sim_speed}")
        except AttributeError:
            pass

    env = ContinuousRoboboEnv(rob)

    model = DuelingDQN(state_size=8, action_size=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state = env.reset()
    print("Model loaded. Running on device:", device)
    print("Press Ctrl+C to stop.\n")

    action_names = ["Forward", "Left", "Right"]

    try:
        while True:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(model(state_t)).item()

            raw_irs = rob.read_irs()
            front_max_raw = max(raw_irs[i] for i in env.front)

            if debug:
                print(f"Front IR max: {front_max_raw:.1f} | Action: {action_names[action]} | "
                      f"IR: {[round(v, 1) for v in raw_irs]}")

            next_state, reward, done = env.step(action)

            if done:
                reason = "collision" if front_max_raw > 90 else "stuck"
                print(f"Episode ended ({reason}). Resetting...")
                state = env.reset()
            else:
                state = next_state

    except KeyboardInterrupt:
        print("\nStopped by user.")


def run_all_actions(rob: IRobobo):
    print("Running trained D3QN model...")
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    run_trained(rob, sim_speed=0.3, debug=True)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()