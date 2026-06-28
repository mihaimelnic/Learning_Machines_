from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)
from robobo_interface import SimulationRobobo, HardwareRobobo

# from .nn_test import reinforcement_learning
# from .nn_newest import reinforcement_learning

def run_all_actions(rob: IRobobo):
    # rob = SimulationRobobo()
    # rob.play_simulation()
    # rob.move_blocking(50, -50, 300)
    # rob.stop_simulation()
    # 
    reinforcement_learning(rob)
