import multiprocessing
from pathlib import Path
from train import run_trial

if __name__ == '__main__':
    multiprocessing.freeze_support()

    OUT_DIR = 'outputs/clean'
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    params_to_try = [
        # Optimal
        {'name': 'opt', 'times': 2, 'ACTION_SELECTION_METHOD': 'optimal_policy', 'USE_SOFT_REWARDS': True, 'epsilons': [0.3, 0.2, 0.1, 0.05], 'num_episodes': 1000},

        # e-greedy
        {'name': 'eg', 'times': 2, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 0, 'ENV_REWARD_SCALE': 1, 'num_episodes': 1000, 'alpha': 1e-3, 'alpha_h': 0.1},

        # TAMER
        {'name': 'tamer', 'times': 2, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 1, 'ENV_REWARD_SCALE': 0, 'num_episodes': 1000, 'alpha': 1e-3, 'alpha_h': 0.1},

        # Reward Shaping
        {'name': 'rs', 'times': 2, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 1, 'ENV_REWARD_SCALE': 0.1, 'num_episodes': 1000, 'alpha': 1e-3, 'alpha_h': 0.1},

        # Action Biasing
        {'name': 'ab', 'times': 2, 'ACTION_SELECTION_METHOD': 'action_biasing', 'DECAY_PARAM': 0.999, 'HUMAN_REWARD_SCALE': 0, 'ENV_REWARD_SCALE': 1, 'num_episodes': 1000, 'alpha': 1e-3, 'alpha_h': 0.1},

        # Action Biasing + Reward Shaping
        {'name': 'abrs', 'times': 2, 'ACTION_SELECTION_METHOD': 'action_biasing', 'DECAY_PARAM': 0.999, 'HUMAN_REWARD_SCALE': 1, 'ENV_REWARD_SCALE': 0.1},

        # Control Sharing
        {'name': 'cs', 'times': 2, 'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 1.0, 'HUMAN_REWARD_SCALE': 0, 'ENV_REWARD_SCALE': 1, 'num_episodes': 1000, 'alpha': 1e-3, 'alpha_h': 0.1},
    ]

    pool = multiprocessing.Pool(8)
    pool.map(run_trial, params_to_try)
