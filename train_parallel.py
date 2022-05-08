import multiprocessing
from pathlib import Path
from train import run_trial

if __name__ == '__main__':
    multiprocessing.freeze_support()

    times = 100
    p_human_errors = [0, 0.1, 0.2, 0.3, 0.4]

    def construct_params(times, p_human_error):
        params_to_try = [
            # Optimal
            {'name': 'opt_', 'times': times, 'ACTION_SELECTION_METHOD': 'optimal_policy', 'num_episodes': 1000},

            # e-greedy
            {'name': 'eg_', 'times': times, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 0, 'ENV_REWARD_SCALE': 1, 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},

            # TAMER
            {'name': 'tamer_', 'times': times, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 1, 'ENV_REWARD_SCALE': 0, 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},

            # Reward Shaping
            {'name': 'rs0_', 'times': times, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 1*0, 'ENV_REWARD_SCALE': 0.1*(1-0), 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},
            {'name': 'rs1_', 'times': times, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 1*0.25, 'ENV_REWARD_SCALE': 0.1*(1-0.25), 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},
            {'name': 'rs2_', 'times': times, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 1*0.5, 'ENV_REWARD_SCALE': 0.1*(1-0.5), 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},
            {'name': 'rs3_', 'times': times, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 1*0.75, 'ENV_REWARD_SCALE': 0.1*(1-0.75), 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},
            {'name': 'rs4_', 'times': times, 'ACTION_SELECTION_METHOD': 'e_greedy', 'epsilons': [0.5, 0.5, 0.5, 0.5], 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 1*1, 'ENV_REWARD_SCALE': 0.1*(1-1), 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},

            # Action Biasing
            {'name': 'ab9_', 'times': times, 'ACTION_SELECTION_METHOD': 'action_biasing', 'DECAY_PARAM': 0.9, 'HUMAN_REWARD_SCALE': 0, 'ENV_REWARD_SCALE': 1, 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},
            {'name': 'ab99_', 'times': times, 'ACTION_SELECTION_METHOD': 'action_biasing', 'DECAY_PARAM': 0.99, 'HUMAN_REWARD_SCALE': 0, 'ENV_REWARD_SCALE': 1, 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},
            {'name': 'ab999_', 'times': times, 'ACTION_SELECTION_METHOD': 'action_biasing', 'DECAY_PARAM': 0.999, 'HUMAN_REWARD_SCALE': 0, 'ENV_REWARD_SCALE': 1, 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},

            # Action Biasing + Reward Shaping
            {'name': 'abrs_', 'times': times, 'ACTION_SELECTION_METHOD': 'action_biasing', 'DECAY_PARAM': 0.99, 'HUMAN_REWARD_SCALE': 1, 'ENV_REWARD_SCALE': 0.1, 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},

            # Control Sharing
            {'name': 'cs_', 'times': times, 'ACTION_SELECTION_METHOD': 'control_sharing', 'P_HUMAN': 1.0, 'DECAY_PARAM': 1.0, 'HUMAN_REWARD_SCALE': 0, 'ENV_REWARD_SCALE': 1, 'num_episodes': 1000, 'alpha': 1e-1, 'alpha_h': 0.1, 'P_HUMAN_ERROR': p_human_error},
        ]
        return params_to_try

    all_params = []
    for idx, p_human_error in enumerate(p_human_errors):
        params = construct_params(times, p_human_error)
        for param in params:
            param['outdir'] = f"noisy{idx}"

        OUT_DIR = f'outputs/noisy{idx}'
        Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

        all_params.extend(params)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(run_trial, all_params)
