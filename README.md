# RL Final Project Code
Kenneth Mitra,
Yancheng Du

### Running the code
To install the required python libraries on your system, run `pip install -r requirements.txt`

To begin running all trials and collecting results, run `train_parallel.py`
## File Structure

### Outputs directory
```
outputs/
├── noisy0/             # Results of trials with noise probability of 0.0
├── noisy1/             # Results of trials with noise probability of 0.1
├── noisy2/             # Results of trials with noise probability of 0.2
├── noisy3/             # Results of trials with noise probability of 0.3
└── noisy4/             # Results of trials with noise probability of 0.4
```

### Main Directory Code
```
.
├── config.json                     # JSON configuration file specifying RoboTaxi environment parameters
├── env.py                          # RoboTaxi environment modified from previous research work
├── train.py                        # Script containing code to run algorithms tested in the paper. Contains run_trial function which accepts a config dictionary and saves trial results to disk
├── train_parallel.py               # Script that iterates over all configurations to run and runs trials in parallel with multiprocessing
└── util.py                         # Contains utility functions for manipulating vectors of observations and states
```

### Old Code / Preliminary Attempts
```
.
├── basic_dqn.py            # Script used to test stable-baselines 3 implementation of algorithms on different environments
├── basic_qlearning.py      # Script used to test q-learning with a neural network in PyTorch
└── Net.py                  # Neural network used in older code
```