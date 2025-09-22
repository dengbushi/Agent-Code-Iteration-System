import random
from pathlib import Path
import numpy as np


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


def set_seed(seed=531):
    random.seed(seed)
    np.random.seed(seed)


set_seed()


config = {
    "exp_name": "ML2025_HW2",
    "data_dir": Path("./dataset").resolve(),
    "task_goal": "Given the survey results from the past two days in a specific state in the U.S.,\
                  predict the probability of testing positive on day 3. \
                  The evaluation metric is Mean Squared Error (MSE).",
    "agent": {
        "steps": 1,
        "search": {
            "debug_prob": 0.5,
            "num_drafts": 1,
        },
    },
}

cfg = Config(config)


