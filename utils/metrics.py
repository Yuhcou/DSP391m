# utils/metrics.py
import numpy as np

def compute_basic_metrics(rewards):
    rewards = np.array(rewards)
    return {
        'mean': float(np.mean(rewards)),
        'median': float(np.median(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
    }
