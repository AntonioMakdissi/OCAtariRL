import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np

log_dir='C:/Users/Personal/Desktop/M1 research project/newOCatariproj/ALE/nomralize_FALSE_missingOb_ZERO_speeds_FALSE'

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]

    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")

name= "Learning curves"
#hard coded
curves_info = [
    {"name": "nomralize_FALSE_missingOb_NEG1_speeds_FALSE", "label": "NFALSE NEG1 SFALSE"},
    {"name": "nomralize_FALSE_missingOb_NEG1_speeds_TRUE", "label": "NFALSE NEG1 STRUE"},
    {"name": "nomralize_FALSE_missingOb_ZERO_speeds_FALSE", "label": "NFALSE ZERO SFALSE"},
    {"name": "nomralize_FALSE_missingOb_ZERO_speeds_TRUE", "label": "NFALSE ZERO STRUE"},
    {"name": "nomralize_TRUE_missingOb_NEG1_speeds_FALSE", "label": "NTRUE NEG1 SFALSE"},
    {"name": "nomralize_TRUE_missingOb_NEG1_speeds_TRUE", "label": "NTRUE NEG1 STRUE"},
    {"name": "nomralize_TRUE_missingOb_ZERO_speeds_FALSE", "label": "NTRUE ZERO SFALSE"},
    {"name": "nomralize_TRUE_missingOb_ZERO_speeds_TRUE", "label": "NTRUE ZERO STRUE"},
    {"name": "ppo", "label": "pixel PPO 500K Timesteps"}
]

for curve_info in curves_info:
    log_dir = 'C:/Users/Personal/Desktop/M1 research project/newOCatariproj/ALE/' + curve_info["name"]
    label = curve_info["label"]
    plot_results(log_dir, label)
plt.legend(labels=[curve_info["label"] for curve_info in curves_info], loc='lower right', bbox_to_anchor=(1, 0))
plt.title("Learning curves")
plt.show()


