import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from tqdm import tqdm
from lccv import LCCV
from IPL import IPL
from surrogate_model import SurrogateModel

DATASET_IDX = 6  # Change this to load another dataset


def get_paths(data_idx):
    # Function for creating working path to load in files
    working_dir = os.getcwd()
    if working_dir.split('\\')[-1] != "Assignment2_LearningCurves":
        path = os.path.join(working_dir, "Assignment2_LearningCurves")
    else:
        path = working_dir

    config_path = os.path.join(path, "lcdb_config_space_knn.json")  # Path to .json configuration file
    performance_path = os.path.join(path, f"config_performances_dataset-{data_idx}.csv")  # Path to .csv dataset file

    return config_path, performance_path


def parse_args():
    config_path, performance_path = get_paths(DATASET_IDX)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default=config_path)
    parser.add_argument('--configurations_performance_file', type=str, default=performance_path)
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--minimal_anchor', type=int, default=256)
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=6)

    return parser.parse_args()


def run(args):

    df = pd.read_csv(args.configurations_performance_file)  # Load in data

    # Set max and min anchor size based on the dataset
    args.max_anchor_size = df['anchor_size'].max()
    args.minimal_anchor = df['anchor_size'].min()

    # Config space + external surrogate
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    # IPL vertical evaluator, we give it an effectively infinite budget
    ipl_eval = IPL(surrogate_model, args.minimal_anchor, args.max_anchor_size, budget=float('inf'))
    best_so_far = None

    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = ipl_eval.evaluate_model(best_so_far, theta_new)

        # IPL may early-stop: last point is either final anchor or last early anchor
        final_result = result[-1][1]
        if best_so_far is None or final_result < best_so_far:
            best_so_far = final_result

        x_values = [a for (a, _) in result]
        y_values = [s for (_, s) in result]

        if len(result) == 1:
            # Only evaluated at final anchor (no early LC) → draw a horizontal line
            plt.hlines(y_values, xmin=args.minimal_anchor, xmax=args.max_anchor_size, linestyles="--")
            plt.scatter([args.max_anchor_size], y_values, marker='o')
        else:
            # Plot partial learning curve used by IPL
            plt.plot(x_values, y_values, linestyle="--")
            # Mark last evaluated anchor (could be early or final)
            plt.scatter(x_values[-1], y_values[-1], marker='o')

    plt.xlabel('Anchor size', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(visible=True, which='both', alpha=0.3)
    # plt.title(f'IPL learning curves on dataset {DATASET_IDX}')
    plt.show()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
