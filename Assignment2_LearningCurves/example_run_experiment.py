import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
import os

from tqdm import tqdm
from lccv import LCCV
from IPL import IPL
from surrogate_model import SurrogateModel

DATASET_IDX = 6 # Change this to corresponding idx of dataset to load in that dataset

def get_paths(data_idx):
    # Function for creating working path to load in files
    working_dir = os.getcwd()
    if working_dir.split('\\')[-1] != "Assignment2_LearningCurves":
        path = os.path.join(working_dir, "Assignment2_LearningCurves")
    else: 
        path = working_dir

    config_path = os.path.join(path, "lcdb_config_space_knn.json") # Path to .json configuration file
    performance_path = os.path.join(path, f"config_performances_dataset-{data_idx}.csv") # Path to .csv dataset file

    return config_path, performance_path

def parse_args():
    config_path, performance_path = get_paths(DATASET_IDX)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default= config_path)
    parser.add_argument('--configurations_performance_file', type=str, default= performance_path)
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--minimal_anchor', type=int, default=256)
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=20)

    return parser.parse_args()


def run(args):
    
    df = pd.read_csv(args.configurations_performance_file) # Load in data

    # Set max, and min anchorsize based on the dataset as this might differ between them
    args.max_anchor_size = df['anchor_size'].max() 
    args.minimal_anchor = df['anchor_size'].min()

    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    print(config_space.get_default_configuration())
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    lccv = IPL(surrogate_model, args.minimal_anchor, args.max_anchor_size)
    best_so_far = None
    
    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = lccv.evaluate_model(best_so_far, theta_new)
        final_result = result[-1][1]
        if best_so_far is None or final_result < best_so_far:
            best_so_far = final_result
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        plt.plot(x_values, y_values, "-o")

    plt.show()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
