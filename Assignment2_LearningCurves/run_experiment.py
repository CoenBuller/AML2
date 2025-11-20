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

# Used for making a df
enc_defaults = {'p': 1, 'pp@kernel_pca_kernel': 'linear', 'pp@kernel_pca_n_components': 0.25, 'pp@poly_degree':2, 'pp@std_with_std':True, 'pp@selectp_percentile':25}


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

def parse_args(dataset_idx=6):
    config_path, performance_path = get_paths(dataset_idx)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default= config_path)
    parser.add_argument('--configurations_performance_file', type=str, default= performance_path)
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--minimal_anchor', type=int, default=256)
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=20)

    return parser.parse_args()


def run(args, filename: str | None =None):
    df = pd.read_csv(args.configurations_performance_file) # Load in data

    # Set max, and min anchorsize based on the dataset as this might differ between them
    args.max_anchor_size = df['anchor_size'].max() 
    args.minimal_anchor = df['anchor_size'].min()

    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    config_default_values = config_space.get_default_configuration()
    HP_space = list(config_space.keys())

    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    budget = 50 * args.max_anchor_size # Chose an arbitrary budget

    print(f"Available budget = {budget}")


    # ------------ Evaluate LCCV with this budget -------------

    print("--------------- Testing LCCV -------------- \n \n")

    lccv_results_df = pd.DataFrame([], columns=list(config_space.keys()) + ['anchor_size', 'score', 'config_id'])

    lccv = LCCV(surrogate_model=surrogate_model, minimal_anchor=args.minimal_anchor, final_anchor=args.max_anchor_size, budget=budget)
    best_f = None
    config_id = 0
    while lccv.budget > df['anchor_size'].unique().sum():
        config = config_space.sample_configuration()

        configs_best = lccv.evaluate_model(best_so_far=best_f, conf=config) # type: ignore
        if configs_best == None:
            print('First train the surrogate model')
            break
        if best_f is None:
            best_f = configs_best
        else:
            best_f = configs_best if configs_best < best_f else best_f # type: ignore

        # Create own learning curve
        for anchor, score in lccv.results[tuple(config.values())]:
            theta_dict = {hp: config_default_values.get(hp, None) for hp in HP_space} 
            theta_dict.update(enc_defaults)
            theta_dict.update(config)
            theta_dict['score'] = score
            theta_dict['anchor'] = anchor
            theta_dict['config_id'] = config_id
        
        lccv_results_df.loc[len(lccv_results_df)] = theta_dict #type: ignore

    if filename is not None:
        lccv_results_df.to_csv(filename+"_lccv.csv") # Save dataframe

    print(f"Best score using LCCV: {best_f}")
    print(f"Cost: {budget - lccv.budget}")

    
    # ------------ Evaluate IPL with this budget -------------

    print("--------------- Testing IPL -------------- \n \n")

    ipl_results_df = pd.DataFrame([], columns=list(config_space.keys()) + ['anchor_size', 'score', 'config_id'])

    best_ipl = None
    anchors = np.linspace(args.minimal_anchor, int(0.5*args.max_anchor_size), 5).astype(np.int32) # Anchor sizes that evaluate per configuration
    ipl = IPL(surrogate_model=surrogate_model, minimal_anchor=args.minimal_anchor, final_anchor=args.max_anchor_size, budget=budget, anchors=anchors)
        
    config_id = 0
    while ipl.budget > np.sum(anchors):
        config = config_space.sample_configuration()
        r = ipl.evaluate_model(best_so_far=best_ipl, conf=dict(config))
        if best_ipl is None:
            best_ipl = r[-1][1]
        else:
            best_ipl = r[-1][1] if r[-1][1] < best_ipl else best_ipl

        # Create own learning curve
        for anchor, score in ipl.results[tuple(config.values())]:
            theta_dict = {hp: config_default_values.get(hp, None) for hp in HP_space} 
            theta_dict.update(enc_defaults)
            theta_dict.update(config)
            theta_dict['score'] = score
            theta_dict['anchor'] = anchor
            theta_dict['config_id'] = config_id
        
        ipl_results_df.loc[len(ipl_results_df)] = theta_dict #type: ignore

    if filename is not None:
        ipl_results_df.to_csv(filename+"_ipl.csv") # Save dataframe
    
    print(f"Best score using IPL: {best_ipl}")
    print(f"Cost: {budget - ipl.budget}")

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args(dataset_idx=6), filename='first_try')
