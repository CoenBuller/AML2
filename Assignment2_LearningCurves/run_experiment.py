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

    budget = 10 * args.max_anchor_size # Chose an arbitrary budget

    # print(f"Available budget = {budget}")


    # ------------ Evaluate LCCV with this budget -------------

    # print("--------------- Testing LCCV -------------- \n \n")

    lccv_results_df = pd.DataFrame([], columns=['anchor', 'score', 'config_id'])

    lccv = LCCV(surrogate_model=surrogate_model, minimal_anchor=args.minimal_anchor, final_anchor=args.max_anchor_size, budget=budget)
    best_f = None
    lccv_id = 0
    while lccv.budget > df['anchor_size'].unique().sum():
        config = config_space.sample_configuration()

        configs_best = lccv.evaluate_model(best_so_far=best_f, conf=config) # type: ignore
        if configs_best == None:
            raise ValueError("Surrogate model is not trained yet. Do that first.")

        if best_f is None or configs_best[-1][1] < best_f:
            best_f = configs_best[-1][1]

        # Create own learning curve
        config_tuple = tuple(config.values())
        theta_dict = {'score': 0., 'anchor': 0, 'config_id':0}
        for anchor, score in lccv.results[config_tuple]:
            theta_dict['score'] = score
            theta_dict['anchor'] = anchor
            theta_dict['config_id'] = lccv_id
            lccv_results_df.loc[len(lccv_results_df)] = theta_dict #type: ignore

        lccv_id += 1
        

    if filename is not None:
        filename_lccv = filename + "_lccv.csv"
        lccv_results_df.to_csv(filename+"_lccv.csv", index=False) # Save dataframe

    # print(f"Best score using LCCV: {best_f}")
    # print(f"Cost: {budget - lccv.budget}")

    
    # ------------ Evaluate IPL with this budget -------------

    # print("--------------- Testing IPL -------------- \n \n")

    ipl_results_df = pd.DataFrame([], columns=['anchor', 'score', 'config_id'])
    best_ipl = None
    anchors = np.linspace(args.minimal_anchor, int(0.4*args.max_anchor_size), 5).astype(np.int32) # Anchor sizes that evaluate per configuration
    ipl = IPL(surrogate_model=surrogate_model, minimal_anchor=args.minimal_anchor, final_anchor=args.max_anchor_size, budget=budget, anchors=anchors)
        
    ipl_id = 0
    while ipl.budget > np.sum(anchors):
        config = config_space.sample_configuration()
        r = ipl.evaluate_model(best_so_far=best_ipl, conf=dict(config))
        if best_ipl is None:
            best_ipl = r[-1][1]
        else:
            best_ipl = r[-1][1] if r[-1][1] < best_ipl else best_ipl

        # Create own learning curve
        theta_dict = {'score': 0., 'anchor': 0, 'config_id':0}
        for anchor, score in ipl.results[tuple(config.values())]:
            theta_dict['score'] = score
            theta_dict['anchor'] = anchor
            theta_dict['config_id'] = ipl_id
            ipl_results_df.loc[len(ipl_results_df)] = theta_dict #type: ignore

        ipl_id += 1

    if filename is not None:
        filename_ipl = filename+"_ipl.csv"
        ipl_results_df.to_csv(filename_ipl, index=False) # Save dataframe
    
    # print(f"Best score using IPL: {best_ipl}")
    # print(f"Cost: {budget - ipl.budget}")

    random_budget = budget
    best_random = float('inf')
    cost = budget - args.max_anchor_size*int(budget/args.max_anchor_size)
    samples = config_space.sample_configuration(int(budget/args.max_anchor_size))
    for sample in samples:
        eval = surrogate_model.predict(sample, args.max_anchor_size)
        if eval < best_random:
            best_random = eval


    # Always return a tuple of filenames (or (None, None) if filename not provided)
    if filename is not None:
        return filename_lccv, filename_ipl, (budget - lccv.budget, best_f, lccv_id+1), (budget - ipl.budget, best_ipl, ipl_id+1), (cost, best_random, int(budget/args.max_anchor_size))  # type: ignore
    return None, None, (budget - lccv.budget, best_f, lccv_id+1), (budget - ipl.budget, best_ipl, ipl_id+1), (cost, best_random, int(budget/args.max_anchor_size))




if __name__ == '__main__':
    lccv_dict = {'best_f': [], 'cost': [], 'num_hpc': []}
    ipl_dict = {'best_f': [], 'cost': [], 'num_hpc': []}
    random_dict = {'best_f': [], 'cost': [], 'num_hpc': []}

    dataset_id = 6
    print(f'Available budget = {10 * 16000}')
    for _ in tqdm(range(20)):

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
        # ax[0].set_xlabel('anchor_size')
        # ax[1].set_xlabel('anchor_size')
        # ax[0].set_ylabel('score')

        # ax[0].set_title(f'LCCV on dataset {dataset_id}')
        # ax[1].set_title(f'IPL on dataset {dataset_id}')

        lccv_file, ipl_file, lccv, ipl, random = run(parse_args(dataset_idx=dataset_id))


        lccv_dict['cost'] += [lccv[0]]
        lccv_dict['best_f'] += [lccv[1]]
        lccv_dict['num_hpc'] += [lccv[2]]
        
        ipl_dict['cost'] += [ipl[0]]
        ipl_dict['best_f'] += [ipl[1]]
        ipl_dict['num_hpc'] += [ipl[2]]

        random_dict['cost'] += [random[0]]
        random_dict['best_f'] += [random[1]]
        random_dict['num_hpc'] += [random[2]]
        # if lccv_file:
        #     lccv_df = pd.read_csv(lccv_file)
        #     for id in lccv_df['config_id'].unique():
        #         selection = (lccv_df['config_id'] == id )
        #         scores = lccv_df[selection]['score'].reset_index(drop=True)
        #         anchors = lccv_df[selection]['anchor'].reset_index(drop=True)
        #         sort = np.argsort(anchors)
        #         if id == 0:
        #             ax[0].hlines(scores, xmin=0, xmax=lccv_df['anchor'].max(), linestyle='--')
        #         else:
        #             ax[0].plot(anchors[sort], scores[sort], linestyle=':')

        # if ipl_file:
        #     ipl_df = pd.read_csv(ipl_file)
        
        #     for id in ipl_df['config_id'].unique():
        #         selection = (ipl_df['config_id'] == id )
        #         scores = ipl_df[selection]['score'].reset_index(drop=True)
        #         anchors = ipl_df[selection]['anchor'].reset_index(drop=True)
        #         sort = np.argsort(anchors)
        #         if id == 0:
        #             ax[1].hlines(scores, xmin=0, xmax=ipl_df['anchor'].max(), linestyle='--')
        #         else:
        #             ax[1].plot(anchors[sort], scores[sort], linestyle=':')
        # plt.show()

    print(f"---------- Experiment results for dataset {dataset_id} --------------- \n")
    print(f"""
    LCCV results \n
    |       | mean | std  | \n 
    | score | {round(np.mean(lccv_dict['best_f']), 3)} | {round(np.std(lccv_dict['best_f']), 3)} | \n
    | cost  | {round(np.mean(lccv_dict['cost']), 3)} | {round(np.std(lccv_dict['cost']), 3)} | \n 
    | #hpc  | {round(np.mean(lccv_dict['num_hpc']), 3)} | {round(np.std(lccv_dict['num_hpc']), 3)} | \n
    """)
        
    print(f"""
    IPL results \n 
    |       | mean | std  | \n 
    | score | {round(np.mean(ipl_dict['best_f']), 3)} | {round(np.std(ipl_dict['best_f']), 3)} | \n
    | cost  | {round(np.mean(ipl_dict['cost']), 3)} | {round(np.std(ipl_dict['cost']), 3)} | \n 
    | #hpc  | {round(np.mean(ipl_dict['num_hpc']), 3)} | {round(np.std(ipl_dict['num_hpc']), 3)} | \n
    """)
        
    print(f"""
    Random results \n 
    |       | avg  | std  | \n 
    | score | {round(np.mean(random_dict['best_f']), 3)} | {round(np.std(random_dict['best_f']), 3)} | \n
    | cost  | {round(np.mean(random_dict['cost']), 3)} | {round(np.std(random_dict['cost']), 3)} | \n
    | #hpc  | {round(np.mean(random_dict['num_hpc']), 3)} | {round(np.std(random_dict['num_hpc']), 3)} | \n
 
    """)

