import argparse
import ConfigSpace
import pandas as pd

from tqdm import tqdm
from scipy.stats import spearmanr
from surrogate_model import SurrogateModel
from sklearn.model_selection import train_test_split



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='Assignment2_LearningCurves/lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='Assignment2_LearningCurves/config_performances_dataset-6.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()


def run(args):

    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Split data into train/test split. 
    train, test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=2025)
    train['score'] = y_train
    test['score'] = y_test

    # Instantiate surrogate model and fit to train data
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(train)

    # Test surrogate model
    y_pred = []
    for i in tqdm(range(len(test))):
        prediction = surrogate_model.predict(test.iloc[i, :-2], test.iloc[i, -2])
        y_pred.append(prediction)

    r = spearmanr(y_pred, test.iloc[:, -1])
    print('Pearson-r correlaton: ', f"{r}")




if __name__ == '__main__':
    run(parse_args())
