import ConfigSpace
import pandas as pd

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None
        self.enc_defaults = {'p': 1, 'pp@kernel_pca_kernel': 'linear', 'pp@kernel_pca_n_components': 0.25, 
                             'pp@poly_degree':2, 'pp@std_with_std':True, 'pp@selectp_percentile':25}

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        self.df = df
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        categorical_col_names = X.select_dtypes(include=['object', 'bool']).columns
        numerical_col_names = X.select_dtypes(exclude=['object', 'bool']).columns


        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, categorical_col_names), #Chance categorical to numerical data
                ("num", "passthrough", numerical_col_names) #Leave the numerical data as it is
            ]
        )

        self.model = Pipeline([
            ('preprocess', preprocessor),
            ('model', RandomForestRegressor(random_state=2025))
            ]        
        )

        self.model.fit(X, y)
        

    def predict(self, theta_new, anchor_size):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """

        if self.model == None:
            raise NotImplementedError("You must must first fit the model to the dataset before you can predict")
        
        HP_space = self.config_space.get_hyperparameter_names()
        default_values = self.config_space.get_default_configuration().get_dictionary()
        theta_dict = {hp: default_values.get(hp, None) for hp in HP_space} 
        theta_dict.update(self.enc_defaults)
        theta_dict.update(theta_new)
        theta_dict['anchor_size'] = anchor_size

        theta_df = pd.DataFrame([theta_dict])

        expected_performance = self.model.predict(theta_df)[0]
        return expected_performance
    
