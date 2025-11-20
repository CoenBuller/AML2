from abc import ABC
import ConfigSpace
import numpy as np
from sklearn.pipeline import Pipeline
from surrogate_model import SurrogateModel
import typing


class VerticalModelEvaluator(ABC):

    def __init__(self, surrogate_model: SurrogateModel, minimal_anchor: int, final_anchor: int, budget: int, results={}, anchors=None) -> None:
        """
        Initialises the vertical model evaluator. Take note of what the arguments are
        
        :param surrogate_model: A sklearn pipeline object, which has already been fitted on LCDB data. You can use the predict model to predict for a numpy array (consisting of configuration information and an anchor size) what the performance of that configuration is. 
        :param minimal_anchor: Smallest anchor to be used
        :param final_anchor: Largest anchor to be used
        """
        self.surrogate_model = surrogate_model
        self.minimal_anchor = minimal_anchor
        self.final_anchor = final_anchor
        self.results = results
        self.budget = budget
        self.anchors = anchors # Initial anchors to evaluate for IPL method

    def evaluate_model(self, best_so_far: None|float, conf: typing.Dict):
        raise NotImplementedError()
    