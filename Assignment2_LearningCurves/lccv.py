import logging
import numpy as np
import typing

from surrogate_model import SurrogateModel
from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    def __init__(self, surrogate_model: SurrogateModel, minimal_anchor: int, final_anchor: int, budget: int, results={}):
        super().__init__(surrogate_model=surrogate_model, minimal_anchor=minimal_anchor, final_anchor=final_anchor, budget=budget, results=results)
    
    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int, previous_performance: float, 
        current_anchor: int, current_performance: float, target_anchor: int
    ) -> float:
        """
        Does the optimistic performance. Since we are working with a simplified
        surrogate model, we can not measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param previous_anchor: See name
        :param previous_performance: Performance at previous anchor
        :param current_anchor: See name
        :param current_performance: Performance at current anchor
        :param target_anchor: the anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        p_T = current_performance + (target_anchor - current_anchor) * (current_performance  - previous_performance)/(current_anchor - previous_anchor)
        return p_T

    def evaluate_model(self, best_so_far: typing.Optional[float], conf: typing.Dict) -> float | None:
        """
        Does a staged evaluation of the model, on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. In case the optimistic extrapolation can not improve
        over the best so far, it stops the evaluation.
        In case the best so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor)

        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """

        configuration = tuple(conf.values())
        if configuration not in self.results:
            self.results[configuration] = []

        if best_so_far is None: # No best yet so evaluate current config at max anchorsize
            best_performance = self.surrogate_model.predict(conf, self.final_anchor)  

            self.results[configuration] += [(self.final_anchor, best_performance)] 
            self.budget -= self.final_anchor 

            return best_performance # type: ignore
        
        try:
            steps = self.surrogate_model.df['anchor_size'].unique() # type: ignore
            for step in steps:

                if len(self.results[configuration]) < 2: # Cannot extrapolate if there arent two points 
                    performance = self.surrogate_model.predict(conf, step)
                    self.results[configuration] += [(step, performance)]

                else:
                    prev1 = self.results[configuration][-1]
                    prev2 = self.results[configuration][-2]
                    
                    p_T = LCCV.optimistic_extrapolation(prev2[0], prev2[1], prev1[0], prev1[1], self.final_anchor)

                    if p_T > best_so_far: # If optimistic extrapolation does not outperform best so far, we can discard this configuration
                        break
                    
                    # If optimistic extrapolation does outperform the best so far, we can evaluate the performance using the surrogate
                    performance = self.surrogate_model.predict(conf, step)
                    self.budget -= step
                    self.results[configuration] += [(step, performance)]

            return self.results[configuration][-1][1] # Return last performance for evaluation
        
        except TypeError:
            print("First fit the surrogate model on the dataset before you try to evaluate a configuration")
            return None
        