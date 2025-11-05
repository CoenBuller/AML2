import logging
import numpy as np
import typing

from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    
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

    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
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
        if best_so_far is None:
            return [(self.final_anchor, self.surrogate_model.predict(configuration, self.final_anchor))] # type: ignore
        try:
            results = []
            steps = self.surrogate_model.df['anchor_size'].unique()
            for step in steps:
                performance = self.surrogate_model.predict(configuration, step)
                if len(results) < 2: # Cannot extrapolate if there arent two points 
                    results.append((step, performance))
                else:
                    prev_step = results[-1]
                    p_T = LCCV.optimistic_extrapolation(prev_step[0], prev_step[1], step, performance, self.final_anchor)
                    results.append((step, performance))

                    if p_T > best_so_far: #If optimistic extrapolation does not outperform best so far, we can discard this configuration
                        return results

            return results
        except TypeError:
            print("First fit the surrogate model on the dataset before you try to evaluate a configuration")
        