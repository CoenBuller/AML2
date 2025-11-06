import numpy as np 
import pandas as pd
import typing
import scipy
import tqdm

from vertical_model_evaluator import VerticalModelEvaluator

class IPL(VerticalModelEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def inverse_power_law(x, a, b):
        f_x = 1/(a * x**b)
        return f_x

    @staticmethod
    def fit_inverse_power(performance: typing.List[typing.Tuple[int, float]]):
        # Fit inverse power law function to observed data
        x = []
        y = []
        for xi, perf in performance:
            x.append(xi)
            y.append(perf)

        popt, pcov = scipy.optimize.curve_fit(IPL.inverse_power_law, x, y)
        return popt, pcov
    
    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        if best_so_far is None: #If no best yet, evaluate with external surrogate at max anchor size
            return [(self.final_anchor, self.surrogate_model.predict(configuration, self.final_anchor))]
        
        train_anchors = np.linspace(0, self.final_anchor*0.4, 10) # We will evaluate 10 anchor sizes uptill 0.4*max_anchor 
        results = []
        for anchor in train_anchors:
            # Evaluate performance of this model on pre-determined anchor sizes
            results.append((anchor, self.surrogate_model.predict(configuration, int(anchor))))
        
        popt, pcov = IPL.fit_inverse_power(performance=results) # Fit inverse power law (IPL) function to data
        best = IPL.inverse_power_law(self.final_anchor, popt[0], popt[1]) # Predict performance at max anchor size using IPL
        if best < best_so_far:
            results.append((self.final_anchor, best))
        return results