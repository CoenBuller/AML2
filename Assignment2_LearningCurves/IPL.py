import numpy as np 
import pandas as pd
import typing
import scipy
import tqdm

from vertical_model_evaluator import VerticalModelEvaluator

class IPL(VerticalModelEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @staticmethod
    # def inverse_power_law(x, a, b):
    #     f_x = 1/(a * x**b)
    #     return f_x

    # Inverse Power Law function: with offset term 
    @staticmethod
    def inverse_power_law(x, a, b, c):
        # y = c + a * x^(-b)
        x = np.asarray(x, dtype=float)
        return c + a * np.power(x, -b)
    

    # @staticmethod
    # def fit_inverse_power(performance: typing.List[typing.Tuple[int, float]]):
    #     # Fit inverse power law function to observed data
    #     x = []
    #     y = []
    #     for xi, perf in performance:
    #         x.append(xi)
    #         y.append(perf)

    #     popt, pcov = scipy.optimize.curve_fit(IPL.inverse_power_law, x, y)
    #     return popt, pcov



    @staticmethod
    def fit_inverse_power(performance: typing.List[typing.Tuple[int, float]]):
        x = np.array([xi for xi, _ in performance], dtype=float)
        y = np.array([yi for _, yi in performance], dtype=float)

        # Ensure positive anchors
        mask = x > 0
        x, y = x[mask], y[mask]

        if len(x) < 3:
            return None, None, None  # too few points to fit

        # Initial guesses using simple heuristics
        c0 = max(0.0, np.percentile(y, 10))
        if np.min(y) > 1e-6:
            c0 = min(c0, np.min(y) - 1e-6)
        else:
            c0 = 0.0

        mask2 = (y - c0) > 1e-8
        if mask2.sum() >= 2:
            X = np.log(x[mask2])
            Y = np.log(y[mask2] - c0)
            B = np.polyfit(X, Y, 1)
            a0 = np.exp(B[1])
            b0 = max(1e-6, -B[0])
        else:
            a0, b0 = np.median(y), 1.0

        p0 = (a0, b0, c0)
        bounds = ((1e-9, 1e-9, 0.0), (1e9, 10.0, 1e6))

        try:
            popt, pcov = curve_fit(IPL.inverse_power_law, x, y, p0=p0, bounds=bounds, maxfev=20000)
        except Exception:
            return None, None, None

        # Goodness-of-fit (R^2)
        y_hat = IPL.inverse_power_law(x, *popt)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return popt, pcov, r2

    def _predict_final_with_unc(self, popt, pcov, x_final: int):
        """Predict y(T) and an uncertainty via the delta method."""
        a, b, c = popt
        T = float(x_final)
        yT = self.inverse_power_law(T, a, b, c)
        if pcov is None or not np.all(np.isfinite(pcov)):
            return float(yT), None
        # y = c + a*T^-b
        Tb = np.power(T, -b)
        grad = np.array([Tb, a * (-np.log(T)) * Tb, 1.0], dtype=float)  # [dy/da, dy/db, dy/dc]
        var = float(grad @ pcov @ grad)
        if var < 0 or not np.isfinite(var):
            return float(yT), None
        return float(yT), float(np.sqrt(var))



    
    # def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
    #     if best_so_far is None: #If no best yet, evaluate with external surrogate at max anchor size
    #         return [(self.final_anchor, self.surrogate_model.predict(configuration, self.final_anchor))]
        
    #     train_anchors = np.linspace(0, self.final_anchor*0.4, 10) # We will evaluate 10 anchor sizes uptill 0.4*max_anchor 
    #     results = []
    #     for anchor in train_anchors:
    #         # Evaluate performance of this model on pre-determined anchor sizes
    #         results.append((anchor, self.surrogate_model.predict(configuration, int(anchor))))
        
    #     popt, pcov = IPL.fit_inverse_power(performance=results) # Fit inverse power law (IPL) function to data
    #     best = IPL.inverse_power_law(self.final_anchor, popt[0], popt[1]) # Predict performance at max anchor size using IPL
    #     if best < best_so_far:
    #         results.append((self.final_anchor, best))
    #     return results

    def evaluate_model(
        self, best_so_far: typing.Optional[float], configuration: typing.Dict
    ) -> typing.List[typing.Tuple[int, float]]:
        # No incumbent yet → evaluate full anchor via surrogate
        if best_so_far is None:
            perf = self.surrogate_model.predict(configuration, self.final_anchor)
            return [(self.final_anchor, perf)]

        # Build schedule of real anchors from the surrogate's data
        if hasattr(self.surrogate_model, "df"):
            anchors_in_data = sorted(set(int(a) for a in self.surrogate_model.df["anchor_size"]))
        else:
            anchors_in_data = [self.minimal_anchor * (2 ** i)
                            for i in range(int(np.log2(self.final_anchor / self.minimal_anchor)) + 1)]

        # Use anchors up to ~60% of the final anchor (captures curvature better)
        early_limit = int(0.6 * self.final_anchor)
        schedule = [a for a in anchors_in_data if self.minimal_anchor <= a <= early_limit]


        # Evaluate surrogate at these anchors
        results = []
        for anchor in schedule:
            perf = self.surrogate_model.predict(configuration, int(anchor))
            results.append((int(anchor), float(perf)))

        # Fit IPL to early points
        popt, pcov, r2 = IPL.fit_inverse_power(results)

        # If fit fails or is unreliable, be conservative
        if popt is None or (r2 is not None and r2 < 0.15):
            if results[-1][1] >= best_so_far:
                return results  # discard (early stop)
            final_perf = self.surrogate_model.predict(configuration, self.final_anchor)
            results.append((self.final_anchor, float(final_perf)))
            return results

        # Predict performance at final anchor, with uncertainty
        pred_final, sigma = self._predict_final_with_unc(popt, pcov, self.final_anchor)

        # use an optimistic threshold; tweak k if too aggressive/conservative
        k = 1.0
        optimistic = pred_final - (k * sigma if sigma is not None else 0.0)

        print(f"[IPL] r²={r2:.3f} | pred={pred_final:.4f} | σ={sigma if sigma is not None else 'NA'} | "
            f"optimistic={optimistic:.4f} | best={best_so_far:.4f}")

        # Early-stop rule (lower is better)
        eps = 1e-4
        if optimistic >= best_so_far - eps:
            return results  # discard
        else:
            final_perf = self.surrogate_model.predict(configuration, self.final_anchor)
            results.append((self.final_anchor, float(final_perf)))
            return results

