import numpy as np 
import pandas as pd
import typing
import scipy
import tqdm

from vertical_model_evaluator import VerticalModelEvaluator
from scipy.optimize import curve_fit   

class IPL(VerticalModelEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def inverse_power_law(x, a, b, c):
        # y = c + a * x^(-b)
        x = np.asarray(x, dtype=float)
        return c + a * np.power(x, -b)

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
            popt, pcov = curve_fit(
                IPL.inverse_power_law, x, y, p0=p0, bounds=bounds, maxfev=20000
            )
        except (RuntimeError, ValueError) as e:
            print(f"[IPL] 3-param fit failed: {e}")

            # Fallback to 2param fit
            def _ipl2(x, a, b):
                x = np.asarray(x, float)
                return a * np.power(x, -b)

            try:
                a0_2 = max(1e-6, float(np.median(y)))
                b0_2 = 1.0
                p0_2 = (a0_2, b0_2)
                bounds_2 = ((1e-9, 1e-9), (1e9, 10.0))
                popt2, pcov2 = curve_fit(_ipl2, x, y, p0=p0_2, bounds=bounds_2, maxfev=20000)
                # Promote to 3-param shape (a,b,c) with c=0
                popt = (popt2[0], popt2[1], 0.0)
                pcov = np.pad(pcov2, ((0, 1), (0, 1)), constant_values=0.0)
                print(f"[IPL] 2-param fallback succeeded: a={popt2[0]:.3f}, b={popt2[1]:.3f}")
            except (RuntimeError, ValueError) as e2:
                print(f"[IPL] 2-param fallback failed: {e2}")
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


    def evaluate_model(
        self,
        best_so_far: typing.Optional[float],
        conf: typing.Dict
    ) -> typing.List[typing.Tuple[int, float]]:
        """
        IPL evaluator with a FIXED learning-curve schedule (as required by the assignment).
        Steps:
        1) Evaluate a fixed set of early anchors for this dataset.
        2) Fit an inverse power law to these points.
        3) Predict performance at the final anchor.
        4) If predicted_final < best_so_far (or best_so_far is None), evaluate final anchor.
            Otherwise, discard (early stop).
        """
        if not hasattr(self.surrogate_model, "df"):
            raise ValueError("SurrogateModel must store the training dataframe as 'self.df' in fit().")
        
        results=[] # Store results. format: (anchor, score)
        conf_tuple = tuple(conf.values())
        if conf_tuple not in self.results:
            self.results[conf_tuple] = []

        # early stopping
        # Case A: no incumbent yet → always evaluate final anchor to establish best_so_far
        if best_so_far is None:
            final_perf = float(self.surrogate_model.predict(conf, self.final_anchor))
            results.append((int(self.final_anchor), float(final_perf)))

            # ------ Added/Changed -------
            self.budget -= self.final_anchor
            self.results[conf_tuple] = results

            return results

        anchors_in_data = sorted(set(int(a) for a in self.surrogate_model.df["anchor_size"]))


        # fixed achoirs -- pak eerste paar

        # ------ Added/Changed -------
        if self.anchors is None:
            schedule = anchors_in_data[:-1]  # all but final
        else:
            schedule = self.anchors
        
        # schedule = anchors_in_data[:5]
        # schedule = [a for a in schedule if a < self.final_anchor]

        #Evaluation
        for anchor in schedule:
            perf = float(self.surrogate_model.predict(conf, anchor))
            results.append((int(anchor), float(perf)))

        # fitting
        popt, pcov, r2 = IPL.fit_inverse_power(results)

        # If fit fails, fall back to "evaluate final if it's better than last point"
        if popt is None:
            last_perf = results[-1][1]
            if best_so_far is None or last_perf < best_so_far:
                final_perf = float(self.surrogate_model.predict(conf, self.final_anchor))
                results.append((int(self.final_anchor), float(final_perf)))

            # ------ Added/Changed -------
            self.budget -= self.final_anchor
            self.results[conf_tuple] = results

            return results

        # Predict final performance using the parametric IPL model
        pred_final = float(IPL.inverse_power_law(self.final_anchor, *popt))

        # print(f"[IPL] r²={r2:.3f} | pred_final={pred_final:.4f} | best={best_so_far if best_so_far is not None else float('inf'):.4f}")

        # Case B: IPL says it's promising → evaluate final
        if pred_final < best_so_far:
            final_perf = float(self.surrogate_model.predict(conf, self.final_anchor))
            results.append((int(self.final_anchor), float(final_perf)))

            
            # ------ Added/Changed -------
            self.budget -= self.final_anchor
            self.results[conf_tuple] = results
            
            return results

        
        # ------ Added/Changed -------
        self.budget -= self.final_anchor
        self.results[conf_tuple] = results
        # Case C: predicted final is not better → early stop
        return results

