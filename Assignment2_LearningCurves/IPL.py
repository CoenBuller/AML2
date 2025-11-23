import numpy as np 
import pandas as pd
import typing

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
        # Fit IPL to a partial learning curve (x = anchors, y = losses)
        # 3 parameter fit and fallback to 2 parameter variant.
        x = np.array([xi for xi, _ in performance], dtype=float)
        y = np.array([yi for _, yi in performance], dtype=float)

        # Only positive achors accepted
        mask = x > 0
        x, y = x[mask], y[mask]

        # Min of 3 anchors
        if len(x) < 3:
            return None, None, None  

        # Initial guessing
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

        # 3 param fit
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
                
                
                # Convert back to 3 parameter shape (a, b, c) with c = 0 for consistency
                popt = (popt2[0], popt2[1], 0.0)
                pcov = np.pad(pcov2, ((0, 1), (0, 1)), constant_values=0.0)
                print(f"[IPL] 2-param fallback succeeded: a={popt2[0]:.3f}, b={popt2[1]:.3f}")
            except (RuntimeError, ValueError) as e2:
                print(f"[IPL] 2-param fallback failed: {e2}")
                return None, None, None



        # R^2
        y_hat = IPL.inverse_power_law(x, *popt)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return popt, pcov, r2

    def _predict_final_with_unc(self, popt, pcov, x_final: int):
        # predict y(T) for final anchor and estimate uncertainty
        a, b, c = popt
        T = float(x_final)
        yT = self.inverse_power_law(T, a, b, c)
        if pcov is None or not np.all(np.isfinite(pcov)):
            return float(yT), None

        Tb = np.power(T, -b)
        grad = np.array([Tb, a * (-np.log(T)) * Tb, 1.0], dtype=float)  
        var = float(grad @ pcov @ grad)
        if var < 0 or not np.isfinite(var):
            return float(yT), None
        return float(yT), float(np.sqrt(var))


    def evaluate_model(self, best_so_far: typing.Optional[float], conf: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        # IPL Evaluator - fixed learning curve schedule
        # 1. Evaluate early anchors
        # 2. Fit IPL
        # 3. Predict final performance
        # 4. check if final full eval is needed
        if not hasattr(self.surrogate_model, "df"):
            raise ValueError("SurrogateModel must store the training dataframe as 'self.df' in fit().")
        
        results=[] # Store results. format: (anchor, score)
        conf_tuple = tuple(conf.values())
        if conf_tuple not in self.results:
            self.results[conf_tuple] = []

        # early stopping
        # Case A: first evaluation , always evaluate at final anchor
        if best_so_far is None:
            final_perf = float(self.surrogate_model.predict(conf, self.final_anchor))
            results.append((int(self.final_anchor), float(final_perf)))

            true_cost = sum(a for a, _ in results)
            self.budget -= true_cost

            self.results[conf_tuple] = results

            return results
        

        anchors_in_data = sorted(set(int(a) for a in self.surrogate_model.df["anchor_size"]))
        final_anchor = anchors_in_data[-1]

        # Fixed early schedule, only use anchors < 40% of final anchor
        if self.anchors is None:
            cutoff = int(0.3 * final_anchor)
            schedule = [a for a in anchors_in_data if a < cutoff]

            if len(schedule) < 3:
                # Fallback: take the first few anchors but never the final one
                schedule = anchors_in_data[:-1]

        else:
            schedule = self.anchors


        # Evaluation
        for anchor in schedule:
            perf = float(self.surrogate_model.predict(conf, anchor))
            results.append((int(anchor), float(perf)))

        # Fitting
        popt, pcov, r2 = IPL.fit_inverse_power(results)

        # If fit fails, fall back: only evaluate final if last point looks promising
        if popt is None:
            last_perf = results[-1][1]
            if best_so_far is None or last_perf < best_so_far:
                final_perf = float(self.surrogate_model.predict(conf, self.final_anchor))
                results.append((int(self.final_anchor), float(final_perf)))

            true_cost = sum(a for a, _ in results)
            self.budget -= true_cost

            self.results[conf_tuple] = results

            return results

        # Predict final performance 
        pred_final = float(IPL.inverse_power_law(self.final_anchor, *popt))

        # print(f"[IPL] r²={r2:.3f} | pred_final={pred_final:.4f} | best={best_so_far if best_so_far is not None else float('inf'):.4f}")

        # Case B: IPL says it's promising, evaluate final
        if pred_final < best_so_far:
            final_perf = float(self.surrogate_model.predict(conf, self.final_anchor))
            results.append((int(self.final_anchor), float(final_perf)))


            true_cost = sum(a for a, _ in results)
            self.budget -= true_cost

            self.results[conf_tuple] = results
            
            return results

    
        true_cost = sum(a for a, _ in results)
        self.budget -= true_cost

        
        self.results[conf_tuple] = results
        # Case C: predicted final is not better, early stop
        return results

