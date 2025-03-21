import numpy as np
from lmfit import Model

def fit_thresholds(x, y, baseline, threshold_factor=2.5, spl_range=[1, 90]):
    # Find where the IO function exceeds the baseline threshold
    # print(grand_df.iloc[i]['ppio'])
    # yd[xn] = p[1] / (1.0 + (p[2]/x[xn])**p[3])
    from lmfit import Model

    def hill(x, vmax, v50, n):
        return vmax / (1.0 + (v50 / x) ** n)

    Hillmodel = Model(hill)
    Hillmodel.set_param_hint("vmax", min=0.0, max=20.0)
    Hillmodel.set_param_hint("v50", min=0.0, max=100.0)
    Hillmodel.set_param_hint("n", min=1.0, max=5.0)
    print("dbs: ", x)
    print("V: ", y)
    params = Hillmodel.make_params(vmax=5, v50=50.0, n=2)
    print("params: ", params)

    result = Hillmodel.fit(y, params, x=x)
    yfit = result.best_fit
    xs = np.linspace(spl_range[0], spl_range[1], sum(spl_range))
    ys = Hillmodel.eval(x=xs, params=result.params)

    # ithr = np.argwhere(np.array(y) > baseline)
    ithr = np.argwhere(np.array(ys) > np.mean(baseline)*threshold_factor)
    if len(ithr) == 0:
        ithr = len(y) - 1
        # print("No threshold found: ", row.Subject)
        interp_thr = 100.0
    else:
        # ithr = ithr[0][0]
        # m = (y[ithr] - y[ithr - 1]) / (x[ithr] - x[ithr - 1])
        # b = y[ithr] - m * x[ithr]
        # bin = 1.0
        # int_thr = (baseline - b) / m
        # interp_thr = np.round(int_thr / bin, 0) * bin
        # print("interp thr: ", row.Subject, int_thr, interp_thr)
        interp_thr = xs[ithr][0][0]
    if interp_thr > 90.0:
        interp_thr = 100.0
    # print("interpthr: ", interp_thr)
    return interp_thr, ithr, (xs, ys)
