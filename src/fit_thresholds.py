import numpy as np
from lmfit import Model

def fit_thresholds(x, y, baseline, threshold_factor=2.5, spl_range=[1, 90]):
    # Find where the IO function exceeds the baseline threshold
    # print(grand_df.iloc[i]['ppio'])
    # yd[xn] = p[1] / (1.0 + (p[2]/x[xn])**p[3])
    from lmfit import Model

    def hill(x, vmax, v50, n):
        return vmax / (1.0 + (v50/x) ** n)
    # def logistic(x, vmax, v50, n):
    #     return vmax / (1. + np.exp(-(x - v50)/n))
    from lmfit.models import ExpressionModel

    logistic_model = ExpressionModel("vmax / (1. + exp(-(x - v50)/n))")

    xs = np.linspace(spl_range[0], spl_range[1], sum(spl_range))
    # if min(x) > 20.:
    #     zx = np.linspace(0, np.min(x), 20)
    #     x = np.hstack((zx, x))
    #     y = np.hstack((np.zeros_like(zx), y))
    # print(x)
    Hillmodel = Model(hill)
    Hillmodel.set_param_hint("vmax", min=0.0, max=20.0)
    Hillmodel.set_param_hint("v50", min=30.0, max=100.0)
    Hillmodel.set_param_hint("n", min=0.1, max=50.0)
    params = Hillmodel.make_params(vmax=4, v50=60.0, n=10.0)
    x = np.array(x)
    y = np.array(y)
    print(len(x), len(y))
    if len(x) > 2:
        result = Hillmodel.fit(y, params, x=x, nan_policy='omit')
        ys = Hillmodel.eval(x=xs, params=result.params)


        # logistic_model = Model(logistic)
        # logistic_model.set_param_hint("vmax", min=0.0, max=20.0)
        # logistic_model.set_param_hint("n", min=0.1, max=50.0)
        # logistic_model.set_param_hint("v50", min=20.0, max=100.0)
        # params = logistic_model.make_params(vmax=4, n=10.0, v50=60.0)
        # result  = logistic_model.fit(y, x=x, amp=np.max(y), n=5, v50=np.mean(x), nan_policy='omit', method="least_squares") # , weights=1.0/x)
        # ys = logistic_model.eval(x=xs, params=result.params)
        # print("result params: ", result.params)
        # print("np.mean(baseline): ", np.mean(baseline), "data: ", y)

        ithr = np.argwhere(np.array(ys) >= np.mean(baseline)*threshold_factor)
        if len(ithr) == 0:
            ithr = len(y) - 1
            interp_thr = 100.0
        else:
            interp_thr = xs[ithr][0][0]
        if interp_thr > 90.0:
            interp_thr = 100.0
    else:
        interp_thr = None
        ithr = 0
        ys = np.zeros_like(xs)
    return interp_thr, ithr, (xs, ys)
