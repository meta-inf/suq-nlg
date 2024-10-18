import numpy as np
from jaxtyping import Array, Float
import pandas as pd
from sklearn import metrics


def area_under_accuracy_coverage_curve(u, a):
    # area under the rejection-VALUE curve, where VALUE could be accuracy, etc.
    # Taken from https://github.com/zlin7/UQ-NLG/blob/main/pipeline/eval_uq.py
    df = pd.DataFrame({"u": u, 'a': a}).sort_values('u', ascending=True)
    df['amean'] = df['a'].expanding().mean()
    return metrics.auc(np.linspace(0,1,len(df)), df['amean'])


def off_diagonal_mean(mat: Float[Array, "nspd nspd"]) -> float:
    mask = (1-np.eye(mat.shape[0])) * (~np.isnan(mat))
    # return np.where(mask, mat, 0).sum() / mask.sum()
    row_mean = np.where(mask, mat, 0).sum(axis=1) / (mask.sum(axis=1) + 1e-7)
    row_mask = mask.sum(axis=1) > 0
    return row_mean[row_mask].mean()


def safe_mean(arr: np.ndarray) -> float:
    mask = ~np.isnan(arr)
    return np.where(mask, arr, 0).sum() / mask.sum()


def eval_ece(pred: Float[Array, "n"], actual: Float[Array, "n"], plot_title=None, n_bins=20):
    grid = np.linspace(0, 1, n_bins+1)
    grid[0] -= 1e-3; grid[-1] += 1e-3
    mean_resp = []
    sd_resp = []
    ece_num, ece_den = 0, 0
    for xs, xe in zip(grid[:-1], grid[1:]):
        mask = (xs < pred) & (pred <= xe)
        if (cnt := mask.sum()) <= 3:
            mean_resp.append(0)
            sd_resp.append(0)
        else:
            m = actual[mask].mean()
            mean_resp.append(m)
            sd_resp.append(actual[mask].std() / (cnt**0.5) * 1.96)
            ece_num += np.abs(m-(xs+xe)/2) * cnt
            ece_den += cnt

    ece = ece_num/ece_den
    
    if plot_title is not None:
        from matplotlib import pyplot as plt
        
        x = (grid[:-1]+grid[1:])/2
        mean_resp, sd_resp = map(np.asarray, (mean_resp, sd_resp))
        plt.bar(x, mean_resp, width=1/n_bins, color='C0')
        plt.plot([0,1], [0,1], linestyle=':', color='C2')
        plt.xlabel('subjective utility')
        plt.ylabel('observed utility')
        plt.title(plot_title) # + f' / ECE={ece:.3f}')
        plt.xlim(-0.01, 1.01)
        plt.ylim(0, 1.01)

    return ece
