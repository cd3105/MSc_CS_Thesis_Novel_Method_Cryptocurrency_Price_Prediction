import pandas as pd
import numpy as np


def save_output_csv(preds, labels, feature, filename, model_type, bivariate=False):
    PATH = f"Original_Code/new_res/{model_type}/Output/output_{filename}.csv"
    if bivariate:
        labels = labels.reshape(-1, preds.shape[1])
        dct = {'avgcpu': preds[:, 0],
               'labelsavgcpu': labels[:, 0],
               'avgmem': preds[:, 1],
               'labelsavgmem': labels[:, 1]
               }
    else:
        try:
            preds = np.concatenate(list(preds), axis=0)
        except:
            pass
        try:
            labels = np.concatenate(list(labels), axis=0)
        except:
            pass
        # try:
        #     dct = {feature: np.concatenate(list(preds), axis=0),
        #            'labels': np.concatenate(list(labels), axis=0)}
        # except:
        dct = {feature: preds,
               'labels': labels}
        #     df = pd.DataFrame(dct)
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_uncertainty_csv(preds, std, labels, feature, filename, model_type, bivariate=False):
    PATH = f"Original_Code/new_res/{model_type}/Output/output_{filename}.csv"
    if bivariate:
        labels = labels.reshape(-1, preds.shape[1])
        dct = {'avgcpu': preds[:, 0],
               'stdavgcpu': std[:, 0],
               'labelsavgcpu': labels[:, 0],
               'avgmem': preds[:, 1],
               'stdavgmem': std[:, 1],
               'labelsavgmem': labels[:, 1],
               }
    else:
        try:
            dct = {feature: np.concatenate(list(preds), axis=0),
                   'std': np.concatenate(list(std), axis=0),
                   'labels': np.concatenate(list(labels), axis=0)}
        except:
            dct = {feature: preds,
                   'std': std,
                   'labels': np.concatenate(labels, axis=0)}

    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_params_csv(p, filename):
    PATH = f"Original_Code/param/p_{filename}.csv"
    df = pd.DataFrame(p, index=[0])
    df.to_csv(PATH)


def save_bayes_csv(preds, min, max, labels, feature, filename, model_type):
    PATH = f"Original_Code/new_res/{model_type}/Bayes/vidp_{filename}.csv"
    dct = {feature: preds,
           'min': min,
           'max': max,
           'labels': labels}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_metrics_csv(mses, maes, rmses, mapes, filename, r2, model_type):
    PATH = f"Original_Code/new_res/{model_type}/Metrics/metrics_{filename}.csv"
    dct = {'MSE': mses,
           'MAE': maes,
           'RMSE': rmses,
           'MAPE': mapes,
           'R2': r2,}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_window_outputs(labels, preds, filename, model_type):
    PATH = f"Original_Code/new_res/iterations/{model_type}/Output/outputs_{filename}.csv"
    predictions, true_values = [], []
    for i, p in enumerate(preds):
        ps = np.concatenate(list(p), axis=0)
        l = labels[i] 
        true = np.concatenate(list(l), axis=0)
        predictions.append(ps)
        true_values.append(true)
  
    dct = {'close': predictions,
        'labels': true_values,
        }
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_iteration_output_csv(preds, labels,  filename, model_type, iterations = 1):
    PATH = f"Original_Code/new_res/{model_type}/Output/output_preds_{filename}.csv"
    p = np.asarray(preds)
    p= p.reshape(iterations, len(p[0]))
    df_preds = pd.DataFrame(p)
    df_preds.to_csv(PATH)
    PATH = f"Original_Code/new_res/{model_type}/Output/output_labels_{filename}.csv"
    l = np.asarray(labels)
    l= l.reshape(iterations, len(l[0]))
    df_labels = pd.DataFrame(l)
    df_labels.to_csv(PATH)

def save_timing(times, filename, model_type, iterations = 1):
    PATH = f"Original_Code/new_res/{model_type}/Timing/timing_{filename}.csv"
    p = np.asarray(times)
    if iterations > 1: p= p.reshape(iterations, len(p[0]))
    df_timing = pd.DataFrame(p)
    df_timing.to_csv(PATH)
   