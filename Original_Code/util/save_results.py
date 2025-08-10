import pandas as pd
import numpy as np
import os


def save_output_csv(preds, 
                    labels, 
                    feature, 
                    filename, 
                    res_map, 
                    model_type,
                    normalized=False,
                    bivariate=False):
    if normalized:
        PATH = f"Original_Code/{res_map}/{model_type}/Output/Normalized/"
    else:
        PATH = f"Original_Code/{res_map}/{model_type}/Output/Regular/"

    if not os.path.exists(PATH):
        os.makedirs(PATH)

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

        dct = {feature: preds,
               'labels': labels}
        
    df = pd.DataFrame(dct)
    df.to_csv(f"{PATH}output_{filename}.csv")


def save_uncertainty_csv(preds, 
                         std, 
                         labels, 
                         feature, 
                         filename, 
                         res_map, 
                         model_type, 
                         bivariate=False):
    PATH = f"Original_Code/{res_map}/{model_type}/Output/"

    if not os.path.exists(PATH):
        os.makedirs(PATH)

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
    df.to_csv(f"{PATH}output_{filename}.csv")


def save_params_csv(p, 
                    filename):
    PATH = f"Original_Code/param/"
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    df = pd.DataFrame(p, index=[0])
    df.to_csv(f"{PATH}p_{filename}.csv")


def save_bayes_csv(preds, 
                   min, 
                   max, 
                   labels, 
                   feature, 
                   filename, 
                   res_map, 
                   model_type):
    PATH = f"Original_Code/{res_map}/{model_type}/Bayes/"

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    dct = {feature: preds,
           'min': min,
           'max': max,
           'labels': labels}
    
    df = pd.DataFrame(dct)
    df.to_csv(f"{PATH}vidp_{filename}.csv")


def save_metrics_csv(mses, 
                     maes, 
                     rmses, 
                     mapes, 
                     filename, 
                     res_map, 
                     r2, 
                     model_type,
                     normalized=False):
    
    if normalized:
        PATH = f"Original_Code/{res_map}/{model_type}/Metrics/Normalized/"
    else:
        PATH = f"Original_Code/{res_map}/{model_type}/Metrics/Regular/"

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    dct = {'MSE': mses,
           'MAE': maes,
           'RMSE': rmses,
           'MAPE': mapes,
           'R2': r2,}
    
    df = pd.DataFrame(dct)
    df.loc['Mean'] = df.mean(axis=0)
    df.to_csv(f"{PATH}metrics_{filename}.csv")


def save_window_outputs(labels, 
                        preds, 
                        filename, 
                        res_map, 
                        model_type):
    PATH = f"Original_Code/{res_map}/iterations/{model_type}/Output/"
    predictions, true_values = [], []

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    for i, p in enumerate(preds):
        ps = np.concatenate(list(p), axis=0)
        l = labels[i] 
        true = np.concatenate(list(l), axis=0)
        predictions.append(ps)
        true_values.append(true)
  
    dct = {'close': predictions,
           'labels': true_values,}
    
    df = pd.DataFrame(dct)
    df.to_csv(f"{PATH}outputs_{filename}.csv")


def save_iteration_output_csv(preds, 
                              labels, 
                              filename, 
                              res_map, 
                              model_type, 
                              iterations=1):
    PATH = f"Original_Code/{res_map}/iterations/{model_type}/Output/"

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    p = np.asarray(preds)
    p= p.reshape(iterations, len(p[0]))
    df_preds = pd.DataFrame(p)
    df_preds.to_csv(f"{PATH}output_preds_{filename}.csv")

    PATH = f"Original_Code/{res_map}/{model_type}/Output/"

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    l = np.asarray(labels)
    l= l.reshape(iterations, len(l[0]))
    df_labels = pd.DataFrame(l)
    df_labels.to_csv(f"{PATH}output_labels_{filename}.csv")


def save_timing(times, 
                filename, 
                res_map, 
                model_type, 
                iterations=1):
    PATH = f"Original_Code/{res_map}/{model_type}/Timing/"

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    p = np.asarray(times)

    if iterations > 1: 
        p=p.reshape(iterations, len(p[0]))

    df_timing = pd.DataFrame(p)
    df_timing.to_csv(f"{PATH}timing_{filename}.csv")
   