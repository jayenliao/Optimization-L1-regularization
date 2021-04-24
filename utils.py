import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *

def _mean_absolute_error(y_true, y_hat):
    return mean_absolute_error(y_true, (y_hat>=.5)*1)

def _mean_squared_error(y_true, y_hat):
    return mean_squared_error(y_true, (y_hat>=.5)*1)

def _accuracy_score(y_true, y_hat):
    return accuracy_score(y_true, (y_hat>=.5)*1)

def _precision_score(y_true, y_hat):
    return precision_score(y_true, (y_hat>=.5)*1)

def _recall_score(y_true, y_hat):
    return recall_score(y_true, (y_hat>=.5)*1)

def _f1_score(y_true, y_hat):
    return f1_score(y_true, (y_hat>=.5)*1)

def set_eval_metric(eval_metric:str):
    if eval_metric.lower() == 'mse':
        return _mean_squared_error
    elif eval_metric.lower() == 'accuracy':
        return _accuracy_score
    elif eval_metric.lower() == 'precision':
        return _precision_score
    elif eval_metric.lower() == 'recall':
        return _recall_score
    elif eval_metric.lower() == 'f1':
        return _f1_score

def set_eval_metrics(eval_metrics:list):
    out = {}
    for m in eval_metrics:
        out[m] = make_scorer(set_eval_metric(m), greater_is_better=True)
    return out