import logging
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from register import Register


register_obj = Register('metric_register')

def compute_metrics(predict_matrix, target_matrix, metrics):
    metric_dict = {}
    for metric in metrics:
        metric_class = register_obj[metric]
        if metric_class is None:
            logging.warning(f'metric:{metric} is not registered.')
        else:
            metric_value = metric_class.compute(predict_matrix, target_matrix)
            metric_dict[metric] = metric_value
    return metric_dict

@register_obj.register
class Accuracy:
    def compute(predict_matrix, target_matrix):
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0] = 1
        return accuracy_score(predict_result, target_matrix)


@register_obj.register
class AUC:
    def compute(predict_matrix, target_matrix):
        return roc_auc_score(target_matrix, predict_matrix)


@register_obj.register
class F1_macro:
    def compute(predict_matrix, target_matrix):
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0] = 1
        return f1_score(target_matrix, predict_result, average='macro')

