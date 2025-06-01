import logging
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from register import Register


register_obj = Register('metric_register')

def compute_metrics(predict_matrix, target_matrix, metrics, env_matrix=None):
    metric_dict = {}
    for metric in metrics:
        metric_class = register_obj[metric]
        if metric_class is None:
            logging.warning(f'metric:{metric} is not registered.')
        else:
            # Check if the metric requires environment information
            if hasattr(metric_class, 'requires_env') and metric_class.requires_env:
                if env_matrix is None:
                    logging.warning(f'metric:{metric} requires environment information but env_matrix is not provided.')
                    continue
                metric_value = metric_class.compute(predict_matrix, target_matrix, env_matrix)
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


@register_obj.register
class PerEnvironmentAccuracy:
    requires_env = True
    
    def compute(predict_matrix, target_matrix, env_matrix):
        """
        Compute accuracy for each environment separately.
        
        Args:
            predict_matrix: Predictions (continuous values)
            target_matrix: True labels (binary)
            env_matrix: Environment labels for each sample
            
        Returns:
            dict: Dictionary with environment IDs as keys and their accuracy as values
        """
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0] = 1
        
        unique_envs = np.unique(env_matrix)
        env_accuracies = {}
        
        for env in unique_envs:
            env_mask = (env_matrix == env)
            if np.sum(env_mask) > 0:  # Ensure there are samples in this environment
                env_predictions = predict_result[env_mask]
                env_targets = target_matrix[env_mask]
                env_accuracy = accuracy_score(env_targets, env_predictions)
                env_accuracies[f'env_{int(env)}'] = env_accuracy
        
        # Also compute overall accuracy for comparison
        overall_accuracy = accuracy_score(target_matrix, predict_result)
        env_accuracies['overall'] = overall_accuracy
        
        return env_accuracies

