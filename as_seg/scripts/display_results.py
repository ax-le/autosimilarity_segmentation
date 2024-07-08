"""
Script to display experimental results in a nice format.

I should probably use scikit-learn's GridSearchCV, TODO.
"""

import pandas as pd
from IPython.display import display
import numpy as np
import itertools

def display_experimental_results(data, conditions, metrics, emphasis = None):
    assert data.shape[-1] == len(metrics), f"Wrong number of metrics: {data.shape[-1]} != {len(metrics)}"
    if type(conditions) == list:
        conditions = np.array(conditions, dtype=object)
    if len(conditions.shape) == 1 and type(conditions[0]) not in [list, tuple, np.ndarray]:
        indexes = conditions
    elif len(conditions.shape) > 1 or (len(conditions.shape) == 1 and type(conditions) in [list, tuple, np.ndarray]):
        indexes = np.array(list(itertools.product(*conditions))) # cartesian_product
    else:
        raise NotImplementedError(f"Wrong format of conditions: {conditions}")
    
    dataframe = pd.DataFrame(data, columns = metrics, index = indexes)
    if emphasis is not None:
        display(dataframe.style.bar(subset=emphasis, color='#5fba7d'))
    else:
        display(dataframe)
    
def find_best_condition(data, conditions):
    if type(conditions) == list:
        conditions = np.array(conditions, dtype=object)
    if len(conditions.shape) == 1 and type(conditions[0]) not in [list, tuple, np.ndarray]:
        best_arg = np.argmax(data)
        best_val = conditions[best_arg]
        return best_val
    elif len(conditions.shape) > 1 or (len(conditions.shape) == 1 and type(conditions) in [list, tuple, np.ndarray]):
        best_arg_flatten = np.argmax(data)
        best_arg_tuple = np.unravel_index(best_arg_flatten, data.shape)
        best_vals = []
        for idx, best_arg in enumerate(best_arg_tuple):
            best_vals.append(conditions[idx][best_arg])
        return np.array(best_vals)

        

