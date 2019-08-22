import numpy as np

'''
    Error Functions that will be used for models:
        MSE: Mean Squared Error
        MAE: Mean Absolute Error

'''

def check_input(true_labels, predict_labels):
    '''
    Basic function to check whether two arrays fit requirement
    Including: 
        whether it is a list
        same size
    Return:
        true_labels: numpy array if input is a list
        predict_labels: numpy array if input is a list
    '''
    if len(true_labels) != len(predict_labels):
        raise ValueError("predictions size {} differ true labels size {}".format(len(predict_labels), len(true_labels)))
    # check whether inputs are list, if so, convert to numpy array
    if isinstance(true_labels, list):
        true_labels_ = np.array(true_labels)
    else:
        true_labels_ = true_labels

    if isinstance(predict_labels, list):
        predict_labels_ = np.array(predict_labels)
    else:
        predict_labels_ = predict_labels

    return true_labels_, predict_labels_

def MSE(true_labels, predict_labels):
    """
    Error function: Mean Squared Error
    e = (true_label - predict_label) ** 2

    """
    true_labels, predict_labels = check_input(true_labels, predict_labels)
    
    squared_error = np.square(true_labels - predict_labels)
    mean_squared_error = np.mean(squared_error)
    if mean_squared_error.shape[1] != 1:
        raise ValueError("Input labels are not 1-d array")
    return mean_squared_error

def MAE(true_labels, predict_labels):
    """
    Error function: Mean Absolute Error
    e = (true_labels - predict_labels)
    """
    true_labels, predict_labels = check_input(true_labels, predict_labels)

    absolute_error = true_labels - predict_labels
    mean_absolute_error = np.mean(absolute_error)
    if mean_absolute_error.shape[1] != 1:
        raise ValueError("Inputlabels are not 1-d array")
    return mean_absolute_error
