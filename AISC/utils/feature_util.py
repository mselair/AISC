# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, cohen_kappa_score

def zscore(x):
    """
    Calculates Z-score
    Parameters
    ----------
    x : numpy ndarray
        shape[n_samples, n_features]

    Returns
    _______
    normalized_features : numpy ndarray
    """

    return (x - x.mean(axis=0).reshape(1, -1)) / x.std(axis=0).reshape(1, -1)

def find_category_outliers(x, y):
    """
    Finds outliers for each category within data.
    Check website: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

    Parameters
    ----------
    x : numpy ndarray
        shape[n_samples, n_features]
    y : list or numpy array
        string or int indexes for each category

    Returns
    -------
    list
        position index list with detected outliers

    """
    ycat = np.unique(y)
    to_del = []

    # extreme values
    for k in range(x.shape[1]):
        to_del = to_del + list(np.where(x[:, k] <= 0)[0])

    x_temp = x.copy()
    for k in range(x_temp.shape[1]):
        to_del = to_del + list(np.where(np.abs(x_temp[:, k]) > 4)[0])

    for yc in ycat:
        positions = np.where(np.array(y) == yc)[0]
        x_sub = x[positions]
        x_sub = zscore(x_sub)
        pred = LocalOutlierFactor().fit_predict(x_sub)
        to_del = to_del + list(positions[np.where(pred == -1)[0]])
    return to_del

def print_classification_scores(Y, YY):
    kappa = cohen_kappa_score(Y, YY)
    cmat = confusion_matrix(Y, YY)

    print('Kappa-score: {:.2f}'.format(kappa))
    print(classification_report(Y, YY))
    print('Confusion Matrix:')
    print(cmat)

def augment_features(x, feature_names=None, feature_indexes=[], operation=None, mutual=False, operation_str = ''):
    """
    Augments features with entered operations (mutual - between features such as *, /, +, -, ....; non mutual - log, exp, power, ...)

    Parameters
    ----------
    x : numpy ndarray
            shape[n_samples, n_features]
    feature_names : list or numpy array of strings, optional
        names of features
    feature_indexes : list or numpy array
        indexes of features which will be augmented
    operation : function
        callable function which will be applied on existing features.
    mutual :  bool
        indicates whether operation is applied on single feature e.g. np.log10, or on 2 parameters e.g. np.divide
        if mutual = True, then applied on all feature combination specified in feature_indexes

    Returns
    -------
    numpy ndarray -> shape[n_samples, n_features]

    """

    if not isinstance(feature_names, type(None)):
        feature_names = list(feature_names)

    # If mutual is true - augments single features
    if mutual is False:
        for idx, ftr_idx in enumerate(feature_indexes):
            temp_x = operation(x[:, ftr_idx])
            x = np.concatenate((x, temp_x.reshape(-1, 1)), axis=1)
            if not isinstance(feature_names, type(None)):
                feature_names = feature_names + [operation_str + feature_names[ftr_idx]]

    # augments feature combination
    else:
        for idx_1, ftr_idx_1 in enumerate(feature_indexes):
            feature_sub_indexes = feature_indexes[idx_1:]
            if not isinstance(feature_names, type(None)):
                feature1_name = feature_names[ftr_idx_1]

            for idx_2, ftr_idx_2 in enumerate(feature_sub_indexes):
                if ftr_idx_1 != ftr_idx_2:
                    temp_x = operation(x[:, ftr_idx_1], x[:, ftr_idx_2])
                    x = np.concatenate((x, temp_x.reshape(-1, 1)), axis=1)

                    if not isinstance(feature_names, type(None)):
                        feature2_name = feature_names[ftr_idx_2]
                        feature_names = feature_names + [feature1_name + ' ' + operation_str + ' ' + feature2_name]

    if not isinstance(feature_names, type(None)):
        return x, feature_names

    return x

def remove_features(x, feature_names=None, to_del=None):
    """
    Removes features

    Parameters
    ----------
    x : numpy ndarray
        shape[n_samples, n_features]
    feature_names : list or numpy array, optional
        names of features
    to_del :

    """
    x = np.delete(x, to_del, 1)
    if not isinstance(feature_names, type(None)):
        feature_names = np.delete(feature_names, to_del, 0)
        return x, feature_names
    return x

def remove_samples(x, y=None, to_del=None):
    """
    Removes samples

    Parameters
    ----------
    x : numpy ndarray / list / pd.DataFrame
        shape[n_samples, n_features]
    y : list or numpy array, optional
        category reference for each sample
    to_del :

    """
    to_del = np.array(to_del)
    if to_del.dtype == np.bool: # if to_del is array of bools
        if x.__len__() != to_del.__len__():
            raise AssertionError('If to_del is a bool array, must be the same length as x')
        to_del = np.where(to_del)[0]


    if isinstance(x, (np.ndarray, list, pd.DataFrame)): # if x is array of parameters / list - can process also reference y
        if isinstance(x, np.ndarray):
            x = np.delete(x, to_del, 0)
        if isinstance(x, list):
            x = [x_ for idx, x_ in enumerate(x) if idx not in to_del]
        if isinstance(x, pd.DataFrame): # if dataframe
            x = x.drop(to_del, axis=0).reset_index(drop=True)

        if not isinstance(y, type(None)):
            y = np.delete(np.array(y), to_del, 0)
            return x, y
        return x

def balance_classes(x, y, std_factor=0.0):
    """
    Balances unbalanced classes in dataset by extending the sample array with same samples, possibly with introduced
    noise. Detects classes from y variable and number of samples per category. Duplicates samples from the categories
    with lower number of samples. std_factor gives the level of noise introduced into duplicated samples relatively to
    the std of a given dimension for a given category.

    Parameters
    ----------
    x : numpy ndarray
        shape[n_samples, n_features]
    y : list or numpy array
        string or int indexes for each category
    std_factor : float
        Amount of noise introduced into duplicated features relatively to std of a given feature within a category.

    Returns
    -------
    numpy ndarray
        x - samples
    list
        y - categories
    """

    # Data augmentation
    cat_members = np.array([(y == c).sum() for c in np.unique(y)])
    for idx, cat in enumerate(np.unique(y)):
        num = (y == cat).sum()
        target_num = cat_members.max()
        generate = target_num - num
        if generate > 0:
            src_idxes = np.array(list(np.where(y == cat)[0]) * int(np.ceil(generate / num)))
            x_aug = x[src_idxes, :]
            x_aug = x_aug + (np.random.randn(x_aug.shape[0], x_aug.shape[1]) * x_aug.std(axis=0) * std_factor)
            x_aug = x_aug[:generate]
            x = np.concatenate((x,  x_aug[:generate]), axis=0)
            y = np.array(list(y) + [cat] * generate)
    return x, y

def replace_annotations(Y, old_key=None, new_key=None):
    Y = list(Y)
    if not isinstance(old_key, list):
        old_key = [old_key]
    Y_new = []
    for Y_ in Y:
        if Y_ in old_key:
            Y_new.append(new_key)
        else:
            Y_new.append(Y_)
    return np.array(Y_new)


