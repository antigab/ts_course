import numpy as np
import holtwinters as hw
from sklearn.metrics import mean_absolute_error as mae


def mase(y_pred, y_true, method='naive', X_test=None, constant=None):
    """
    Mean absolute scaled error. MAE error of your predictions, normalized by
    MAE error of different methods predictions.

    Parameters
    -----------
    y_pred : sequence
        Predictions you want to compare to with different methods.
    y_true: sequence
        True values
    method: {'naive', 'exp_smooth', 'mean', 'median', 'constant'}
        The method used to generate y_method which is predictions to compare to
        predictions of your method
    X_test: pd.Dataframe object, optional
        Must be provided when using all methods but naive and constant
    constant: int, optional
        Must be provided if method arg is set to constant

    Returns
    --------
    mase_score : range(0,1)
        The score, that is computed as following -
        mae(y_true, y_pred)/mae(y_true, y_method). For example if method
        is 'naive' and mase score is 0.25, that means that your method is 4
        times more accurate, then the naive one.
    """

    y_method = y_pred
    if method is 'naive':
        y_method = y_true.shift()
        y_method.fillna(y_method.mean(), inplace=True)
    if method is not 'naive':
        if X_test is None:
            print('You should provide X_test to evaluate predict')
        X_test.drop([label for label in X_test.columns if 'lag_' in label],
                    inplace=True, axis=1)
    if method is 'exp_smooth':
        num_lags = len(X_test.columns)
        y_method = [hw.additive(list(lags[1].values), num_lags, 1)[0][0]
                    for lags in X_test.iterrows()]
    if method is 'mean':
        y_method = X_test.mean(axis=1).values
    if method is 'median':
        y_method = X_test.mean(axis=1).values
    if method is 'constant':
        y_method = np.full(y_true.shape, constant)
    return mae(y_true, y_pred) / mae(y_true, y_method)  # todo fix division by zero
