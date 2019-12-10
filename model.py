import pandas as pd
from datetime import timedelta
from monthdelta import monthdelta
from copy import deepcopy

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline


from transformations import TimeSeriesTransformer, FeatureGenerator


class TimeSeriesPredictor:
    def __init__(self, num_lags=14, granularity='hour', sigma=2.3):
        self.num_lags = num_lags
        self.granularity = granularity
        self.model = None
        self.sigma = sigma
        self.std = None

    def transform(self, ts):
        transformer = TimeSeriesTransformer(num_lags=self.num_lags)
        lag_matrix = transformer.transform(ts)
        return lag_matrix

    def enrich(self, matrix):
        enricher = FeatureGenerator()
        feature_matrix = enricher.transform(matrix)
        return feature_matrix

    def generate_next_row(self, ts):
        """
        Takes time-series as an input and returns next raw, that is fed to the fitted model,
        when predicting next value.
        Parameters
        ----------
        ts : pd.Series(values, timestamps)
            Time-series to detect on
        num_lags : int, default=14
            Defines the number of lag features
        granularity : str, default='day'
            Defines timedelta of your series
        Returns
        ---------
        feature_matrix : pd.DataFrame
            Pandas dataframe, which contains feature lags of
            shape(1, num_lags)
        """
        if not self.granularity:
            raise ValueError('No granularity provided')
        if self.granularity == 'hour':
            delta = timedelta(hours=1)
        if self.granularity == 'day':
            delta = timedelta(days=1)
        if self.granularity == 'month':
            delta = monthdelta(1)
        next_timestamp = pd.to_datetime(ts.index[-1]) + delta
        lag_dict = {'lag_{}'.format(i): [ts[-i]] for i in range(1, self.num_lags + 1)}
        lag_dict.update({'season_lag': ts[-self.num_lags]})
        df = pd.DataFrame.from_dict(lag_dict)
        df.index = [next_timestamp]
        return df

    def fit(self, ts, **kwargs):
        lag_matrix = self.transform(ts)
        feature_matrix = self.enrich(lag_matrix)
        X, y = feature_matrix.drop('y', axis=1), feature_matrix['y']
        model = LinearRegression(**kwargs)
        # model = GradientBoostingRegressor()
        model.fit(X, y)
        self.model = model

    def predict_batch(self, ts, ts_batch=pd.Series([])):
        if not self.model:
            raise ValueError('Model is not fitted yet')

        unite_ts = ts.append(ts_batch)
        matrix = self.enrich(self.transform(unite_ts))

        data_batch = matrix[-len(ts_batch):]
        preds = self.model.predict(data_batch.drop('y', axis=1))

        return pd.Series(index=data_batch.index, data=preds)

    def predict_next(self, ts, k=1):
        if not self.model:
            raise ValueError('Model is not fitted yet')
        ts = deepcopy(ts)
        predictions = pd.Series()
        for _ in range(k):
            row = self.generate_next_row(ts)
            row = self.enrich(row)
            value = self.model.predict(row)
            ts = ts.append(pd.Series(value, index=row.index))
            predictions = predictions.append(pd.Series(value, index=row.index))
        return predictions

    def fit_statistics(self, ts):
        preds = self.predict_batch(ts)
        std = (ts - preds).std()
        self.std = std

    def fit_seasonal_statistics(self, ts):
        return None

    def analyze(self, ts_true, ts_pred):
        lower, upper = self.get_prediction_intervals(ts_pred)
        return ts_true[(ts_true < lower) | (ts_true > upper)]

    def get_prediction_intervals(self, y_pred):
        lower, upper = y_pred - self.sigma * self.std, y_pred + self.sigma * self.std
        return lower, upper