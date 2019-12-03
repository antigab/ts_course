import pandas as pd
from datetime import timedelta

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline


from transformations import TimeSeriesTransformer, FeatureGenerator


class TimeSeriesPredictor:
    def __init__(self, num_lags=14, granularity='hour'):
        self.num_lags = num_lags
        self.granularity = granularity
        self.model = None

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
        delta = timedelta(hours=1) if self.granularity == 'hour' else timedelta(days=1)
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
        # model = LinearRegression(**kwargs)
        model = GradientBoostingRegressor()
        model.fit(X, y)
        self.model = model

    def predict_batch(self, ts, ts_batch):
        if not self.model:
            raise ValueError('Model is not fitted yet')

        unite_ts = ts.append(ts_batch)

        data_batch = self.transformer.transform(unite_ts)[-len(ts_batch):]
        preds = self.model.predict(data_batch.drop('y', axis=1))

        return pd.Series(index=data_batch.index, data=preds)

    def predict_next(self, ts):
        if not self.model:
            raise ValueError('Model is not fitted yet')
        row = self.generate_next_row(ts)
        row = self.enrich(row)
        value = self.model.predict(row)
        return pd.Series(value, index=row.index)
