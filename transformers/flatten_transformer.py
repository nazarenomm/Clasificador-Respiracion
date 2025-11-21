from sklearn.base import BaseEstimator, TransformerMixin


class FlattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        '''Transformador para aplanar matrices multidimensionales.
        '''
        return self

    def transform(self, X):
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)