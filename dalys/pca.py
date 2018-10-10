import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dalys.base_estimator import BaseEstimator


class PCATool(BaseEstimator):
    def __init__(self, samples, labels, n_components=2, style=None, labels_unique_name=None,
                 scale_axis=0, scaled=False):
        super().__init__(samples, labels, n_components, style, labels_unique_name, scale_axis, scaled)
        self._ca = PCA(n_components=self._n_components)
        self._reduce = self._ca.fit_transform(self._scaled_data)
        self._fill_components()
        self._extract_classes()
        self._pca = PCA(n_components=len(self._scaled_data[0])).fit(self._scaled_data)

    def set_params(self, n_components, scale_axis=0, scaled=False):
        self.__init__(self._samples, self._labels, n_components=n_components, style=self._style,
                      labels_unique_name=self._labels_unique_name, scale_axis=scale_axis, scaled=scaled)

    def explained_variance(self):
        return self._ca.explained_variance_

    def explained_deviation(self):
        return np.sqrt(self._ca.explained_variance_)

    def explained_variance_ratio(self):
        return self._ca.explained_variance_ratio_

    def explained_variance_ratio_cumsum(self):
        return self._ca.explained_variance_ratio_.cumsum()

    def explained_plot(self):
        plt.plot(self._pca.explained_variance_ratio_.cumsum())
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    def components_for_explained_variance(self, explained_variance=0.80):
        return next(x[0] for x in enumerate(self._pca.explained_variance_ratio_.cumsum())
                    if x[1] > explained_variance)

    def variance_explained_init(self, explained_variance=0.80, scale_axis=0, scaled=False):
        n_components = self.components_for_explained_variance(explained_variance)
        self.set_params(n_components, scale_axis=scale_axis, scaled=scaled)