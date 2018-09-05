import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from base_estimator import BaseEstimator


class PCATool(BaseEstimator):
    def __init__(self, samples, labels, labels_unique_name=None, n_components=2, scale_axis=0, scaled=False):
        super().__init__(samples, labels, n_components, labels_unique_name=labels_unique_name,
                         scale_axis=scale_axis, scaled=scaled)
        self._ca = PCA(n_components=self._n_components)
        self._reduce = self._ca.fit_transform(self._scaled_data)
        self._fill_components()
        self._extract_classes()
        self._pca = PCA(n_components=len(self._scaled_data[0])).fit(self._scaled_data)

    def set_params(self, n_components, scale_axis=0, scaled=False):
        self.__init__(self._samples, self._labels, n_components=n_components,
                      scale_axis=scale_axis, scaled=scaled)

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

    def variance_explained_init(self, explained_variance=0.80, scale_axis=0):
        n_components = self.components_for_explained_variance(explained_variance)
        self.set_params(n_components, scale_axis=scale_axis)


if __name__ == '__main__':
    #import mnist
    #mnist_train_images = mnist.train_images()[:1000]
    #mnist_train_labels = mnist.train_labels()[:1000]
    #mnist_test_images = mnist.test_images()[:1000]
    #mnist_test_labels = mnist.test_labels()[:1000]

    from sklearn.datasets import load_digits
    digits = load_digits()
    x = digits.data
    y = digits.target
    names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    pca = PCATool(x, y, labels_unique_name=names, n_components=2)
    pca.projections_plot()
