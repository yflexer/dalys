import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from base_estimator import BaseEstimator


class LDATool(BaseEstimator):
    def __init__(self, samples, labels, labels_unique_name=None, n_components=3, scale_axis=0, scaled=False):
        super().__init__(samples, labels, labels_unique_name=labels_unique_name,
                         scale_axis=scale_axis, scaled=scaled)
        self._n_components = n_components
        self._ca = LinearDiscriminantAnalysis(n_components=self._n_components)
        self._components_list = list([list() for i in range(self._n_components)])
        self._reduce = self._ca.fit_transform(self._scaled_data, self._labels)
        self._fill_components()
        self._extract_classes()

    def set_params(self, n_components, scale_axis=0, scaled=False):
        self.__init__(self._samples, self._labels, n_components=n_components,
                      scale_axis=scale_axis, scaled=scaled)


if __name__ == '__main__':
    '''
    import mnist
    mnist_train_images = mnist.train_images()[:1000]
    mnist_train_labels = mnist.train_labels()[:1000]
    mnist_test_images = mnist.test_images()[:1000]
    mnist_test_labels = mnist.test_labels()[:1000]
    '''
    from sklearn.datasets import load_iris
    iris = load_iris()
    x = iris.data
    y = iris.target

    names = ['Setosa', 'Versicolour', 'Virginica']
    #mnist_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    lda = LDATool(x, y, labels_unique_name=names, n_components=2)
    lda.projections_plot()