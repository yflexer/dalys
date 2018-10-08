from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dalys.base_estimator import BaseEstimator


class LDATool(BaseEstimator):
    def __init__(self, samples, labels, n_components=3, style=None, labels_unique_name=None, scale_axis=0,
                 scaled=False):
        super().__init__(samples, labels, n_components, style, labels_unique_name, scale_axis, scaled)
        self._ca = LinearDiscriminantAnalysis(n_components=self._n_components)
        self._reduce = self._ca.fit_transform(self._scaled_data, self._labels)
        self._fill_components()
        self._extract_classes()

    def set_params(self, n_components, scale_axis=0, scaled=False):
        self.__init__(self._samples, self._labels, n_components=n_components, style=self._style,
                      labels_unique_name=self._labels_unique_name,
                      scale_axis=scale_axis, scaled=scaled)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    names = ['Setosa', 'Versicolour', 'Virginica']
    style = [('red', '.'), ('blue', '.'), ('green', '.')]

    plt.scatter(X[y == 0, 3], X[y == 0, 1], color='red')
    plt.scatter(X[y == 1, 3], X[y == 1, 1], color='blue')
    plt.scatter(X[y == 2, 3], X[y == 2, 1], color='green')
    plt.ylabel('sepal width')
    plt.xlabel('petal width')
    plt.legend(names)

    lda = LDATool(X, y, style=style, labels_unique_name=names, n_components=2)
    lda.projections_plot()

    lda.set_params(n_components=1)
    lda.projections_plot()
