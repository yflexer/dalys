from sklearn.decomposition import KernelPCA
from base_estimator import BaseEstimator


class kPCATool(BaseEstimator):
    def __init__(self, samples, labels, labels_unique_name=None, n_components=2, kernel='rbf',
                 fit_inverse_transform=True, gamma=1, scale_axis=0, scaled=False):
        super().__init__(samples, labels, n_components=n_components, labels_unique_name=labels_unique_name,
                         scale_axis=scale_axis, scaled=scaled)
        self._ca = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=fit_inverse_transform,
                             gamma=gamma)
        self._reduce = self._ca.fit_transform(self._scaled_data)
        self._fill_components()
        self._extract_classes()

    def set_params(self, n_components, kernel='rbf', fit_inverse_transform=True,
                   gamma=1, scale_axis=0, scaled=False):
        self.__init__(self._samples, self._labels, n_components=n_components, kernel=kernel,
                      fit_inverse_transform=fit_inverse_transform, gamma=gamma, scale_axis=scale_axis,
                      scaled=scaled)

    def inverse_transform(self, x):
        return self._ca.inverse_transform(x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
    plt.show()
    kpca = kPCATool(X, y, n_components=2, kernel='rbf', gamma=2, scaled=True)
    kpca.projections_plot(style=[('red', '.'), ('blue', '.')])
