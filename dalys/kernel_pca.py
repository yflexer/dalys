from sklearn.decomposition import KernelPCA
from dalys.base_estimator import BaseEstimator


class KPCATool(BaseEstimator):
    def __init__(self, samples, labels, n_components=2, style=None, labels_unique_name=None, kernel='rbf',
                 fit_inverse_transform=True, gamma=1, scale_axis=0, scaled=False):
        super().__init__(samples, labels, n_components, style, labels_unique_name, scale_axis, scaled)
        self._ca = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=fit_inverse_transform,
                             gamma=gamma)
        self._reduce = self._ca.fit_transform(self._scaled_data)
        self._fill_components()
        self._extract_classes()

    def set_params(self, n_components, kernel='rbf', fit_inverse_transform=True,
                   gamma=1, scale_axis=0, scaled=False):
        self.__init__(self._samples, self._labels, n_components=n_components, style=self._style,
                      labels_unique_name=self._labels_unique_name, kernel=kernel,
                      fit_inverse_transform=fit_inverse_transform, gamma=gamma, scale_axis=scale_axis,
                      scaled=scaled)

    def inverse_transform(self, x):
        return self._ca.inverse_transform(x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.datasets import make_circles, make_moons
    from sklearn.metrics.pairwise import rbf_kernel

    style = [('red', '.'), ('blue', '.')]

    # circles
    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')

    # circles get rbf kernel
    rbf_x_b = [item[0] for i, item in enumerate(X) if y[i] == 1]
    rbf_y_b = [item[1] for i, item in enumerate(X) if y[i] == 1]
    rbf_x_r = [item[0] for i, item in enumerate(X) if y[i] == 0]
    rbf_y_r = [item[1] for i, item in enumerate(X) if y[i] == 0]
    z_b = [np.sum(item) for item in rbf_kernel(list(zip(rbf_x_b, rbf_y_b)), gamma=2)]
    z_r = [np.sum(item) for item in rbf_kernel(list(zip(rbf_x_r, rbf_y_r)), gamma=2)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rbf_x_b, rbf_y_b, z_b, color='blue')
    ax.scatter(rbf_x_r, rbf_y_r, z_r, color='red')

    # circles 3 components projections
    kpca = KPCATool(X, y, style=style, n_components=3, kernel='rbf', gamma=2, scaled=True)
    kpca.projections_plot(grid=22)

    # circles 1 components
    kpca.set_params(n_components=1, kernel='rbf', gamma=2, scaled=True)
    kpca.projections_plot()

    # moons
    X, y = make_moons(n_samples=100, shuffle=False, noise=0)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')

    # moons 3 components projections
    kpca = KPCATool(X, y, style=style, n_components=3, kernel='rbf', gamma=100, scaled=True)
    kpca.projections_plot(grid=22)

    # moons 1 components
    kpca.set_params(n_components=1, kernel='rbf', gamma=100, scaled=True)
    kpca.projections_plot()
