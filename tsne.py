from sklearn.manifold import TSNE
from base_estimator import BaseEstimator


class TSNEAnalisys(BaseEstimator):
    def __init__(self, samples, labels, n_components=2, perplexity=30,
                 init='random', n_iter=1000, n_iter_without_progress=300,
                 early_exaggeration=12, verbose=0, scale_axis=0, scaled=False):
        super().__init__(samples, labels, scale_axis=scale_axis, scaled=scaled)
        self._n_components = n_components
        self._ca = TSNE(n_components=self._n_components, perplexity=perplexity,
                        init=init, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress,
                        early_exaggeration=early_exaggeration, verbose=verbose)
        self._components_list = list([list() for i in range(self._n_components)])
        self._reduce = self._ca.fit_transform(self._scaled_data)
        self._fill_components()
        self._extract_classes()

    def set_params(self, n_components=2, perplexity=30, init='random',
                   n_iter=1000, n_iter_without_progress=300, early_exaggeration=12,
                   verbose=0, scale_axis=0, scaled=False):
        self.__init__(self._samples, self._labels, n_components=n_components, perplexity=perplexity,
                      init=init, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress,
                      early_exaggeration=early_exaggeration, verbose=verbose, scale_axis=scale_axis,
                      scaled=scaled)


if __name__ == '__main__':
    import mnist
    from pca import PCAAnalisys

    mnist_train_images = mnist.train_images()[:1000].reshape(1000, 28*28)
    mnist_train_labels = mnist.train_labels()[:1000]
    mnist_test_images = mnist.test_images()[:1000].reshape(1000, 28*28)
    mnist_test_labels = mnist.test_labels()[:1000]

    pca = PCAAnalisys(mnist_train_images, mnist_train_labels, n_components=50)
    data = pca.get_reduce_data()
    tsne = TSNEAnalisys(data, mnist_train_labels, n_components=2, perplexity=40, scaled=True, verbose=1)
    tsne.projections_plot()

    '''
    print(mnist_test_images.shape)
    row, col, _ = mnist_test_images.shape
    mnist_test_images = mnist_test_images.reshape(row, 28*28)
    my_pca = PCAAnalisys(mnist_test_images, mnist_test_labels)
    my_pca.variance_explained_init(explained_variance=0.95)
    tsne = TSNE(n_components=2, perplexity=10)
    reduce = my_pca.get_reduce_data()
    tsne_data = tsne.fit_transform(reduce)
    print(tsne_data)
    x = [item[0] for item in tsne_data]
    y = [item[1] for item in tsne_data]
    plt.scatter(x, y)
    plt.show()
    '''
