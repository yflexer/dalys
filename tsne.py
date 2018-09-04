from sklearn.manifold import TSNE
from base_estimator import BaseEstimator


class TSNETool(BaseEstimator):
    def __init__(self, samples, labels, labels_unique_name=None, n_components=2, perplexity=30,
                 init='random', n_iter=1000, n_iter_without_progress=300,
                 early_exaggeration=12, verbose=0, scale_axis=0, scaled=False):
        super().__init__(samples, labels, n_components, labels_unique_name=labels_unique_name,
                         scale_axis=scale_axis, scaled=scaled)
        self._ca = TSNE(n_components=self._n_components, perplexity=perplexity,
                        init=init, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress,
                        early_exaggeration=early_exaggeration, verbose=verbose)
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
    '''
    import mnist
    from pca import PCAAnalisys

    mnist_train_images = mnist.train_images()[:1000].reshape(1000, 28*28)
    mnist_train_labels = mnist.train_labels()[:1000]
    mnist_test_images = mnist.test_images()[:1000].reshape(1000, 28*28)
    mnist_test_labels = mnist.test_labels()[:1000]

    pca = PCAAnalisys(mnist_train_images, mnist_train_labels, n_components=100)
    data = pca.get_reduce_data()
    tsne = TSNEAnalisys(data, mnist_train_labels, n_components=2, perplexity=20, scaled=True, verbose=1, n_iter=10000)
    tsne.projections_plot()
    '''
    from sklearn.datasets import load_digits
    digits = load_digits()
    x = digits.data
    y = digits.target

    names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    #pca = PCAAnalisys(mnist_train_images, mnist_train_labels, labels_unique_name=names, n_components=3)
    tsne = TSNETool(x, y, labels_unique_name=names, n_components=2)
    tsne.projections_plot()
