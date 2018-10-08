from sklearn.manifold import TSNE
from dalys.base_estimator import BaseEstimator


class TSNETool(BaseEstimator):
    def __init__(self, samples, labels, n_components=2, style=None, labels_unique_name=None, perplexity=30,
                 init='random', n_iter=1000, n_iter_without_progress=300,
                 early_exaggeration=12, verbose=0, scale_axis=0, scaled=False):
        super().__init__(samples, labels, n_components, style, labels_unique_name, scale_axis, scaled)
        self._ca = TSNE(n_components=self._n_components, perplexity=perplexity,
                        init=init, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress,
                        early_exaggeration=early_exaggeration, verbose=verbose)
        self._reduce = self._ca.fit_transform(self._scaled_data)
        self._fill_components()
        self._extract_classes()

    def set_params(self, n_components, perplexity=30, init='random',
                   n_iter=1000, n_iter_without_progress=300, early_exaggeration=12,
                   verbose=0, scale_axis=0, scaled=False):
        self.__init__(self._samples, self._labels, n_components=n_components, style=self._style,
                      labels_unique_name=self._labels_unique_name, perplexity=perplexity,
                      init=init, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress,
                      early_exaggeration=early_exaggeration, verbose=verbose, scale_axis=scale_axis,
                      scaled=scaled)


if __name__ == '__main__':
    from sklearn.datasets import load_digits

    digits = load_digits()
    x = digits.data
    y = digits.target

    names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    tsne = TSNETool(x, y, labels_unique_name=names, n_components=3, n_iter=5000, n_iter_without_progress=1000)
    tsne.projections_plot(grid=22)
    tsne.projections_plot_3d()
