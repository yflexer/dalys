import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from itertools import permutations
from utils import marker_dict, colors
from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    def __init__(self, samples, labels, scale_axis=0, scaled=False):
        self._markers = [key for key in marker_dict.keys()]
        self._samples = samples
        self._labels = labels
        self._unique_labels = np.unique(labels)
        self._n_classes = len(self._unique_labels)
        self._scaled = scaled
        self._scale_axis = scale_axis
        self._ca = None
        self._class_list = None
        self._scaled_data = None
        self._style = None
        self._reduce = None
        self._get_scale_data(samples)

    @abstractmethod
    def set_params(self):
        pass

    def _generate_styles(self):
        self._style, previous = list(), None
        while len(self._style) != self._n_classes:
            color = np.random.choice(colors)
            marker = np.random.choice(self._markers)
            if previous == color:
                continue
            previous = color
            self._style.append((color, marker))

    def _get_scale_data(self, samples):
        self._scaled_data = scale([item.flatten() for item in samples],
                                  axis=self._scale_axis) if not self._scaled else samples

    def _fill_components(self):
        for i in range(len(self._scaled_data)):
            for j in range(self._n_components):
                self._components_list[j].append(self._reduce[i][j])

    def _extract_classes(self):
        self._class_list = list([[list() for i in range(self._n_components)]
                                for j in range(self._n_classes)])
        for i in range(len(self._labels)):
            for j in range(self._n_components):
                for item in self._unique_labels:
                    if self._labels[i] == item:
                        self._class_list[item][j].append(self._components_list[j][i])

    def projections_plot(self):
        plt.rcParams.update({'figure.max_open_warning': 0})
        self._generate_styles()
        s_labels = ''.join(str(c) for c in range(self._n_components))
        perms = np.sort(list(permutations(s_labels, 2)))
        perms = list(set((int(a), int(b)) if a <= b else (int(a), int(b)) for a, b in perms))
        for i in range(len(perms)):
            k = perms[i][0]
            j = perms[i][1]
            plt.figure(i)
            for m in range(self._n_classes):
                item = self._class_list[m]
                color, marker = self._style[m]
                plt.scatter(item[k], item[j], c=color, marker=marker)
                plt.legend(self._unique_labels)
        plt.show()


if __name__ == '__main__':
    import mnist
    from sklearn.datasets import load_digits
    mnist_train_images = mnist.train_images()[:7000]
    mnist_train_labels = mnist.train_labels()[:7000]
    mnist_test_images = mnist.test_images()[:1000]
    mnist_test_labels = mnist.test_labels()[:1000]
    digits = load_digits()
    ca = BaseEstimator(mnist_train_images, mnist_train_labels)