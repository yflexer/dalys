import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from itertools import permutations
from utils import marker_dict, colors
from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    def __init__(self, samples, labels, n_components, labels_unique_name=None, scale_axis=0, scaled=False):
        self._markers = [key for key in marker_dict.keys()]
        self._samples = samples
        self._labels = labels
        self._unique_labels = np.unique(labels)
        self._labels_unique_name = labels_unique_name
        self._legend = self._labels_unique_name if labels_unique_name else self._unique_labels
        self._n_classes = len(self._unique_labels)
        self._scaled = scaled
        self._scale_axis = scale_axis
        self._n_components = n_components
        self._ca = None
        self._class_list = None
        self._scaled_data = None
        self._style = None
        self._reduce = None
        self._components_list = list([list() for i in range(self._n_components)])
        self._get_scale_data(samples)

    @abstractmethod
    def set_params(self):
        pass

    def get_reduce_data(self):
        return self._reduce

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
                    self._class_list[self._labels[i]][j].append(self._components_list[j][i])

    def projections_plot(self):
        plt.rcParams.update({'figure.max_open_warning': 0})
        self._generate_styles()
        if len(self._reduce[0]) > 1:
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
                    plt.legend(self._legend)
            plt.show()
            return
        for m in range(self._n_classes):
            item = self._class_list[m]
            color, marker = self._style[m]
            plt.scatter(item[0], np.zeros(len(item[0])), c=color, marker=marker)
            plt.legend(self._legend)
        plt.show()

