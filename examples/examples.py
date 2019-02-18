import matplotlib.pyplot as plt
from dalys.lda import LDATool
from dalys.pca import PCATool
from dalys.kernel_pca import KPCATool
from dalys.tsne import TSNETool
from dalys.umap_algo import UMAPTool
from sklearn.datasets import make_circles, make_moons, load_digits, load_iris

# Kernel PCA for moon and circles data

style = [('red', '.'), ('blue', '.')]

# circles
x, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue')

# circles 3 components projections
kpca = KPCATool(x, y, style=style, n_components=3, kernel='rbf', gamma=2, scaled=True)
kpca.projections_plot(grid=22)

# circles 1 components
kpca.set_params(n_components=1, kernel='rbf', gamma=2, scaled=True)
kpca.projections_plot()

# moons
x, y = make_moons(n_samples=100, shuffle=False, noise=0)

plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue')

# moons 3 components projections
kpca = KPCATool(x, y, style=style, n_components=3, kernel='rbf', gamma=100, scaled=True)
kpca.projections_plot(grid=22)

# moons 1 components
kpca.set_params(n_components=1, kernel='rbf', gamma=100, scaled=True)
kpca.projections_plot()

########################################################################################

# LDA for iris dataset

iris = load_iris()
x = iris.data
y = iris.target

names = ['Setosa', 'Versicolour', 'Virginica']
style = [('red', '.'), ('blue', '.'), ('green', '.')]

plt.scatter(x[y == 0, 3], x[y == 0, 1], color='red')
plt.scatter(x[y == 1, 3], x[y == 1, 1], color='blue')
plt.scatter(x[y == 2, 3], x[y == 2, 1], color='green')
plt.ylabel('sepal width')
plt.xlabel('petal width')
plt.legend(names)

lda = LDATool(x, y, style=style, labels_unique_name=names, n_components=2, scaled=True)
lda.projections_plot()

lda.set_params(n_components=1)
lda.projections_plot()

########################################################################################

# PCA for MNIST dataset

digits = load_digits()
x = digits.data
y = digits.target

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
pca = PCATool(x, y, labels_unique_name=names, n_components=3, scaled=True)
pca.projections_plot(grid=22)

########################################################################################

# TSNE for MNIST dataset

digits = load_digits()
x = digits.data
y = digits.target

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
tsne = TSNETool(x, y, labels_unique_name=names, n_components=3, n_iter=5000,
                n_iter_without_progress=1000, scaled=True)
tsne.projections_plot(grid=22)
tsne.projections_plot_3d()

########################################################################################

# UMAP for MNIST dataset

digits = load_digits()
x = digits.data
y = digits.target

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
umap_tool = UMAPTool(x, y, labels_unique_name=names, n_neighbors=150,
                     min_dist=0.1, metric='euclidean', scaled=True)
umap_tool.projections_plot()
