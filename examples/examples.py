import matplotlib.pyplot as plt
from dalys.lda import LDATool
from dalys.pca import PCATool
from dalys.kernel_pca import KPCATool
from dalys.tsne import TSNETool
from sklearn.datasets import make_circles, make_moons, load_digits, load_iris

# Kernel PCA for moon and circles data

style = [('red', '.'), ('blue', '.')]

# circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')

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

########################################################################################

# LDA for iris dataset

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

########################################################################################

# PCA for MNIST dataset

digits = load_digits()
x = digits.data
y = digits.target

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
pca = PCATool(x, y, labels_unique_name=names, n_components=3)
pca.projections_plot(grid=22)

########################################################################################

# TSNE for MNIST dataset

digits = load_digits()
x = digits.data
y = digits.target

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
tsne = TSNETool(x, y, labels_unique_name=names, n_components=3, n_iter=5000, n_iter_without_progress=1000)
tsne.projections_plot(grid=22)
tsne.projections_plot_3d()
