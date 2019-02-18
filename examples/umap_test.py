from dalys.umap_algo import UMAPTool
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

names = ['Setosa', 'Versicolour', 'Virginica']
styles = [('red', '.'), ('blue', '.'), ('green', '.')]

umap_tool = UMAPTool(x, y, labels_unique_name=names, style=styles, n_neighbors=50,
                     min_dist=0.1, metric='euclidean', preprocessing='norm_l1')
umap_tool.projections_plot()