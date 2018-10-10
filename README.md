# Dalys - library for visualising data analysis algorithms

Dalys - library allows visualised many algorithms that need for statistical data processing and machine learning

## Getting Started

Dalys is a wrapper over the library [scikit-learn](scikit-learn.org/) and you can use [scikit-learn](scikit-learn.org/) datasets 
for visualize desired algorithm.

Example:

```python
from sklearn.datasets import make_circles
from dalys.kernel_pca import KPCATool


X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

style = [('red', '.'), ('blue', '.')]
label_names = ['red dotes', 'blue dotes']

kpca = KPCATool(X, y, style=style, labels_unique_name=label_names, n_components=3,
                kernel='rbf', gamma=2, scaled=True)
kpca.projections_plot(grid=22)
```
Example of gridplot with *grid*=22:

![Kernel PCA gridplot](img/KPCATool_2D_subplot.png)

There is a number of parameters that can be set for the KPCATool class;
the major ones are as follows:

* *style* - set marker styles and colors for data representation, 
example: style = [('red', '.'), ('blue', '.')] or
style = [('red', 'd'), ('blue', 'd')]. If you need random colors and markers - leave this argument with default parameter (default = None).

* *labels_unique_name* - set list with class names, such as labels_unique_name = ['red dotes', 'blue dotes']. If *labels_unique_name* is *None*, this means that labels tags will be numbered in ascending order.

* *scaled* - if your data is already transformed(normalized, standardize or whitening and so all) - you must set "True" for this flag to prevent data preprocessing,
if you need to standardize your data - you must use tools with default argument (default = False).

* *scale_axis* - set axis for data standardize (default = 0)


### Installing

Requirements:
* numpy >= 0.15.1
* scikit-learn >= 0.19.0
* matplotlib >= 2.2.2

## Authors

* **Timothy Tkachenko** - [Jazzros company](http://www.jazzros.com/ua) researcher