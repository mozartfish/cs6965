"""
The data set used was kdcup99 smt dataset from scikitlearn 

"""
# sphinx_gallery_thumbnail_path = '../examples/breast-cancer/breast-cancer-d3.png'

import sys

try:
    import pandas as pd
except ImportError as e:
    print(
        "pandas is required for this example. Please install with `pip install pandas` and then try again."
    )
    sys.exit()

import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble
from sklearn.datasets import fetch_kddcup99

# For data we use the Wisconsin Breast Cancer Dataset
# Via:
data = fetch_kddcup99(subset='smtp').data


# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)


lens = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())



# Create the simplicial complex
graph = mapper.map(
    lens,
    data,
    cover=km.Cover(n_cubes=15, perc_overlap=0.4),
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
)

# Visualization
mapper.visualize(
    graph,
    path_html="output/kdd_cup.html",
    title="kddCup Dataset",
)





import matplotlib.pyplot as plt

km.draw_matplotlib(graph)
plt.show()
