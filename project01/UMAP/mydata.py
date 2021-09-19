from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper, ColumnDataSource
from bokeh.palettes import Category10

import umap.umap_ as umap
from sklearn.datasets import load_wine

wine = load_wine()
embedding = umap.UMAP(
    n_neighbors=30, learning_rate=0.5, init="random", min_dist=0.001
).fit_transform(wine.data)

output_file("wine.html")


targets = [str(d) for d in wine.target_names]

source = ColumnDataSource(
    dict(
        x=[e[0] for e in embedding],
        y=[e[1] for e in embedding],
        label=[targets[d] for d in wine.target],
    )
)

cmap = CategoricalColorMapper(factors=targets, palette=Category10[10])

p = figure(title="Test UMAP on Wine dataset")
p.circle(
    x="x",
    y="y",
    source=source,
    color={"field": "label", "transform": cmap},
    legend="label",
)

show(p)
