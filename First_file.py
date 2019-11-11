import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine


wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)


print(wine_df.columns)

sns.set()

#wine_sns = sns.load_dataset("wine_df")

g = sns.lmplot(x=wine_df["alcohol"], y=wine_df["malic_acid"], hue=wine_df["hue"], data=wine_df)

g.set_axis_labels("Alcohol", "Malic Acid")

plt.show()
