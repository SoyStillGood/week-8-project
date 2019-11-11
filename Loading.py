import numpy as np
import pandas as pd
from sklearn.datasets import load_wine


wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
print(df.head())
