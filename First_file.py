import argparse
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns

# import scikit-learn as skt


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", dest="file", help="input data file")

args = parser.parse_args()
file = args.file

wine = pd.read_csv(filepath_or_buffer=file, sep=',', header=0)
wine_df = pd.DataFrame(data=wine)
#print(wine_df.columns)

sns.set()

#wine_sns = sns.load_dataset("wine_df")

g = sns.lmplot(x=wine_df[" Alcohol"], y=wine_df[" Malic Acid"], hue=wine_df["Instance"], data=wine_df)

g.set_axis_labels("Alcohol", "Malic Acid")

plt.show()
