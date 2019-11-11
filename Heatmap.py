import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.datasets import load_wine


wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
print(wine.target)

print(wine_df.columns)


os.makedirs('plots/seaborn_heatmap', exist_ok=True)

sns.set()

fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(wine_df.corr(), annot=True, ax=ax, cmap='ocean', fmt='.2f', annot_kws={"size": 15}, linewidths=.05)
ax.set_xticklabels(wine_df.columns, rotation=45)
ax.set_yticklabels(wine_df.columns, rotation=45)
fig.subplots_adjust(top=.75)
plt.savefig('plots/seaborn_heatmap/wine_heatmap.png')
# plt.tight_layout()
plt.close()

wine_df.hist(bins=14, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=6, ylabelsize=6, grid=False)
# plt.tight_layout(rect=(0, 0, 1.2, 1.2))
plt.savefig('plots/seaborn_heatmap/wine_histogram.png')

pairs_plot = sns.pairplot(wine_df)
plt.savefig('plots/seaborn_heatmap/wine-pairsplot.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Alcohol % by Class', fontsize=14)

sns.boxplot(x=wine.target, y="alcohol", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Alcohol %",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-alcohol.png')


f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Malic Acid by Class', fontsize=14)

sns.boxplot(x=wine.target, y="malic_acid", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Malic Acid",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-malic-acid.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Ash by Class', fontsize=14)

sns.boxplot(x=wine.target, y="ash", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Ash",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-ash.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Alcalinity of Ash by Class', fontsize=14)

sns.boxplot(x=wine.target, y="alcalinity_of_ash", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Alcalinity of Ash",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-alc-ash.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Magnesium by Class', fontsize=14)

sns.boxplot(x=wine.target, y="magnesium", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Magenesium",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-mag.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Proanthocyanins by Class', fontsize=14)

sns.boxplot(x=wine.target, y="proanthocyanins", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Proanthocyanins",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-pro.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Flavanoids by Class', fontsize=14)

sns.boxplot(x=wine.target, y="flavanoids", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Flavanoids",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-flavanoids.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Non Flavanoid Phenols by Class', fontsize=14)

sns.boxplot(x=wine.target, y="nonflavanoid_phenols", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Non Flavanoid Phenols",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-NFP.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Total Phenols by Class', fontsize=14)

sns.boxplot(x=wine.target, y="total_phenols", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Total Phenols",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-TotalPhenols.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Color Intensity by Class', fontsize=14)

sns.boxplot(x=wine.target, y="color_intensity", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Color Intensity",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-color.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Hue by Class', fontsize=14)

sns.boxplot(x=wine.target, y="hue", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Hue",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-boxplot-hue.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Dilution by Class', fontsize=14)

sns.boxplot(x=wine.target, y="od280/od315_of_diluted_wines", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Dilution",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-dilution.png')

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Proline by Class', fontsize=14)

sns.boxplot(x=wine.target, y="proline", data=wine_df,  ax=ax)
ax.set_xlabel("Class",size = 12,alpha=0.8)
ax.set_ylabel("Proline",size = 12,alpha=0.8)

plt.savefig('plots/seaborn_heatmap/wine-proline.png')



