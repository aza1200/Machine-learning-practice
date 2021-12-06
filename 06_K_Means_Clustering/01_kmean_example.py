import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(columns=['height','weight'])
df.loc[0] = [185,60]
df.loc[1] = [180,60]
df.loc[2] = [185,70]
df.loc[3] = [165,63]
df.loc[4] = [155,48]
df.loc[5] = [170,75]
df.loc[6] = [175,80]


data_points = df.values
kmeans = KMeans(n_clusters=3).fit(data_points)

df['cluster_id'] = kmeans.labels_

sns.lmplot(x='height',y='weight',
           data=df,fit_reg=False,
           scatter_kws={"s":150},
           hue='cluster_id')

plt.show()

