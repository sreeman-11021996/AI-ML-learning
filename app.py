import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kmeans import KMeans

data = pd.read_csv("student_clustering.csv")
km = KMeans(n_clusters=4,max_iter=100)
y_means = km.fit_predict(data)
print("Inertia - ",km.inertia_)
# visualize
#sns.scatterplot(data.iloc[:,0],data.iloc[:,1])
#plt.show()

# plot
plt.scatter(data.iloc[y_means == 0,0], data.iloc[y_means == 0,1])
plt.scatter(data.iloc[y_means == 1,0], data.iloc[y_means == 1,1])
plt.scatter(data.iloc[y_means == 2,0], data.iloc[y_means == 2,1])
plt.scatter(data.iloc[y_means == 3,0], data.iloc[y_means == 3,1])
plt.show()