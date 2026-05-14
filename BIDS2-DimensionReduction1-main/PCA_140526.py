import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA, KernelPCA
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler

#importing the data
df = pd.read_excel("../Data/COVID19_proteomics.xlsx")
df.head(20)

# run PCA with 4 components
pca_res = PCA(n_components=4) #running PCA with 4 components, we can choose the number of components based on the number of dimensions we want to visualise or based on the proportion of variance explained by the components
pca_transform = pca_res.fit_transform(df.iloc[:, 3:])
print(pca_transform.shape)
print(type(pca_transform))

# Scatterplot
sns.set_style("ticks")
sns.set_context("notebook")
plt.figure(figsize=(8, 8))
p = sns.scatterplot(x=pca_transform[:, 0], y=pca_transform[:, 1], hue=df[1])
p.set_xlabel("PC1")
p.set_ylabel("PC2")

plt.show()


#Store PCA result in pca_df dataframe 
pca_df = pd.DataFrame(pca_transform, columns=["PC"+str(i) for i in range(1, pca_transform.shape[1]+1)])
pca_df["status"] = df[1]

pca_df

# Pairplot 
sns.pairplot(data=pca_df, hue="status")
plt.show()
sns.set_style("ticks")
sns.set_context("notebook")
plt.figure(figsize=(8, 8))
p = sns.scatterplot(x=pca_transform[:, 0], y=pca_transform[:, 1], hue=df[1])

# Label pairplot % variance explained
pvars = pca_res.explained_variance_ratio_[:2] * 100

p.set_xlabel(("PC1: " "{:.2f}%".format(pvars[0])))
p.set_ylabel(("PC2: " "{:.2f}%".format(pvars[1])))

plt.show()

# Scree plot
# PCA with 20 components
pca_df = PCA(n_components=20)
pca_df.fit(df.iloc[:, 3:])

# use the attribute .explained_variance_ratio_ to get the eigenvalues
variance_per_component = pca_df.explained_variance_ratio_

# sum the eigenvalues to get the cumulative variance explained for each component
cumulative_variance = np.cumsum(variance_per_component)
components = list(range(1, 21))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.barplot(x=components, y=variance_per_component, palette="viridis", ax=ax, hue=components)

# show the cumulative variance with a blue line
sns.pointplot(x=components, y=cumulative_variance, ax=ax, color="blue", label="Cumulative variance")

plt.xlabel("Components")
plt.ylabel("Eigenvalue (% variance explained)")
plt.show()