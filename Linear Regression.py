import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns

def pca_manual(X, n_components):
    X_meaned = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    return X_reduced, sorted_eigenvalue, eigenvector_subset

def linear_regression_manual(X, y):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

X_pca, eigen_values, eigen_vectors = pca_manual(X, n_components=2)

beta = linear_regression_manual(X_pca, y)

correlation_matrix = pd.DataFrame(X_pca).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

print("Correlation Coefficients:")
print(beta)
