import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Step 1: Find the mean of data
iris = load_iris()
data = iris.data
mean_data = np.mean(data, axis=0)

# Convert data to DataFrame
df = pd.DataFrame(data, columns=iris.feature_names)
print(df.cov())
# Step 2: Find covariance matrix
covariance_matrix = np.cov(np.array(df), rowvar=False)

print(covariance_matrix)
# Step 3: Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print(eigenvalues, eigenvectors)
# Step 4: Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Find principal components
# Use only the top k eigenvectors, where k is the number of features in the original data
k = data.shape[1]
principal_components = np.dot(data - mean_data, eigenvectors[:, :k])

# Step 6: Plot scatter plot for the first 2 principal components
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=iris.target, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of First 2 Principal Components')
plt.show()

# Step 7: Plot explained variance for the components
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='-')

#to print the feature compressed details

"""print("Order of Principal Components:")
for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues_sorted, eigenvectors_sorted.T)):
    print(f"Principal Component {i + 1}: Eigenvalue = {eigenvalue:.4f}")
    print("   Feature Names: ", [f"{feature_name} ({weight:.4f})" for feature_name, weight in zip(df.columns, eigenvector)])

# Step 5: Find principal components
k = data.shape[1]"""
principal_components = np.dot(data - mean_data, eigenvectors_sorted[:, :k])
