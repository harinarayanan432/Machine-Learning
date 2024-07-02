import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Apply PCA to reduce the dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Print the names of the principal components
pc_names = [f'Principal Component {i+1}' for i in range(X_pca.shape[1])]
print("Names of Principal Components:")
for i, pc_name in enumerate(pc_names):
    print(f"{pc_name}: {feature_names[pca.components_[i].argmax()]}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Fit logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Fit Linear Discriminant Analysis model
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

# Plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired, marker='o', s=60)
    plt.title(title)
    plt.xlabel('mean concave points')
    plt.ylabel('mean fractal dimension')
    plt.show()

# Plot decision boundaries for Logistic Regression
plot_decision_boundary(logistic_model, X_train, y_train, title='Logistic Regression Decision Boundary')

# Plot decision boundaries for Linear Discriminant Analysis
plot_decision_boundary(lda_model, X_train, y_train, title='Linear Discriminant Analysis Decision Boundary')


from sklearn.metrics import classification_report, confusion_matrix
y_pred_lda = lda_model.predict(X_test)

# Print the classification report
print("Classification Report for Linear Discriminant Analysis:")

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Generate points for the plot
x_vals = np.linspace(-5, 5, 100)
y_vals = sigmoid_function(x_vals)

# Plot the sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Sigmoid Function', color='blue')

# Mark the data points with circles
plt.scatter(feature_values, np.zeros_like(feature_values), c=y, cmap='viridis', edgecolors='w', marker='o', label='Data Points')
plt.xlabel('Standardized Mean Concave Points')
plt.ylabel('Probability')
plt.title('Sigmoid Function for Mean Concave Points')
plt.legend()
plt.show()

print(classification_report(y_test, y_pred_lda))
