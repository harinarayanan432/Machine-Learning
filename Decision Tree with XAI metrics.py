from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training set
dt_classifier.fit(X_train, y_train)

# Get information about the root node
root_node = dt_classifier.tree_

# Print information about the root node
print("Root Node Information:")
print(f"Feature Index: {root_node.feature[0]}")
print(f"Threshold Value: {root_node.threshold[0]}")
print(f"Impurity: {root_node.impurity[0]}")
print(f"Number of Samples: {root_node.n_node_samples[0]}")
print(f"Value (class distribution): {root_node.value[0]}")
print()

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.show()

# New point for prediction
new_point = [[5.0, 3.5, 1.5, 0.2]]  # Replace with your own values

# Predict the class for the new point
predicted_class = dt_classifier.predict(new_point)

# Print prediction flow through the decision tree
print("\nPrediction Flow:")
node_index = 0
while True:
    feature_index = root_node.feature[node_index]
    threshold = root_node.threshold[node_index]
    print(f"Node {node_index}: {iris.feature_names[feature_index]} <= {threshold}")
    
    if new_point[0][feature_index] <= threshold:
        node_index = root_node.children_left[node_index]
    else:
        node_index = root_node.children_right[node_index]
    
    if root_node.children_left[node_index] == root_node.children_right[node_index] == -1:
        break

print(f"\nPredicted Class: {iris.target_names[predicted_class[0]]}")

tree_rules = export_text(dt_classifier, feature_names=['Feature 1', 'Feature 2'])
print("Decision Tree Rules:")
print(tree_rules)


