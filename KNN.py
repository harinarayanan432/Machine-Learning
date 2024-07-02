
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_df = pd.read_csv(url, header=None, names=column_names)
print(iris_df.head())
X = iris_df.drop("class", axis=1)
y = iris_df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = list(range(1,100))
accuracy_scores = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, linestyle='-', color='b')
plt.title('KNN Test Accuracy for Different Values of k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.show()

best_k = 5 
print(f"The best k value is: {best_k}")

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

new_data_point = np.array([6.5, 3.8, 1.5, 0.2]).reshape(1, -1)  
predicted_class = final_model.predict(new_data_point)
print(f"The predicted class for the new data point is: {predicted_class[0]}")


y_test_pred = final_model.predict(X_test)
classification_report_output = classification_report(y_test, y_test_pred)
print("Classification Report:\n", classification_report_output)


plt.figure(figsize=(12, 8))

# Plot data points for each class
for class_label in iris_df['class'].unique():
    class_data = iris_df[iris_df['class'] == class_label]
    plt.scatter(class_data['sepal_length'], class_data['sepal_width'], label=class_label)

indices = np.where(y_test != y_test_pred)[0]
points = X_test.iloc[indices]
plt.scatter(6.5, 3.8, marker='x', color='red',s=200, label='New Data Point')

plt.title('Scatter Plot of Iris Dataset with Misclassified Points')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.grid(True)
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris_df['class'].unique(), yticklabels=iris_df['class'].unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


