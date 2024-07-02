import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def rbf_kernel(x1, x2, gamma=1.0):
    if x1.ndim == 1:
        x1 = x1[np.newaxis, :]
    if x2.ndim == 1:
        x2 = x2[np.newaxis, :]
    diff = x1[:, np.newaxis] - x2[np.newaxis, :]
    return np.exp(-gamma * np.sum(diff ** 2, axis=2))

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2.T) + coef0) ** degree

def sigmoid_kernel(x1, x2, coef0=1):
    return np.tanh(np.dot(x1, x2.T) + coef0)

def fit(X, y, kernel, C=1.0, tol=1e-3, max_passes=5):
    m, n = X.shape
    alpha = np.zeros(m)
    b = 0
    passes = 0
    kernel_function = kernel_functions[kernel]

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            K = kernel_function(X, X[i])
            Ei = np.dot((alpha * y), K) + b - y[i]
            if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
                j = np.random.choice([x for x in range(m) if x != i])
                Kj = kernel_function(X, X[j])
                Ej = np.dot((alpha * y), Kj) + b - y[j]
                old_alpha_i, old_alpha_j = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                if L == H:
                    continue
                eta = 2 * K[j] - K[i] - kernel_function(X[j], X[j])
                if eta >= 0:
                    continue
                alpha[j] -= y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                if abs(alpha[j] - old_alpha_j) < tol:
                    continue
                alpha[i] += y[i] * y[j] * (old_alpha_j - alpha[j])
                b1 = b - Ei - y[i] * (alpha[i] - old_alpha_i) * K[i] - y[j] * (alpha[j] - old_alpha_j) * K[j]
                b2 = b - Ej - y[i] * (alpha[i] - old_alpha_i) * kernel_function(X[i], X[j]) - y[j] * (alpha[j] - old_alpha_j) * kernel_function(X[j], X[j])
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b

def predict(X_train, y_train, alpha, b, X, kernel):
    kernel_function = kernel_functions[kernel]
    K = kernel_function(X_train, X)
    return np.sign(np.dot((alpha * y_train), K) + b)

X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
y[y == 0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

kernel_functions = {'linear': linear_kernel, 'rbf': rbf_kernel, 'poly': polynomial_kernel, 'sigmoid': sigmoid_kernel}
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

for kernel in kernels:
    alpha, b = fit(X_train, y_train, kernel)
    y_pred = predict(X_train, y_train, alpha, b, X_test, kernel)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {kernel} kernel')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print(f'Classification Report for {kernel} kernel:')
    print(cr)
