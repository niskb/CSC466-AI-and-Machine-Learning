import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.classes_ = None

    def _get_cls_map(self, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM requires exactly two classes")
        return np.where(y == self.classes_[0], -1, 1)
    
    def fit(self, X, y):
        y_ = self._get_cls_map(y)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.w) + self.b
            conditions = y_ * linear_output >= 1
            misclassified_indices = np.where(~conditions)[0]
            
            hinge_grade_w = np.sum(-y_[misclassified_indices, np.newaxis] * X[misclassified_indices], axis=0)
            dw = 2 * self.lambda_param * self.w + (1/n_samples) * hinge_grade_w
            
            hinge_grade_b = np.sum(-y_[misclassified_indices])
            db = (1/n_samples) * hinge_grade_b

            self.w -= self.lr * dw
            self.b -= self.lr * db
            
    def predict(self, X):
        if self.w is None or self.b is None:
            raise RuntimeError("You must call fit before prediction")
        linear_output = np.dot(X, self.w) + self.b
        prediction_interval = np.where(linear_output >= 0, 1, -1)
        prediction_original = np.where(prediction_interval == 1, self.classes_[1], self.classes_[0])
        return prediction_original

def plot_svm_boundary(svm_model, X, y, scaler):
    plt.figure(figsize=(10, 6))
    X_scaled = scaler.transform(X)
    y_internal = svm_model._get_cls_map(y)

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = np.dot(xy, svm_model.w) + svm_model.b
    Z = Z.reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    decision_values = y_internal * (np.dot(X_scaled, svm_model.w) + svm_model.b)
    support_vector_indices = np.where(np.abs(decision_values) <= 1 + 1e-5)[0]
    
    if len(support_vector_indices):
        ax.scatter(X_scaled[support_vector_indices, 0], 
                  X_scaled[support_vector_indices, 1], 
                  s=100, 
                  facecolors='none', 
                  edgecolors='r', 
                  label='Support Vectors')
        ax.legend()

    plt.title('SVM Decision Boundary with Margins (Scaled Data)')
    plt.xlabel('Feature 1 (Scaled)')
    plt.ylabel('Feature 2 (Scaled)')
    plt.show()

if __name__ == "__main__":
    X, y = make_blobs(n_samples=150, centers=2, n_features=2, cluster_std=1.0, random_state=42, center_box=(-10, 10))
    print("Original Labels:", np.unique(y))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVM(learning_rate=0.1, lambda_param=0.01, n_iters=500)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("SVM Classifier trained.")
    print("Learned Weights [w]:", svm.w)
    print("Learned Bias [b]:", svm.b)
    print("Test Accuracy:", accuracy * 100)
    print("Predicted Labels on test set:", y_pred)
    print("Actual Labels on test set:", y_test)
    
    plot_svm_boundary(svm, X, y, scaler)