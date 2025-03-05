# Niski
# Part 1 #
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # Pre-pruning condition
        if len(unique_labels) == 1 or num_samples < self.min_samples_split or (self.max_depth is not None and depth == self.max_depth):
            return Node(value=Counter(y).most_common(1)[0][0])

        best_feature, best_threshold = self._best_split(X, y, num_features)

        if best_feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        left_index = X[:, best_feature] < best_threshold
        right_index = ~left_index
        left_child = self._grow_tree(X[left_index], y[left_index], depth + 1)
        right_child = self._grow_tree(X[right_index], y[right_index], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_y, right_y = y[X_column < threshold], y[X_column >= threshold]
        n, n_left, n_right = len(y), len(left_y), len(right_y)
        if n_left == 0 or n_right == 0:
            return 0

        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)
        return parent_entropy - child_entropy

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / np.sum(counts)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def prune(self, X_val, y_val):
        self._prune_tree(self.root, X_val, y_val)

    def _prune_tree(self, node, X_val, y_val):
        if node.is_leaf():
            return

        # Recursively prune left and right children
        self._prune_tree(node.left, X_val, y_val)
        self._prune_tree(node.right, X_val, y_val)

        # Check if we can prune this node
        if node.left.is_leaf() and node.right.is_leaf():
                        # Create a temporary leaf node
            temp_value = Counter([node.left.value, node.right.value]).most_common(1)[0][0]
            temp_node = Node(value=temp_value)

            # Evaluate the accuracy of the tree before and after pruning
            original_accuracy = accuracy_score(y_val, self.predict(X_val))
            # Replace the current node with the temporary leaf node
            node.value = temp_value
            node.left = None
            node.right = None
            pruned_accuracy = accuracy_score(y_val, self.predict(X_val))

            # If pruning does not improve accuracy, revert the change
            if pruned_accuracy < original_accuracy:
                node.value = temp_value  # Restore the original leaf value
                node.left = node.left  # Restore left child
                node.right = node.right  # Restore right child

if __name__ == "__main__":
    # Part 2 #
    # Load the breast cancer dataset instead
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Part 3 # 
    # Without pruning
    data = load_breast_cancer()
    print(data.target)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTree(max_depth = 3)
    tree.fit(X_train, y_train)

    prediction = tree.predict(X_test)

    print(prediction)
    print(y_test)

    accuracy = np.mean(prediction == y_test)
    print(f"Part 3 Accuracy Without Pruning: {accuracy:.2f}")

    # Part 3 #
    # With pruning
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Check for missing values (not applicable for this dataset)
    print("Missing values in each column:", np.isnan(X).sum(axis=0))

    # Standardize the features (optional but recommended for many algorithms)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)    # Same test size and seed
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)     # Same test size and seed

    tree = DecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(X_train, y_train)

    # Output the shapes of the resulting datasets
    print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
    print(f"Testing set shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

    # Prune the tree using the validation set
    tree.prune(X_val, y_val)

    # Test the tree on the validation set
    prediction = tree.predict(X_val)

    print(prediction)
    print(y_val)

    accuracy = np.mean(prediction == y_val)
    print(f"Part 3 Accuracy with Pruning: {accuracy:.2f}")

    # Part 4 #
    # Load the breast cancer dataset
    print("Part 4")
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)    # Same test size and seed
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)     # Same test size and seed

    # Initialize and fit the custom decision tree
    custom_tree = DecisionTree(max_depth=3, min_samples_split=2)
    custom_tree.fit(X_train, y_train)

    # Prune the custom tree using the validation set
    custom_tree.prune(X_val, y_val)

    # Test the custom tree on the testing set
    custom_predictions = custom_tree.predict(X_test)
    custom_accuracy = accuracy_score(y_test, custom_predictions)

    # Initialize and fit the sklearn DecisionTreeClassifier
    sklearn_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2, random_state=42)
    sklearn_tree.fit(X_train, y_train)

    # Test the sklearn tree on the testing set
    sklearn_predictions = sklearn_tree.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    # Output the accuracies
    print(f"Custom Decision Tree Accuracy: {custom_accuracy:.2f}")
    print(f"Sklearn Decision Tree Accuracy: {sklearn_accuracy:.2f}")

    # Bonus #
    # Implement Cross-Validation and Feature Importance Analysis
    print("Bonus")
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)    #Same test size and seed

    # Set up the parameter grid for hyperparameter tuning
    param_grid = {
        'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 5, 10, 15, 20]
    }

    # Initialize the DecisionTreeClassifier
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Perform GridSearchCV for hyperparameter optimization
    grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Score: {best_score:.2f}")

    # Train the best model on the entire training set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Test the best model on the testing set
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Step 2: Feature Importance Analysis
    feature_importances = best_model.feature_importances_
    features = data.feature_names

    # Sort feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. {features[indices[f]]}: {feature_importances[indices[f]]:.4f}")

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()