import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create a simple Decision Rule Classifier class
class DecisionRuleClassifier:
    def __init__(self):
        self.rules = []  # Store rules as (feature_index, threshold, class_label)
    
    def fit(self, X, y):
        """Train the classifier by creating simple threshold-based rules."""
        self.rules = []
        unique_classes = np.unique(y)
        
        for feature_index in range(X.shape[1]):
            for cls in unique_classes:
                class_samples = X[y == cls, feature_index]
                threshold = np.mean(class_samples)
                self.rules.append((feature_index, threshold, cls))
    
    def predict(self, X):
        """Predict labels based on the learned rules."""
        predictions = []
        for sample in X:
            class_votes = {}
            
            for feature_index, threshold, cls in self.rules:
                if sample[feature_index] >= threshold:
                    class_votes[cls] = class_votes.get(cls, 0) + 1
            
            predictions.append(max(class_votes, key=class_votes.get))
        
        return np.array(predictions)
    
    def print_rules(self):
        """Print the learned decision rules."""
        for feature_index, threshold, cls in self.rules:
            print(f"If feature[{feature_index}] >= {threshold:.2f}, predict class {cls}")

# Step 2: Generate a synthetic Rose dataset
np.random.seed(42)
num_samples = 150

# Features: [sepal_length, sepal_width, petal_length, petal_width]
X = np.random.rand(num_samples, 4) * 10  # Values between 0 and 10

y = np.random.choice(['Red Rose', 'White Rose', 'Yellow Rose'], num_samples)  # Target classes

# Convert to NumPy array for easy indexing
X = np.array(X)
y = np.array(y)

# Step 3: Train and evaluate the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionRuleClassifier()
classifier.fit(X_train, y_train)
classifier.print_rules()  # Print the learned rules
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")