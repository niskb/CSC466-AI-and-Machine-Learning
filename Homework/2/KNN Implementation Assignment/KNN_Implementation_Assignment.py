# Brian Niski
import math
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.training_data = []
        self.training_labels = []

    def fit(self, X, y):
        self.training_data = X
        self.training_labels = y

    def euclidean_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):  # Iterate over feature dimensions
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)

    # My distance function
    def manhattan_distance(self, point1, point2):
        """Calculate Manhattan distance between two points."""
        distance = 0
        for i in range(len(point1)):
            distance += abs(point1[i] - point2[i]) # Manhattan distance between two points
            return distance

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []
            for i, train_point in enumerate(self.training_data):
                dist = self.manhattan_distance(test_point, train_point) # Manhattan Distance instead of Euclidean Distance
                distances.append((dist, self.training_labels[i]))
            distances.sort(key=lambda x: x[0])
            k_nearest_neighbors = distances[:self.k]
            labels = [label for _, label in k_nearest_neighbors]
            most_common_label = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return predictions

# Example Usage
if __name__ == "__main__":
    # Open CSV file
    import csv
    with open('iris-dataset.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader) # Convert to list
        del data[0] # Delete first row
    # Add the 4 features to X_train and its Class label to y_train
    X_train = []
    Y_train = []
    # Iterate through the data and insert them into the training lists
    for i in range(len(data)):
        # Insert Features (there are 4 of them... I hardcoded the appends... also the data is all float)
        X_train.append([float(data[i][0]), float(data[i][1]), float(data[i][2]), float(data[i][3])])
        # Insert class label
        Y_train.append(data[i][4]) # Add class label to Y_train

    print(X_train)
    print(Y_train)
    # Example dataset: [feature1, feature2, feature3, feature4]
    X_test = [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 5.1, 1.8], [2.0, 4.3, 8.1, 0.1]]
    kvalue = int(math.sqrt(len(X_train)))
    print("K:", kvalue)
    knn = KNN(k=kvalue)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_test)

    print("Predictions:", predictions)