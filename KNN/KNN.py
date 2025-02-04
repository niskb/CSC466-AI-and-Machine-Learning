import math
from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.training_data = []
        self.testing_label = []
    
    def fit(self, x,y):
        self.training_data = x
        self.training_label = y

    def ecludian_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)

    def predict(self, x_test):
        predictions = []
        for test_point in x_test:
            distances = []
            for i, train_point in enumerate(self.training_data):
                dist = self.ecludian_distance(test_point, train_point)
                distances.append((dist, self.training_label[i]))
                # print(dist)
                # print(distances)
            distances.sort(key=lambda x:x[0])
            knn = distances[:self.k]
            # print(knn)
            # labels = [label for _, label in knn]
            # print(labels)
            most_common_label = Counter(knn).most_common(1)[0][0]

            predictions.append(most_common_label)
        return predictions

if __name__ == "__main__":
    x_train = [[2,4],[4,6],[4,2],[6,4],[6,6],[8,8]]
    y_train = ['A', 'A', 'B', 'B', 'A', 'B']

    x_test = [[5,5],[7,7],[3,3]]

    knn = KNN(k=6)
    knn.fit(x_train, y_train)
    predictions = knn.predict([[3,9]])
    print("Prediction [[3,9]]", predictions)
    predictions = knn.predict(x_test)
    print(x_test, predictions)