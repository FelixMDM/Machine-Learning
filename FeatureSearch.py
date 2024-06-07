import random 
import copy
import numpy as np
from collections import Counter
from operator import itemgetter
import os

class Node():
    def __init__(self, features=None, score=0):
        self.features = []
        # If features is already a list, extend self.features with it, otherwise append
        if isinstance(features, list):
            self.features.extend(features)
        else:
            self.features.append(features)
        self.score = score

    def add_features(self, features):
        self.features.append(features)

    def remove_features(self, feature):
        if self.features and feature < len(self.features):
            self.features.pop(feature)

    def get_features(self):
        return self.features
    
    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score
    
    def copy_node(self):
        return copy.deepcopy(self)

    def print_node(self):
        print(f"\tUsing feature(s): {self.features} | Accuracy is: {self.score*100}%\n")
    
def dummy_evaluation(features):
    return round(random.uniform(0.0, 1.0), 3)

def BE(num_features):
    individual_scores = [Node([i for i in range(1, num_features+1)]) for _ in range(num_features)]
    best_features = Node()

    for i in range(0, num_features):
        individual_scores[i].remove_features(i)
        individual_scores[i].set_score(dummy_evaluation(individual_scores[i].get_features()))
        best_features.add_features(i+1)

    best_features.set_score(dummy_evaluation(best_features.get_features()))
    individual_scores = sorted(individual_scores, key=lambda node: node.get_score())

    default_rate = dummy_evaluation(0)
    print(f"\nUsing no features and 'random' evaluation, I get an accuracy of {default_rate*100}%\n")

    print(f"Model accuracy considering ALL features: {best_features.get_score()*100}%\n")
    print(f"Beginning search.\n")
    for i in range(0, num_features):
        individual_scores[i].print_node()
 
    if individual_scores[num_features-1].get_score() >= 0.95 * best_features.get_score():
        best_features = individual_scores[num_features-1]
    else:
        print(f"(Warning accuracy has decreased)\n")
        return best_features

    for _ in range(num_features):
        print(f"Feature set: {best_features.get_features()} was best, accuracy is {best_features.get_score()*100}%\n")
        current_tree_level = []

        for feature in range(0, len(best_features.get_features())):
            temp_node = best_features.copy_node()
            temp_node.remove_features(feature)
            temp_node.set_score(dummy_evaluation(temp_node.get_features()))

            current_tree_level.append(temp_node)
            temp_node.print_node()

        if current_tree_level:
            current_tree_level = sorted(current_tree_level, key=lambda node: node.get_score())
        else:
            return best_features
        
        threshold = 0.95 * best_features.get_score()
        local_max = current_tree_level[len(current_tree_level) - 1]

        print(f"Threshold: {threshold} | Local Max: {local_max.get_score()}")
        if local_max.get_score() >= threshold:
            best_features = local_max
        else:
            print(f"\n(Warning, Accuracy has decreased!)\n")
            return best_features
    return best_features

def GFS(num_features):
    individual_scores = []

    for i in range(1, num_features+1):
        individual_scores.append(Node(i, dummy_evaluation(i)))
    individual_scores = sorted(individual_scores, key=lambda node: node.get_score())
    
    zero_features = dummy_evaluation(0)
    print(f"\nUsing no features and 'random' evaluation, I get an accuracy of {zero_features*100}%\n")

    print(f"Beginning search.\n")
    for i in range(0, num_features):
        individual_scores[i].print_node()
    best_features = individual_scores[num_features-1].copy_node()

    if best_features.get_score() < zero_features:
        print(f"\n(Warning, Accuracy has decreased!)\n")
        return Node(0, zero_features)

    for _ in range(0, num_features):
        print(f"Feature set: {best_features.get_features()} was best, accuracy is {best_features.get_score()*100}%\n")
        current_tree_level = []

        for feature in range(0, num_features):
            temp = best_features.get_features()

            if feature not in temp:
                temp_node = best_features.copy_node()
                temp_node.add_features(feature)
                temp_node.set_score(dummy_evaluation(temp_node.get_features()))

                current_tree_level.append(temp_node)
                temp_node.print_node()

        length = len(current_tree_level)
        if length != 0:
            current_tree_level = sorted(current_tree_level, key=lambda node: node.get_score())
        else:
            return best_features
        
        local_max = current_tree_level[length-1]
        if local_max.get_score() > best_features.get_score():
            best_features = local_max
        else:
            print(f"\n(Warning, Accuracy has decreased!)\n")
            return best_features
    return best_features

def p_norm_distance(vec1, vec2, power):
    sum_dist = 0
    # Iterate over the coordinates of both vectors
    for v1, v2 in zip(vec1, vec2):
        # Calculate the distance component for the current coordinate pair
        component_distance = abs(v1 - v2) ** power
        # Add the component distance to the total sum
        sum_dist += component_distance
    # Return the p-norm distance between the two vectors
    return sum_dist ** (1 / power)

def find_nearest_neighbors(train_data, test_instance, train_labels, power):
    k = 1  # Number of nearest neighbors to consider
    distances = []  # List to store distances and corresponding labels
    # Iterate over the training data
    for idx, train_instance in enumerate(train_data):
        # Calculate distance from the test instance to the current training instance
        distance = p_norm_distance(train_instance, test_instance, power)
        # Append the distance and the corresponding label to the list
        distances.append((distance, train_labels[idx]))
    # Sort the list of distances in ascending order
    distances.sort(key=itemgetter(0))
    # Extract the labels of the k nearest neighbors
    nearest_labels = [label for _, label in distances[:k]]
    # Return the list of nearest neighbors' labels
    return nearest_labels

def knn_classifier(test_instance, train_data, train_labels, power):
    # Find the k nearest neighbors for the test instance
    neighbors = find_nearest_neighbors(train_data, test_instance, train_labels, power)
    # Determine the most common label among the neighbors
    most_common = Counter(neighbors).most_common(1)[0][0]
    # Return the most common label as the prediction
    return most_common

def leave_one_out_cross_validation(data):
    predictions = []  # List to store predicted values
    actual_labels = [row[0] for row in data]  # Extract actual labels from the data
    # Iterate over the dataset
    for i in range(len(data)):
        # Select the current instance as the test point
        test_instance = data[i, 1:]
        # Determine the training data by excluding the current instance
        if i == 0:
            train_data = data[1:]
        elif i == len(data) - 1:
            train_data = data[:i]
        else:
            train_data = np.concatenate((data[:i], data[i + 1:]), axis=0)
        # Extract the labels for the training data
        train_labels = [row[0] for row in train_data]
        # Remove the labels from the training data
        train_data = train_data[:, 1:]
        # Predict the label for the test point using the nearest neighbor classifier
        predicted_label = knn_classifier(test_instance, train_data, train_labels, 2)
        # Append the predicted label to the list of predictions
        predictions.append(predicted_label)
    # Calculate the accuracy of the predictions
    accuracy = calculate_accuracy(predictions, actual_labels)
    # Return the calculated accuracy
    return accuracy

def calculate_accuracy(predictions, true_labels):
    # Count the number of correct predictions
    correct_count = sum(1 for pred, actual in zip(predictions, true_labels) if pred == actual)
    # Calculate the accuracy as a percentage
    accuracy_percentage = (correct_count / len(true_labels)) * 100
    # Return the accuracy percentage
    return accuracy_percentage

def main():
    print(f"Feature Search Algorithms\n")
    print(f"\t1. 'Greedy' Forward Selection")
    print(f"\t2. Backwards Elimination")
    print(f"\t3. Validate Feature Subset")
    
    user_selection = int(input("Enter the #(number) of which algorithm you'd like to run: "))
    features = int(input("Enter the #(number) of features you would like to search over: "))

    if user_selection == 1:
        best_subset = GFS(features)
        if best_subset:
            print(f"Finished search!! The best feature subset is {best_subset.get_features()}, which had an accuracy of {best_subset.get_score()*100}%\n")

    if user_selection == 2:
        best_subset = BE(features)
        if best_subset: 
            print(f"Finished search!! The best feature subset is {best_subset.get_features()}, which had an accuracy of {best_subset.get_score()*100}%\n")

    if user_selection == 3:
        data_file = input("Type in the name of the file to test: ")
        data_path = os.path.join(os.getcwd(), data_file)
        data = np.genfromtxt(data_path)
        accuracy = leave_one_out_cross_validation(data)
        print(f"Accuracy with all features: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
