import random 
import copy
import numpy as np
from collections import Counter
from operator import itemgetter
import os
import math
import pandas as pd

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
        print(f"\tUsing feature(s): {self.features} | Accuracy is: {(self.score*100):.2f}%\n")
    
def dummy_evaluation(features):
    return round(random.uniform(0.0, 1.0), 3)

def BE(data):
    max_feature = data.shape[1] - 1
    #individual_scores = [Node([i for i in range(1, max_feature)]) for _ in range(max_feature)]
    individual_scores = []
    best_features = Node(list(range(1, max_feature+1)))

    best_features.set_score(leave_one_out_cross_validation(data, best_features.get_features()))

    #for i in range(1, max_feature):
    #    individual_scores[i].remove_features(i + 1)
    #    individual_scores[i].set_score(leave_one_out_cross_validation(data, individual_scores[i].get_features()))
    #    best_features.add_features(i+1)

    #best_features.set_score(leave_one_out_cross_validation(data, best_features.get_features()))

    #default_rate = dummy_evaluation(0)

    print(f"Model accuracy considering ALL features: {best_features.get_score()*100}%\n")
    print(f"Beginning search.\n")
    for i in range(0, max_feature - 1):
        temp_node = Node(list(np.delete(best_features.get_features(), i)))
        temp_node.set_score(leave_one_out_cross_validation(data, temp_node.get_features()))
        temp_node.print_node()
        individual_scores.append(temp_node)
 
    individual_scores = sorted(individual_scores, key=lambda node: node.get_score())

    if individual_scores[len(individual_scores) - 1].get_score() >= best_features.get_score():
        best_features = individual_scores[len(individual_scores) - 1]
    else:
        print(f"(Warning accuracy has decreased)\n")
        return best_features

    for _ in range(max_feature):
        print(f"Feature set: {best_features.get_features()} was best, accuracy is {best_features.get_score()*100}%\n")
        current_tree_level = []

        for feature in range(0, len(best_features.get_features())):
            temp_node = best_features.copy_node()
            temp_node.remove_features(feature)
            temp_node.set_score(leave_one_out_cross_validation(data, temp_node.get_features()))

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

def GFS(data):
    individual_scores = []
    max_feature = data.shape[1]

    for i in range(1, max_feature):
        temp_node = Node(i, leave_one_out_cross_validation(data, [i]))
        temp_node.print_node()
        individual_scores.append(temp_node)
    
    individual_scores = sorted(individual_scores, key=lambda node: node.get_score())

    #zero_features = leave_one_out_cross_validation(data, [])
    #print(f"\nUsing no features and 'random' evaluation, I get an accuracy of {zero_features*100}%\n")

    print(f"Beginning search.\n")
    for i in range(len(individual_scores)):
        individual_scores[i].print_node()
    
    best_features = individual_scores[len(individual_scores)-1].copy_node()

    #if best_features.get_score() < zero_features:
    #    print(f"\n(Warning, Accuracy has decreased!)\n")
    #    return Node(0, zero_features)

    for _ in range(1, max_feature):
        print(f"Feature set: {best_features.get_features()} was best, accuracy is {(best_features.get_score()*100):.2f}%\n")
        current_tree_level = []

        for feature in range(1, max_feature):
            temp = best_features.get_features()

            if feature not in temp:
                temp_node = best_features.copy_node()
                temp_node.add_features(feature)
                temp_node.set_score(leave_one_out_cross_validation(data, temp_node.get_features()))

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

def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def normalize_data(data):
    class_labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    
    # Calculate means and standard deviations, ignoring NaNs
    means = features.mean()
    stds = features.std()
    
    # Normalize the features
    normalized_features = (features - means) / stds
    
    # Combine class labels with normalized features
    normalized_df = pd.concat([class_labels, normalized_features], axis=1)
    
    return normalized_df

class NNClassifier:
    trainingData = []
    trainingLabels = []

    def Train(self, data):
        self.trainingData = np.delete(data, 0, 1)
        self.trainingLabels = data[:,0]

    
    def Test(self, instance):
        # classification = instance[0]
        instance = np.delete(instance, 0)
        minDist = math.inf
        minPoint = 0
        for idx, point in enumerate(self.trainingData):
            tempDist = euclidean_distance(instance, point)
            if tempDist < minDist:
                minDist = tempDist
                minPoint = idx
        return self.trainingLabels[minPoint]

class LOOValidator:
    def __init__(self, subset, classifier, data):
        self.subset = [0] + subset
        self.classifier = classifier
        self.data = data.to_numpy()
        # self.subset = [0] + subset
        # self.classifier = classifier
        # self.data = data
    
    def validate(self):
        total = 0
        correct = 0
        for i in range(self.data.shape[0]):
            total += 1
            tempClassifier = self.classifier()
            tempClassifier.Train(np.delete(self.data[:,self.subset], i, 0))
            if tempClassifier.Test(self.data[i,self.subset]) == self.data[i,0]:
                correct += 1
        return correct / total


def leave_one_out_cross_validation(data, feature_set):
    return LOOValidator(feature_set, NNClassifier, data).validate()

def main():
    print(f"Feature Search Algorithms\n")
    print(f"\t1. 'Greedy' Forward Selection")
    print(f"\t2. Backwards Elimination")
    #print(f"\t3. Validate Feature Subset")
    
    user_selection = int(input("\nEnter the #(number) of which algorithm you'd like to run: "))

    data_file = input("Type in the name of the file to test: ")
    data = pd.read_csv(data_file,sep=r'\s+',header=None)
    # data_path = os.path.join(os.getcwd(), data_file)
    # data = np.genfromtxt(data_path)

    # print(f"{data}")
    normalized_data = normalize_data(data)
    # print(f"{normalized_data}\n")

    if user_selection == 1:
        best_subset = GFS(normalized_data)
        if best_subset:
            print(f"Finished search!! The best feature subset is {best_subset.get_features()}, which had an accuracy of {best_subset.get_score()*100}%\n")

    if user_selection == 2:
        best_subset = BE(normalized_data)
        if best_subset: 
            print(f"Finished search!! The best feature subset is {best_subset.get_features()}, which had an accuracy of {best_subset.get_score()*100}%\n")

    #if user_selection == 3:
        
        #request feature subset
        #accuracy = LOOValidator([3, 5], NNClassifier, data).validate()
        #print(f"Accuracy with all features: {(accuracy*100):.2f}%")

if __name__ == "__main__":
    main()