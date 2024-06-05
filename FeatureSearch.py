import random 
import copy

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
    #create an array 'individual scores' that holds 'num_features' amount of 'Node' instances, each with a feature subset of [1..num_features]
    individual_scores = [Node([i for i in range(1, num_features+1)]) for _ in range(num_features)]
    best_features = Node()

    #loop through every 'Node' contained within 'individual scores' and remove a unique feature from every subset, assigning every node a score
    #initialize 'best_features' as a node containing the subset of features [1..num_features], i.e. all features included
    for i in range(0, num_features):
        individual_scores[i].remove_features(i)
        individual_scores[i].set_score(dummy_evaluation(individual_scores[i].get_features()))
        best_features.add_features(i+1)

    best_features.set_score(dummy_evaluation(best_features.get_features()))
    individual_scores = sorted(individual_scores, key=lambda node: node.get_score())

    #generate default rate NOTE: I don't think we actually end up using default rate here, rather threshold --will followup
    default_rate = dummy_evaluation(0)
    print(f"\nUsing no features and 'random' evaluation, I get an accuarcy of {default_rate*100}%\n")

    print(f"Model accuracy considering ALL features: {best_features.get_score()*100}%\n")
    print(f"Beginning search.\n")
    for i in range(0, num_features):
        individual_scores[i].print_node()
 
    #first iteration happens outside loop
    #LHS: maximum performing subset, minus one feature
    #RHS: threshold to beat (must be 5% greater than best score thus far)
    if individual_scores[num_features-1].get_score() >= 0.95 * best_features.get_score():
        best_features = individual_scores[num_features-1]
    else:
        print(f"(Warning accuracy has decreased)\n")
        return best_features

    for _ in range(num_features):
        #print the performance of the last subset
        print(f"Feature set: {best_features.get_features()} was best, accuracy is {best_features.get_score()*100}%\n")
        current_tree_level = []

        #populate the 'level' of this tree with Node instances containing feature subsets of [best_features - `some` feature]
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
        
        #if the best performing subset, the local max, outperforms the threshold, then we update the global max and continue searching, otherwise break and return
        threshold = 0.95 * best_features.get_score()
        local_max = current_tree_level[len(current_tree_level) - 1]

        print(f"Threshold: {threshold} | Local Max: {local_max.get_score()}")
        if local_max.get_score() >= threshold:
            best_features = local_max
        else:
            print(f"\n(Warning, Accuracy has decreased!)\n")
            return best_features
    #exhausted all feature subsets to search
    return best_features

def GFS(num_features):
    individual_scores = []

    #loop through every feature, creating node, assigning score, and appending to array [individual features]
    for i in range(1, num_features+1):
        individual_scores.append(Node(i, dummy_evaluation(i)))
    individual_scores = sorted(individual_scores, key=lambda node: node.get_score())
    
    #generate random evaluation for using 'zero features' i.e. the default rate
    zero_features = dummy_evaluation(0)
    print(f"\nUsing no features and 'random' evaluation, I get an accuarcy of {zero_features*100}%\n")

    print(f"Beginning search.\n")
    for i in range(0, num_features):
        individual_scores[i].print_node()
    best_features = individual_scores[num_features-1].copy_node()

    if best_features.get_score() < zero_features:
        print(f"\n(Warning, Accuracy has decreased!)\n")
        return Node(0, zero_features)

    for _ in range(0, num_features):
        #print the performance of the last subset
        print(f"Feature set: {best_features.get_features()} was best, accuracy is {best_features.get_score()*100}%\n")
        current_tree_level = []

        #for each feature '_' create a subset of all other features except self, append each node to the array representing the level of this 'tree'
        for feature in range(0, num_features):
            temp = best_features.get_features()

            if feature not in temp:
                temp_node = best_features.copy_node()
                temp_node.add_features(feature)
                temp_node.set_score(dummy_evaluation(temp_node.get_features()))

                current_tree_level.append(temp_node)
                temp_node.print_node()

        #sort tree to acces best performing subset
        length = len(current_tree_level)
        if length != 0:
            current_tree_level = sorted(current_tree_level, key=lambda node: node.get_score())
        else:
            return best_features
        
        #if the best performing subset, the local max, outperforms the global max, then we update the global max and continue searching, otherwise break and return
        local_max = current_tree_level[length-1]
        if local_max.get_score() > best_features.get_score():
            best_features = local_max
        else:
            print(f"\n(Warning, Accuracy has decreased!)\n")
            return best_features
    return best_features

def main():
    print(f"Feature Search Algorithms\n")
    print(f"\t1. 'Greedy' Forward Selection")
    print(f"\t2. Backwards Elimination\n")
    
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

if __name__ == "__main__":
    main()