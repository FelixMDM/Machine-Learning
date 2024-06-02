import random 
import copy

class Node():
    def __init__(self, features=None, score=0):
        if features is None:
            features = []
        else:
            features = [features]
        self.features = features
        self.score = score

    def add_features(self, features):
        self.features.append(features)

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
        return None

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

#TODO: implement main function to run algorithms in a better way
best_subset = GFS(9)
if best_subset:
    print(f"Finished search!! The best feature subset is {best_subset.get_features()}, which had an accuracy of {best_subset.get_score()*100}%\n")
else:
    print(F"Best subset is zerp features, random evaluation\n")