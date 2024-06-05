import random 
import copy

class Node():
    def __init__(self, features=None, score=0):
        if features is None:
            self.features = []
        else:
            self.features = features
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
    #create an array 'num_features' size holding nodes of subset [1,...n]
    individual_scores = [Node([i for i in range(1, num_features+1)]) for _ in range(num_features)]
    best_features = Node()

    #loop through every potential feature, and remove one from each subset in 'individual_scores,' populate best_features [1...n]
    for i in range(0, num_features):
        individual_scores[i].remove_features(i)
        individual_scores[i].set_score(dummy_evaluation(individual_scores[i].get_features()))
        best_features.add_features(i+1)

    best_features.set_score(dummy_evaluation(best_features.get_features()))
    individual_scores = sorted(individual_scores, key=lambda node: node.get_score())
    
    #generate random evaluation for using 'zero features' i.e. the default rate
    default_rate = dummy_evaluation(0)
    print(f"\nUsing no features and 'random' evaluation, I get an accuarcy of {default_rate*100}%\n")

    print(f"Score with all features: {best_features.get_score()*100}%")
    print(f"Beginning search.\n")
    for i in range(0, num_features):
        individual_scores[i].print_node()
 
    #for the time being if the subset of features with the best score is not better than the default rate just return 
    # if best_features.get_score() < default_rate:
    #     print(f"\n(Warning, Accuracy has decreased!)\n")
    #     return None

    temp_max = individual_scores[num_features-1]
    if best_features.get_score() < temp_max.get_score():
        best_features = temp_max
    else:
        print(f"Here\n")
        return best_features

    for _ in range(num_features):
        #print the performance of the last subset
        print(f"Feature set: {best_features.get_features()} was best, accuracy is {best_features.get_score()*100}%\n")
        current_tree_level = []

        #for each feature '_' create a subset of all other features except self, append each node to the array representing the level of this 'tree'
        for feature in range(0, num_features):
            temp = best_features.get_features()

            temp_node = best_features.copy_node()
            temp_node.remove_features(feature)
            temp_node.set_score(dummy_evaluation(temp_node.get_features()))

            current_tree_level.append(temp_node)
            temp_node.print_node()

        if current_tree_level:
            current_tree_level = sorted(current_tree_level, key=lambda node: node.get_score())
        else:
            return best_features
        
        #if the best performing subset, the local max, outperforms the global max, then we update the global max and continue searching, otherwise break and return
        threshold = 0.95 * best_features.get_score()
        local_max = current_tree_level[num_features-1]
        print(f"Threshold: {threshold} | Local Max: {local_max.get_score()}")
        if local_max.get_score() >= threshold:
            best_features = local_max
        else:
            print(f"\n(Warning, Accuracy has decreased!)\n")
            return best_features

#TODO: build starting subset as all features, remove one instead of adding one
best_subset = BE(9)
if best_subset:
    print(f"Finished search!! The best feature subset is {best_subset.get_features()}, which had an accuracy of {best_subset.get_score()*100}%\n")
else:
    print(F"Best subset is zerp features, random evaluation\n")