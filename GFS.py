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
        print(f"Using feature(s): {self.features} | Accuracy is: {self.score*100}%\n")
    
def dummy_evaluation(features):
    return round(random.uniform(0.0, 1.0), 4)

def GFS(num_features):
    individual_scores = []

    for i in range(1, num_features+1):
        individual_scores.append(Node(i, dummy_evaluation(i))) #create starting features + values
    individual_scores = sorted(individual_scores, key=lambda node: node.get_score())
    
    for i in range(0, num_features):
        individual_scores[i].print_node()

    print(f"##### BEST NODE #####\n")
    best_features = individual_scores[num_features-1].copy_node()
    best_features.print_node()
    print(f"#####################\n")


    for _ in range(0, num_features): #fix this to be a while loop that terminates at SOME points idk maybe the threshold value

        current_tree_level = []

        for feature in range(1, num_features+1):
            #here i wanna add every feature thats not the current feature to the level
            temp = best_features.get_features()

            #check for membership of feature in temp
            if feature not in temp:
                temp_node = best_features.copy_node()
                temp_node.add_features(feature)
                temp_node.set_score(dummy_evaluation(temp_node.get_features()))

                current_tree_level.append(temp_node)
                temp_node.print_node()

        #finished creating the tree level, if its empty that means we plateued?
        length = len(current_tree_level)
        if length != 0:
            current_tree_level = sorted(current_tree_level, key=lambda node: node.get_score())
        else:
            return best_features
        
        local_max = current_tree_level[length-1]
        #now we have a local max & a overall 'best' node -> compare?
        if local_max.get_score() > best_features.get_score():
            print(f"\nFOUND: {local_max.get_features()}\n")
            best_features = local_max
        else:
            print(f"\nDid not find better subset\n")
            return best_features

best_subset = GFS(9)

print(f"##### BEST NODE #####\n")
best_subset.print_node()
print(f"#####################\n")