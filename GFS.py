import random 
import operator

def dummy_evaluation(features):
    return round(random.uniform(0.0, 1.0), 4)

def GFS(features):
    #we wanna store the score of the individual features first
    individual_scores = []

    for i in range(1, features+1):
        individual_scores.append((i, dummy_evaluation(i)))
        print(f"Score of feature{i-1}: {individual_scores[i-1]}\n")

    individual_scores.sort(key=lambda x: x[1])

    print("_________SORTED__________\n")
    for i in range(0, features):
        print(f"Sorted score of feature{i}: {individual_scores[i]}\n")

    best_score = float("-inf")
    best_features = [individual_scores[features-1]]


    #TODO: fix the actual feature selection for this algorithm: thus far the only thing that it does is hold the sorted individual scores + initialize starting values

    #for however many features we have, temp = best + another feature, check if thats greater
    for i in range(1, features):
        temp_features = [best_features] + [individual_scores[i:]]
        temp_score = dummy_evaluation(temp_features)

        if temp_score > best_score:
            #if the score we found is a better score, then update our variables to keep track --later we will want to check if it meets 'threshold'
            best_score = temp_score
            best_features.append(i)
            print(f"Found better feature set: {best_features} | with a score of: {best_score}\n")

GFS(9)