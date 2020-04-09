from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
 

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


# Edit test and train dataset
def edit_data(dataset):
    del dataset[0]
    for val in dataset:
        del val[0]
    for i in range(len(dataset[0])-1):
    	str_column_to_float(dataset, i)
    str_column_to_int(dataset, len(dataset[0])-1) 

        
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
      
        
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
		print('[%s] => %d' % (value, i))
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
    
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(train_set, test_set, algorithm, k_list):
    scores = list()
    for k in k_list:
        actual = list()
        predicted = algorithm(train_set, test_set, k)
        for val in test_set:
            actual.append(val[-1])
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
 
    
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
    
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
    
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
    
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)  

     
train_set = load_csv('iristrain.csv')
test_set = load_csv('iristest.csv')

edit_data(train_set)
edit_data(test_set)

#k_list = list()
#number = int(input("how many value you want in a list: "))
#for i in range(0,number):    
#    numbers = int(input("enter your choice number:"))
#    k_list.append(numbers)
#    
k_list = list(range(1,50,2))

for num, val in enumerate(k_list):
    print("K-Value {}..: ".format(num+1),val)
    
    
# evaluate algorithm
scores = evaluate_algorithm(train_set, test_set, k_nearest_neighbors, k_list)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

## define a new record
#row = [5.7,2.9,4.2,1.3]
#num_neighbors = 5
## predict the label
#label = predict_classification(train_set, row, num_neighbors)
#print('Data=%s, Predicted: %s' % (row, label))

# PLOT 
# =============================================================================
plt.figure()
plt.bar(k_list,scores, label = "Scores for K-Values")
plt.ylim(90,100)
plt.ylabel('Accuracy Scores')
plt.xlabel('K-Values')
plt.title('KNN')
plt.legend()
plt.show()
# =============================================================================





