import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function for normalizing mnist data to be 0-1
def normalize_mnist_data(file_path):
    with open(file_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            for i in range(1,len(row)):
                row[i] = round(float(row[i])/255.0,2)
            append_list_as_row("data/normalized_mnist_test.csv", row)

# Function used for squashing on the sigmoid for the evaluations during a forward pass.
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Gets data from the csv file and returns a numpy array of the inputs.
def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', header=None, dtype=float)

    return df.values


# This looks at an image and gets the value of the handwritten digit and the pixels of the image used as inputs
def get_inputs_and_target(index, data):
    return data[index][1:], data[index][0]


# Function for generating random weights from -0.5-0.5
def generate_weights(N):
    # calculation: 10 rows for each digit 0-9.
    # number of weights is number of inputs * number of hidden layer
    # neurons + # of hidden layer neurons.
    input_to_hidden_weights = np.random.uniform(-0.5,0.5, (N,785))
    hidden_to_output_weights = np.random.uniform(-0.5, 0.5, (10,N+1))

    return input_to_hidden_weights, hidden_to_output_weights


# function for doing a forward pass. Works for both input to hidden and also hidden to output.
def forward_pass_evaluation(inputs, weights):
    return sigmoid(np.dot(inputs, np.transpose(weights[:-1])) + weights[-1])


# Function used to compute the error on the output layer
def compute_error_outputs(outputs, target_digit):
    errors = np.zeros(10, dtype=float)
    for i in range(len(outputs)):
        if i == int(target_digit): t = 0.9
        else: t = 0.1

        if t-np.absolute(np.round(outputs[i], 1)) == 0.0: errors[i] = 0
        else: errors[i] = outputs[i] * (1-outputs[i]) * (t-outputs[i])
    return errors


# Function used to compute the error on the hidden layer
def compute_error_hiddens(hiddens, weights, errors, N):
    herrors = np.zeros(N+1, dtype=float)

    for i in range(N):
        w = [j[i] for j in weights]
        herrors[i] = hiddens[i] * (1-hiddens[i]) * np.dot(w, np.transpose(errors))

    return herrors


# This function will update the weights from the hidden layer to the output layer.
# Important here that if you want to change the momentum you need to update line 77 and change the 0.9 to the desired value
# Ajust the momentum from 0.9 to some other value used in experiment 3
def update_hidden_to_output_weights(ho_weights, n, h_inputs, output_nerons, oerrors, hidden_to_output_weight_change):
    for i in range(len(output_nerons)):
        initial_change = np.dot(n * oerrors[i], np.append(h_inputs, 1))
        hidden_to_output_weight_change[i] = initial_change + (0.9 * hidden_to_output_weight_change[i])
    i = 0

    for i in range(len(output_nerons)):
        ho_weights[i] = np.add(ho_weights[i], hidden_to_output_weight_change[i])

    return ho_weights, hidden_to_output_weight_change


# This function will update the weights from the input layer to the hidden layer.
# Important here that if you want to change the momentum you need to update line 92 and change the 0.9 to the desired value
# Ajust the momentum from 0.9 to some other value used in experiment 3
def update_input_to_hidden_weights(ih_weights, n, inputs, hidden_nerons, herrors, input_to_hidden_weight_change):
    for i in range(len(hidden_nerons)):
        initial_change = np.dot(n * herrors[i], np.append(inputs, 1.0))
        input_to_hidden_weight_change[i] = initial_change + (0.9 * input_to_hidden_weight_change[i])
    i = 0

    for i in range(len(hidden_nerons)):
        ih_weights[i] = np.add(ih_weights[i], input_to_hidden_weight_change[i])

    return ih_weights, input_to_hidden_weight_change


# This function checks to see if the weights are where they should be or not.
def check_imaged_finished(evaluations, target_digit):
    for i in range(len(evaluations)):
        if i == target_digit:
            if np.round(evaluations[i],1) == .9: continue
            else: return False
        else:
            if np.round(evaluations[i],1) == 0.1: continue
            else: return False
    return True


# Helper functions for passing the current weights into a MLP and getting the accuracy
def get_accuracy(data, input_to_hidden_weights, hidden_to_output_weights):
    global confusion_matrix

    correct = 0

    for l in range(len(data)):
        hidden_evaluations = np.zeros(hidden_layers)
        output_evaluations = np.zeros(10)

        image_inputs, target_digit = get_inputs_and_target(l, data)

        # do a forward pass and set the hidden and output evaluations
        for i in range(hidden_layers):
            hidden_evaluations[i] = forward_pass_evaluation(image_inputs, input_to_hidden_weights[i])
        i = 0
        for i in range(10):
            output_evaluations[i] = forward_pass_evaluation(hidden_evaluations, hidden_to_output_weights[i])
        i = 0

        max_index = np.argmax(output_evaluations)

        if int(max_index) == int(target_digit):
            correct += 1

        confusion_matrix[int(target_digit)][int(max_index)] += 1

    accuracy = correct / len(data)

    return accuracy



# Note: I am using an alraedy normalized set in csv format. My function for normalization is in the functions above.
#       I did not want to do the normalization each time so that is why it is not in my main here.
#       the csv files can be found at https://pjreddie.com/projects/mnist-in-csv/.  All i did was normalize the value between 0-1     

# get each image from the train and test data
data = load_data("data/MNIST/normalized_mnist_train.csv")
data_test = load_data("data/MNIST/normalized_mnist_test.csv")

# set the number for n and the learning rate. Adjust the hidden_layers to set the n value in Experiment 1
hidden_layers = 2           # This is the n value.  The hidden_layers name a little confusion, but it is the nubmer of hidden_neurons in the hidden layer
learning_rate = 0.1         # Learning rate which was always constant at 0.1

# initialize confusion matrix
confusion_matrix = np.zeros((10,10), dtype=int)

# generate random weights
input_to_hidden_weights, hidden_to_output_weights = generate_weights(hidden_layers)

# initialize variables for the change in weights used in adjusting the weights later
hidden_to_output_weight_change = np.zeros((10,hidden_layers+1), dtype=float)
input_to_hidden_weight_change = np.zeros((hidden_layers,785), dtype=float)

# initialize lists for accuracy of the train and test accuracy.  Very helpful for plotting later.
acc = []
acc_test = []

# run on epoch 0 and get the accuracy
a = np.round(get_accuracy(data, input_to_hidden_weights, hidden_to_output_weights),4)
acc.append(a)
b = np.round(get_accuracy(data_test, input_to_hidden_weights, hidden_to_output_weights),4)
acc_test.append(b)
print("0: ",a,b)


# Run for the rest of the 50 epochs
for f in range(2):         
    for l in range(len(data)):  # For experiment 3 i just change the for loop range to be "int(len(data)/2)" for half of the data set.
        hidden_evaluations = np.zeros(hidden_layers)
        output_evaluations = np.zeros(10)


        # get the image inputs and target digit
        image_inputs, target_digit = get_inputs_and_target(l, data)

        # do a forward pass and set the hidden and output evaluations
        for i in range (hidden_layers):
            hidden_evaluations[i] = forward_pass_evaluation(image_inputs, input_to_hidden_weights[i])
        i = 0
        for i in range (10):
            output_evaluations[i] = forward_pass_evaluation(hidden_evaluations, hidden_to_output_weights[i])
        i = 0

        # Compute the errors for the output and hidden layers
        output_errors = compute_error_outputs(output_evaluations, target_digit)
        hidden_errors = compute_error_hiddens(hidden_evaluations, hidden_to_output_weights, output_errors, hidden_layers)

        # adjust the weights
        hidden_to_output_weights, hidden_to_output_weight_change = update_hidden_to_output_weights(hidden_to_output_weights, learning_rate, hidden_evaluations, output_evaluations, output_errors, hidden_to_output_weight_change)
        input_to_hidden_weights, input_to_hidden_weight_change = update_input_to_hidden_weights(input_to_hidden_weights, learning_rate, image_inputs, hidden_evaluations, hidden_errors, input_to_hidden_weight_change)

    # Compute the accuracy for the current epoch on both the train and test data
    a = np.round(get_accuracy(data, input_to_hidden_weights, hidden_to_output_weights),4)
    acc.append(a)
    b = np.round(get_accuracy(data_test, input_to_hidden_weights, hidden_to_output_weights), 4)
    acc_test.append(b)
    print(str(f+1) + ": ", a,b)

# Print the confusion matrix (included in my report)
print(confusion_matrix)


# This is an initialization of the x axis for the plots.  Probably a much better way of doing this, but this is what I came up with.
# x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
x = [0,1,2]


# graph the results
plt.ylim(0,1)
plt.xlim(0,3)
plt.title("Experiment 1: n=2")
plt.plot(x, acc)
plt.plot(x, acc_test)
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()