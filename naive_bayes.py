import pandas as pd
import numpy as np


# Function for computing the prior probabilty of spam or not spam.
def compute_prior_probability(data, number):
  is_number = 0.0
  not_number = 0.0

  for item in data:
    if item[0] == number: is_number += 1.0
    else: not_number += 1.0

  is_number_probability = is_number / (is_number + not_number)
  not_number_probability = not_number / (is_number + not_number)

  return is_number_probability, not_number_probability


def compute_prior_probabilities(data):
  number_priors = np.zeros((10,2))

  for i in range(10):
    number_priors[i][0], number_priors[i][1] = compute_prior_probability(data, float(i))

  return number_priors


def split_data(data, number):
  is_number_list = []
  not_number_list = []

  for item in data:
    if item[0] == number:
      is_number_list.append(item)
    else:
      not_number_list.append(item)

  is_number_arr = np.asanyarray(is_number_list)
  not_number_arr = np.asanyarray(not_number_list)

  return is_number_arr, not_number_arr


# Get all the the means.
def get_means(is_number_data, not_number_data):
  is_number_data_without_label = is_number_data[..., 1:]
  not_number_data_without_label = not_number_data[..., 1:]

  is_number_sums = np.sum(is_number_data_without_label, axis=0)
  not_number_sums = np.sum(not_number_data_without_label, axis=0)

  is_number_means = np.divide(is_number_sums, len(is_number_data))
  not_number_means =  np.divide(not_number_sums, len(not_number_data))

  return is_number_means, not_number_means


# get all of the std deviations
def get_std_devs(is_number_means, not_number_means, is_number_arr, not_number_arr):
  is_number_subtracted_means = np.zeros((len(is_number_arr), len(is_number_means)))
  not_number_subtracted_means = np.zeros((len(not_number_arr), len(not_number_means)))

  for i in range(len(is_number_arr)):
    is_number_subtracted_means[i] = np.subtract(is_number_arr[i][1:], is_number_means)
  for i in range(len(not_number_arr)):
    not_number_subtracted_means[i] = np.subtract(not_number_arr[i][1:], not_number_means)

  is_number_squared_subtracted_means = np.zeros((len(is_number_arr), len(is_number_means)))
  not_number_squared_subtracted_means = np.zeros((len(not_number_arr), len(not_number_means)))

  for i in range(len(is_number_subtracted_means)):
    is_number_squared_subtracted_means[i] = np.square(is_number_subtracted_means[i])
  for i in range(len(not_number_subtracted_means)):
    not_number_squared_subtracted_means[i] = np.square(not_number_subtracted_means[i])

  is_number_sum_squared_subtracted_means = np.sum(is_number_squared_subtracted_means, axis=0)
  not_number_sum_squared_subtracted_means = np.sum(not_number_squared_subtracted_means, axis=0)

  is_number_std_dev = np.sqrt(np.divide(is_number_sum_squared_subtracted_means, len(is_number_arr)))
  not_number_std_dev = np.sqrt(np.divide(not_number_sum_squared_subtracted_means, len(not_number_arr)))

  for i in range(len(is_number_std_dev)):
    if is_number_std_dev[i] == 0.0: is_number_std_dev[i] = .00001
  for i in range(len(not_number_std_dev)):
    if not_number_std_dev[i] == 0.0: not_number_std_dev[i] = .00001

  return is_number_std_dev, not_number_std_dev


# Compute conditional probability
def conditional_probability_log_input(ip, mean, std_dev):
  first_part = 1/(np.sqrt(2.0 * np.pi)*std_dev)
  second_part = np.exp(-1 * ((np.square(ip - mean))/(2 * np.square(std_dev))))
  second_part[second_part == 0.0] = .00001
  return np.log(first_part * second_part)


# Use pandas to read the csv file into a variable we can use
df_train = pd.read_csv("data/MNIST/normalized_mnist_train.csv", sep=',', header=None, dtype=float) # use the path to your data file here.  Needs to be csv(can change .data to .csv).
train_data = df_train.values

df_test = pd.read_csv("data/MNIST/normalized_mnist_test.csv", sep=',', header=None, dtype=float)
test_data = df_test.values


# Part 2

accuracies = np.zeros(10)

for j in range(10):
  # Compute prior probabilities
  prior_probabilities = compute_prior_probabilities(train_data)

  # For figuring out the means and std deviations it is helpful to split into 2 list of spam and not spam.
  is_number_arr, not_number_arr = split_data(train_data, float(j))

  # get the means
  is_number_means, not_number_means = get_means(is_number_arr, not_number_arr)

  # # get the std devs
  is_number_std_devs, not_number_std_devs = get_std_devs(is_number_means, not_number_means, is_number_arr, not_number_arr)


  # Part 3

  # Use the Gaussian Naïve Bayes algorithm to classify the instances in your test 

  correct = 0
  not_correct = 0

  predicted_is_number = 0

  is_number_count = 0

  outcome = -1.0


  predicted_count = 0
  # loop once over every test input
  for i in range(len(test_data)):


    # The last number of each test input is the label of the target.
    target = test_data[i][0]

    # Used to track what the model predicts
    prediction = -1.0


    ip = test_data[i][1:]   # input is everything but the last value 

    is_number_probs = conditional_probability_log_input(ip, is_number_means, is_number_std_devs) # get all the probabilites for the spam subset.
    is_number_probs_sum = np.sum(is_number_probs) # take the sum (because already have taken the log above)
    final_is_number = np.log(1) + is_number_probs_sum # add log(1)

    not_number_probs = conditional_probability_log_input(ip, not_number_means, not_number_std_devs)  # get all the probabilites for the not spam subset.
    not_number_probs_sum = np.sum(not_number_probs) # take the sum (because already have taken the log above)
    final_not_number = .00001 + not_number_probs_sum # add log(0) or close to it we use .00001



    if final_is_number > final_not_number:
      prediction = float(j)
      predicted_count +=1

      if target == prediction:
        correct += 1
    else:
      prediction = -1.0

      if target == float(j):
        not_correct += 1


  accuracies[j] = correct / (correct + not_correct)

print(np.sum(accuracies)/10)