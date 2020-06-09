# -*- coding: utf-8 -*-
"""programming2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19hhBOVcmzm0j5z2iZT1YBgC9Dly2avzm

# Programming 2

### Intialization

Imports
"""

import pandas as pd
import numpy as np

"""Load data from spambase"""

# Use pandas to read the csv file into a variable we can use
df = pd.read_csv("/content/drive/My Drive/data/spambase.csv", sep=',', header=None, dtype=float) # use the path to your data file here.  Needs to be csv(can change .data to .csv).
data = df.values

"""### Functions"""

# Function for computing the prior probabilty of spam or not spam.
def compute_prior_probability(data):
  spam_counter = 0.0
  nt_spam_counter = 0.0

  for item in data:
    if item[-1] == 1.0: spam_counter += 1.0
    else: nt_spam_counter += 1.0

  train_set_prior_probability_1 = spam_counter / (spam_counter + nt_spam_counter)
  train_set_prior_probability_0 = nt_spam_counter / (spam_counter + nt_spam_counter)

  return train_set_prior_probability_1, train_set_prior_probability_0

# Split a chunk of data into spams and not spams.
# Really useful for computing means and std deviations
def split_classes(data):
  spams = []
  nt_spams = []

  for i in range(len(data)):
    if data[i][-1] == 1.0: spams.append(i)
    else: nt_spams.append(i)

  spams_arr = np.zeros((len(spams), len(data[0])))
  nt_spams_arr = np.zeros((len(nt_spams), len(data[0])))

  for i in range(len(spams)):
    spams_arr[i] = data[spams[i]]

  for i in range(len(nt_spams)):
    nt_spams_arr[i] = data[nt_spams[i]]

  # here we do not need the last digit since we are splitting.
  adjusted_spams_arr = np.zeros((len(spams_arr),len(spams_arr[0])-1))
  adjusted_nt_spams_arr = np.zeros((len(nt_spams_arr),len(nt_spams_arr[0])-1))

  for i in range(len(spams_arr)):
    adjusted_spams_arr[i] = spams_arr[i][:-1]
  for i in range(len(nt_spams_arr)):
    adjusted_nt_spams_arr[i] = nt_spams_arr[i][:-1]

  return adjusted_spams_arr, adjusted_nt_spams_arr

# Get all of the means.
def get_means(spam_data, nt_spam_data):
  spam_sums = np.sum(spam_data, axis=0)
  nt_spam_sums = np.sum(nt_spam_data, axis=0)

  spam_means = np.divide(spam_sums, len(spam_data))
  nt_spam_means =  np.divide(nt_spam_sums, len(nt_spam_data))

  return spam_means, nt_spam_means

# get all of the std deviations
def get_std_devs(spam_means, nt_spam_means, spams_arr, nt_spams_arr):
  spam_subtracted_means = np.zeros((len(spams_arr), len(spam_means)))
  nt_spam_subtracted_means = np.zeros((len(nt_spams_arr), len(nt_spam_means)))

  for i in range(len(spams_arr)):
    spam_subtracted_means[i] = np.subtract(spams_arr[i], spam_means)
  for i in range(len(nt_spams_arr)):
    nt_spam_subtracted_means[i] = np.subtract(nt_spams_arr[i], nt_spam_means)

  spam_squared_subtracted_means = np.zeros((len(spams_arr), len(spam_means)))
  nt_spam_squared_subtracted_means = np.zeros((len(nt_spams_arr), len(nt_spam_means)))

  for i in range(len(spam_subtracted_means)):
    spam_squared_subtracted_means[i] = np.square(spam_subtracted_means[i])
  for i in range(len(nt_spam_subtracted_means)):
    nt_spam_squared_subtracted_means[i] = np.square(nt_spam_subtracted_means[i])

  spam_sum_squared_subtracted_means = np.sum(spam_squared_subtracted_means, axis=0)
  nt_spam_sum_squared_subtracted_means = np.sum(nt_spam_squared_subtracted_means, axis=0)

  spam_std_dev = np.sqrt(np.divide(spam_sum_squared_subtracted_means, len(spams_arr)))
  nt_spam_std_dev = np.sqrt(np.divide(nt_spam_sum_squared_subtracted_means, len(nt_spams_arr)))

  for i in range(len(spam_std_dev)):
    if spam_std_dev[i] == 0.0: spam_std_dev[i] = .00001
  for i in range(len(nt_spam_std_dev)):
    if nt_spam_std_dev[i] == 0.0: nt_spam_std_dev[i] = .00001

  return spam_std_dev, nt_spam_std_dev

# Compute conditional probability
def conditional_probability_log_input(ip, mean, std_dev):
  first_part = 1/(np.sqrt(2.0 * np.pi)*std_dev)
  second_part = np.exp(-1 * ((np.square(ip - mean))/(2 * np.square(std_dev))))
  second_part[second_part == 0.0] = .00001
  return np.log(first_part * second_part)

"""### Part 1

Split data into a training and test set.
Each should have roughly 2300 instances with 40% spam and 60% not-spam.
"""

# create some temporary lists for splitting the data.
spam_indices = []
nt_spam_indices = []

# split the data into the temp lists
for i in range(len(data)):
  if data[i][len(data[i])-1] == 1.0: spam_indices.append(i)
  else: nt_spam_indices.append(i)

# Do some calculations for how to build test and train sets
mid_spam = int(np.floor(len(spam_indices)/2))
mid_nt_spam = int(np.floor(len(nt_spam_indices)/2))

remainder_spam = int(len(spam_indices) - mid_spam)
remainder_nt_spam = int(len(nt_spam_indices) - mid_nt_spam)

# Initialize the train set to 0
train_set = np.zeros((mid_spam+mid_nt_spam,len(data[0])))

# Build the train set
for i in range(0,mid_spam):
  train_set[i] = data[spam_indices[i]]
for i in range (0,mid_nt_spam):
  train_set[mid_spam+i] = data[nt_spam_indices[i]]

# Initialize the test set to 0
test_set = np.zeros((remainder_spam+remainder_nt_spam,len(data[0])))

# Build the test set.
for i in range(0,remainder_spam):
  test_set[i] = data[spam_indices[mid_spam + i]]
for i in range (0,remainder_nt_spam):
  test_set[remainder_spam+i] = data[nt_spam_indices[mid_nt_spam + i]]

"""### Part 2

Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in the training data. As described in part 1, P(1) should be about 0.4.
"""

# Compute prior probabilities
spam_prior, nt_spam_prior = compute_prior_probability(train_set)

"""For each of the 57 features, compute the mean and standard deviation in the training set of the values given each class. If any of the features has zero standard deviation, assign it a “minimal” standard deviation (e.g., 0.0001) to avoid a divide-by- zero error in Gaussian Naïve Bayes."""

# For figuring out the means and std deviations it is helpful to split into 2 list of spam and not spam.
# Split
spams_arr, nt_spams_arr = split_classes(train_set)

# get the means
spam_means, nt_spam_means = get_means(spams_arr, nt_spams_arr)

# get the std devs
spam_std_devs, nt_spam_std_devs = get_std_devs(spam_means, nt_spam_means, spams_arr, nt_spams_arr)

"""### Part 3

Use the Gaussian Naïve Bayes algorithm to classify the instances in your test 
![alt text](https://i.ibb.co/d7ZrKqN/Screen-Shot-2020-05-13-at-2-50-41-PM.png)
"""

# Initialization for tracking and computation below.
correct = 0

is_spam = 0
predicted_spam = 0

is_nt_spam = 0
predicted_nt_spam = 0

false_negative = 0

confusion_matrix = np.zeros((2,2), dtype=int)

# loop once over every test input
for i in range(len(test_set)):
  # The last number of each test input is the label of the target.
  target = test_set[i][-1]

  # Used to track what the model predicts
  outcome = -1.0


  ip = test_set[i][:-1]   # input is everything but the last value 

  spam_probs = conditional_probability_log_input(ip, spam_means, spam_std_devs) # get all the probabilites for the spam subset.
  spam_probs_sum = np.sum(spam_probs) # take the sum (because already have taken the log above)
  final_spam = np.log(1) + spam_probs_sum # add log(1)

  nt_spam_probs = conditional_probability_log_input(ip, nt_spam_means, nt_spam_std_devs)  # get all the probabilites for the not spam subset.
  nt_spam_probs_sum = np.sum(nt_spam_probs) # take the sum (because already have taken the log above)
  final_nt_spam = .00001 + nt_spam_probs_sum # add log(0) or close to it we use .00001

  # Track accuracy, precision, and recall.
  if final_spam > final_nt_spam: 
    outcome = 1.0

    predicted_spam += 1
    if target == 1.0: 
      is_spam += 1
      confusion_matrix[1][1] += 1
    else:
      confusion_matrix[1][0] += 1

  else: 
    outcome = 0.0

    if target == 0.0:
      confusion_matrix[0][0] += 1
    else:
      false_negative += 1
      confusion_matrix[0][1] += 1

  if outcome == target: correct +=1

# Compute accuracy, precision, and recall.
accuracy = correct/len(test_set)
spam_precision = is_spam/predicted_spam
recall = is_spam/(is_spam + false_negative)

"""### Results"""

print('Accuracy:')
print(np.round(accuracy, 4))
print('\nPrecsion:')
print(np.round(spam_precision, 4))
print('\nRecall:')
print(np.round(recall, 4))
print('\nConfusion matrix:')
print(np.round(confusion_matrix, 4))