# ML_Final_Project
Comparison of the difference between Neural Networks and Naive Bayes over the MNIST dataset.

## MNIST
Support for MNIST can be found [here](http://yann.lecun.com/exdb/mnist/).
For ease of use during the project I used a csv of the MNIST found [here](https://pjreddie.com/projects/mnist-in-csv/)

## Running
You will first need to unzip the compressed data.zip file.  These programs require python 3 to be able to run. 

### naive_bayes.py
```bash
python3 naive_bayes.py
```
Inside the program you will need to make sure your path is pointing to the correct input files.  If you unzipped the data file in the same directory it should automatically point to it. 

You can update "test_data" on this line:
```python
is_number_arr, not_number_arr = split_data(test_data, float(j))
```
If you want to run on a quarter of the data to see the results you can use the variable "quarter_data"

### neural-networks.py
```bash
python3 neural-networks.py
```
Inside the program you will need to make sure your path is pointing to the correct input files.  If you unzipped the data file in the same directory it should automatically point to it. Change the variable "hidden_layers" to specify how many neurons you want in the hidden layer (value for n in report). Change the value in the "f" "for loop" for the number of epochs you want to run (currently set to 20). Update the "learning_rate" variable to set the learning rate (currently set at 0.1).