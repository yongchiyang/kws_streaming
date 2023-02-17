import csv
import os
import numpy as np
from tensorflow.compat.v1 import keras

def loadCSV(csvf):
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels

def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1
    datamat = datamat.reshape(1, row)
    return datamat

class IEGM_DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size, shuffle, size, n_classes=2):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = size
        self.n_classes = n_classes
        self.on_epoch_end()
        
    
    def __len__(self):
        # Denotes the number of batches per epoch
         return int(np.floor(len(self.list_IDs) / self.batch_size))
     
    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(25)
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, batchsz):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((batchsz, self.size))
        y = np.empty((batchsz), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x[i] = txt_to_numpy(os.path.join('./dataset', ID), self.size)
            # Store class
            y[i] = self.labels[ID]

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __getitem__(self, index):
        
        list_IDs_temp = [self.list_IDs[k] for k in range(24588)]

        x, y = self.__data_generation(list_IDs_temp, 24588)

        return x, y  
    
class IEGM_DataGenerator_test(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size, shuffle, size, n_classes=2):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = size
        self.n_classes = n_classes
        self.on_epoch_end()
        
    
    def __len__(self):
        # Denotes the number of batches per epoch
         return int(np.floor(len(self.list_IDs) / self.batch_size))
     
    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(25)
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, batchsz):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        x = np.empty((batchsz, self.size))
        y = np.empty((batchsz), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x[i] = txt_to_numpy(os.path.join('./dataset', ID), self.size)
            # Store class
            y[i] = self.labels[ID]

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __getitem__(self, index):

        list_IDs_temp = [self.list_IDs[k] for k in range(5625)]

        x, y = self.__data_generation(list_IDs_temp, 5625)
        return x, y





def count_list(y_predict, y_label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i, (a, b) in enumerate(y_label):
        (c, d) = y_predict[i]
        if a == 0:
            if c == 0:
                TP += 1
            else:
                FN += 1
        else:
            if c == 1:
                TN += 1
            else:
                FP += 1
    return ([TP, FN, FP, TN])



#   print('\n')
#   # Compare prediction results with ground truth labels to calculate accuracy.
#   prediction_digits = np.array(prediction_digits)
#   accuracy = (prediction_digits == test_labels).mean()
#   return accuracy

def evaluate_model(interpreter, test_data, test_labels):
    # Get input and output tensors.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    prediction_digits = []
    for i, test_d in enumerate(test_data):
        # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
        test_d = np.expand_dims(test_d, axis=0).astype(np.int8)
        interpreter.set_tensor(input_index, test_d)

        # Run inference.
        interpreter.invoke()
        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    # print("test_labels.shape=",test_labels.shape)
    prediction_digits = keras.utils.to_categorical(prediction_digits, num_classes=2)
    # print("prediction_digits.shape=",prediction_digits.shape)
    accuracy = (prediction_digits == test_labels).mean()

    score_list = count(prediction_digits, test_labels)
    fb = FB(score_list)
    print("quan_int8_list: ", score_list)
    print("after quantization: ")
    print("acc: ", accuracy)
    print("fb-score: ", fb)
            

def convertmax(y_predict):
    y = np.empty((5625), dtype=int)
    for i, (a,b) in enumerate(y_predict):
        if(a >= b):
            y[i] = 0
        else:
            y[i] = 1
    return keras.utils.to_categorical(y, num_classes=2)

def count(y_predict, y_label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i, (a, b) in enumerate(y_label):
        (c, d) = y_predict[i]
        if a == 0:
            if c == 0:
                TP += 1
            else:
                FN += 1
        else:
            if c == 1:
                TN += 1
            else:
                FP += 1

    return ([TP, FN, FP, TN])

def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        fb = 0
    else:
        fb = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return fb

def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]

    if tp + fn == 0:
        ppv = 1
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv

def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity
