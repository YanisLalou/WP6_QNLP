import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../../models/quantum/alpha/module/")
from alpha_1_2_model import Alpha_1_2_model
from alpha_3_model import Alpha_3_model
from utils import seed_everything, preprocess_train_test_dataset, preprocess_train_test_dataset_words


import torch
import torch.nn as nn
import time
import numpy as np


def compute_inference_time(model_number, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device):

    if model_number == 1:
        version_original = True
    else:
        version_original = False

    if model_number == 1 or model_number == 2:
        # load the dataset
        if version_original:
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = preprocess_train_test_dataset_words(self.train_path, self.val_path, self.test_path, self.reduced_word_embedding_dimension)
        else:
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = preprocess_train_test_dataset(self.train_path, self.val_path, self.test_path)

        self.train_diagrams, self.train_label, self.X_train ,self.Y_train = self.create_diagrams(self.X_train, self.Y_train)
        self.val_diagrams, self.val_labels, self.X_val, self.Y_val = self.create_diagrams(self.X_val, self.Y_val)
        self.test_diagrams, self.test_labels, self.X_test, self.Y_test = self.create_diagrams(self.X_test, self.Y_test)

        print("Number of training diagrams: ", len(self.train_diagrams))
        print("Number of validation diagrams: ", len(self.val_diagrams))
        print("Number of test diagrams: ", len(self.test_diagrams))
        print("Number of training labels: ", len(self.Y_train))
        print("Number of validation labels: ", len(self.Y_val))
        print("Number of test labels: ", len(self.Y_test))

        train_circuits = self.create_circuits(self.train_diagrams, self.ansatz, self.n_layers, self.n_single_qubit_params, self.qn, self.qs)
        val_circuits = self.create_circuits(self.val_diagrams, self.ansatz, self.n_layers, self.n_single_qubit_params, self.qn, self.qs)
        test_circuits = self.create_circuits(self.test_diagrams, self.ansatz, self.n_layers, self.n_single_qubit_params, self.qn, self.qs)

        # Mandatory to add the test circuits to the list of all circuits even if its not good practice
        # Otherwise the model will raise an error if it encounter an unseen symbol in the test set
        self.all_circuits = train_circuits + val_circuits + test_circuits
        
    # initialise model
    if model_number == 1:
        model = Alpha_1_2_model.from_diagrams(all_circuits, probabilities=True, normalize=True, 
                                                                version_original = True, reduced_word_embedding_dimension = reduced_word_embedding_dimension)
    elif model_number == 2:
        model = Alpha_1_2_model.from_diagrams(all_circuits, probabilities=True, normalize=True, 
                                                                version_original = False, reduced_word_embedding_dimension = reduced_word_embedding_dimension)
    else:
        model = Alpha_3_model(n_qubits, 0.01, device)



    if model_number == 1:
        dummy_input = torch.randn(optimal_batch_size, 3, reduced_word_embedding_dimension, dtype=torch.float).to(device) # We assume we have 3 words per sentence as a mean of the whole dataset
    elif model_number == 2:
        dummy_input = torch.randn(optimal_batch_size, 768, dtype=torch.float).to(device)
    else:
        dummy_input = torch.randn(optimal_batch_size, 768, dtype=torch.float).to(device)


    #self.model = nn.DataParallel(self.model)
    model.to(device)


    measured_inference_time = []

    if device == "cuda:0":
        repetitions=100

        #GPU-WARM-UP
        for _ in range(10):
            _ = model(dummy_input)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                measured_inference_time.append(curr_time/optimal_batch_size*1e6)


        print('Inference time mean (micro-seconds):', np.mean(measured_inference_time))
        print('Inference time std (micro-seconds):', np.std(measured_inference_time))

    else:
        repetitions=100

        #CPU-WARM-UP, Dont know if that exists for CPU or not
        for _ in range(10):
            _ = model(dummy_input)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter = time.time()
                _ = model(dummy_input)
                ender = time.time()
                curr_time = ender - starter
                measured_inference_time.append(curr_time/optimal_batch_size*1e6)


        print('Inference time mean (micro-seconds):', np.mean(measured_inference_time))
        print('Inference time std (micro-seconds):', np.std(measured_inference_time))


def main():
    #Experiment 1
    dataset_name = '3k'
    reduced_word_embedding_dimension = 22
    n_qubits = 3
    optimal_batch_size = 8192
    device = "cpu"

    print("Experiment 1, model 1")
    compute_inference_time(1, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device)
    print("Experiment 1, model 2")
    compute_inference_time(2, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device)
    print("Experiment 1, model 3")
    compute_inference_time(3, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device)

    #Experiment 2
    dataset_name = '20k'
    reduced_word_embedding_dimension = 22
    n_qubits = 3
    optimal_batch_size = 8192
    device = "cpu"

    print("Experiment 2, model 1")
    compute_inference_time(1, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device)
    print("Experiment 2, model 2")
    compute_inference_time(2, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device)
    print("Experiment 2, model 3")
    compute_inference_time(3, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device)

    #Experiment 3
    dataset_name = '20k'
    reduced_word_embedding_dimension = 22
    n_qubits = 3
    optimal_batch_size = 16.384
    device = "cuda:0"

    print("Experiment 3, model 3")
    compute_inference_time(3, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device)

if __name__ == "__main__":
    main()              
    