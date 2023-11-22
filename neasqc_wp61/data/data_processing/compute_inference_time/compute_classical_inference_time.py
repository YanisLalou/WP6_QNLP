import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../../models/quantum/alpha/module/")
from alpha_classical_counterparts_model import alpha_classical_counterparts_model

import torch
import torch.nn as nn
import time
import numpy as np


def compute_inference_time(model_number, dataset_name, reduced_word_embedding_dimension, n_qubits, optimal_batch_size, device):
    
    # initialise model
    model = alpha_classical_counterparts_model(n_qubits, model_number, dataset_name, reduced_word_embedding_dimension)


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
    