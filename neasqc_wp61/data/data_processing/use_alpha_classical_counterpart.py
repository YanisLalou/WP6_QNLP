import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/alpha/module/")
import argparse

import json
import numpy as np

import random, os
import numpy as np
import torch
import time
import git

from alpha_classical_counterparts_trainer import Alpha_classical_counterparts_trainer
from save_json_output import JsonOutputer



parser = argparse.ArgumentParser()

# To chose the model

parser.add_argument("-s", "--seed", help = "Seed for the initial parameters", type = int, default = 0)
parser.add_argument("-i", "--iterations", help = "Number of iterations of the optimiser", type = int, default = 100)
parser.add_argument("-r", "--runs", help = "Number of runs", type = int, default = 1)
parser.add_argument("-tr", "--train", help = "Directory of the train dataset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_train.csv')
parser.add_argument("-val", "--val", help = "Directory of the validation dataset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_dev.csv')
parser.add_argument("-te", "--test", help = "Directory of the test dataset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_test.csv')
parser.add_argument("-o", "--output", help = "Output directory with the predictions", type = str, default = "../../benchmarking/results/raw/")

parser.add_argument("-b", "--batch_size", help = "Batch size", type = int, default = 2048)

# Hyperparameters
parser.add_argument("-lr", "--lr", help = "Learning rate", type = float, default = 2e-3)
parser.add_argument("-wd", "--weight_decay", help = "Weight decay", type = float, default = 0.0)
parser.add_argument("-slr", "--step_lr", help = "Step size for the learning rate scheduler", type = int, default = 20)
parser.add_argument("-g", "--gamma", help = "Gamma for the learning rate scheduler", type = float, default = 0.5)

# Choose betweeb Alpha 1 - 2 - 3 counterparts
parser.add_argument("-c", "--counterpart", help = "Choose between Alpha 1 - 2 - 3 counterparts", type = int, default = 3)
parser.add_argument("-pca", "--pca", help = "Choose the reduced dimension for the word embeddings", type = int, default = 22)

args = parser.parse_args()



def main(args):
    random.seed(args.seed)
    seed_list = random.sample(range(1, int(2**32 - 1)), int(args.runs))
    
    model_name = "alpha_" + str(args.counterpart) + "_classical_counterpart"

    best_val_acc_all_runs = 0
    best_run = 0

    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Create the JsonOutputer object
    json_outputer = JsonOutputer(model_name, timestr, args.output)

    for i in range(args.runs):
        t_before = time.time()
        print("\n")
        print("-----------------------------------")
        print("run = ", i+1)
        print("-----------------------------------")
        print("\n")

        trainer = Alpha_classical_counterparts_trainer(args.iterations, args.train, args.val, args.test, seed_list[i],
                                          args.batch_size, args.lr, args.weight_decay, args.step_lr, args.gamma, args.pca, model_number = args.counterpart)
        
        training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, best_val_acc, best_model = trainer.train()

        t_after = time.time()
        print("Time taken for this run = ", t_after - t_before, "\n")
        time_taken = t_after - t_before

        prediction_list, inference_time = trainer.predict()
        prediction_list = prediction_list.tolist()

        test_loss, test_acc = trainer.compute_test_logs(best_model)

        if best_val_acc > best_val_acc_all_runs:
            best_val_acc_all_runs = best_val_acc
            best_run = i

        # Save the results of each run in a json file
        json_outputer.save_json_output_run_by_run(args, prediction_list, time_taken, inference_time = inference_time,
                    best_val_acc=best_val_acc_all_runs, best_run = best_run, seed_list=seed_list[i],
                    test_acc=test_acc, test_loss=test_loss,
                    val_acc=validation_acc_list, val_loss=validation_loss_list,
                    train_acc=training_acc_list, train_loss=training_loss_list
                    )





if __name__ == "__main__":
    args = parser.parse_args()
    main(args)              
    