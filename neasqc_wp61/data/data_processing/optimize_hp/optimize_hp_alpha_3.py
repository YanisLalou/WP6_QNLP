import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../../models/quantum/alpha/module/")
import argparse

import json
import numpy as np

import random, os
import numpy as np
import torch
import time
import git

from alpha_pennylane_trainer import Alpha_pennylane_trainer


from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


parser = argparse.ArgumentParser()

# To chose the model

parser.add_argument("-s", "--seed", help = "Seed for the initial parameters", type = int, default = 0)
parser.add_argument("-i", "--iterations", help = "Number of iterations of the optimiser", type = int, default = 100)
parser.add_argument("-r", "--runs", help = "Number of runs", type = int, default = 1)
parser.add_argument("-tr", "--train", help = "Directory of the train dataset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_train.csv')
parser.add_argument("-val", "--val", help = "Directory of the validation dataset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_dev.csv')
parser.add_argument("-te", "--test", help = "Directory of the test dataset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_test.csv')
parser.add_argument("-o", "--output", help = "Output directory with the predictions", type = str, default = "../../benchmarking/results/raw/")

parser.add_argument("-nq", "--n_qubits", help = "Number of qubits in our circuit", type = int, default = 3)
parser.add_argument("-qd", "--q_delta", help = "Initial spread of the parameters", type = float, default = 0.01)
parser.add_argument("-b", "--batch_size", help = "Batch size", type = int, default = 2048)

# Hyperparameters
parser.add_argument("-lr", "--lr", help = "Learning rate", type = float, default = 2e-3)
parser.add_argument("-wd", "--weight_decay", help = "Weight decay", type = float, default = 0.0)
parser.add_argument("-slr", "--step_lr", help = "Step size for the learning rate scheduler", type = int, default = 20)
parser.add_argument("-g", "--gamma", help = "Gamma for the learning rate scheduler", type = float, default = 0.5)

args = parser.parse_args()



def main(args):
    random.seed(args.seed)
    seed_list = random.sample(range(1, int(2**32 - 1)), int(args.runs))
    
    model_name = "alpha_3"
    
    all_training_loss_list = []
    all_training_acc_list = []
    all_validation_loss_list = []
    all_validation_acc_list = []

    all_prediction_list = []
    all_time_list = []

    all_best_model_state_dict = []

    best_val_acc_all_runs = 0
    best_run = 0

    timestr = time.strftime("%Y%m%d-%H%M%S")


    # Bounded region of parameter space
    pbounds = {'lr': (1e-3, 1e-1), 'weight_decay': (0, 1e-2), 'gamma': (0.1, 1)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    logger = JSONLogger(path=f'{model_name}_{timestr}_logs.log')
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
    init_points=10,
    n_iter=100,
)


def black_box_function(lr, weight_decay, gamma):
    train_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_train_sentence.csv'
    val_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_validation_sentence.csv'
    test_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_test_sentence.csv'

    trainer = Alpha_pennylane_trainer(100, train_path, val_path, test_path, 0, 3, 0.1,
                                        2048, lr, weight_decay, 15, gamma)

    training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, best_val_acc, best_model = trainer.train()

    print("LAST VAL ACC = ", validation_acc_list[-1])
    return -validation_loss_list[-1]
    


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)              
    


class Custom_JSONLogger(JSONLogger):
    def __init__(self, path, reset=True):
        super(JSONLogger, self).__init__()

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }

            if "allowed" in data: # fix: github.com/fmfn/BayesianOptimization/issues/361
                data["allowed"] = bool(data["allowed"])

            with open(self._path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, instance)