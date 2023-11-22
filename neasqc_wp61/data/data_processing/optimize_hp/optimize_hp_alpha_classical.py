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

from alpha_classical_counterparts_trainer import Alpha_classical_counterparts_trainer


from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


parser = argparse.ArgumentParser()

# To chose the model


def main(args):
    random.seed(0)
    
    model_name = "alpha_" + str(1) + "_classical_counterpart_20K"
    
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
    #pbounds = {'lr': (1e-3, 1e-1), 'weight_decay': (0, 1e-2), 'gamma': (0.1, 1)}
    pbounds = {'lr': (1e-3, 5e-1)}

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


def black_box_function(lr, weight_decay=0, gamma=1):
    # train_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_train_sentence.csv'
    # val_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_validation_sentence.csv'
    # test_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_test_sentence.csv'

    # train_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_train_word.csv'
    # val_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_validation_word.csv'
    # test_path = '../../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_test_word.csv'

    #train_path = '../../datasets/alpha_report_datasets/alpha_20k_dataset/amazonreview_train_sentence.csv'
    #val_path = '../../datasets/alpha_report_datasets/alpha_20k_dataset/amazonreview_dev_sentence.csv'
    #test_path = '../../datasets/alpha_report_datasets/alpha_20k_dataset/amazonreview_test_sentence.csv'

    train_path = '../../datasets/alpha_report_datasets/alpha_20k_dataset/amazonreview_train_word.csv'
    val_path = '../../datasets/alpha_report_datasets/alpha_20k_dataset/amazonreview_dev_word.csv'
    test_path = '../../datasets/alpha_report_datasets/alpha_20k_dataset/amazonreview_test_word.csv'

    trainer = Alpha_classical_counterparts_trainer(80, train_path, val_path, test_path, 0, 3, 0.1,
                                        2048, lr, weight_decay, 15, gamma, 22,  model_number = 1)

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