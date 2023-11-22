import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from alpha_classical_counterparts_model import alpha_classical_counterparts_model
from utils import seed_everything, preprocess_train_test_dataset_for_alpha_3, BertEmbeddingDataset, preprocess_train_test_dataset_words, preprocess_train_test_dataset
import time
import lambeq

class Alpha_classical_counterparts_trainer():
    def __init__(self, number_of_epochs: int, train_path: str, val_path: str, test_path: str, seed: int,
                 batch_size: int, lr: float, weight_decay: float, step_lr: int, gamma: float, reduced_word_embedding_dimension: int, model_number: int):
        

        """
            This class is the trainer for the Alpha 1 - 2 - 3 classical counterparts.

            For the Classical 1 and Classical 2, It is hard-coded to work with the 3k and 20k datasets only!
            Indeed the NN architecture for the Classical 1 and Classical 2 depends on the number of parameters in the lambeq model associated to a certain dataset.

            For the Classical 3, it is hard-coded to work with the n_qubits = 3 only!
            Again the NN architecture for the Classical 3 depends on n_qubits.
        """
        
        self.number_of_epochs = number_of_epochs
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.seed = seed
        self.batch_size = batch_size

        self.reduced_word_embedding_dimension = reduced_word_embedding_dimension

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_lr = step_lr
        self.gamma = gamma

        #Model number
        self.model_number = model_number


        # seed everything
        seed_everything(self.seed)


        if self.train_path == '../datasets/alpha_report_datasets/alpha_20k_dataset/amazonreview_train_sentence.csv' or self.train_path == '../datasets/alpha_report_datasets/alpha_20k_dataset/amazonreview_train_word.csv':
            dataset_name = "20k"
        elif self.train_path == '../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_train_sentence.csv' or self.train_path == '../datasets/alpha_report_datasets/alpha_report_3K/reduced_amazonreview_train_word.csv':
            dataset_name = "3k"
        else:
            dataset_name = "3k"
            # No worries for the toy dataset, the results will be ignored anyway
            #raise Exception('Wrong dataset path')


         # load the dataset
        if self.model_number == 1:
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = preprocess_train_test_dataset_words(self.train_path, self.val_path, self.test_path, self.reduced_word_embedding_dimension)
        elif self.model_number == 2:
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = preprocess_train_test_dataset(self.train_path, self.val_path, self.test_path)
        else:
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = preprocess_train_test_dataset_for_alpha_3(self.train_path, self.val_path, self.test_path)
       


        # initialise datasets and optimizers as in PyTorch

        if self.model_number == 1:

            self.training_dataloader = lambeq.Dataset(self.X_train['sentence_vectorized'].values.tolist(),
                                    self.Y_train,
                                    batch_size=self.batch_size)
            self.validation_dataloader = lambeq.Dataset(self.X_val['sentence_vectorized'].values.tolist(),
                                    self.Y_val,
                                    batch_size=self.batch_size)
            self.test_dataloader = lambeq.Dataset(self.X_test['sentence_vectorized'].values.tolist(),
                                    self.Y_test,
                                    batch_size=self.batch_size)

        elif self.model_number == 2:
            self.training_dataloader = lambeq.Dataset(self.X_train['sentence_embedding'].values.tolist(),
                                    self.Y_train,
                                    batch_size=self.batch_size)
            self.validation_dataloader = lambeq.Dataset(self.X_val['sentence_embedding'].values.tolist(),
                                    self.Y_val,
                                    batch_size=self.batch_size)
            self.test_dataloader = lambeq.Dataset(self.X_test['sentence_embedding'].values.tolist(),
                                    self.Y_test,
                                    batch_size=self.batch_size)
        else:
            self.train_dataset = BertEmbeddingDataset(self.X_train, self.Y_train)
            self.validation_dataset = BertEmbeddingDataset(self.X_val, self.Y_val)
            self.test_dataset = BertEmbeddingDataset(self.X_test, self.Y_test)

            self.training_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            # Shuffle is set to False for the validation dataset because in the predict function we need to keep the order of the predictions
            self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

            




        # initialise the device
        if self.model_number == 3:
            # Only the Classical 3 counterpart should be trained on GPU if possible (to follow Experiment 3 requirements)
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            print('Device: {}'.format(self.device))
        else:
            self.device = torch.device("cpu")
            print('Device: {}'.format(self.device))


        # initialise model
        self.model = alpha_classical_counterparts_model(self.model_number, dataset_name, self.reduced_word_embedding_dimension)


        # initialise loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_lr, gamma=self.gamma)
	
        #self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.criterion.to(self.device)

        

    def train(self):
        training_loss_list = []
        training_acc_list = []

        validation_loss_list = []
        validation_acc_list = []

        best_val_acc = 0.0


        for epoch in range(self.number_of_epochs):
            print('Epoch: {}'.format(epoch))
            running_loss = 0.0
            running_corrects = 0

            self.model.train()

            for inputs, labels in self.training_dataloader:
                batch_size_ = len(inputs)

                if self.model_number == 1:
                    # Word version
                    #import pdb; pdb.set_trace()
                    embedding_list = [torch.tensor(embedding).float() for embedding in inputs]  

                    outputs = self.model(embedding_list)
                    outputs = torch.flatten(outputs)

                    labels = torch.FloatTensor(labels)

                elif self.model_number == 2:
                    embeddings_tensor = torch.stack([torch.tensor(embedding) for embedding in inputs])
                    outputs = self.model(embeddings_tensor)
                    outputs = torch.flatten(outputs)

                    labels = torch.FloatTensor(labels)

                else:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)


                self.optimizer.zero_grad()
                
                loss = self.criterion(outputs, labels)
                loss.backward()

            
                self.optimizer.step()

                # Print iteration results
                running_loss += loss.item()*batch_size_

                if self.model_number == 1 or self.model_number == 2:
                    batch_corrects = (torch.round(outputs) == labels).sum().item()
                else:
                    batch_corrects = torch.sum(preds == torch.max(labels, 1)[1]).item()

                running_corrects += batch_corrects

            
            # Print epoch results
            if self.model_number == 1 or self.model_number == 2:
                train_loss = running_loss / len(self.training_dataloader)
                train_acc = running_corrects / len(self.training_dataloader)
            else:
                train_loss = running_loss / len(self.training_dataloader.dataset)
                train_acc = running_corrects / len(self.training_dataloader.dataset)
            
            training_loss_list.append(train_loss)
            training_acc_list.append(train_acc)

            running_loss = 0.0
            running_corrects = 0

            self.model.eval()

            with torch.no_grad():
                for inputs, labels in self.validation_dataloader:
                    batch_size_ = len(inputs)

                if self.model_number == 1:
                    # Word version
                    #import pdb; pdb.set_trace()
                    embedding_list = [torch.tensor(embedding).float() for embedding in inputs]  

                    outputs = self.model(embedding_list)
                    outputs = torch.flatten(outputs)

                    labels = torch.FloatTensor(labels)

                elif self.model_number == 2:
                    embeddings_tensor = torch.stack([torch.tensor(embedding) for embedding in inputs])
                    outputs = self.model(embeddings_tensor)
                    outputs = torch.flatten(outputs)

                    labels = torch.FloatTensor(labels)
                    
                else:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)

                self.optimizer.zero_grad()
                
                loss = self.criterion(outputs, labels)
                
                # Print iteration results
                running_loss += loss.item()*batch_size_
                
                if self.model_number == 1 or self.model_number == 2:
                    batch_corrects = (torch.round(outputs) == labels).sum().item()
                else:
                    batch_corrects = torch.sum(preds == torch.max(labels, 1)[1]).item()


                running_corrects += batch_corrects


            
            if self.model_number == 1 or self.model_number == 2:
                validation_loss = running_loss / len(self.validation_dataloader)
                validation_acc = running_corrects / len(self.validation_dataloader)
            else:
                validation_loss = running_loss / len(self.validation_dataloader.dataset)
                validation_acc = running_corrects / len(self.validation_dataloader.dataset)


            validation_loss_list.append(validation_loss)
            validation_acc_list.append(validation_acc)

            if validation_acc > best_val_acc:
                best_val_acc = validation_acc
                best_model = self.model.state_dict()
                

            self.lr_scheduler.step()
            
            print('Train loss: {}'.format(train_loss))
            print('Valid loss: {}'.format(validation_loss))
            print('Train acc: {}'.format(train_acc))
            print('Valid acc: {}'.format(validation_acc))

            print('-'*20)

        return training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, best_val_acc, best_model

    def predict(self):
        t_before = time.time()
        prediction_list = torch.tensor([]).to(self.device)
        
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.validation_dataloader:

                if self.model_number == 1:
                    # Word version
                    embedding_list = [torch.tensor(embedding).float() for embedding in inputs]  

                    outputs = self.model(embedding_list)
                    outputs = torch.flatten(outputs)
                    preds = torch.round(outputs)

                    labels = torch.FloatTensor(labels)

                elif self.model_number == 2:
                    embeddings_tensor = torch.stack([torch.tensor(embedding) for embedding in inputs])
                    outputs = self.model(embeddings_tensor)
                    outputs = torch.flatten(outputs)
                    preds = torch.round(outputs)

                    labels = torch.FloatTensor(labels)
                    
                else:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)

                prediction_list = torch.cat((prediction_list, torch.round(torch.flatten(preds))))

            
        t_after = time.time()
        print("Time taken for this run = ", t_after - t_before, "\n")
        inference_time = t_after - t_before

        return prediction_list.detach().cpu().numpy(), inference_time


    def compute_test_logs(self, best_model):
        running_loss = 0.0
        running_corrects = 0

        # Load the best model found during training
        self.model.load_state_dict(best_model)
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                batch_size_ = len(inputs)
                if self.model_number == 1:
                    # Word version
                    embedding_list = [torch.tensor(embedding).float() for embedding in inputs]  

                    outputs = self.model(embedding_list)
                    outputs = torch.flatten(outputs)

                    labels = torch.FloatTensor(labels)

                elif self.model_number == 2:
                    embeddings_tensor = torch.stack([torch.tensor(embedding) for embedding in inputs])
                    outputs = self.model(embeddings_tensor)
                    outputs = torch.flatten(outputs)

                    labels = torch.FloatTensor(labels)
                    
                else:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)


                self.optimizer.zero_grad()
                
                loss = self.criterion(outputs, labels)
                
                # Print iteration results
                running_loss += loss.item()*batch_size_
                
                if self.model_number == 1 or self.model_number == 2:
                    batch_corrects = (torch.round(outputs) == labels).sum().item()
                else:
                    batch_corrects = torch.sum(preds == torch.max(labels, 1)[1]).item()


                running_corrects += batch_corrects


    
            
        if self.model_number == 1 or self.model_number == 2:
            test_loss = running_loss / len(self.test_dataloader)
            test_acc = running_corrects / len(self.test_dataloader)
        else:
            test_loss = running_loss / len(self.test_dataloader.dataset)
            test_acc = running_corrects / len(self.test_dataloader.dataset)

        print('Run test results:')
        print('Test loss: {}'.format(test_loss))
        print('Test acc: {}'.format(test_acc))

        return test_loss, test_acc

