from torch import nn
import torch
import pennylane as qml
from pennylane import numpy as np

class alpha_classical_counterparts_model(nn.Module):
    def __init__(self, model_number, dataset_name, reduced_word_embedding_dimension = None): 
        """
        Definition of the classical counterpart of the quantum model.
        """

        super().__init__()


        self.model_number = model_number
        self.dataset_name = dataset_name
        self.reduced_word_embedding_dimension = reduced_word_embedding_dimension


        if self.model_number == 1:
            if self.reduced_word_embedding_dimension is None:
                pre_net_input_size = 768
            else:
                pre_net_input_size = self.reduced_word_embedding_dimension
        else:
            # BERT embedding size = 768
            pre_net_input_size = 768


        if self.model_number == 1 or self.model_number == 2:
            self.pre_net = nn.Linear(pre_net_input_size, 19) # 19 comes from the max number of parameters in the circuit
        else:
            self.pre_net = nn.Linear(pre_net_input_size, 3)

        print("Model number = ", self.model_number)
        print("Dataset name = ", self.dataset_name)
            

        if (self.model_number == 1 or self.model_number == 2) & (self.dataset_name == '3k'):
            print("1st layer size = ", 728)
            self.counter_part_net = nn.Sequential(nn.Linear(19, 728, bias=False),
                                        nn.LeakyReLU(0.01),
                                        nn.Linear(728, 2, bias=False),
                                        nn.LeakyReLU(0.01))

            self.post_net = nn.Linear(2, 1)
            self.last_activation_layer = nn.Sigmoid()

            # 19*728 + 728*2 = 15,288. We want 15,304 trainable parameters
        elif (self.model_number == 1 or self.model_number == 2) & (self.dataset_name == '20k'):
            print("1st layer size = ", 2673)
            self.counter_part_net = nn.Sequential(nn.Linear(19, 2673, bias=False),
                                        nn.LeakyReLU(0.01),
                                        nn.Linear(2673, 2, bias=False),
                                        nn.LeakyReLU(0.01))

            # 19*2673 + 2673*2 = 56133. We want 56,142 trainable parameters

            self.post_net = nn.Linear(2, 1)
            self.last_activation_layer = nn.Sigmoid()

        elif model_number == 3:
            print("1st layer size = ", 2)
            self.counter_part_net = nn.Sequential(nn.Linear(3, 2, bias=False),
                                        nn.LeakyReLU(0.01),
                                        nn.Linear(2, 2, bias=False),
                                        nn.LeakyReLU(0.01),
                                        nn.Linear(2, 3, bias=False),
                                        nn.LeakyReLU(0.01))
            #3*2 + 2*2 +0 2*3 = 16 trainable weights --> 16 trainable parameters

            self.post_net = nn.Linear(3, 2)
            self.last_activation_layer = nn.Softmax()
        else:
            raise ValueError("Model number must be 1, 2 or 3.")

        


    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the classical counterpart of the quantum model.
        """       

        if self.model_number == 1:
            # Here we have a list of embeddings for each word in the sentence
            pre_out = []


            for embedding in input_features:
                post_qc_output = self.pre_net(embedding)

                post_qc_output = torch.sum(post_qc_output, dim=0)
                pre_out.append(post_qc_output)

            pre_out = torch.stack(pre_out)

        else:
            # Here we have a list of embeddings for each sentence
            pre_out = self.pre_net(input_features)



        counterpart_out = self.counter_part_net(pre_out)

        # return the two-dimensional prediction from the postprocessing layer
        return self.last_activation_layer(self.post_net(counterpart_out))