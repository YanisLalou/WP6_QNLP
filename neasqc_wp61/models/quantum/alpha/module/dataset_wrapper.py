from transformers import BertTokenizer
from transformers import BertModel
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )

from discopy import grammar
from pytket.circuit.display import render_circuit_jupyter

from lambeq import BobcatParser
from lambeq.ansatz.circuit import IQPAnsatz
from lambeq.core.types import AtomicType

from sympy import default_sort_key

import time

import re

import json
import pandas as pd

parser = BobcatParser()

class dataset_wrapper():
    """Generates BERT embeddings for each sentence. Also hold sentences, sentence_types, sentence_labels

    ........

    Attributes
    ----------
    
    file: str
        Inputted file name 
    sentences: list
        list of sentences
    sentence_types: list
        list of sentence types(grammatical structures)
    sentence_labels: list
        list of sentence classification labels
    bert_embeddings: list[list[floats]]
        The BERT embeddings for each word in each sentence.

    """
    
    def __init__(self, filename: str):
        """Initialises dataset_wrapper.

        Takes in a dataset of sentences and finds the sentences, sentence structures(types), sentence classification labels and the BERT embeddings for each word in each sentence.
        

        Parameters
        ----------
        filename : str
            Input data json file
            

        """
        self.file=filename
        
        self.sentences, self.sentence_types, self.sentence_labels = self.data_parser()
        
        self.bert_embeddings = self.data_preparation()
        
        
    def data_preparation(self):
        """Transforms sentences into Qsentences.

        Takes sentence train and test data along with their repective true or false labels and transforms each sentence into a so-called Qsentence.:

        Parameters
        ----------
        
        Returns
        -------
        Dataset: list[list[floats]]
            Nested list of BERT embeddings for each word in each sentence.
            
        """
    
        Dataset = []  
        for sentence in self.sentences:
            print("Sentence = ", sentence)
            Dataset.append(self.get_sentence_BERT_embeddings(sentence_string=sentence))
            print("BERT Embedding Obtained")
        return Dataset
    
    def data_parser(self):
        """Parses the elements of the dataset and returns them as lists.

        Takes the dataset and returns a tuple of three lists, the sentences, sentence_types and sentence_labels.:

        Parameters
        ----------

        Returns
        -------
        sentences, sentence_types, sentence_labels: tuple(list,list,list)
           
        """
        with open(self.file) as f:
            data = json.load(f)
        dftrain = pd.DataFrame(data['train_data'])
        dftrain["truth_value"]= dftrain["truth_value"].map({True: [1,0], False: [0,1]})
        dftest = pd.DataFrame(data['test_data'])
        dftest["truth_value"]= dftest["truth_value"].map({True: [1,0], False: [0,1]})
        
        sentences = []
        sentence_types = []
        sentence_labels = []

        #for sentence, sentence_type, label in zip(dftrain["sentence"], dftrain["structure_tilde"],dftrain["truth_value"]):
        for sentence, sentence_type, label in zip(dftrain["sentence"], dftrain["sentence_type"],dftrain["truth_value"]):
            if sentence == 'Borring as hell':
                sentence = 'Boring as hell'
            sentences.append(sentence)
            sentence_types.append(sentence_type)
            sentence_labels.append(label)
        return sentences, sentence_types, sentence_labels

    def get_sentence_BERT_embeddings(self, sentence_string):
        """Returns word embedding for each sentence.

        Takes a list of sentences and find a Bert embedding for each.:


        Returns
        -------
        Sentences_Embeddings: list
            List consisting of word embeddings for each sentence.

        """
        ellipses = ['.'*i for i in range(1,30,1)]
        
        SentenceList = sentence_string
        Sentences_Embeddings = []
        if type(SentenceList) == str:
            SentenceList = [SentenceList]
        for sentence in SentenceList:
            tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(sentence, tokenizer)
            list_token_embeddings = self.get_bert_embeddings(tokens_tensor, segments_tensors, model)
            nwords = len(sentence.split(" "))
            
            print("tokenized_text = ",tokenized_text)
            word_embeddings = []
            
            
            #for word in sentence.split(" "):
            for word in re.split(r' |-|&',sentence):
                for ellipse in ellipses:
                    if word==ellipse:
                        word="."
                if word.lower() == 'amazon.com':
                    word = 'amazon'
                if word.lower() == 'asp.net':
                    word = 'asp'
                if word.lower() == ':)':
                    word = ':'
                if word == "PC'd":
                    word = 'pc'
                if word == '':
                    word = '-'
                #word_index = tokenized_text.index(word.lower().replace(".",""))
                print("word = ", word)
                print("word.lower() = ", word.lower())
                word_index = tokenized_text.index(word.lower())
                word_embeddings.append(list_token_embeddings[word_index])

            Sentences_Embeddings.append(word_embeddings)
        return Sentences_Embeddings
        
        


    def bert_text_preparation(self, text: str, tokenizer = tokenizer)->tuple:
        """Tokenises sentence.

        Uses Bert Tokeniser to tokenise a sentence(string). It returns the tokenized text, tokens tensor and segments tensors.:

        Parameters
        ----------
        text : str
            A sentence to be tokenised.
        tokeniser : tokenizer
            The tokeniser being used. The default is set to the transformers pretrained bert tokeniser.

        Returns
        -------
        (tokenized_text, tokens_tensor, segments_tensors): tuple  

        """

        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        
        ## This loop eliminates the appearance of "#" symbols in the tokenised text due to the tokenizer.tokenize() process above
        del_list = []
        for i,x in enumerate(tokenized_text):
            if x[0]=='#':
                tokenized_text[i] = tokenized_text[i-1] + tokenized_text[i][2:]
                del_list.append(i-1)
                
        tokenized_text = [tokenized_text[i] for i in range(len(tokenized_text)) if i not in del_list]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)


        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors


    def get_bert_embeddings(self, tokens_tensor, segments_tensors, model = model)->list:
        """Word embeddings for each word in a sentence.

        Returns word embeddings for each token in a sentence.:

        Parameters
        ----------
        tokens_tensor:tokens_tensor
            Tensor of tokens for a sentence

        segments_tensors:segments_tensor
            Tensor of segments of a sentence

        model:Embedding_Model
            Word embedding model to be used. The default is set to the Bert Model.



        Returns
        -------
        list_token_embeddings: list
            List consisting of word embeddings for each token in the sentence.

        """

        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)

            hidden_states = outputs[2][1:]


        token_embeddings = hidden_states[-1]

        token_embeddings = torch.squeeze(token_embeddings, dim=0)

        list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

        return list_token_embeddings