# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2014 Siddharth Agrawal
# Code written by : Siddharth Agrawal
# Email ID : siddharth.950@gmail.com

#
##
###
#### MODIFIED by Nick Powell - Grakn Labs 2017
###
##
#

from argparse import ArgumentParser
from datetime import datetime
from scipy.linalg import blas as FB
import insertLexicon as graql
import math
import pickle
import numpy as np
import random
import scipy.optimize
import scipy.sparse as sp
import sys


_NEW_RELATIONS_PATH = "insertNewRelations.txt"

###########################################################################################
""" For formatting print statements """
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


#######################################################################################
""" Add contribution of 'W[i]' term in the entity vectors' gradient """
def addWeightToGradient(i, k, W_slice, entity_vectors, e1, e2, e1_neg, e2_neg, temp_pos, temp_neg, e1_sparse, e2_sparse, e1_neg_sparse, e2_neg_sparse):
    return ((np.dot(W_slice, entity_vectors[:, e2.tolist()]) * temp_pos) * e1_sparse \
        + (np.dot(W_slice.T, entity_vectors[:, e1.tolist()]) * temp_pos) * e2_sparse \
        + (np.dot(W_slice, entity_vectors[:, e2_neg.tolist()]) * temp_neg) * e1_neg_sparse \
        + (np.dot(W_slice.T, entity_vectors[:, e1_neg.tolist()]) * temp_neg) * e2_neg_sparse)


###########################################################################################
""" The Neural Tensor Network class """
class NeuralTensorNetwork(object):

    #######################################################################################
    """ Initialization of the network """
    def __init__(self, program_parameters):

        """ Extract program parameters from the passed dictionary """
        self.num_words           = program_parameters['num_words']
        self.embedding_size      = program_parameters['embedding_size']
        self.num_entities        = program_parameters['num_entities']
        self.num_relations       = program_parameters['num_relations']
        self.batch_size          = program_parameters['batch_size']
        self.slice_size          = program_parameters['slice_size']
        self.word_indices        = program_parameters['word_indices']
        self.activation_function = program_parameters['activation_function']
        self.lamda               = program_parameters['lamda']

        """ Initialize word vectors randomly with each element in the range [-r, r] """
        r = 0.0001
        word_vectors = np.random.random((self.embedding_size, self.num_words)) * 2 * r - r

        """ 'r' value for initializing 'W' """
        r = 1 / math.sqrt(2 * self.embedding_size)

        """ Initialize the parameter dictionaries """
        W = {}; V = {}; b = {}; U = {};

        for i in range(self.num_relations):

            """ Initialize tensor network parameters """
            W[i] = np.random.random((self.embedding_size, self.embedding_size, self.slice_size)) * 2 * r - r
            W[i] = np.full((self.embedding_size, self.embedding_size, self.slice_size), 0.65, dtype=float) * 2 * r - r
            V[i] = np.zeros((2 * self.embedding_size, self.slice_size))
            b[i] = np.zeros((1, self.slice_size))
            U[i] = np.ones((self.slice_size, 1))

        """ Unroll the parameters into a vector """
        self.theta, self.decode_info = self.stackToParams(W, V, b, U, word_vectors)

    #######################################################################################
    """ Unroll the passed parameters into a vector """
    def stackToParams(self, *arguments):

        """ Initialize the 'theta' vector and 'decode_info' for the network configuration """
        theta       = []
        decode_info = {}

        for i in range(len(arguments)):
            """ Extract the 'i'the argument """
            argument = arguments[i]

            if isinstance(argument, dict):
                """ If the argument is a dictionary, store configuration as dictionary """
                decode_cell = {}

                for j in range(len(argument)):
                    """ Store the configuration and concatenate to the unrolled vector """
                    decode_cell[j] = argument[j].shape
                    theta          = np.concatenate((theta, argument[j].flatten()))

                """ Store the configuration dictionary of the argument """
                decode_info[i] = decode_cell

            else:
                """ If not a dictionary, simply store the configuration and unroll """
                decode_info[i] = argument.shape
                theta          = np.concatenate((theta, argument.flatten()))

        return theta, decode_info

    #######################################################################################
    """ Returns a stack of parameters, using an unrolled vector """
    def paramsToStack(self, theta):

        """ Initialize an empty stack """
        stack = []
        index = 0

        for i in range(len(self.decode_info)):
            """ Get the configuration of the 'i'th parameter to be put """
            decode_cell = self.decode_info[i]

            if isinstance(decode_cell, dict):
                """ If the cell is a dictionary, the 'i'th stack element is a dictionary """
                param_dict = {}

                for j in range(len(decode_cell)):
                    """ Extract the parameter matrix from the unrolled vector """
                    param_dict[j] = theta[index : index + np.prod(decode_cell[j])].reshape(decode_cell[j])
                    index        += np.prod(decode_cell[j])

                stack.append(param_dict)

            else:
                """ If not a dictionary, simply extract the parameter matrix """
                stack.append(theta[index : index + np.prod(decode_cell)].reshape(decode_cell))
                index += np.prod(decode_cell)

        return stack

    #######################################################################################
    """ Applies the activation function to the passed matrix """
    def activationFunction(self, x):

        if self.activation_function == 0:
            """ If activation function is tanh """
            return np.tanh(x)

        elif self.activation_function == 1:
            """ If activation function is sigmoid """
            return (1 / (1 + np.exp(-x)))

    #######################################################################################
    """ Returns differential of the activation function for the passed matrix """
    def activationDifferential(self, x):

        if self.activation_function == 0:
            """ If activation function is tanh """
            return (1 - np.power(x, 2))

        elif self.activation_function == 1:
            """ If activation function is sigmoid """
            return (x * (1 - x))

    #######################################################################################
    """ Add contribution of 'W[i]' term in the entity vectors' gradient """
    def addWeightToGradient(self, i, k, W_slice, entity_vectors, e1, e2, e1_neg, e2_neg, temp_pos, temp_neg, e1_sparse, e2_sparse, e1_neg_sparse, e2_neg_sparse):
        return ((np.dot(W_slice, entity_vectors[:, e2.tolist()]) * temp_pos) * e1_sparse \
            + (np.dot(W_slice.T, entity_vectors[:, e1.tolist()]) * temp_pos) * e2_sparse \
            + (np.dot(W_slice, entity_vectors[:, e2_neg.tolist()]) * temp_neg) * e1_neg_sparse \
            + (np.dot(W_slice.T, entity_vectors[:, e1_neg.tolist()]) * temp_neg) * e2_neg_sparse)



    #######################################################################################
    """ Returns cost and parameter gradients for a given batch of data """
    def neuralTensorNetworkCost(self, theta, data_batch, flip):
        """ Get stack of network parameters """
        W, V, b, U, word_vectors = self.paramsToStack(theta)

        """ Initialize entity vectors and their gradient as matrix of zeros """
        entity_vectors = np.zeros((self.embedding_size, self.num_entities))
        entity_vector_grad = np.zeros((self.embedding_size, self.num_entities))

        """ Assign entity vectors to be the mean of word vectors involved """
        for entity in range(self.num_entities):
            entity_vectors[:, entity] = np.mean(word_vectors[:, self.word_indices[entity]], axis = 1)

        """ Initialize cost as zero """
        cost = 0

        """ Make dictionaries for parameter gradients """
        W_grad = {}; V_grad = {}; b_grad = {}; U_grad = {}

        for i in range(self.num_relations):

            """ Make a list of examples for the 'i'th relation """
            rel_i_list = (data_batch['rel'] == i)
            num_rel_i = np.sum(rel_i_list)

            """ Get entity lists for examples of 'i'th relation """
            e1 = data_batch['e1'][rel_i_list].astype(int)
            e2 = data_batch['e2'][rel_i_list].astype(int)
            e3 = data_batch['e3'][rel_i_list].astype(int)

            """ Get entity vectors for examples of 'i'th relation """
            entity_vectors_e1 = entity_vectors[:, e1.tolist()]
            entity_vectors_e2 = entity_vectors[:, e2.tolist()]
            entity_vectors_e3 = entity_vectors[:, e3.tolist()]

            """ Choose entity vectors and lists based on 'flip' """
            if flip:
                entity_vectors_e1_neg = entity_vectors_e1
                entity_vectors_e2_neg = entity_vectors_e3
                e1_neg = e1
                e2_neg = e3

            else:
                entity_vectors_e1_neg = entity_vectors_e3
                entity_vectors_e2_neg = entity_vectors_e2
                e1_neg = e3
                e2_neg = e2

            """ Initialize pre-activations of the tensor network as matrix of zeros"""
            preactivation_pos = np.zeros((self.slice_size, num_rel_i))
            preactivation_neg = np.zeros((self.slice_size, num_rel_i))

            """ Add contributuion of term containing 'W' """
            for slice in range(self.slice_size):
                preactivation_pos[slice, :] = np.sum(entity_vectors_e1 *
                    np.dot(W[i][:, :, slice], entity_vectors_e2), axis = 0)
                preactivation_neg[slice, :] = np.sum(entity_vectors_e1_neg *
                    np.dot(W[i][:, :, slice], entity_vectors_e2_neg), axis = 0)

            """ Add contributions of terms containing 'V' and 'b' """
            preactivation_pos += b[i].T + np.dot(V[i].T, np.vstack((entity_vectors_e1, entity_vectors_e2)))
            preactivation_neg += b[i].T + np.dot(V[i].T, np.vstack((entity_vectors_e1_neg, entity_vectors_e2_neg)))

            """ Apply the activation funtion """
            activation_pos = self.activationFunction(preactivation_pos)
            activation_neg = self.activationFunction(preactivation_neg)

            """ Calculate scores for positive and negative examples """
            score_pos = np.dot(U[i].T, activation_pos)
            score_neg = np.dot(U[i].T, activation_neg)

            """ Filter for examples that contribute to error """
            wrong_filter = (score_pos + 1 > score_neg)[0]

            """ Add max-margin term to the cost """
            cost += np.sum(wrong_filter * (score_pos - score_neg + 1)[0])

            """ Initialize 'W[i]' and 'V[i]' gradients as matrix of zeros """
            W_grad[i] = np.zeros(W[i].shape)
            V_grad[i] = np.zeros(V[i].shape)

            """ Number of examples contributing to error """
            num_wrong = np.sum(wrong_filter)

            """ Filter matrices using 'wrong_filter' """
            activation_pos            = activation_pos[:, wrong_filter]
            activation_neg            = activation_neg[:, wrong_filter]
            entity_vectors_e1_rel     = entity_vectors_e1[:, wrong_filter]
            entity_vectors_e2_rel     = entity_vectors_e2[:, wrong_filter]
            entity_vectors_e1_rel_neg = entity_vectors_e1_neg[:, wrong_filter]
            entity_vectors_e2_rel_neg = entity_vectors_e2_neg[:, wrong_filter]

            """ Filter entity lists using 'wrong_filter' """
            e1     = e1[wrong_filter]
            e2     = e2[wrong_filter]
            e1_neg = e1_neg[wrong_filter]
            e2_neg = e2_neg[wrong_filter]

            """ Calculate 'U[i]' gradient """
            U_grad[i] = np.sum(activation_pos - activation_neg, axis = 1).reshape(self.slice_size, 1)

            """ Calculate U * f'(z) terms useful for gradient calculation """
            temp_pos_all = U[i] * self.activationDifferential(activation_pos)
            temp_neg_all = - U[i] * self.activationDifferential(activation_neg)

            """ Calculate 'b[i]' gradient """
            b_grad[i] = np.sum(temp_pos_all + temp_neg_all, axis = 1).reshape(1, self.slice_size)

            """ Variables required for sparse matrix calculation """
            values = np.ones(num_wrong)
            rows   = np.arange(num_wrong + 1)

            """ Calculate sparse matrices useful for gradient calculation """
            e1_sparse     = sp.csr_matrix((values, e1, rows), shape = (num_wrong, self.num_entities))
            e2_sparse     = sp.csr_matrix((values, e2, rows), shape = (num_wrong, self.num_entities))
            e1_neg_sparse = sp.csr_matrix((values, e1_neg, rows), shape = (num_wrong, self.num_entities))
            e2_neg_sparse = sp.csr_matrix((values, e2_neg, rows), shape = (num_wrong, self.num_entities))

            for k in range(self.slice_size):
                """ U * f'(z) values corresponding to one slice """
                temp_pos = temp_pos_all[k, :].reshape(1, num_wrong)
                temp_neg = temp_neg_all[k, :].reshape(1, num_wrong)

                """ Calculate 'k'th slice of 'W[i]' gradient """
                W_grad[i][:, :, k] = np.dot(entity_vectors_e1_rel * temp_pos, entity_vectors_e2_rel.T) \
                    + np.dot(entity_vectors_e1_rel_neg * temp_neg, entity_vectors_e2_rel_neg.T)

                """ Calculate 'k'th slice of 'V[i]' gradient """
                V_grad[i][:, k] = np.sum(np.vstack((entity_vectors_e1_rel, entity_vectors_e2_rel)) * temp_pos
                    + np.vstack((entity_vectors_e1_rel_neg, entity_vectors_e2_rel_neg)) * temp_neg, axis = 1)

                """ Add contribution of 'V[i]' term in the entity vectors' gradient """
                V_pos = V[i][:, k].reshape(2*self.embedding_size, 1) * temp_pos
                V_neg = V[i][:, k].reshape(2*self.embedding_size, 1) * temp_neg

                entity_vector_grad += V_pos[:self.embedding_size, :] * e1_sparse + V_pos[self.embedding_size:, :] * e2_sparse \
                    + V_neg[:self.embedding_size, :] * e1_neg_sparse + V_neg[self.embedding_size:, :] * e2_neg_sparse

                ########################################       
                W_slice = np.array(W[i][:, :, k], order="F")
                entity_vector_grad += self.addWeightToGradient(i, k, W_slice, entity_vectors, e1, e2, e1_neg, e2_neg, temp_pos, temp_neg, e1_sparse, e2_sparse, e1_neg_sparse, e2_neg_sparse)

            """ Normalize the gradients with the training batch size """
            W_grad[i] /= self.batch_size
            V_grad[i] /= self.batch_size
            b_grad[i] /= self.batch_size
            U_grad[i] /= self.batch_size

        """ Initialize word vector gradients as a matrix of zeros """
        word_vector_grad = np.zeros(word_vectors.shape)

        """ Calculate word vector gradients from entity gradients """
        for entity in range(self.num_entities):
            entity_len = len(self.word_indices[entity])
            word_vector_grad[:, self.word_indices[entity]] += \
                np.tile(entity_vector_grad[:, entity].reshape(self.embedding_size, 1) / entity_len, (1, entity_len))

        """ Normalize word vector gradients and cost by the training batch size """
        word_vector_grad /= self.batch_size
        cost             /= self.batch_size

        """ Get unrolled gradient vector """
        theta_grad, d_t = self.stackToParams(W_grad, V_grad, b_grad, U_grad, word_vector_grad)

        """ Add regularization term to the cost and gradient """
        cost       += 0.5 * self.lamda * np.sum(theta * theta)
        theta_grad += self.lamda * theta

        return cost, theta_grad

    #######################################################################################
    """ Calculates the best thresholds for classification """
    def computeBestThresholds(self, dev_data, dev_labels):

        """ Get stack of network parameters """
        W, V, b, U, word_vectors = self.paramsToStack(self.theta)

        """ Initialize entity vectors as matrix of zeros """
        entity_vectors = np.zeros((self.embedding_size, self.num_entities))

        """ Assign entity vectors to be the mean of word vectors involved """
        for entity in range(self.num_entities):
            entity_vectors[:, entity] = np.mean(word_vectors[:, self.word_indices[entity]], axis = 1)

        """ Initialize prediction scores as matrix of zeros """
        dev_scores = np.zeros(dev_labels.shape)

        for i in range(dev_data.shape[0]):
            """ Extract required information from 'dev_data' """
            rel = dev_data[i, 1]
            entity_vector_e1  = entity_vectors[:, dev_data[i, 0].astype(int)].reshape(self.embedding_size, 1)
            entity_vector_e2  = entity_vectors[:, dev_data[i, 2].astype(int)].reshape(self.embedding_size, 1)

            """ Stack the entity vectors one over the other """
            entity_stack = np.vstack((entity_vector_e1, entity_vector_e2))

            """ Calculate the prediction score for the 'i'th example """
            for k in range(self.slice_size):

                dev_scores[i, 0] += U[rel][k, 0] * \
                   (np.dot(entity_vector_e1.T, np.dot(W[rel][:, :, k], entity_vector_e2)) +
                    np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k])

        """ Minimum and maximum of the prediction scores """
        score_min = np.min(dev_scores)
        score_max = np.max(dev_scores)

        """ Initialize thresholds and accuracies """
        best_thresholds = np.empty((self.num_relations, 1))
        best_accuracies = np.empty((self.num_relations, 1))

        for i in range(self.num_relations):
            best_thresholds[i, :] = score_min
            best_accuracies[i, :] = -1

        score_temp = score_min
        interval   = 0.01

        """ Check for the best accuracy at intervals between 'score_min' and 'score_max' """
        while(score_temp <= score_max):
            for i in range(self.num_relations):
                """ Check accuracy for 'i'th relation at 'score_temp' """
                rel_i_list    = (dev_data[:, 1] == i)
                predictions   = (dev_scores[rel_i_list, 0] <= score_temp) * 2 - 1
                temp_accuracy = np.mean((predictions == dev_labels[rel_i_list, 0]))

                """ If the accuracy is better, update the threshold and accuracy values """
                if(temp_accuracy > best_accuracies[i, 0]):
                    best_accuracies[i, 0] = temp_accuracy
                    best_thresholds[i, 0] = score_temp

            score_temp += interval
        best_thresholds = self.activationFunction(best_thresholds)

        """ Store the threshold values to be used later """
        self.best_thresholds = best_thresholds


    #######################################################################################
    """ Returns predictions for the passed test data """
    def getPredictions(self, test_data):

        """ Get stack of network parameters """
        W, V, b, U, word_vectors = self.paramsToStack(self.theta)

        """ Initialize entity vectors as matrix of zeros """
        entity_vectors = np.zeros((self.embedding_size, self.num_entities))

        """ Assign entity vectors to be the mean of word vectors involved """
        for entity in range(self.num_entities):
            entity_vectors[:, entity] = np.mean(word_vectors[:, self.word_indices[entity]], axis = 1)

        """ Initialize predictions as an empty array """
        predictions = np.empty((test_data.shape[0], 1))

        """ Initialize array for indices in test set to inspect in Grakn """
        indices = []

        for i in range(test_data.shape[0]):
            """ Extract required information from 'test_data' """
            rel = test_data[i, 1]
            entity_vector_e1  = entity_vectors[:, test_data[i, 0].astype(int)].reshape(self.embedding_size, 1)
            entity_vector_e2  = entity_vectors[:, test_data[i, 2].astype(int)].reshape(self.embedding_size, 1)

            """ Stack the entity vectors one over the other """
            entity_stack = np.vstack((entity_vector_e1, entity_vector_e2))
            test_score   = 0

            """ Calculate the prediction score for the 'i'th example """
            for k in range(self.slice_size):
                activated_matrix = self.activationFunction((np.dot(entity_vector_e1.T, np.dot(W[rel][:, :, k], entity_vector_e2)) +
                    np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k]))
                test_score += U[rel][k, 0] * activated_matrix         

            """ Give predictions based on previously calculate thresholds """
            if(test_score <= self.best_thresholds[rel.astype(int), 0]):
                #print("True pred: Line " + str(i) + " of test data, with score of " + str(test_score))
                predictions[i, 0] = 1
            else:
                #print("False pred: Line " + str(i) + " of test data, with score of " + str(test_score))
                indices.append(i)
                predictions[i, 0] = -1

        return (predictions, indices)

    #######################################################################################
    """ Loops through and finds a set of likely entity-relation-entity triplets, writing them to file to be added later """
    def findLikelyEntities(self, entity_dictionary, relation_dictionary):
        print "---> Checking for likeliest entities"
        """ Get stack of network parameters """
        W, V, b, U, word_vectors = self.paramsToStack(self.theta)

        """ Initialize entity vectors as matrix of zeros """
        entity_vectors = np.zeros((self.embedding_size, self.num_entities))

        """ Set parameter for how many of the 30000 entities will be paired with each e1 with each iteration"""
        size_of_sample = 100

        """ Assign the key/value pairs of the parameter dicts to a variable in working memory """
        entity_tuples = entity_dictionary.items()
        relation_tuples = relation_dictionary.items()

        """ Assign entity vectors to be the mean of word vectors involved """
        for entity in range(self.num_entities):
            entity_vectors[:, entity] = np.mean(word_vectors[:, self.word_indices[entity]], axis = 1)

        """ Initialize string for insertion of new relations that we will write later """
        new_relations_string = ""

        before_entity_loop = datetime.now()
        """ Now that entity vectors are initialized, we loop through all entities again and calculate scores """
        for entity in range(self.num_entities):

            """ Creates a random sample of e2 entities to check against each e1 entity """
            entity_sample = random.sample(xrange(self.num_entities), size_of_sample)
            
            e1_tuple = entity_tuples[entity]    
            entity_vector_e1 = entity_vectors[:, e1_tuple[1]].reshape(self.embedding_size, 1)

            """ As a quirk of this implementation, the best triplets have the lowest scores """
            best_entity_score = sys.maxint
            best_entity_triplet = (' ', ' ', ' ')

            for rel in range(self.num_relations):
                """ Extracts the relation 'rel' and saves its embedding """
                relation_tuple = relation_tuples[rel]
                rel_embedding = relation_tuple[1]


                for index in entity_sample:
                    e2_tuple = entity_tuples[index]
                    entity_vector_e2 = entity_vectors[:, e2_tuple[1]].reshape(self.embedding_size, 1)

                    """ Stack the entity vectors one over the other """
                    entity_stack = np.vstack((entity_vector_e1, entity_vector_e2))
                    test_score   = 0

                    """ Calculate the prediction score for the generated (e1, R, e2) triplet """
                    for k in range(self.slice_size):
                        activated_matrix = self.activationFunction((np.dot(entity_vector_e1.T, np.dot(W[rel_embedding][:, :, k], entity_vector_e2)) +
                            np.dot(V[rel_embedding][:, k].T, entity_stack) + b[rel_embedding][0, k]))
                        test_score += U[rel][k, 0] * activated_matrix

                    """ Update most likely """
                    if (test_score < best_entity_score):
                        best_entity_score = test_score
                        best_entity_triplet = (e1_tuple[0], relation_tuple[0], e1_tuple[0])

            new_relations_string += '\t'.join(best_entity_triplet) + '\n'

            """ Add potential reciprocal relation """
            recip_relation = graql.getReciprocal(best_entity_triplet[1])
            if recip_relation:
                best_entity_recip = (best_entity_triplet[2], recip_relation, best_entity_triplet[0])
                new_relations_string += '\t'.join(best_entity_recip) + '\n'


        after_entity_loop = datetime.now()

        """ Sanity check """
        print "     Finding likeliest entities took " + str((after_entity_loop-before_entity_loop).total_seconds())

        """ Write to file """
        f = open(_NEW_RELATIONS_PATH, 'w')
        f.write(new_relations_string)
        f.close()
                        

###########################################################################################
""" Read and construct test data, using entity and relation dictionaries """
def getTestData(file_name, entity_dictionary, relation_dictionary):

    """ Read and split data linewise """
    file_object = open(file_name, 'r')
    data        = file_object.read().splitlines()

    """ Initialize test data and labels as empty matrices """
    num_entries = len(data)
    test_data   = np.empty((num_entries, 3))
    labels      = np.empty((num_entries, 1))

    index = 0

    for line in data:
        """ Obtain relation example text by splitting line """
        entity1, relation, entity2, label = line.split()

        """ Assign indices to the obtained entities and relation """
        test_data[index, 0] = entity_dictionary[entity1]
        test_data[index, 1] = relation_dictionary[relation]
        test_data[index, 2] = entity_dictionary[entity2]

        """ Label value for the example """
        if label == '1':
            labels[index, 0] = 1
        else:
            labels[index, 0] = -1

        index += 1
    return test_data, labels

###########################################################################################
""" Get indices of words in the entities """
def getWordIndices(file_name):

    """ Load the pickled data file """
    word_dictionary = pickle.load(open(file_name, 'rb'))

    """ Extract the number of words and the word indices from the dictionary """
    num_words    = word_dictionary['num_words']
    word_indices = word_dictionary['word_indices']

    return word_indices, num_words

###########################################################################################
""" Read and construct training data, using entity and relation dictionaries """
def getTrainingData(file_name, entity_dictionary, relation_dictionary):

    """ Read and split data linewise """
    file_object = open(file_name, 'r')
    data        = file_object.read().splitlines()

    """ Initialize training data as an empty matrix """
    num_examples  = len(data)
    training_data = np.empty((num_examples, 3))

    index = 0

    for line in data:
        """ Obtain relation example text by splitting line """
        entity1, relation, entity2 = line.split()

        """ Assign indices to the obtained entities and relation """
        training_data[index, 0] = entity_dictionary[entity1]
        training_data[index, 1] = relation_dictionary[relation]
        training_data[index, 2] = entity_dictionary[entity2]

        index += 1

    return training_data, num_examples

###########################################################################################
""" Create a numerical mapping of entity/relation data elements """
def getDictionary(file_name):

    """ Read and split data linewise """
    file_object = open(file_name, 'r')
    data = file_object.read().splitlines()

    """ Initialize dictionary to store the mapping """
    dictionary = {}
    index = 0

    for entity in data:
        """ Assign unique index to every entity """
        dictionary[entity] = index
        index += 1

    """ Number of entries in the data file """
    num_entries = index

    return dictionary, num_entries

###########################################################################################
""" Defines and returns the program parameters """
def getProgramParameters():

    """ Initialize dictionary of program parameters """
    program_parameters = {}

    """ Set program parameters """
    program_parameters['embedding_size']      = 100    # size of a single word vector
    program_parameters['slice_size']          = 3      # number of slices in tensor
    program_parameters['num_iterations']      = 200    # number of optimization iterations
    program_parameters['batch_size']          = 20000  # training batch size
    program_parameters['corrupt_size']        = 10     # corruption size
    program_parameters['activation_function'] = 0      # 0 - tanh, 1 - sigmoid
    program_parameters['lamda']               = 0.0001 # regularization parameter
    program_parameters['batch_iterations']    = 5      # optimization iterations for each batch
    program_parameters['num_graql_updates']   = 7     # number of iterations of graql insertions

    return program_parameters

###########################################################################################
""" Creates ontology and rule set, and inserts the dataset into the graph """
def createGraqlGraph(entity_file, relation_file, graql_keyspace):

    print color.GREEN + "-------------------- BUILDING  GRAPH --------------------" + color.END
    print "---> Inserting schema"
    graql.insertOntology(graql_keyspace)

    print "---> Inserting entities"
    graql.insertEntities(graql_keyspace, entity_file)

    print "---> Inserting relations"
    graql.insertRelations(graql_keyspace, relation_file)

    print color.GREEN + "---------------------- GRAPH BUILT ----------------------" + color.END


###########################################################################################
""" Trains the network, and calculates accuracy on test dataset """
def neuralTensorNetwork(graql_keyspace, build_graph):
    """ Add entities and relations into a Graql graph with the keyspace parameter, if provided """
    if build_graph:
        createGraqlGraph('entities.txt', 'train.txt', graql_keyspace)

    """ Get the program parameters """
    program_parameters = getProgramParameters()

    """ Extract information useful within the scope of this function """
    num_iterations   = program_parameters['num_iterations']
    batch_size       = program_parameters['batch_size']
    corrupt_size     = program_parameters['corrupt_size']
    batch_iterations = program_parameters['batch_iterations']
    num_graql_updates = program_parameters['num_graql_updates']

    """ Get entity and relation data dictionaries """
    entity_dictionary, num_entities    = getDictionary('entities.txt')
    relation_dictionary, num_relations = getDictionary('relations.txt')

    """ Get training data using entity and relation dictionaries """
    training_data, num_examples = getTrainingData('train.txt', entity_dictionary, relation_dictionary)

    """ Get word indices for all the entities in the data """
    word_indices, num_words = getWordIndices('wordIndices.p')

    """ Store newly learned data in the dictionary """
    program_parameters['num_entities']  = num_entities
    program_parameters['num_relations'] = num_relations
    program_parameters['num_examples']  = num_examples
    program_parameters['num_words']     = num_words
    program_parameters['word_indices']  = word_indices

    """ Create a NeuralTensorNetwork object """
    network = NeuralTensorNetwork(program_parameters)

    print "---> Training " + str(num_iterations) + " epochs"

    for i in range(num_iterations):
        """ Create a training batch by picking up random samples from training data """
        batch_indices = np.random.randint(num_examples, size = batch_size)
        data          = {}
        data['rel']   = np.tile(training_data[batch_indices, 1], (1, corrupt_size)).T
        data['e1']    = np.tile(training_data[batch_indices, 0], (1, corrupt_size)).T
        data['e2']    = np.tile(training_data[batch_indices, 2], (1, corrupt_size)).T
        data['e3']    = np.random.randint(num_entities, size = (batch_size * corrupt_size, 1))

        """ Optimize the network using the training batch """
        clock_start = datetime.now()
        if np.random.random() < 0.5:
            opt_solution = scipy.optimize.minimize(network.neuralTensorNetworkCost, network.theta,
                args = (data, 0,), method = 'L-BFGS-B', jac = True, options = {'maxiter': batch_iterations})
        else:
            opt_solution = scipy.optimize.minimize(network.neuralTensorNetworkCost, network.theta,
                args = (data, 1,), method = 'L-BFGS-B', jac = True, options = {'maxiter': batch_iterations})
        clock_end = datetime.now()
        diff = clock_end - clock_start
        if i%10 == 9:
            print("Network has been optimized with " + str(i+1) + " iterations complete; most recent iteration took " 
                + str(diff.total_seconds()) + " seconds")

        """ Store the optimized theta value """
        network.theta = opt_solution.x

    print "\n"

    """ Get test data to calculate predictions """
    dev_data, dev_labels   = getTestData('dev.txt', entity_dictionary, relation_dictionary)
    test_data, test_labels = getTestData('test.txt', entity_dictionary, relation_dictionary)

    """ Compute the best thresholds for classification, and get predictions on test data """
    network.computeBestThresholds(dev_data, dev_labels)

    print "---> Calculating predictions..."
    """ Calculate predictions, and return set of predictions alongside list of indices to inspect with Grakn """
    (predictions, close_test_indices) = network.getPredictions(test_data)

    """ Print accuracy of the obtained predictions """
    print color.BOLD + "------> Initial neural network accuracy: " + str(np.mean((predictions == test_labels))) + color.END

    incorrect_preds = [x for i,x in enumerate(predictions) if predictions[i] != test_labels[i]]
    print "        Only " + str(len([y for y in incorrect_preds if y == -1])) + " false negatives out of " + str(np.sum((predictions == -1))) + " total negatives "

    """ Now begin the Graql loop """
    for r in range(num_graql_updates):
        """ len(graql_predictions) = len(close_test_indices) """
        graql_predictions = graql.checkGraqlGraph(graql_keyspace, close_test_indices)

        """ Return indices that graql found true """
        graql_predictions_true = [i for i,x in enumerate(graql_predictions) if x == 1]
        indices_predicted_true = [close_test_indices[j] for j in graql_predictions_true]
        indices_actual_value = test_labels[indices_predicted_true]

        """ Calculate the accuracy of the inferences made by Grakn """
        num_correct_graql = np.sum((indices_actual_value == 1))
        num_wrong_graql = np.sum((indices_actual_value == -1))
        num_total_graql = len(indices_predicted_true)
        print "---> " + str(num_correct_graql) + " correct inferences found; " + str(num_wrong_graql) + " incorrect inferences found"

        """ Mask the NN predictions/labels arrays to not include inferences made by Grakn"""
        masked_predictions = np.ma.array(predictions, mask=False)
        masked_labels = np.ma.array(test_labels, mask=False)
        masked_predictions.mask[indices_predicted_true] = True
        masked_labels.mask[indices_predicted_true] = True

        """ Calculate accuracy on these masked arrays """
        num_correct_nn = np.sum((masked_predictions == masked_labels)) 
        num_total_nn = masked_predictions.count()

        """ Calculate total Grakn-improved accuracy """
        improved_accuracy = (num_correct_nn + num_correct_graql) / float(num_total_nn + num_total_graql)
        print color.BOLD + "------> Improved accuracy: " + str(improved_accuracy) + color.END

        """ If we are at the last iteration of the for loop, there's no need to find likely triplets again """
        if r < num_graql_updates - 1:
            network.findLikelyEntities(entity_dictionary, relation_dictionary)
            graql.insertRelations(graql_keyspace, _NEW_RELATIONS_PATH)

        print color.GREEN + "---------------------------------------------------------" + color.END

###########################################################################################
""" Main function """
if __name__=="__main__":
    parser = ArgumentParser(
        description="neuralTensorNetwork -k KEYSPACE [--buildGraph]")
    parser.add_argument('-k', '--keyspace', help="Graql keyspace to use", required=True)
    parser.add_argument('--buildGraph', dest='buildGraph', help="Optional flag to build the ontology and ruleset if it has not already \
        been built", action='store_true')
    parser.set_defaults(buildGraph = False)
    sysargs = parser.parse_args()

    neuralTensorNetwork(sysargs.keyspace, sysargs.buildGraph)
