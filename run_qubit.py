import argparse

import warnings
warnings.filterwarnings("ignore")

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from qubit_models import *
from helpers import *
from datasets import *

# store the s_params and w_params required for any scheme
scheme_config = {}
scheme_config['A'] = [1,1]
scheme_config['B'] = [1,3]
scheme_config['C'] = [0,3]
scheme_config['D'] = [2,3]
scheme_config['E'] = [2,2]
scheme_config['F'] = [3,3]
scheme_config['G'] = [1,2]

# default scheme: B
config = {}
config['dataset'] = 'circular'
config['encoding_and_rotation_scheme'] = 'B'
config['s_params_size'] = 1
config['w_params_size'] = 3

f = open("./logs/qubit_run.txt","a")


dev = qml.device("default.qubit", wires=1)
@qml.qnode(dev)
def vqc_model(x_i, params):
    s_params,w_params = params[:config['s_params_size']], params[config['s_params_size']:]
    scheme = config['encoding_and_rotation_scheme']
    if scheme == 'A':
        scheme_a(x_i,s_params,w_params)
    elif scheme == 'B':
        scheme_b(x_i,s_params,w_params)
    elif scheme == 'C':
        scheme_c(x_i,w_params)
    elif scheme == 'D':
        scheme_d(x_i,s_params,w_params)
    elif scheme == 'E':
        scheme_e(x_i,s_params,w_params)
    elif scheme == 'F':
        scheme_f(x_i,s_params,w_params)
    elif scheme == 'G':
        scheme_g(x_i,s_params,w_params)
        print("Yet to implement the scheme G with 3d data")
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def get_state(x_i,params):
    s_params,w_params = params[:config['s_params_size']], params[config['s_params_size']:]
    scheme = config['encoding_and_rotation_scheme']
    if scheme == 'A':
        scheme_a(x_i,s_params,w_params)
    elif scheme == 'B':
        scheme_b(x_i,s_params,w_params)
    elif scheme == 'C':
        scheme_c(x_i,w_params)
    elif scheme == 'D':
        scheme_d(x_i,s_params,w_params)
    elif scheme == 'E':
        scheme_e(x_i,s_params,w_params)
    elif scheme == 'F':
        scheme_f(x_i,s_params,w_params)
    elif scheme == 'G':
        scheme_g(x_i,s_params,w_params)
        print("Yet to implement the scheme G with 3d data")
    return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]


def loss(data, labels, model, params):    
    loss_sum = []
    for idx in range(len(data)):
        data_point = data[idx]
        true_label = labels[idx]
        model_output = model(data_point, params)

        if (model_output<0 and true_label>0) or (model_output>0 and true_label<0):
            loss_sum.append((model_output - true_label) ** 2)

    return sum(loss_sum)/len(data)


def make_prediction(model, data_point, params):
    measurement_result = model(data_point, params)
    if measurement_result < 0:
        return -1
    return 1


def compute_accuracy(data, labels, model, params):
    n_samples = len(data)
    return np.sum(
        [make_prediction(model, data[x], params) == labels[x] for x in range(n_samples)
    ]) / n_samples


def run(dataset='circular', encoding_and_rotation_scheme='B',dataset_size=200,num_its=220,train_X=None,test_X=None, train_y=None, test_y=None,f=None,is_default=True):

    if f is None:
        f = open("./logs/qubit_run.txt","a")
    print("Dataset:",dataset)
    print("Size:",dataset_size)
    print("Scheme:",encoding_and_rotation_scheme)
    print("Iters:",num_its)    

    config = {}
    config['dataset'] = dataset
    config['encoding_and_rotation_scheme'] = encoding_and_rotation_scheme
    config['s_params_size'], config['w_params_size'] = scheme_config[encoding_and_rotation_scheme]

    if train_X is None:
        if dataset == 'xor':
            train_X, test_X, train_y, test_y = get_xor_data(dataset_size)
        elif dataset == 'circular':
            train_X, test_X, train_y, test_y = get_circular_boundary_dataset(dataset_size)
        elif dataset == 'moon':
            train_X, test_X, train_y, test_y = get_moon_dataset(dataset_size)
    else:
        is_default = False
        
    s_params_size, w_params_size = scheme_config[encoding_and_rotation_scheme]
    params = np.random.normal(size=(s_params_size+w_params_size))

    f.writelines("Dataset: "+dataset+"\n")
    f.writelines("Dataset size: "+str(dataset_size)+"\n")
    f.writelines("Encoding scheme: "+str(encoding_and_rotation_scheme)+"\n")


    print("Initial parameters:",params)
    f.writelines("Initial parameters: "+str(params)+"\n")
    f.writelines("Number of iterations: "+str(num_its)+"\n")

    opt = qml.GradientDescentOptimizer(stepsize=0.009)
    loss_over_time = []
    for itr in range(num_its):
        (_, _, _, params), _loss = opt.step_and_cost(loss, train_X, train_y, vqc_model, params)
        loss_over_time.append(_loss)
        if (itr+1)%20 == 0:
            print("Iteration:",itr+1,"/",num_its,"Loss:",_loss)
    
    print("Final params:",params)
    f.writelines(str(loss_over_time)+"\n")
    f.writelines("Final params:"+str(params)+"\n")
    
    training_accuracy = compute_accuracy(train_X, train_y, vqc_model, params)
    testing_accuracy = compute_accuracy(test_X, test_y, vqc_model, params)

    print(f"Training accuracy = {training_accuracy}")
    f.writelines("Training accuracy:"+str(training_accuracy)+"\n")

    print(f"Testing accuracy = {testing_accuracy}")
    f.writelines("Testing accuracy:"+str(testing_accuracy)+"\n")
    f.writelines("------------------------------------\n")
    
    if is_default == True:
        f.close()