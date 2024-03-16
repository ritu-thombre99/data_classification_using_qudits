import argparse

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from qutrit_models import *
from helpers import *
from datasets import *
from helpers import *
from sklearn.decomposition import PCA


# store the s_params and w_params required for any scheme
scheme_config = {}
scheme_config['A'] = [2,2]
scheme_config['B'] = [4,4]
scheme_config['C'] = [4,4]
scheme_config['D1'] = [1,7]
scheme_config['D2'] = [4,4]
scheme_config['D3'] = [8,1]

# default scheme: B
config = {}
config['dataset'] = 'circular'
config['encoding_and_rotation_scheme'] = 'B'
config['s_params_size'] = 4
config['w_params_size'] = 4
config["binary_classifier"] = True

dev = qml.device("default.qutrit", wires=1)
@qml.qnode(dev)
def vqc_model(x_i, params):
    s_params,w_params = params[:config['s_params_size']], params[config['s_params_size']:]
    scheme = config['encoding_and_rotation_scheme']
    if scheme == 'A':
        scheme_a(x_i,s_params,w_params)
    elif scheme == 'B':
        scheme_b(x_i,s_params,w_params)
    elif scheme == 'C':
        scheme_c(x_i,s_params,w_params)
    elif scheme == 'D1':
        scheme_d1(x_i,s_params,w_params)
    elif scheme == 'D2':
        scheme_d2(x_i,s_params,w_params)
    elif scheme == 'D3':
        scheme_d3(x_i,s_params,w_params)
    obs = qml.GellMann(0,3)+np.sqrt(3)*qml.GellMann(0,8)
    return qml.expval(obs)



def loss(data, labels, model, params):    
    loss_sum = []
    for idx in range(len(data)):
        data_point = data[idx]
        true_label = labels[idx]
        model_output = model(data_point, params)
        print(model_output,end=' ')
        if config["binary_classifier"]  == True:
            print("Bin class")
            if (model_output<0 and true_label>0) or (model_output>0 and true_label<0):
                loss_sum.append((model_output - true_label) ** 2)
        else:
            model_class = 0
            if -2 <= model_output and model_output < (-2/3):
                model_class = -2
            elif (2/3) <= model_output and model_output <= 2:
                model_class = 2

            if model_class != true_label:
                loss_sum.append((model_output - true_label) ** 2)

    return sum(loss_sum)/len(data)

def make_prediction(model, data_point, params):
    model_output = model(data_point, params)
    if config["binary_classifier"]  == True:
        if model_output < 0:
            return -1
        return 1
    else:
        if -2 <= model_output and model_output < (-2/3):
            return -2
        elif (2/3) <= model_output and model_output <= 2:
            return 2
        return 0

def compute_accuracy(data, labels, model, params):
    n_samples = len(data)
    for x in range(n_samples):
        print(make_prediction(model, data[x], params),labels[x])
    return np.sum(
        [make_prediction(model, data[x], params) == labels[x] for x in range(n_samples)
    ]) / n_samples

def main():
    binary_datasets = ['moon','xor','circular']
    ternary_datasets = ['wine','iris','xor']
    train_X, test_X, train_y, test_y = None, None, None, None
    binary_classifier = (input("Enter True for binary classifier, False for ternary classifier: "))
    if binary_classifier == "True":
        print("Enter the dataset from the following: [moon,circular,xor]")
        dataset = input("Deualt is circular: ")
        if dataset not in binary_datasets:
            print("Enter valid dataset")
            return 
        config['dataset'] =  dataset
        try:
            dataset_size = int(input("Enter the dataset size: "))
        except Exception as e:
            print(e)
            return
        encoding_and_rotation = input("Enter encoding and rotation scheme from [A,B,C]: ")
        if encoding_and_rotation not in ['A','B','C']:
            print("Enter valid scheme")
            return 
        config['encoding_and_rotation_scheme'] = encoding_and_rotation
        config['s_params_size'], config['w_params_size'] = scheme_config[config['encoding_and_rotation_scheme']]

        if config['dataset'] == 'xor':
            train_X, test_X, train_y, test_y = get_xor_data(dataset_size)
        elif config['dataset'] == 'circular':
            train_X, test_X, train_y, test_y = get_circular_boundary_dataset(dataset_size)
        elif config['dataset'] == 'moon':
            train_X, test_X, train_y, test_y = get_moon_dataset(dataset_size)
    elif binary_classifier == "False":
        config["binary_classifier"] = False
        dataset = input("Enter the dataset from the following: [wine,iris,xor]: ")
        if dataset not in ternary_datasets:
            print("Enter valid dataset")
            return 
        config['dataset'] = dataset 
        
        encoding_and_rotation = input("Enter encoding and rotation scheme from [A,B,C,D1,D2,D3] XOR wont works with D1,D2,D3: ")
        if encoding_and_rotation not in ['A','B','C','D1','D2','D3']:
            print("Enter valid scheme")
        config['encoding_and_rotation_scheme'] = encoding_and_rotation
        config['s_params_size'], config['w_params_size'] = scheme_config[config['encoding_and_rotation_scheme']]

        if config['dataset'] == 'wine':
            train_X, test_X, train_y, test_y = get_wine_dataset()
            if encoding_and_rotation in ['A','B','C']:
                pca = PCA(2)
                train_X = pca.fit_transform(train_X)
                test_X = pca.fit_transform(test_X)
            else:
                pca = PCA(8)
                train_X = pca.fit_transform(train_X)
                test_X = pca.fit_transform(test_X)

        elif config['dataset'] == 'iris':
            train_X, test_X, train_y, test_y = iris_dataset()
            if encoding_and_rotation in ['A','B','C']:
                pca = PCA(2)
                train_X = pca.fit_transform(train_X)
                test_X = pca.fit_transform(test_X)
                print("After PCA:",train_X.shape, test_X.shape)
            else:
                train_X = np.pad(train_X, ((0,0),(0,4)), 'constant')
                test_X = np.pad(test_X, ((0,0),(0,4)), 'constant')
                print("After reshape:",train_X.shape, test_X.shape)


        elif config['dataset'] == 'xor':
            if config['encoding_and_rotation_scheme'] in ['D1','D2','D3']:
                print("Cannot use D1,D2,D3 encoding scheme with XOR data, only A,B,C works")
                return 
            try:
                dataset_size = int(input("Enter the dataset size: "))
            except Exception as e:
                print(e)
                return
            train_X, test_X, train_y, test_y = get_three_class_xor_data(dataset_size) 
    else:
        print("Invalid input")
        return 

    
    # opt = qml.AdamOptimizer(stepsize=0.00087)
    opt = qml.GradientDescentOptimizer(stepsize=0.009)
    num_its = 220
    try:
        num_its = int(input("Enter number of iterations (Default:220): "))
    except Exception as e:
        print(e)
        return
    s_params_size, w_params_size = config['s_params_size'], config['w_params_size']
    params = np.random.normal(size=(s_params_size+w_params_size))#*100
    init_loss = loss(train_X, train_y, vqc_model, params)
    # while init_loss>500:
    #     print("Loss too big:",init_loss)
    #     print("Generating new params")
    #     params = np.random.normal(size=(s_params_size+w_params_size))#*100

    print("Initial parameters:",params)
    loss_over_time = []
    for itr in range(num_its):
        (_, _, _, params), _loss = opt.step_and_cost(loss, train_X, train_y, vqc_model, params)
        loss_over_time.append(_loss)
        print("Iteration:",itr+1,"/",num_its,"Loss:",_loss)
    
    training_accuracy = compute_accuracy(train_X, train_y, vqc_model, params)
    testing_accuracy = compute_accuracy(test_X, test_y, vqc_model, params)

    print(f"Training accuracy = {training_accuracy}")
    print(f"Testing accuracy = {testing_accuracy}")
    
    plt.plot(loss_over_time)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss for "+config['dataset']+" using single qubit with scheme "+config['encoding_and_rotation_scheme'])
    plt.show()
    
    # op_state = []
    # for i in range(len(train_X)):
    #     x,y,z = (get_state(train_X[i],params))
    #     op_state.append([y,z])
    # op_state = np.array(op_state)
    # plot_classified_data_on_bloch(op_state,train_y)
    # plt.show()
if __name__ == "__main__":
    main()
    