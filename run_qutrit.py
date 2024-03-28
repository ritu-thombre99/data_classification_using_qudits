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
scheme_config['custom_ternary'] = [1,4]
scheme_config['custom_binary'] = [6,3]

# default scheme: B
config = {}
config['dataset'] = 'moon'
config['encoding_and_rotation_scheme'] = 'B'
config['s_params_size'] = 4
config['w_params_size'] = 4
config["binary_classifier"] = True

dev = qml.device("default.qutrit", wires=1)
@qml.qnode(dev) 
def vqc_model(x_i, params): # change return value qml.probs
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
    elif scheme == 'custom_ternary':
        custom_ternary(x_i,s_params,w_params)
    elif scheme == 'custom_binary':
        custom_binary(x_i,s_params,w_params)
    if config["binary_classifier"]  == True:
        obs = qml.GellMann(0,3)+np.sqrt(3)*qml.GellMann(0,8)
        return qml.expval(obs)
    else:
        return qml.probs()

@qml.qnode(dev)
def get_state(x_i, params,obs_name):
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
    elif scheme == 'custom_ternary':
        custom_ternary(x_i,s_params,w_params)
    elif scheme == 'custom_binary':
        custom_binary(x_i,s_params,w_params)
    obs = qml.GellMann(0,3)+np.sqrt(3)*qml.GellMann(0,8)
    if obs_name == 'L1':
        obs = qml.GellMann(0,1)+qml.GellMann(0,6)
    if obs_name == 'L2':
        obs = qml.GellMann(0,2)+qml.GellMann(0,7)
    return qml.expval(obs)



def loss(data, labels, model, params):    
    loss_sum = []
    for idx in range(len(data)):
        data_point = data[idx]
        true_label = labels[idx]
        model_output = model(data_point, params)
        if config["binary_classifier"]  == True:
            if (model_output<0 and true_label>0) or (model_output>0 and true_label<0):
                loss_sum.append((model_output - true_label) ** 2)
        else:
            loss = 0
            if true_label == -2:
                loss = -1*np.log(model_output[0])
            elif true_label == 0:
                loss = -1*np.log(model_output[1])
            elif true_label == 2:
                loss = -1*np.log(model_output[2])
            loss_sum.append(loss)
            # if -2 <= model_output and model_output < (-2/3):
            #     model_class = -2
            # elif (2/3) <= model_output and model_output <= 2:
            #     model_class = 2

            # if model_class != true_label:
            #     loss_sum.append((model_output - true_label) ** 2)
            

    return sum(loss_sum)/len(data)

def make_prediction(model, data_point, params):
    model_output = model(data_point, params)
    if config["binary_classifier"]  == True:
        if model_output < 0:
            return -1
        return 1
    else:
        pred = np.argmax(model_output)
        if pred == 0:
            return -2
        if pred == 1:
            return 0
        if pred == 2:
            return 2
        # if -2 <= model_output and model_output < (-2/3):
        #     return -2
        # elif (2/3) <= model_output and model_output <= 2:
        #     return 2
        # return 0

def compute_accuracy(data, labels, model, params):
    n_samples = len(data)
    # for x in range(n_samples):
    #     print(make_prediction(model, data[x], params),labels[x])
    return np.sum(
        [make_prediction(model, data[x], params) == labels[x] for x in range(n_samples)
    ]) / n_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding_and_rotation",type=str,
                        choices=['A','B','C','D1','D2','D3','E','custom_binary','custom_ternary'],help="choose the encoding and rotation scheme, default B")
    parser.add_argument("--dataset",type=str,
                        choices=['xor','noisy_xor','moon','wine','iris'],help="choose the dataset, default moon")
    parser.add_argument("--num_itr",type=int,help="Enter number of iterations for training")
    args = parser.parse_args()
    
    if args.encoding_and_rotation:
        config['encoding_and_rotation_scheme'] = args.encoding_and_rotation
    if args.dataset:
        config['dataset'] = args.dataset
    
    if config['encoding_and_rotation_scheme'] not in ['A','B','C','D1','D2','D3','custom_binary','custom_ternary']:
        print("Invalid Encoding and Rotation scheme. Allowed Schemes: A,B,C,D1,D2,D3,custom_binary,custom_ternary")
        return
    if config['dataset'] not in ['xor','noisy_xor','moon','wine','iris']:
        print("Invalid dataset. Choose from [XOR,Noisy XOR,Moon,Iris,Wine]")
        return
    
    if config['encoding_and_rotation_scheme'] != 'B':
        config['s_params_size'], config['w_params_size'] = scheme_config[config['encoding_and_rotation_scheme']]
        
    train_X, test_X, train_y, test_y = None, None, None, None
    dataset_size = 300
    if config['dataset'] == 'xor':
        train_X, test_X, train_y, test_y = get_xor_data(dataset_size)
    elif config['dataset'] == 'moon':
        train_X, test_X, train_y, test_y = get_moon_dataset(dataset_size)
    elif config['dataset'] == 'noisy_xor':
        config["binary_classifier"] = False
        train_X, test_X, train_y, test_y = get_three_class_xor_data(dataset_size)
        plot_2d_data(train_X,train_y)
        plt.show()
    elif config['dataset'] == 'iris': 
        config["binary_classifier"] = False
        train_X, test_X, train_y, test_y = iris_dataset()
    elif config['dataset'] == 'wine': 
        config["binary_classifier"] = False
        train_X, test_X, train_y, test_y = get_wine_dataset() 
    
    if config['encoding_and_rotation_scheme'] in ['A','B','C','custom_binary'] and config['dataset'] in ['iris','wine']:
        print("Wine and Iris cannot be used witj encoding schemes A,B,C,custom_binary")
        return
    if config['encoding_and_rotation_scheme'] in ['D1','D2','D3','custom_ternary'] and config['dataset'] not in ['iris','wine']:
        print("XOR and Moon cannot be used with encoding schemes D1,D2,D3,custom_ternary")
        return
    if config['encoding_and_rotation_scheme'] in ['D1','D2','D3'] and config['dataset'] == 'iris':
        print("Iris (4D) does not work with D1,D2,D3")
        return
    # preprocessing wine
    if config['dataset'] == 'wine':
        if config['encoding_and_rotation_scheme'] in ['D1','D2','D3']:
            pca = PCA(8)
            train_X = pca.fit_transform(train_X)
            test_X = pca.fit_transform(test_X)
        elif config['encoding_and_rotation_scheme'] == 'custom_ternary':
            pca = PCA(4)
            train_X = pca.fit_transform(train_X)
            test_X = pca.fit_transform(test_X)


    # opt = qml.AdamOptimizer(stepsize=0.00087)
    opt = qml.GradientDescentOptimizer(stepsize=0.009)
    # opt = qml.GradientDescentOptimizer(stepsize=0.03)
    num_its = 220
    if type(args.num_itr) == str and args.num_itr.isdigit() == True:
        args.num_itr = int(args.num_itr)
    if type(args.num_itr) == int:
        num_its = args.num_itr
    else:
        print("Number of itreations should be integer, using default num_itr=220")

    s_params_size, w_params_size = config['s_params_size'], config['w_params_size']
    params = np.random.normal(size=(s_params_size+w_params_size))#*100
    
    
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
    title = "Loss for "+config['dataset']+" using single qutrit with scheme "+config['encoding_and_rotation_scheme']
    plt.title(title)
    plt.grid()
    plt.show()
    plt.savefig("./Figs/qutrit/"+title+".png",bbox_inches='tight')
    plt.close()
    
    op_state = []
    for i in range(len(train_X)):
        x = (get_state(train_X[i],params,'L1'))
        y = (get_state(train_X[i],params,'L2'))
        z = (get_state(train_X[i],params,'L3'))
        op_state.append([y,z])
    op_state = np.array(op_state)
    plot_classified_data_on_bloch(op_state,train_y)
    plt.grid()
    title = "SU(2) bloch for "+config['dataset']+" with "+config['encoding_and_rotation_scheme']+" Itr:"+str(num_its)
    plt.title(title)
    plt.show()
    plt.savefig("./Figs/qutrit/"+title+".png",bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    main()
