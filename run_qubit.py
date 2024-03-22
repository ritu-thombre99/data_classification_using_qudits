import argparse

import warnings
warnings.filterwarnings("ignore")

import pennylane as qml
# from pennylane import numpy as np

import jax, jaxopt
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

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


dev = qml.device("default.qubit.jax", wires=1)

@jax.jit
@qml.qnode(dev, interface='jax')
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

@jax.jit
@qml.qnode(dev, interface='jax')
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

@jax.jit
def loss(params, data, labels):    
    loss_sum = []
    for idx in range(len(data)):
        data_point = data[idx]
        true_label = labels[idx]
        model_output = vqc_model(data_point, params)

        # jax.lax.cond((model_output<0 and true_label>0) || (),  print_training, print_fn, lambda: None)
        # if (model_output<0 and true_label>0) or (model_output>0 and true_label<0):
        loss_sum.append((model_output - true_label) ** 2)
        # jax.lax.cond(true_label>0, lambda:jax.lax.cond(model_output<0,loss_sum.at[idx].set((model_output - true_label) ** 2), lambda:None), lambda: None)
    loss_sum = jnp.asarray(loss_sum)
    return jnp.sum(loss_sum)/len(data)

# djax loss
def loss_and_grad(params, data, labels, i, print_training=True):
    loss_val, grad_val = jax.value_and_grad(loss)(params, data, labels)

    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

    # if print_training=True, print the loss every 5 steps
    jax.lax.cond((jnp.mod(i, 5) == 0) & print_training, print_fn, lambda: None)

    return loss_val, grad_val


def make_prediction(model, data_point, params):
    measurement_result = model(data_point, params)
    if measurement_result < 0:
        return -1
    return 1


def compute_accuracy(data, labels, model, params):
    n_samples = len(data)
    return sum(
        [make_prediction(model, data[x], params) == labels[x] for x in range(n_samples)
    ]) / n_samples

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding_and_rotation",type=str,
                        choices=['A','B','C','D','E','F','G'],help="choose the encoding and rotation scheme, default B")
    parser.add_argument("--dataset",type=str,
                        choices=['xor','moon','circular'],help="choose the dataset, default circular boundary")
    parser.add_argument("--dataset_size",type=int,help="Enter the dataset size")
    parser.add_argument("--num_itr",type=int,help="Enter number of iterations for training")
    args = parser.parse_args()
    
    if args.encoding_and_rotation:
        config['encoding_and_rotation_scheme'] = args.encoding_and_rotation
    if args.dataset:
        config['dataset'] = args.dataset
    
    if config['encoding_and_rotation_scheme'] not in ['A','B','C','D','E','F','G']:
        print("Invalid Encoding and Rotation scheme. Allowed Schemes: A,B,C,D,E,F,G")
        return
    if config['dataset'] not in ['xor','moon','circular']:
        print("Invalid dataset. Choose from [Moon,XOR,Circular boundary]")
        return
    
    if config['encoding_and_rotation_scheme'] != 'B':
        config['s_params_size'], config['w_params_size'] = scheme_config[config['encoding_and_rotation_scheme']]
        
    train_X, test_X, train_y, test_y = None, None, None, None
    dataset_size = 200
    if type(args.dataset_size) == str and args.dataset_size.isdigit() == True:
        args.dataset_size = int(args.dataset_size)
    if type(args.dataset_size) == int:
        dataset_size = args.dataset_size
    if config['dataset'] == 'xor':
        train_X, test_X, train_y, test_y = get_xor_data(dataset_size)
    elif config['dataset'] == 'circular':
        train_X, test_X, train_y, test_y = get_circular_boundary_dataset(dataset_size)
    elif config['dataset'] == 'moon':
        train_X, test_X, train_y, test_y = get_moon_dataset(dataset_size)
        
    s_params_size, w_params_size = config['s_params_size'], config['w_params_size']
    # arams = jnp.random.normal(size=(s_params_size+w_params_size))#*100
    params = jnp.asarray(np.random.normal(size=(s_params_size+w_params_size)))#*100
    # key = jax.random.PRNGKey(758493)
    # params = jax.random.normal(key,shape=(s_params_size+w_params_size))#*100

    f.writelines("Dataset: "+config['dataset']+"\n")
    f.writelines("Dataset size: "+str(dataset_size)+"\n")
    f.writelines("Encoding scheme: "+str(config['encoding_and_rotation_scheme'])+"\n")


    print("Initial parameters:",params)
    f.writelines("Initial parameters: "+str(params)+"\n")
    # opt = qml.AdamOptimizer(stepsize=0.00087)
    # opt = qml.GradientDescentOptimizer(stepsize=0.009)
    num_its = 220
    if type(args.num_itr) == str and args.num_itr.isdigit() == True:
        args.num_itr = int(args.num_itr)
    if type(args.num_itr) == int:
        num_its = args.num_itr
    else:
        print("Number of itreations should be integer, using default num_itr=220")

    f.writelines("Number of iterations: "+str(num_its)+"\n")

    loss_over_time = []
    # for itr in range(num_its):
    #     (_, _, _, params), _loss = opt.step_and_cost(loss, train_X, train_y, vqc_model, params)
    #     loss_over_time.append(_loss)
    #     if (itr+1)%20 == 0:
    #         print("Iteration:",itr+1,"/",num_its,"Loss:",_loss)
    opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.009, value_and_grad=True)
    opt_state = opt.init_state(params)

    for i in range(100):
        #params, opt_state = opt.update(params, opt_state, train_X, train_y, i)
        params, opt_state = opt.update(params, opt_state, train_X, train_y, i)
    
    f.writelines(str(loss_over_time)+"\n")
    f.writelines("Final params:"+str(params)+"\n")
    
    training_accuracy = compute_accuracy(train_X, train_y, vqc_model, params)
    testing_accuracy = compute_accuracy(test_X, test_y, vqc_model, params)

    print(f"Training accuracy = {training_accuracy}")
    f.writelines("Training accuracy:"+str(training_accuracy)+"\n")

    print(f"Testing accuracy = {testing_accuracy}")
    f.writelines("Testing accuracy:"+str(testing_accuracy)+"\n")
    
    plt.plot(loss_over_time)
    plt.plot([], [], ' ', label="Training Accuracy:"+str(training_accuracy))
    plt.plot([], [], ' ', label="Testing Accuracy:"+str(testing_accuracy))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    title = "Loss for "+config['dataset']+" with scheme "+config['encoding_and_rotation_scheme']+" itr: "+str(num_its)
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
    plt.savefig("./Figs/qubit/"+title+".png",bbox_inches = 'tight')
    
    yz_op_state = []
    xz_op_state = []
    for i in range(len(train_X)):
        x,y,z = (get_state(train_X[i],params))
        yz_op_state.append([y,z])
        xz_op_state.append([x,z])
    yz_op_state = jnp.array(yz_op_state)
    xz_op_state = jnp.array(xz_op_state)
    plot_classified_data_on_bloch(yz_op_state,train_y)
    plt.show()
    title = "Bloch YZ for "+config['dataset']+" with scheme "+config['encoding_and_rotation_scheme']+" itr: "+str(num_its)
    plt.savefig("./Figs/qubit/"+title+".png")

    plot_classified_data_on_bloch(xz_op_state,train_y)
    plt.show()
    title = "Bloch XZ for "+config['dataset']+" with scheme "+config['encoding_and_rotation_scheme']+" itr: "+str(num_its)
    plt.savefig("./Figs/qubit/"+title+".png")
if __name__ == "__main__":
    main()
    f.writelines("--------------------------------------------\n")
    f.close()
