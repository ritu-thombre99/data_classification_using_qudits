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
from functools import partial
import optax

# store the s_params and w_params required for any scheme
scheme_config = {}
scheme_config['A'] = [1,1]
scheme_config['B'] = [1,3]
scheme_config['C'] = [0,3]
scheme_config['D'] = [2,3]
scheme_config['E'] = [2,2]
scheme_config['F'] = [3,3]
scheme_config['G'] = [1,2]

config = {}
config['dataset'] = 'circular'
config['encoding_and_rotation_scheme'] = 'B'
config['s_params_size'] = 1
config['w_params_size'] = 3

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
    loss_sum = jnp.asarray([0.])
    for idx in range(len(data)):
        data_point = data[idx]
        true_label = labels[idx]
        model_output = vqc_model(data_point, params)
        @jax.jit
        def append_loss_fn():
            temp_loss = jax.numpy.square(jax.numpy.subtract(model_output,true_label))
            return temp_loss
        def false_fn():
            return 0.

        # if (model_output<0 and true_label>0) or (model_output>0 and true_label<0):
        # (model_output<0 and true_label>0)
        temp_loss = jax.lax.cond(
                        (jax.numpy.greater(0,model_output) & jax.numpy.greater(true_label,0)),
                        append_loss_fn,
                        false_fn,
                    )
        loss_sum = loss_sum.at[0].add(temp_loss)

        # (model_output>0 and true_label<0)
        temp_loss = jax.lax.cond(
                        (jax.numpy.greater(model_output,0) & jax.numpy.greater(0,true_label)),
                        append_loss_fn,
                        false_fn,
                    )
        loss_sum = loss_sum.at[0].add(temp_loss)

    return jnp.sum(loss_sum)/len(data)

# jax loss for jax gradient descent opt
# def loss_and_grad(params, data, labels, i, print_training=True):
#     loss_val, grad_val = jax.value_and_grad(loss)(params, data, labels)

#     def print_fn():
#         jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

#     # if print_training=True, print the loss every 5 steps
#     jax.lax.cond((jnp.mod(i, 5) == 0) & print_training, print_fn, lambda: None)

#     return loss_val, grad_val

def update_step(opt, params, opt_state, data, labels):
    loss_val, grads = jax.value_and_grad(loss)(params, data, labels)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val



# @jax.jit
def make_prediction(model, data_point, params):
    measurement_result = model(data_point, params)

    # @jax.jit
    # def true_fn():
    #     return -1
    # def false_fn():
    #     return 1
    # return jax.lax.cond(
    #                 jax.numpy.greater(0,measurement_result),
    #                 true_fn,
    #                 false_fn,
    #             )

    if measurement_result < 0:
        return -1
    return 1

# @jax.jit
def compute_accuracy(data, labels, model, params):
    # accuracy_sum = jnp.asarray([0.])
    # @jax.jit
    # def true_fn():
    #     return 1.
    # def false_fn():
    #     return 0.

    # for x in range(len(data)):
    #     retval = jax.lax.cond(
    #                     jax.numpy.equal(make_prediction(model, data[x], params),labels[x] ),
    #                     true_fn,
    #                     false_fn,
    #                 )
    #     accuracy_sum = accuracy_sum.at[0].add(retval)


    # return jnp.sum(accuracy_sum)/len(data)
    preds = jnp.asarray([make_prediction(model, data[x], params) == labels[x] for x in range(len(data))])

    return jnp.sum(preds) / len(data)

# @jax.jit
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
    params = jnp.asarray(np.random.normal(size=(s_params_size+w_params_size)))

    f.writelines("Dataset: "+dataset+"\n")
    f.writelines("Dataset size: "+str(dataset_size)+"\n")
    f.writelines("Encoding scheme: "+str(encoding_and_rotation_scheme)+"\n")


    print("Initial parameters:",params)
    f.writelines("Initial parameters: "+str(params)+"\n")
    f.writelines("Number of iterations: "+str(num_its)+"\n")

    # opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.009, value_and_grad=True)
    # opt_state = opt.init_state(params)
    opt = optax.sgd(learning_rate=0.003)


    @jax.jit
    def update_step_jit(i, args):
        params, opt_state, data, labels, print_training = args

        loss_val, grads = jax.value_and_grad(loss)(params, data, labels)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        def print_fn():
            jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

        # if print_training=True, print the loss every 5 steps
        jax.lax.cond((jnp.mod(i, 20) == 0) & print_training, print_fn, lambda: None)

        return (params, opt_state, data, labels, print_training)
    @jax.jit
    def optimization_jit(params, data, targets, print_training=False):

        opt_state = opt.init(params)
        args = (params, opt_state, data, targets, print_training)
        (params, opt_state, _, _, _) = jax.lax.fori_loop(0, 100, update_step_jit, args)

        return params
    # params = optimization_jit(params, train_X, train_y, True)

    opt_state = opt.init(params)
    loss_over_time = []
    for i in range(num_its):
        # params, opt_state = opt.update(params, opt_state, train_X, train_y, i)
        params, opt_state, loss_val = update_step(opt, params, opt_state, train_X, train_y)

        if i % 5 == 0:
            print(f"Step: {i} Loss: {loss_val}")

        loss_over_time.append(loss_val)
    
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
    
    plt.plot(loss_over_time)
    training_accuracy = int((10**5)*training_accuracy)/(10**5)
    testing_accuracy = int((10**5)*testing_accuracy)/(10**5)
    plt.plot([], [], ' ', label="Training Accuracy:"+str(training_accuracy))
    plt.plot([], [], ' ', label="Testing Accuracy:"+str(testing_accuracy))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    title = "Loss for "+config['dataset']+" with scheme "+config['encoding_and_rotation_scheme']+" itr: "+str(num_its)
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
    plt.savefig("./Figs/qubit/"+title+".png",bbox_inches = 'tight')
    plt.close()
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
    plt.close()
    plot_classified_data_on_bloch(xz_op_state,train_y)
    plt.show()
    title = "Bloch XZ for "+config['dataset']+" with scheme "+config['encoding_and_rotation_scheme']+" itr: "+str(num_its)
    plt.savefig("./Figs/qubit/"+title+".png")
    plt.close()
    if is_default == True:
        f.close()

