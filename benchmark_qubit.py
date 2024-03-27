import warnings, os
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib, shutil
matplotlib.use('Agg')

from run_qubit import *
datasets = ['xor','moon','circular']
rotation_schemes = ['A','B','C','D','E','F']
# num_itrs = [100,200,300,400,500] # to do
num_itrs = [50] # to do
dataset_size = 300

def delete_figures(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

print("Deleting previous plots")
delete_figures('./Figs/qubit/')
print("Deleting previous logs")
delete_figures('./logs/')

f = open("./logs/qubit_run.txt","w")

for dataset in datasets:
    if dataset == 'xor':
        train_X, test_X, train_y, test_y = get_xor_data(dataset_size)
    elif dataset == 'circular':
        train_X, test_X, train_y, test_y = get_circular_boundary_dataset(dataset_size)
    elif dataset == 'moon':
        train_X, test_X, train_y, test_y = get_moon_dataset(dataset_size)
    print("Benchmarking for dataset:",dataset)
    for sch in rotation_schemes:
        for itr in num_itrs:
            command = "python run_qubit.py --dataset "+dataset
            command = command + " --encoding_and_rotation "+sch
            command = command + " --dataset_size "+str(dataset_size)
            command = command + " --num_itr "+str(itr)
            print(command)
            run(dataset,sch,dataset_size,itr,train_X, test_X, train_y, test_y,f)
f.close()



import numpy as np
import pandas as pd
f = open('./logs/qubit_run.txt')
lines = f.readlines()
lines = ''.join(lines)
lines = lines.split("------------------------------------\n")
df = pd.DataFrame()
for l in lines:
    row = {}
    l = (l.split('\n'))
    if len(l) == 10:
        dataset = l[0][len('Dataset: '):]
        row['Dataset'] = dataset

        dataset_size = int(l[1][len('Dataset size: '):])
        row['Dataset Size'] = dataset_size

        scheme = l[2][len('Encoding scheme: '):]
        row['Scheme'] = scheme

        init_params = l[3][len('Initial parameters: '):].replace('[','').replace(']','')
        init_params = init_params.split(" ")
        while '' in init_params:
            init_params.remove('')
        init_params = (np.asarray(init_params,dtype=float))

        final_params = l[6][len('Final params:'):].replace('[','').replace(']','')
        final_params = final_params.split(" ")
        while '' in final_params:
            final_params.remove('')
        final_params = (np.asarray(final_params,dtype=float))
        row['final_params'] = final_params

        num_its = int(l[4][len('Number of iterations: '):])
        row['Iterations'] = num_its

        loss = l[5].replace('dtype=float64)','').replace('Array(','').replace('[','').replace(']','')
        loss = loss.split(", ")
        while '' in loss:
            loss.remove('')
        loss = (np.asarray(loss,dtype=float))
        row['Loss'] = loss


        training_accuracy = float(l[7][len('Training accuracy:'):])
        row['Training accuracy'] = training_accuracy
        test_accuracy = float(l[8][len('Testing accuracy:'):])
        row['Test accuracy'] = test_accuracy
        row = pd.DataFrame([row])
        df = pd.concat([df, row], ignore_index=True)
        
xor_df = df.groupby('Dataset').get_group('xor')
circular_df = df.groupby('Dataset').get_group('circular')
moon_df = df.groupby('Dataset').get_group('moon')

import pennylane as qml
from qubit_models import *
scheme_config = {}
scheme_config['A'] = [1,1]
scheme_config['B'] = [1,3]
scheme_config['C'] = [0,3]
scheme_config['D'] = [2,3]
scheme_config['E'] = [2,2]
scheme_config['F'] = [3,3]
scheme_config['G'] = [1,2]

dev = qml.device("default.qubit", wires=1)
@qml.qnode(dev)
def get_state(x_i,params,scheme):
    s_params_size = scheme_config[scheme][0]
    s_params,w_params = params[:s_params_size], params[s_params_size:]
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


import matplotlib.pyplot as plt
import sns
from datasets import *
from helpers import *
def plot_datafram(df,dataset):
    print(dataset)
    def get_loss(df):
        unique_schemes = df['Scheme'].unique()
        return_val = []
        for s in unique_schemes:
            scheme_df = df.groupby('Scheme').get_group(s)
            return_val.append(scheme_df['Loss'].iloc[0])
        return unique_schemes,return_val
    def get_accuracies(df):
        unique_schemes = df['Scheme'].unique()
        tr,ts = [],[]
        print("Unique schemes:",unique_schemes)
        for s in unique_schemes:
            scheme_df = df.groupby('Scheme').get_group(s)
            tr_acc = scheme_df['Training accuracy'].iloc[0]
            ts_acc = scheme_df['Test accuracy'].iloc[0]
            tr_acc = (int(tr_acc*10))/10
            ts_acc = (int(ts_acc*10))/10
            tr.append(tr_acc)
            ts.append(ts_acc)
            
        return unique_schemes,tr,ts
        
    unique_itrs = df['Iterations'].unique()
    for itr in unique_itrs:
        print("Iterations:",itr)
        itr_df = df.groupby('Iterations').get_group(itr)
        schemes,losses = get_loss(itr_df)
        
        x = list(range(1,itr+1))
        for l,s in zip(losses,schemes):
            plt.plot(x,l,label="Scheme:"+s)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid()
        title = "Loss vs Iters ("+str(itr)+") Dataset:"+dataset
        plt.title(title)
        plt.show()
        plt.savefig("./Figs/qubit/"+title+".png",bbox_inches = 'tight')
        plt.close()


        X,train,test = get_accuracies(itr_df)
        bar_df = pd.DataFrame(np.c_[train,test], index=X,columns=['Train Accuracy','Test Accuracy'])
        ax = bar_df.plot.bar()
        for container in ax.containers:
            ax.bar_label(container)
        plt.grid()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        title = "Accuracy vs Schemes ("+str(itr)+") Dataset:"+dataset
        plt.title(title)
        plt.show()
        plt.savefig("./Figs/qubit/"+title+".png",bbox_inches = 'tight')
        plt.close()

        if dataset == 'xor':
            train_X, test_X, train_y, test_y = get_xor_data(dataset_size)
        elif dataset == 'circular':
            train_X, test_X, train_y, test_y = get_circular_boundary_dataset(dataset_size)
        elif dataset == 'moon':
            train_X, test_X, train_y, test_y = get_moon_dataset(dataset_size)
        

        test_max_indices = [X[i] for i, x in enumerate(test) if x == max(test)]
        train_max_indices = [X[i] for i, x in enumerate(train) if x == max(train)]
        intersection = (list(set(train_max_indices).intersection(test_max_indices)))
        print("Max test accuracy for schemes:",test_max_indices)
        print("Max train accuracy for schemes:",train_max_indices)
        print("Intersections:",intersection)
        best_schemes = test_max_indices
        if len(intersection) != 0:
            best_schemes = intersection
        for best_scheme in best_schemes:
            final_params = (itr_df[itr_df['Scheme']==best_scheme]['final_params'].iloc[0])
            yz_op_state = []
            xz_op_state = []
            for i in range(len(train_X)):
                x,y,z = (get_state(train_X[i],final_params,best_scheme))
                yz_op_state.append([y,z])
                xz_op_state.append([x,z])
            for i in range(len(test_X)):
                x,y,z = (get_state(test_X[i],final_params,best_scheme))
                yz_op_state.append([y,z])
                xz_op_state.append([x,z])

            yz_op_state = np.array(yz_op_state)
            xz_op_state = np.array(xz_op_state)
            classes = np.concatenate((train_y,test_y), axis=None)
            
            plot_classified_data_on_bloch(yz_op_state,classes)
            title = "Bloch YZ for "+dataset+" with scheme "+best_scheme+" itr: "+str(itr)
            plt.title(title)
            plt.grid()
            plt.show()
            plt.savefig("./Figs/qubit/"+title+".png",bbox_inches = 'tight')
            plt.close()

            plot_classified_data_on_bloch(xz_op_state,classes)
            title = "Bloch XZ for "+dataset+" with scheme "+best_scheme+" itr: "+str(itr)
            plt.title(title)
            plt.grid()
            plt.show()
            plt.savefig("./Figs/qubit/"+title+".png",bbox_inches = 'tight')
            plt.close()


plot_datafram(moon_df,'moon')
plot_datafram(circular_df,'circular')
plot_datafram(xor_df,'xor')
