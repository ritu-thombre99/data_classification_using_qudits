import warnings, os
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib, shutil
matplotlib.use('Agg')

from run_qubit import *
datasets = ['xor','moon','circular']
rotation_schemes = ['A','B','C','D','E','F']
num_itrs = [100,300,500] # to do
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

import matplotlib.pyplot as plt
import sns
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
        plt.savefig("./Figs/qubit/"+title+".png")
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
        plt.savefig("./Figs/qubit/"+title+".png")
        plt.close()

plot_datafram(moon_df,'moon')
plot_datafram(circular_df,'circular')
plot_datafram(xor_df,'xor')
