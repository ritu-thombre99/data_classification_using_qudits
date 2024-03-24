import warnings, os
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from run_qubit import *
datasets = ['xor','moon','circular']
rotation_schemes = ['A','B','C','D','E','F']
num_itrs = [100,200,300,400,500] # to do
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
