import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os 
datasets = ['xor','moon','circular']
rotation_schemes = ['A','B','C','D','E','F']
num_itrs = [100,150,200,250,300,350,400,450,500] # to do
dataset_size = 200

def delete_figures():
    folder = './Figs/qubit/'
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
delete_figures()
for dataset in datasets:
    print("Benchmarking for dataset:",dataset)
    for sch in rotation_schemes:
        command = "python run_qubit.py --dataset "+dataset
        command = command + " --encoding_and_rotation "+sch
        command = command + " --dataset_size "+str(dataset_size)
        print(command)
        os.system(command)
