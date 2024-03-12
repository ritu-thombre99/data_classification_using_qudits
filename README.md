# Qubit Models:

Qubit Encoding and Rotate: A-G
Datasets: Moon,XOR,Circular Boundary

Command to run:
```python 
Example: python run_qubit.py --encoding_and_rotation A --dataset xor --dataset_size 100

python run_qubit.py --help:                                                 
usage: run_qubit.py [-h] [--encoding_and_rotation {A,B,C,D,E,F,G}] [--dataset {xor,moon}] [--dataset_size DATASET_SIZE]

options:
  -h, --help            show this help message and exit
  --encoding_and_rotation {A,B,C,D,E,F,G}
                        choose the encoding and rotation scheme, default B
  --dataset {xor,moon}  choose the dataset, default circular boundary
  --dataset_size DATASET_SIZE
                        Enter the dataset size

```