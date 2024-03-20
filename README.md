# Setup

Create environment:
```conda create --name cpen400q -c anaconda python=3.11.5```

Activate:
```conda activate cpen400q```

Requirements
```cd /pat/to/repo```
```pip install -r requirements.txt```

Deactivate Env:
```conda deactivate cpen400q```

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
  --num_itr NUM_ITR
            Enter number of iterations for training

```

# Questions to ask Instructor
1. For qubit encoding and rotation G, the input data is 3-dimensional $\{x_0,x_1,x_2\}$ with binary classification. But any of the dataset used in the paper (i.e. circular boundary, moon, XOR/Noisy XOR, wine, iris) satisfy this criteria of 3d input 2d output
2. Section 3 to discuss lossless dimension, author use 50 different initial points from the input dataset (they mention somewhere they use 4% of 2000 XOR points as initial datapoint, randomly), where they change the initial datapoints if previous initial datapoints did not give good results. Do we have to do this? We're planning to experiment more with the combinations of optimizers, number of iterations, and encoding and rotation schemes.
3. The way lossless dimension ($D_{LM}$) is dervied is by trail and error of some initial datapoints (empirically), there is no formula to derive it mathemtically. Do we need to do this as well?
4. MSE loss (exp_val-true_label)**2 for incorrectly classified points works fine in qubit binary classifier case, unclear on how to implement cross-entropy loss (using negative log-likelihood). Cross entriopy uses probability of classified into particular class (we are not sure how to get this)
5. In qutrit case, encoding schemes 
   + A,B,C use 2D data $\{x_0,x_1\}$ for the input, 
   + And encoding schemes D1,D2,D3 use 8D data $\{x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7\}$ for the input. 
   + XOR,Noisy XOR,Circular,Moon can be used with schemes A,B,C with classification of $[-2,0)$ as class -1 and $[0,2]$ as class 1.
   + But Wine is contains 13 features (i.e 13 dimensional data)
   + Iris contains 4 features (i.e. 4 dimensional data)
   + Both Iris and Wine need ternary classification (so we have to go with qutrit schemes)
   + In the preprocessing table 3:

  | Dataset | Features | Classes | Samples | Pre-processing |
  | :---: | :---: | :---: | :---: | :---: |
  | IRIS | 4 | 3 | 150 | none |
  | WINE | 13 | 3 | 178 | PCA |

  + How is it possible to encode 4D iris data without preprocessing into qutrit encoding?
  + We wither need to use PCA on Iris to reduce dimensions to 2 so that it can be encoded with schemes A,B,C
  
  Or We need to pad some 0s to Iris to increase dimensions to 8 so that it can be encoded with schemes D1,D2,D3
