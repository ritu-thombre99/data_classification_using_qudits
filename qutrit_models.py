import warnings
warnings.filterwarnings("ignore")
import pennylane as qml
from pennylane import numpy as np

def expm(H):
    eigval, eigvec = np.linalg.eig(H)
    exp_evals = np.exp(1j * eigval)
    unitary = eigvec @ np.diag(exp_evals) @ np.linalg.inv(eigvec)
    return unitary

lambdas = {}
for i in range(8):
    lambdas[i+1] = qml.matrix(qml.GellMann(0,i+1))

# binary
def scheme_a(x_i,s_params,w_params): 
    # input vector x   
    H = (x_i[0]*s_params[0]*lambdas[6])+(x_i[1]*s_params[1]*lambdas[7]) # encode
    H = H + (w_params[0]*lambdas[1])+(w_params[1]*lambdas[4]) # rotate
    U = expm(H)
    qml.QutritUnitary(U,wires=0)

# binary
def scheme_b(x_i,s_params,w_params): 
    # input vector x   
    U1 = x_i[0]*((s_params[0]*lambdas[1]) + (s_params[1]*lambdas[2]))
    U2 = x_i[1]*((s_params[2]*lambdas[3]) + (s_params[3]*lambdas[4]))
    H = U1+U2

    for i in range(4):
        H = H + (w_params[i]*lambdas[i+1])
    U = expm(H)
    qml.QutritUnitary(U,wires=0)
             
# binary
def scheme_c(x_i,s_params,w_params): 
    # input vector x   
    U1 = x_i[0]*((s_params[0]*lambdas[5]) + (s_params[1]*lambdas[6]))
    U2 = x_i[1]*((s_params[2]*lambdas[7]) + (s_params[3]*lambdas[8]))
    H = U1+U2

    for i in range(4):
        H = H + (w_params[i]*lambdas[i+1])
    U = expm(H)
    qml.QutritUnitary(U,wires=0)

# x_i has 8 features           
def scheme_d1(x_i,s_params,w_params): 
    # input vector x   

    H = x_i[0]*lambdas[1]
    for i in range(1,8):
        H = H + x_i[i]*lambdas[i+1]
    H = s_params[0]*H

    for i in range(7):
        H = H + w_params[i]*lambdas[i+1]
    U = expm(H)
    qml.QutritUnitary(U,wires=0)


# x_i has 8 features           
def scheme_d2(x_i,s_params,w_params): 
    # input vector x   

    H = s_params[0]*x_i[0]*lambdas[1]
    for i in range(1,8):
        H = H + s_params[i%4]*x_i[i]*lambdas[i+1]

    for i in range(4):
        H = H + w_params[i]*(lambdas[i+1]+lambdas[4+i+1])
    U = expm(H)
    qml.QutritUnitary(U,wires=0)



# x_i has 8 features           
def scheme_d3(x_i,s_params,w_params): 
    # input vector x   

    H = s_params[0]*x_i[0]*lambdas[1]
    for i in range(1,8):
        H = H + s_params[i]*x_i[i]*lambdas[i+1]

    H = H + w_params[0]*lambdas[1]
    U = expm(H)
    qml.QutritUnitary(U,wires=0)
             

# x_i has 4 features           
def custom_ternary(x_i,s_params,w_params): 
    # input vector x   

    H = x_i[0]*lambdas[1]
    for i in range(1,4):
        H = H + x_i[i]*lambdas[i+1]
    H = s_params[0]*H

    for i in range(4):
        H = H + w_params[i]*lambdas[8]
    U = expm(H)
    qml.QutritUnitary(U,wires=0)

# x_i has 2 features           
def custom_binary(x_i,s_params,w_params): 
    # input vector x   

    H = x_i[0]*(s_params[0]*lambdas[3] + s_params[1]*lambdas[5] + s_params[2]*lambdas[7])
    H = H +  x_i[1]*(s_params[3]*lambdas[4] + s_params[4]*lambdas[6] + s_params[5]*lambdas[8])
    
    L1 = lambdas[1]+lambdas[6]
    L2 = lambdas[2]+lambdas[7]
    L3 = lambdas[3]+np.sqrt(3)*lambdas[8]

    H = H + w_params[0]*L1 + w_params[1]*L2 + w_params[2]*L3
    U = expm(H)
    qml.QutritUnitary(U,wires=0)