import warnings
warnings.filterwarnings("ignore")
import pennylane as qml
from pennylane import numpy as np

lambdas = {}
for i in range(8):
    lambdas[i+1] = qml.matrix(qml.GellMann(0,i+1))

# binary
def scheme_a(x_i,s_params,w_params): 
    # input vector x   
    U = (x_i[0]*s_params[0]*lambdas[6])+(x_i[1]*s_params[1]*lambdas[7])
    encode_U = np.exp(1j*U)
    qml.QutritUnitary(encode_U,wires=0)

    U = (w_params[0]*lambdas[1])+(w_params[1]*lambdas[4])
    rotate_U = np.exp(1j*U)
    qml.QutritUnitary(rotate_U,wires=0)

# binary
def scheme_b(x_i,s_params,w_params): 
    # input vector x   
    U1 = x_i[0]*((s_params[0]*lambdas[1]) + (s_params[1]*lambdas[2]))
    U2 = x_i[1]*((s_params[2]*lambdas[3]) + (s_params[3]*lambdas[4]))
    encode_U = np.exp(1j* (U1+U2))
    qml.QutritUnitary(encode_U,wires=0)

    U = w_params[0]*lambdas[1]
    for i in range(1,4):
        U = U + w_params[i]*lambdas[i+1]
    rotate_U = np.exp(1j*U)
    qml.QutritUnitary(rotate_U,wires=0)
             
# binary
def scheme_c(x_i,s_params,w_params): 
    # input vector x   
    U1 = x_i[0]*((s_params[0]*lambdas[5]) + (s_params[1]*lambdas[6]))
    U2 = x_i[1]*((s_params[2]*lambdas[7]) + (s_params[3]*lambdas[8]))
    encode_U = np.exp(1j* (U1+U2))
    qml.QutritUnitary(encode_U,wires=0)

    U = w_params[0]*lambdas[1]
    for i in range(1,4):
        U = U + w_params[i]*lambdas[i+1]
    rotate_U = np.exp(1j*U)
    qml.QutritUnitary(rotate_U,wires=0)

# x_i has 8 features           
def scheme_d1(x_i,s_params,w_params): 
    # input vector x   

    U = x_i[0]*lambdas[1]
    for i in range(1,8):
        U = U + x_i[i]*lambdas[i+1]
    U = s_params[0]*U
    encode_U = np.exp(1j*U)
    qml.QutritUnitary(encode_U,wires=0)

    U = w_params[0]*lambdas[1]
    for i in range(1,7):
        U = U + w_params[i]*lambdas[i+1]
    rotate_U = np.exp(1j*U)
    qml.QutritUnitary(rotate_U,wires=0)


# x_i has 8 features           
def scheme_d2(x_i,s_params,w_params): 
    # input vector x   

    U = s_params[0]*x_i[0]*lambdas[1]
    for i in range(1,8):
        U = U + s_params[i%4]*x_i[i]*lambdas[i+1]
    encode_U = np.exp(1j*U)
    qml.QutritUnitary(encode_U,wires=0)

    U = w_params[0]*(lambdas[1]+lambdas[5])
    for i in range(1,4):
        U = U + w_params[i]*(lambdas[i+1]+lambdas[4+i+1])
    rotate_U = np.exp(1j*U)
    qml.QutritUnitary(rotate_U,wires=0)



# x_i has 8 features           
def scheme_d3(x_i,s_params,w_params): 
    # input vector x   

    U = s_params[0]*x_i[0]*lambdas[1]
    for i in range(1,8):
        U = U + s_params[i]*x_i[i]*lambdas[i+1]
    encode_U = np.exp(1j*U)
    qml.QutritUnitary(encode_U,wires=0)

    U = w_params[0]*lambdas[1]
    rotate_U = np.exp(1j*U)
    qml.QutritUnitary(rotate_U,wires=0)
             
