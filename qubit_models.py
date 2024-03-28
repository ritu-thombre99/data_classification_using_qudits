import pennylane as qml
from pennylane import numpy as np

# import numpy as np
import jax.numpy as jnp
pauli_x = qml.matrix(qml.PauliX(0))
pauli_y = qml.matrix(qml.PauliY(0))
pauli_z = qml.matrix(qml.PauliZ(0))
def expm(H):
    eigval, eigvec = jnp.linalg.eigh(H)
    exp_evals = jnp.exp(1j * eigval)
    unitary = eigvec @ jnp.diag(exp_evals) @ jnp.linalg.inv(eigvec)
    return unitary

def scheme_a(x_i,s_params,w_params): 
    # input vector x   
    H = s_params[0]*x_i[0]*pauli_x + s_params[0]*x_i[1]*pauli_y # encoding
    H = H + w_params[0]*pauli_x # rotation
    U = expm(H)
    qml.QubitUnitary(U,wires=0)

def scheme_b(x_i,s_params,w_params): 
    # input vector x   
    H = s_params[0]*x_i[0]*pauli_x + s_params[0]*x_i[1]*pauli_y # encoding
    H = H + w_params[0]*pauli_x + w_params[1]*pauli_y + w_params[2]*pauli_z
    U = expm(H)
    qml.QubitUnitary(U,wires=0)
             
             
def scheme_c(x_i,w_params): 
    # input vector x   
    H = x_i[0]*pauli_x + x_i[1]*pauli_y
    H = H + w_params[0]*pauli_x + w_params[1]*pauli_y + w_params[2]*pauli_z
    U = expm(H)
    qml.QubitUnitary(U,wires=0)
             
def scheme_d(x_i,s_params,w_params): 
    # input vector x   
    H = s_params[0]*x_i[0]*pauli_x + s_params[1]*x_i[1]*pauli_y
    H = H + w_params[0]*pauli_x + w_params[1]*pauli_y + w_params[2]*pauli_z
    U = expm(H)
    qml.QubitUnitary(U,wires=0)
             
             
def scheme_e(x_i,s_params,w_params):
    H = s_params[0]*(x_i[0]*pauli_x + x_i[1]*pauli_y)
    H = H + w_params[0]*pauli_x

    H = H + s_params[1]*(x_i[0]*pauli_y + x_i[1]*pauli_z)
    H = H + w_params[1]*pauli_y

    U = expm(H)
    qml.QubitUnitary(U,wires=0)
                  
             
def scheme_f(x_i,s_params,w_params):
    H = s_params[0]*(x_i[0]*pauli_x + x_i[1]*pauli_y)
    H = H + w_params[0]*pauli_x

    H = H + s_params[1]*(x_i[0]*pauli_y + x_i[1]*pauli_z)
    H = H + w_params[1]*pauli_y

    H = H + s_params[2]*(x_i[0]*pauli_x + x_i[1]*pauli_y)
    H = H + w_params[2]*pauli_x
    
    U = expm(H)
    qml.QubitUnitary(U,wires=0) 
             
def scheme_g(x_i,s_params,w_params):
    H = s_params[0]*(x_i[0]*pauli_x + x_i[1]*pauli_y + x_i[2]*pauli_z)
    H = H + w_params[0]*pauli_x + w_params[1]*pauli_y
    U = expm(H)
    qml.QubitUnitary(U,wires=0) 
