import pennylane as qml
from pennylane import numpy as np

def scheme_a(x_i,s_params,w_params): 
    # input vector x   
    rx_angle = -2*s_params[0]*x_i[0]    
    ry_angle = -2*s_params[0]*x_i[1]
    qml.RY(ry_angle,0)
    qml.RX(rx_angle,0)
    qml.RX(-2*w_params[0],0)

def scheme_b(x_i,s_params,w_params): 
    # input vector x   
    rx_angle = -2*s_params[0]*x_i[0]    
    ry_angle = -2*s_params[0]*x_i[1]
    qml.RY(ry_angle,0)
    qml.RX(rx_angle,0)
    qml.RX(-2*w_params[0],0)    
    qml.RY(-2*w_params[1],0)
    qml.RZ(-2*w_params[2],0)    
             
             
def scheme_c(x_i,w_params): 
    # input vector x   
    rx_angle = -2*x_i[0]    
    ry_angle = -2*x_i[1]
    qml.RY(ry_angle,0)
    qml.RX(rx_angle,0)
    qml.RX(-2*w_params[0],0)    
    qml.RY(-2*w_params[1],0)
    qml.RZ(-2*w_params[2],0)    
             
def scheme_d(x_i,s_params,w_params): 
    # input vector x   
    rx_angle = -2*s_params[0]*x_i[0]    
    ry_angle = -2*s_params[1]*x_i[1]
    qml.RY(ry_angle,0)
    qml.RX(rx_angle,0)
    qml.RX(-2*w_params[0],0)    
    qml.RY(-2*w_params[1],0)
    qml.RZ(-2*w_params[2],0)   
             
             
def scheme_e(x_i,s_params,w_params):
    rx_angle = -2*s_params[0]*x_i[0]    
    ry_angle = -2*s_params[0]*x_i[1]
    qml.RY(ry_angle,0)
    qml.RX(rx_angle,0)
    qml.RX(-2*w_params[0],0)

    ry_angle = -2*s_params[1]*x_i[0]    
    rz_angle = -2*s_params[1]*x_i[1]
    qml.RZ(rz_angle,0)
    qml.RY(ry_angle,0)
    qml.RY(-2*w_params[1],0)
                  
             
def scheme_f(x_i,s_params,w_params):
    rx_angle = -2*s_params[0]*x_i[0]    
    ry_angle = -2*s_params[0]*x_i[1]
    qml.RY(ry_angle,0)
    qml.RX(rx_angle,0)
    qml.RX(-2*w_params[0],0)

    ry_angle = -2*s_params[1]*x_i[0]    
    rz_angle = -2*s_params[1]*x_i[1]
    qml.RZ(rz_angle,0)
    qml.RY(ry_angle,0)
    qml.RY(-2*w_params[1],0)

    rx_angle = -2*s_params[2]*x_i[0]    
    ry_angle = -2*s_params[2]*x_i[1]
    qml.RY(ry_angle,0)
    qml.RX(rx_angle,0)
    qml.RX(-2*w_params[2],0)    
             
def scheme_g(x_i,s_params,w_params):
    rx_angle = -2*s_params[0]*x_i[0]    
    ry_angle = -2*s_params[0]*x_i[1]
    rz_angle = -2*s_params[0]*x_i[2]
    qml.RZ(rz_angle,0)
    qml.RY(ry_angle,0)
    qml.RX(rx_angle,0)
    qml.RX(-2*w_params[0],0) 
    qml.RY(-2*w_params[1],0)
            
 