import numpy as np
import pandas as pd
from copy import deepcopy

WTS = "w"
BIAS = "b"
BIAS_KEY = "bias"
OUTPUT = "a"
X_ROW = 2
X_COL = 1
LAYER = "layer"
NODE = "node"
INPUT_FEAT = "input_feat"

class sequential_:

    def __init__ (self):
        self.layers_ = 0
        self.params_ = {}
        self.nodes_ = []

    def add (self,nodes=5,activation = "linear",input_dim=None):
        # add one layer
        self.layers_ += 1
        # add the nodes
        self.nodes_.append(nodes)

        if input_dim:  # Input Layer
            prev_nodes = input_dim
        else:
            prev_nodes = self.nodes_[self.layers_-2]
        curr_nodes = nodes
        wts, bias = self.init_params(curr_nodes,prev_nodes)

        # add the wts and bias to self.params_
        self.params_[WTS+str(self.layers_)] = wts
        self.params_[BIAS+str(self.layers_)] = bias


    def init_params (self,curr_layer_nodes:int,prev_layer_nodes:int):
        # Initializing wt = 0.1 and bias = 0
        wts = np.ones((curr_layer_nodes, prev_layer_nodes))*0.1
        bias = np.zeros((curr_layer_nodes,1))
        return wts,bias

    def compile(self,loss="mean_squared_error",learning_rate=0.01):
        # Loss Function
        self.loss = loss
        # Gradient Descent params
        self.lr = learning_rate
        self.outputs = {}

    def summation_ (self,wts:np.ndarray,bias:np.ndarray,X:np.ndarray):
        z = np.dot(wts,X) + bias
        return z

    def forward_prop_ (self,X,params=None)->float:
        if params:
            parameters = params
        else:
            parameters = self.params_
        A = X
        for l in range(1,self.layers_+1):
            wts = parameters[WTS + str(l)]
            bias = parameters[BIAS + str(l)]
            A = self.summation_(wts,bias,A)
            self.outputs[OUTPUT+str(l)] = A
        return A[0][0]

    def mse_loss (self,y_pred,y)->float:
        # mean squared loss
        mse = (y_pred - y)**2
        return mse

    def diffp(self,func,idx:dict,*args):
        """
        :param func: function to find y_pred
        :param idx: dictionary {"layer" : <layer_no>:int,"node": <node_no>:int,
        "input_feat" : <input_feat_no>:int,"bias":<True/False>}
        :return:
        """
        y = func(*args)
        delta = 0.000000000001

        if idx[BIAS_KEY]:
            # update bias
            arg = BIAS
        else:
            # update wts
            arg = WTS
        # Deep Copy self.params_ and add delta to req. wt or bias
        params_copy = deepcopy(self.params_)
        params_copy[arg + str(idx[LAYER])][idx[NODE]] \
            [idx[INPUT_FEAT]] += delta

        y1 = func(*args,parameters=params_copy)
        return (y1 - y)/delta

    def Loss_func_ (self,X,y,parameters=None,y_pred=None)->float:
        if not y_pred:
            y_pred = self.forward_prop_(X,params=parameters)
        loss_func = self.mse_loss(y_pred,y)
        return loss_func


    def fit(self,X:pd.core.frame.DataFrame,y:pd.core.frame.DataFrame,epochs=100):
        X = np.array(X)
        y = np.array(y)
        self.loss_ = []

        for epoch in range(epochs):
            loss_arr = []
            for i in range(X.shape[0]):
                # Get a random number
                Xi = X[i].reshape(X_ROW,X_COL)
                yi = y[i][0] # scalar

                # Predict for this value
                y_predi = self.forward_prop_(Xi)

                # Calculate the loss
                loss = self.Loss_func_(Xi,yi,y_pred=y_predi)
                loss_arr.append(loss)

                # Update the weights and bias
                for layer in range(1,self.layers_+1):
                    for node in range(self.nodes_[layer-1]):
                        for input_feat in range(len(self.params_[WTS+str(
                                layer)][node])):
                            # update wts
                            pass
                        # update bias
                        bias_idx = {LAYER:layer,NODE:node,
        "input_feat" : <input_feat_no>:int,"bias":<True/False>}
                        """self.params_[BIAS+str(layer)][node] = self.params_[
                            BIAS+str(layer)][node] - self.lr*self.diffp(
                            self.Loss_func_,)"""


            self.loss_.append(np.mean(loss_arr))