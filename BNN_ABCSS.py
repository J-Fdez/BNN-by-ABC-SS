
"""
@author: Juan Fernandez Salas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from ABC_aux_func import actf_dic, metric_dic 

###################### README ########################
"""
This code is a basic implementation of the methodology presented in the research
article "Uncertainty quantification in Neural Networks by Approximate Bayesian
Computation: Application to fatigue in composite materials" in the journal 
Engineering Applications of Artificial Intelligence.
https://doi.org/10.1016/j.engappai.2021.104511 


Python version 3.7 was used, more recent versions might cause problems when using
'plot_pred_pdf' and 'plot_pred_pdf' functions below to plot some figures with 
the seaborn package. To solve this, either use version 3.7 or modify the code 
below to suit your python version.
 

I hope you find BNN by ABC-SS useful, and please share your ideas if you find new ways 
of improving its performance.

For more details, refer to the research article and the references provided.
"""

######################################################

"""
Input features to Class BNN_ABCSS:
N - Number of samples in ABC-SS. The more complex the task and/or the neural network
    architecture the more samples are needed.
recover - Name of the file where the weights and bias were stored using the save_weights
    function below, format: 'file_name.npz' 
mlp_neurons - Neural network architecture in the form 
    [Input neurons, neurons hidden layer 1,...,Output neurons]
mlp_act_f - Activation functions for each layer in the form 
    [act_func in hidden layer 1,...,act_func in Output layer]
    Activation functions and their derivatives are saved on ABC_aux_func. If other
    activation functions are needed they can be added to that file.
    
Three 'training modes' can be used, depending on whether the tolerance value epsilon
is defined, the number of simulation levels is fixed, or a gradient directed ABC-SS
is used (this last one is not included in the research article mentioned above, and it
does not follow the gradient-free principles of ABC-SS).
The hyperparameters for ABC-SS training are explained below.

"""
class BNN_ABCSS:
    def __init__(self, N=10000, recover=None, mlp_neurons=[1,10,10,1], mlp_act_f =['relu', 'relu','linear']):
        self.recover=recover
        self.act_f_list=mlp_act_f
        if self.recover==None:
            self.N=N
            self.mlp_neurons=mlp_neurons
            self.mlp_act_f =[]
            for i in mlp_act_f:
                self.mlp_act_f.append(actf_dic[i])
            self.nW = 0 
            self.nb = 0
            for i in range (len(self.mlp_neurons)-1):
                p = self.mlp_neurons[i+1]*self.mlp_neurons[i]
                self.nW=self.nW+p # Number of weights
                self.nb=self.nb+self.mlp_neurons[i+1] # Number of bias
            # Initialization of Matrix where all parameters and metrics will
            # be stored each simulation level (+1 is for the metric)
            self.ABCSampAcc = np.zeros(shape = (self.N , self.nW+self.nb+1))
            # Fill in the ABCSampAcc Matrix with the "Prior" parameters
            for i in range(self.nW+self.nb):
                self.ABCSampAcc[: , i] = stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=self.N) 
        else:
            with np.load(self.recover) as data:
                self.ABCSampAcc =data['ABC_Matrix']
                self.N =data['n_samples']
                self.mlp_neurons =data['neurons']
                self.mlp_act_f =[]
                for i in data['act_f']:
                    self.mlp_act_f.append(actf_dic[i])
                self.nW =data['nw']
                self.nb =data['nb']
                
        self.Ranked_ABCSamoACC = self.ABCSampAcc

    # Function to transform the array (nW,1) containing the weights into the
    # matrix needed for the forward pass        
    def MatrixW(self,W,neurons): 
        MatW = []
        ref=0
        for i in range(len(neurons)-1):
            x = W[ref:ref+(neurons[i+1]*neurons[i])]
            x = x.reshape((neurons[i], neurons[i+1]))
            MatW.append(x)
            ref=ref+(neurons[i+1]*neurons[i])
        return MatW
    
    # Function to transform the vector (nb,1) containing the bias into the matrix
    # needed for the forward pass
    def Matrixb(self,b,neurons): 
        Matb = []
        ref=0
        for i in range(len(neurons)-1):
            x = b[ref:ref+(neurons[i+1])]
            x = x.reshape((1, neurons[i+1]))
            Matb.append(x)
            ref=ref+(neurons[i+1])
        return Matb
    
    # Forward pass with weights matrix "W", bias matrix "b" and Input Data X.
    # Remember that weights and bias from ABC-SS matrix need to be reshaped before
    # the forward pass using the functions above.
    def mlp_fpass(self,X, W, b): 
        output = [X] 
        for count in range(len(self.mlp_neurons)-1):
            z = output[-1] @ W[count] + b[count]
            a = self.mlp_act_f[count](z)[0]
            output.append(a)
        return output[-1]
    
    # Forward pass with weights matrix "W", bias matrix "b", Input Data X and 
    # Target Data Y. In this case, the gradient of an auxiliary loss function with
    # respect to each weight and bias is also calculated. This is used for the 
    # Gradient directed ABC-SS training below (train_gd_loop).
    # Remember that weights and bias from ABC-SS matrix need to be reshaped before
    # the forward pass using the functions above
    def gd_mlp_fpass(self,X, Y, W, b): 
        output = [X]
        delta = [] 
        Grad_W=[]
        Grad_b=[]
        back = list(range(len(self.mlp_neurons)-1)) 
        back.reverse() 
        for count in range(len(self.mlp_neurons)-1):
            z = output[-1] @ W[count] + b[count]
            a = self.mlp_act_f[count](z)[0]
            output.append(a)
        for layer in back:
            a = output[layer+1]
            if layer == back[0]: 
                x = metric_dic['mse'](a, Y)[1] * self.mlp_act_f[layer](a)[1]
                delta.append(x)
            else: 
                x = delta[-1] @ W[layer+1].T * self.mlp_act_f[layer](a)[1] 
                delta.append(x)

            sw = (output[layer].T @ delta[-1]).reshape((-1))
            sb = np.mean(delta[-1], axis = 0, keepdims = True).reshape((-1))
            Grad_W=np.concatenate((sw,Grad_W))
            Grad_b=np.concatenate((sb,Grad_b))

        return output[-1], Grad_W, Grad_b
    
    
# Features and Hyperparameters in train_loop below:   
# X_train and Y_train - Training data with shapes (n_data_points,n_input_features)
#   and (n_data_points,n_output_features) respectively.
# P0 - Conditional probability. See original manuscript for further information.
#   It needs to be 0.1, 0.2, 0.25 or 0.5. It affects the quantification of the 
#   uncertainty and training performance.
# num_sim - Maximum number of simulations allowed. This is to stop the algorithm
#   in case the threshold cannot be reached.
# max_iter - Number of iterations of the training algorithm. If the threshold 
#   has not been reached after the maximum number of simulation levels, then it 
#   starts all over again using the posterior distribution of the parameters as 
#   the new prior distribution.
# start_prop_std - This is the starting point for the standard deviation of the proposal 
#   function for sampling new parameters (gaussian in this case). 
# perc - Decrease rate of the standard deviation in the proposal function 
#   per simulation level.
# threshold - Intermediate epsilon required by the user. In simpler terms, the tolerance 
#   value, or the maximum error that is acceptable for the user.
# metric - metric function, rho in the research article (different metrics can
#   be found in ABC_aux_func, a new ones could be added if necessary).
# save_levels - In case intermediate distributions of the parameters are required to 
#   be stored. This may be useful to understand how the neural network learns step by step.


    def train_loop(self, X_train, Y_train, P0=0.2, num_sim=10, max_iter=1, start_prop_std=0.9, perc=0.5, threshold=0.007, metric='mse', save_levels=False):
        
        self.metric=metric_dic[metric]
        if save_levels==True:
            self.ACC_Interm_SS = []
        
        # Simulation level 0, with the parameters sampled from the prior distribution.
        for i in range(self.N): 
            W = self.ABCSampAcc[i , 0 : self.nW]
            b = self.ABCSampAcc[i , self.nW : self.nW+self.nb]
            MW = self.MatrixW(W,self.mlp_neurons)
            Mb = self.Matrixb(b,self.mlp_neurons)
            Ypred = self. mlp_fpass(X_train,MW,Mb)
            e = self.metric(Ypred, Y_train)[0]
            self.ABCSampAcc[i , self.nW+self.nb] = e
        
        # List to store the percentage of accepted samples for each simulation level
        accept_persim = [] 
        
        # Levels counter
        l = 0
        # Iteration counter
        iteration = 0
        
        print('Training starts')
        
        while iteration < max_iter:
            
            # Standard deviation in proposal function for sampling 
            prop_std = start_prop_std
            
            for ii in range(num_sim):
            
                # Rank Matrix by metric
                SampSort = self.ABCSampAcc[self.ABCSampAcc[:,-1].argsort()]
                if save_levels==True:
                    self.ACC_Interm_SS.append(SampSort)
                
                # Find the metric for the worst valid prediction
                per_ind = np.rint(P0*self.N).astype(int)  
                intr_eps = (SampSort[per_ind, -1 ]) 
                
                # Standard deviation in proposal function for sampling 
                prop_std = prop_std * perc
                
                # Create an empty matrix for the subsets and the seeds matrix
                seeds = SampSort[0 : per_ind , :]
                SubSetsABC = np.zeros(shape = (self.N - per_ind, self.nW+self.nb+1 ))
                
                # Counter used in the subsets.
                ll = 0 
                # Acceptance accumulator is initialised, so the percentage of 
                # accepted samples from the seed (accept_persim) can be calculated 
                accept_m = 0 
                for k in range (0, per_ind):
                    
                    # Select the parameters to be used as seeds 
                    prev_W = seeds[k , 0 : self.nW ]
                    prev_b = seeds[k , self.nW : self.nW+self.nb ]
                    prev_metric = seeds[k , -1 ]
                    
                    # Create the subsets with the sample generated from the seeds 
                    # if accepted or the seeds themselves if the new sample was not accepted.
                    for j in range(1 , np.rint(1/P0).astype(int)):
                        New_W = prop_std*np.random.randn(self.nW) + prev_W
                        New_b = prop_std*np.random.randn(self.nb) + prev_b
                        
                        MW = self.MatrixW(New_W,self.mlp_neurons)
                        Mb = self.Matrixb(New_b,self.mlp_neurons)
                        
                        Ypred = self. mlp_fpass(X_train,MW,Mb)
                        
                        e = self.metric(Ypred, Y_train)[0]
                            
                        
                        if (e <= intr_eps):
                            accept_m = accept_m + 1  
                            
                            SubSetsABC[ll , 0:self.nW] = New_W[:]
                            SubSetsABC[ll , self.nW:self.nW+self.nb] = New_b[:]
                            
                            SubSetsABC[ll , -1] = e
                            
                            prev_W = New_W
                            prev_b = New_b
                            prev_metric = e
                            
                        else:
                            SubSetsABC[ll , 0:self.nW] = prev_W[:]
                            SubSetsABC[ll , self.nW:self.nW+self.nb] = prev_b[:]
                                
                            SubSetsABC[ll , -1] = prev_metric
                        
                        ll = ll + 1
                
                # Join the Subsets created and the seeds
                self.ABCSampAcc  = np.concatenate((seeds, SubSetsABC), axis = 0)                
                
                # Calculation of the percentage of accepted samples per simulation
                accept_persim.append(float(accept_m)/float(self.N))
                l = l+1
                print('End of level:', l, '  Intermediate epsilon:', intr_eps)
                if intr_eps < threshold:
                    break
            
            iteration=iteration+1
            print('End of iteration:', iteration)
            if intr_eps < threshold:
                iteration = max_iter+1
                print('End of Training')
        
        self.Ranked_ABCSamoACC = self.ABCSampAcc[self.ABCSampAcc[:,-1].argsort()]
        if save_levels==True:
            self.ACC_Interm_SS.append(self.Ranked_ABCSamoACC)
            self.ACC_Interm_SS=np.stack(self.ACC_Interm_SS)
            
        return self.Ranked_ABCSamoACC
    
# Features and Hyperparameters in train_levels below:     
# X_train and Y_train - Training data with shapes (n_data_points,n_input_features)
#   and (n_data_points,n_output_features).
# P0 - Conditional probability. See original manuscript for further information.
#   It needs to be 0.1, 0.2, 0.25 or 0.5. It affects the quantification of the 
#   uncertainty and the training performance.
# simlevls - Number of simulations to be carried out.
# metric - metric function, rho in the research article (the range of metrics can
#   be found in ABC_aux_func).
# save_levels - In case intermediate distrubutions of the parameters are required to 
#   be stored. This may be useful to understand how the neural network learns step by step.
    
    def train_levels(self, X_train, Y_train, P0=0.2, simlevls=10, save_levels=False, metric='mse'):
   
        self.metric=metric_dic[metric]
        if save_levels==True:
            self.ACC_Interm_SS = []
        
        # Simulation level 0, with the parameters sampled from the prior distribution.
        for i in range(self.N): 
            W = self.ABCSampAcc[i , 0 : self.nW]
            b = self.ABCSampAcc[i , self.nW : self.nW+self.nb]
            MW = self.MatrixW(W,self.mlp_neurons)
            Mb = self.Matrixb(b,self.mlp_neurons)
            Ypred = self. mlp_fpass(X_train,MW,Mb)
            e = self.metric(Ypred, Y_train)[0]
            self.ABCSampAcc[i , self.nW+self.nb] = e

        # List to store the acceptation rate (metric) for each simulation level,
        # based on the percentile of accepted samples in the previous simulation
        acc_intr_eps = np.zeros(shape = (simlevls))
        # List to store the percentage of accepted samples for each simulation level.
        accept_persim = [] 

        
        start_std=(simlevls+1)*0.1
        
        print('Training starts')
        
        for l in range(simlevls):
            
            # Standard deviation in proposal function for sampling 
            prop_std = start_std - (l+1)*0.1
            
            # Rank Matrix by metric
            SampSort = self.ABCSampAcc[self.ABCSampAcc[:,-1].argsort()]
            if save_levels==True:
                    self.ACC_Interm_SS.append(SampSort)
            
            # Find the metric for the worst valid prediction
            per_ind = np.rint(P0*self.N).astype(int)  
            intr_eps = (SampSort[per_ind, -1 ])
            acc_intr_eps[l] =  intr_eps
            
            
            # Create an empty matrix for the subsets and the seeds matrix
            seeds = SampSort[0 : per_ind , :]
            SubSetsABC = np.zeros(shape = (self.N - per_ind, self.nW+self.nb+1 ))
            
            # Counter used in the subsets.
            ll = 0 
            # Acceptance accumulator is initialised, so the percentage of accepted
            # samples from the seed (accept_persim) can be calculated for each simulation level
            accept_m = 0 
            for k in range (0, per_ind):
                
                # Select the parameters to be used as seeds
                prev_W = seeds[k , 0 : self.nW ]
                prev_b = seeds[k , self.nW : self.nW+self.nb ]
                prev_metric = seeds[k , -1 ]
                
                # Create the subsets with the sample generated from the seeds 
                # if accepted or the seeds themselves if the new sample was not accepted.
                for j in range(1 , np.rint(1/P0).astype(int)):
                    New_W = prop_std*np.random.randn(self.nW) + prev_W
                    New_b = prop_std*np.random.randn(self.nb) + prev_b
                    
                    MW = self.MatrixW(New_W,self.mlp_neurons)
                    Mb = self.Matrixb(New_b,self.mlp_neurons)
                    
                    Ypred = self. mlp_fpass(X_train,MW,Mb)
                    
                    e = self.metric(Ypred, Y_train)[0]
                        
                    
                    if (e <= intr_eps):
                        accept_m = accept_m + 1  
                        
                        SubSetsABC[ll , 0:self.nW] = New_W[:]
                        SubSetsABC[ll , self.nW:self.nW+self.nb] = New_b[:]
                        
                        SubSetsABC[ll , -1] = e
                        
                        prev_W = New_W
                        prev_b = New_b
                        prev_metric = e
                        
                    else:
                        SubSetsABC[ll , 0:self.nW] = prev_W[:]
                        SubSetsABC[ll , self.nW:self.nW+self.nb] = prev_b[:]
                            
                        SubSetsABC[ll , -1] = prev_metric
                    
                    ll = ll + 1
            
            # Join the Subsets created and the seeds
            self.ABCSampAcc  = np.concatenate((seeds, SubSetsABC), axis = 0)
            
            
            # Calculation of the percentage of accepted samples per simulation
            accept_persim.append(float(accept_m)/float(self.N))
            print('End of level:', l+1, '  Intermediate epsilon:', intr_eps)
        
        print('End of Training')
        self.Ranked_ABCSamoACC = self.ABCSampAcc[self.ABCSampAcc[:,-1].argsort()]
        if save_levels==True:
            self.ACC_Interm_SS.append(self.Ranked_ABCSamoACC)
            self.ACC_Interm_SS=np.stack(self.ACC_Interm_SS)
        return self.Ranked_ABCSamoACC
    
    
# Features and Hyperparameters in train_gd_loop below:   
# X_train and Y_train - Training data with shapes (n_data_points,n_input_features)
#   and (n_data_points,n_output_features) respectively.
# P0 - Conditional probability. See original manuscript for further information.
#   It needs to be 0.1, 0.2, 0.25 or 0.5. It affects the quantification of the 
#   uncertainty and training performance.
# num_sim - Maximum number of simulations allowed. This is to stop the algorithm
#   in case the threshold cannot be reached.
# max_iter - Number of iterations of the training algorithm. If the threshold 
#   has not been reached after the maximum number of simulation levels, then it 
#   starts all over again using the posterior distribution of the parameters as 
#   the new prior distribution.
# start_prop_std - This is the starting point for the standard deviation of the proposal 
#   function for sampling new parameters (gaussian in this case). 
# perc - Decrease rate of the standard deviation in the proposal function 
#   per simulation level.
# threshold - Intermediate epsilon required by the user. In simpler terms, the tolerance 
#   value, or the maximum error that is acceptable for the user.
# metric - metric function, rho in the research article (different metrics can
#   be found in ABC_aux_func, a new ones could be added if necessary).
# save_levels - In case intermediate distributions of the parameters are required to 
#   be stored. This may be useful to understand how the neural network learns step by step.

    def train_gd_loop(self, X_train, Y_train, P0=0.2, num_sim=10, max_iter=1, start_prop_std=0.9, perc=0.5, threshold=0.007, metric='mse', save_levels=False):
        
        self.metric=metric_dic[metric]
        Grad_Matrix=np.zeros(shape = (self.N , self.nW+self.nb+1))
        if save_levels==True:
            self.ACC_Interm_SS = []
        
        # Simulation level 0, with the parameters sampled from the prior distrubution.
        for i in range(self.N): 
            W = self.ABCSampAcc[i , 0 : self.nW]
            b = self.ABCSampAcc[i , self.nW : self.nW+self.nb]
            MW = self.MatrixW(W,self.mlp_neurons)
            Mb = self.Matrixb(b,self.mlp_neurons)
            Ypred,Grad_Matrix[i,0:self.nW],Grad_Matrix[i,self.nW:self.nW+self.nb] = self.gd_mlp_fpass(X_train, Y_train, MW, Mb)
            e = self.metric(Ypred, Y_train)[0]
            self.ABCSampAcc[i , self.nW+self.nb] = Grad_Matrix[i,self.nW+self.nb] = e

        # List to store the percentage of accepted samples for each simulation level
        accept_persim = [] 
        
        # Levels counter
        l = 0
        # Iteration counter
        iteration = 0
        
        print('Training starts')
        
        while iteration < max_iter:
            
            # Standard deviation in proposal function for sampling 
            prop_std = start_prop_std
            
            for ii in range(num_sim):
            
                # Rank Matrix by metric
                SampSort = self.ABCSampAcc[self.ABCSampAcc[:,-1].argsort()]
                Sortgrad = Grad_Matrix[Grad_Matrix[:,-1].argsort()]
                if save_levels==True:
                    self.ACC_Interm_SS.append(SampSort)
                
                # Find the metric for the worst valid prediction
                per_ind = np.rint(P0*self.N).astype(int)  
                intr_eps = (SampSort[per_ind, -1 ])                
                
                # Standard deviation in proposal function for sampling 
                prop_std = prop_std * perc
                
                # Create an empty matrix for the subsets and the seeds matrix
                seeds = SampSort[0 : per_ind , :]
                seedsgrad = Sortgrad[0 : per_ind , :]
                SubSetsABC = np.zeros(shape = (self.N - per_ind, self.nW+self.nb+1 ))
                SubSetsgrad = np.zeros(shape = (self.N - per_ind, self.nW+self.nb+1 ))
                
                # Counter used in the subsets 
                ll = 0 
                # Acceptance accumulator is initialised, so the percentage of accepted
                # samples is calculated for each simulation level
                accept_m = 0 
                for k in range (0, per_ind):
                    
                    # Select the parameters to be used as seeds (row with all W and b)
                    prev_W = seeds[k , 0 : self.nW ]
                    prev_b = seeds[k , self.nW : self.nW+self.nb ]
                    prev_metric = seeds[k , -1 ]
                    prev_grad_W = seedsgrad[k , 0 : self.nW ]
                    prev_grad_b = seedsgrad[k , self.nW : self.nW+self.nb ]

                    
                    # Create the subsets with the sample generated from the seeds 
                    # if accepted or the seeds themselves if the new sample was not accepted.
                    for j in range(1 , np.rint(1/P0).astype(int)):
                        New_W = np.zeros(shape = (self.nW))
                        New_b = np.zeros(shape = (self.nb))
                        for i in range(self.nW):
                            if prev_grad_W[i]==0.0:
                                New_W[i] = prop_std*np.random.randn() + prev_W[i]
                            else:
                                New_W[i] = prev_W[i] - ((prev_grad_W[i])/np.absolute(prev_grad_W[i]))*prop_std*np.absolute(np.random.randn())
                        for r in range(self.nb):
                            if prev_grad_b[r]==0.0:
                                New_b[r] = prop_std*np.random.randn() + prev_b[r]
                            else:
                                New_b[r] = prev_b[r] - ((prev_grad_b[r])/np.absolute(prev_grad_b[r]))*prop_std*np.absolute(np.random.randn())

                        
                        MW = self.MatrixW(New_W,self.mlp_neurons)
                        Mb = self.Matrixb(New_b,self.mlp_neurons)
                        
                        Ypred, New_grad_W, New_grad_b  = self.gd_mlp_fpass(X_train, Y_train, MW, Mb)
                        
                        e = self.metric(Ypred, Y_train)[0]
                            
                        
                        if (e <= intr_eps):
                            accept_m = accept_m + 1  
                            
                            SubSetsABC[ll , 0:self.nW] = New_W[:]
                            SubSetsABC[ll , self.nW:self.nW+self.nb] = New_b[:]
                            SubSetsgrad[ll , 0:self.nW] = New_grad_W[:]
                            SubSetsgrad[ll , self.nW:self.nW+self.nb] = New_grad_b[:]

                            
                            SubSetsABC[ll , -1] = SubSetsgrad[ll , -1] =  e
                            
                            prev_W = New_W
                            prev_b = New_b
                            prev_metric = e
                            prev_grad_W = New_grad_W
                            prev_grad_b = New_grad_b

                            
                        else:
                            SubSetsABC[ll , 0:self.nW] = prev_W[:]
                            SubSetsABC[ll , self.nW:self.nW+self.nb] = prev_b[:]
                            SubSetsgrad[ll , 0:self.nW] = prev_grad_W[:]
                            SubSetsgrad[ll , self.nW:self.nW+self.nb] = prev_grad_b[:]

                                
                            SubSetsABC[ll , -1] = SubSetsgrad[ll , -1] =  prev_metric
                        
                        ll = ll + 1
                
                # Join the Subsets created and the seeds
                self.ABCSampAcc  = np.concatenate((seeds, SubSetsABC), axis = 0)
                Grad_Matrix  = np.concatenate((seedsgrad, SubSetsgrad), axis = 0)
                
                
                # Calculation of the percentage of accepted samples per simulation
                accept_persim.append(float(accept_m)/float(self.N))
                l = l+1
                print('End of level:', l, '  Intermediate epsilon:', intr_eps)
                if intr_eps < threshold:
                    break
            
            iteration=iteration+1
            print('End of iteration:', iteration)
            if intr_eps < threshold:
                iteration = max_iter+1
                
        print('End of training')
        self.Ranked_ABCSamoACC = self.ABCSampAcc[self.ABCSampAcc[:,-1].argsort()]
        if save_levels==True:
            self.ACC_Interm_SS.append(self.Ranked_ABCSamoACC)
            self.ACC_Interm_SS=np.stack(self.ACC_Interm_SS)
        return self.Ranked_ABCSamoACC
    
# Function to save the weights and bias ABC-SS matrix after they have been trained.
# They can then be loaded and used directly to make predictions with 
# the 'predict' function below. The 'name' feature has to be entered as a string
# such as 'file_name'. A .npz file will be created automatically with that name.   
    def save_weights (self,name):
        ABC_Matrix=self.Ranked_ABCSamoACC
        n_samples=self.N
        nw=self.nW
        nb=self.nb
        act_f=self.act_f_list
        neurons=self.mlp_neurons
        np.savez(name, ABC_Matrix=ABC_Matrix,nw=nw,nb=nb,n_samples=n_samples,act_f=act_f,neurons=neurons)

        
# Function for making predictions once the weights and bias have been trained, or 
# they have been loaded. X must be shaped (n_data_points,n_input_features)     
    def predict (self, X):
        Output=np.zeros(shape=(self.mlp_neurons[-1],len(X[:,0]), self.N))
        out_median=np.zeros(shape=(len(X[:,0]), self.mlp_neurons[-1]))
        out_p95=np.zeros(shape=(len(X[:,0]), self.mlp_neurons[-1]))
        out_p5=np.zeros(shape=(len(X[:,0]), self.mlp_neurons[-1]))
        out_mean=np.zeros(shape=(len(X[:,0]), self.mlp_neurons[-1]))
        for i in range(self.N):
            W=self.Ranked_ABCSamoACC[i,0:self.nW]
            b=self.Ranked_ABCSamoACC[i,self.nW:self.nW+self.nb]
            MW=self.MatrixW(W,self.mlp_neurons)
            Mb=self.Matrixb(b,self.mlp_neurons)
            out_temp=self.mlp_fpass(X, MW, Mb)
            for j in range(self.mlp_neurons[-1]):
                Output[j,:,i]=out_temp[:,j]
        for n in range(self.mlp_neurons[-1]):
            out_median[:,n]=np.percentile(Output[n,:,:],50, axis=1)
            out_p95[:,n]=np.percentile(Output[n,:,:],95, axis=1)
            out_p5[:,n]=np.percentile(Output[n,:,:],5, axis=1)
            out_mean[:,n]=np.mean(Output[n,:,:], axis=1)
        return Output, out_median, out_p95, out_p5, out_mean
            
# Function for plotting the PDF of a prediction. It is coded with python 3.7,
# newer versions might give errors. 
# results - tuple generated by the 'predict' function above
# Y - Target data array with shape (n_data_points,n_output_features) 
# data_point - specific prediction to be ploted within the output array
    def plot_pred_pdf (self, results, Y, data_point, kde_color='lightgrey'):
        for i in range(self.mlp_neurons[-1]):
            fig, ax = plt.subplots(figsize=(1.61803398*4 , 4), dpi = 600, nrows = 1, ncols = 1)
            image=sns.kdeplot(data=results[0][i,data_point,:],ax=ax, color = kde_color, fill=True, alpha=0.45, linewidth=0.5, )
            image.tick_params(labelsize=15)
            sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})
            sns.despine(top=False,left=False, bottom=False, right=False)
            image.set_xlabel(None)
            image.set_ylabel(None)
            image.set(yticks=[])
            plt.axvline(Y[data_point,i], linestyle='dashed', color='black', linewidth=3, label='Test data')
            median=results[1][:,i]
            plt.axvline(median[data_point], color='black', linewidth=3, label='Prediction (median)')
            plt.show()
            return
        
# Function for plotting the PDF of a weight or bias. It is coded with python 3.7,
# newer versions might give errors.
# weight_num - Number of weight to be plotted, from 0 to nW+nb, 
# where nW is the total number of weights and nb the total number of bias.          
    def plot_weight_pdf(self,weight_num,kde_color='lightgrey'):
        fig, ax = plt.subplots(figsize=(1.61803398*4 , 4), dpi = 600, nrows = 1, ncols = 1)
        image=sns.kdeplot(data=self.Ranked_ABCSamoACC[:,weight_num],ax=ax, color = kde_color, fill=True, alpha=0.45, linewidth=0.5, )
        image.tick_params(labelsize=15)
        sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})
        sns.despine(top=False,left=False, bottom=False, right=False)
        image.set_xlabel(None)
        image.set_ylabel(None)
        image.set(yticks=[])
        plt.show()
        

      
    
    

    
          
