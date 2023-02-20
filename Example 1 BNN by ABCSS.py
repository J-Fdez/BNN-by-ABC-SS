
"""
@author: Juan Fernandez Salas - iPMLab
"""
import numpy as np
from BNN_ABCSS import BNN_ABCSS as abc
import matplotlib.pyplot as plt

###################### IMPORTANT ########################

# Python version 3.7 was used, more recent versions might cause problems when plotting
# some figures with the seaborn package. To solve this, either use version 3.7  
# or modify the source code in file BNN_ABCSS to suit your python version and seaborn package


######################## DATA ########################
N_Inputs = 200
noise = 0.1
X = np.linspace(-3, 3, num=N_Inputs)
Y = np.cos(X) + np.random.normal(0, noise, size=N_Inputs)
X = X.astype(np.float32).reshape((N_Inputs, 1))
Y = Y.astype(np.float32).reshape((N_Inputs, 1))
Y_true=np.cos(X)

###################### README ########################

# This example aims at illustrating how artificial neurals networks are trained
# using Approximate Bayesian Computation by SubSet Simulation. In this case, the 
# neural network  will fit some synthetic data created using a cosine
# function with noise, as shown above.  

# For more info about the hyperparameters and other features, such as saving or 
# loading weights, check the source code BNN_ABCSS.

# I hope you find BNN by ABC-SS useful, and please share your ideas if you find new ways 
# of improving its performance.

# For more details, refer to "Uncertainty quantification in Neural Networks by 
# Approximate Bayesian Computation: Application to fatigue in composite materials"
# https://doi.org/10.1016/j.engappai.2021.104511

######################################################


# In this example we are defining the number of simulation levels, instead of 
# the epsilon (threshold) desired.
Simulation_levels=6 
                    

# Build a very simple model with only 2 hidden neurons using class BNN_ABCSS.
model=abc(N=5000, mlp_neurons=[1,2,1], mlp_act_f =['tanh', 'tanh'])

# Train the Bayesian neural network with 'train_levels' mode
# Activate 'save_levels=True' so we can monitor the weights on every simulation level.
model.train_levels(X_train=X, Y_train=Y, simlevls=Simulation_levels, save_levels=True)

# Make predictions to verify how the model fit the cosine function
# The 'predict' function will provide us a tuple including a matrix with predictions 
# from all N samples, the median value of the predictions, the 95 precentile,
# the 5 precentile and the average.
results=model.predict(X)


# Plot the prediction on every level of simulation, so we can see how the
# Bayesian neural network learns level by level
Y_Inter = np.zeros(shape = (N_Inputs , model.N))
Y_Int_Mean=np.zeros(shape = (N_Inputs , Simulation_levels + 1)) 

for j in range (Simulation_levels + 1):            
    for i in range(model.N):
        Wtemp = model.ACC_Interm_SS[j, i , 0 : model.nW]
        btemp = model.ACC_Interm_SS[j, i , model.nW : model.nW+model.nb]
        MWtemp = model.MatrixW(Wtemp,model.mlp_neurons)
        Mbtemp = model.Matrixb(btemp,model.mlp_neurons)
        Ypred_temp = model.mlp_fpass(X, MWtemp, Mbtemp)
        Y_Inter[: , i ] = Ypred_temp[:, 0]
    Y_Int_Mean[:, j] = np.mean(Y_Inter, axis = 1)

line_width = [0.33, 0.66, 0.99, 1.33, 1.66, 2.0]
fig, ax =plt.subplots(figsize=(1.61803398*4 , 4), dpi = 600, nrows = 1, ncols = 1)   
ax.plot(X, Y, color = 'black', marker='+', mfc='none', linestyle = 'None')
ax.plot(X, results[1], color = 'darkred', linewidth=2.0)
for i in range(Simulation_levels):
    ax.plot(X,  Y_Int_Mean[:, i], color = "indianred", linewidth=line_width[i], alpha=.6)
ax.fill_between(X[:,0], results[2][:,0], results[3][:,0], facecolor='lightgrey',edgecolor='lightgrey',alpha=.5)
plt.show()

# Plot a prediction at a certain data point to compare it against
# the true target. Grey PDF represents the uncertainty bounds of the prediction,
# dashed line is the target value, and black line is the median of the prediction PDF
model.plot_pred_pdf(results=results, Y=Y_true, data_point=60)


# Plot the PDF of a weight to see its distribution. These PDFs become much more 
# complex when the architecture of the neural network is larger with more neurons and layers.
model.plot_weight_pdf(weight_num=2)

# Save weights and bias 
model.save_weights('Cosine_Weights_Bias_ABCSS')









