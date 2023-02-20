"""
@author: Juan Fernandez Salas
"""
import numpy as np
from BNN_ABCSS import BNN_ABCSS as abc
import matplotlib.pyplot as plt

###################### IMPORTANT ########################

# Python version 3.7 was used, more recent versions might cause problems when plotting
# some figures with the seaborn package. To solve this, either use version 3.7  
# or modify the source code in file BNN_ABCSS to suit your python version and seaborn package

######################## DATA ########################
N_Inputs = 100
noise=1.0
def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * (x)) + epsilon

X = np.linspace(-0.5, 0.5, num=N_Inputs)
Y = f(X, sigma=noise)
X = X.astype(np.float32).reshape((N_Inputs, 1))
Y = Y.astype(np.float32).reshape((N_Inputs, 1))

noise=0.0
Xt = np.linspace(-1.5, 1.5, num=N_Inputs)
Yt = f(Xt, sigma=noise)
Xt = Xt.astype(np.float32).reshape((100, 1))
Yt = Yt.astype(np.float32).reshape((100, 1))


###################### README ########################

# This example aims at illustrating how the uncertainty is quantified when ABC-SS
# is used as the training algorithm. A synthetic training dataset is created using a sine 
# function with noise. Then, another dataset is created, with the same function
# but this time the input domain is wider, so we can evaluate the difference between
# making predictions within the training domain, and outside of it. 

# For more info about the hyperparameters and other features, such as saving or 
# loading weights, check the source code BNN_ABCSS.

# I hope you find BNN by ABC-SS useful, and please share your ideas if you find new ways 
# of improving its performance.

# For more details, refer to "Uncertainty quantification in Neural Networks by 
# Approximate Bayesian Computation: Application to fatigue in composite materials"
# https://doi.org/10.1016/j.engappai.2021.104511

######################################################

# In this example we are defining the tolerance threshold, instead of 
# the number of simulations levels.
# For more precise results, chose a lower threshold, and increase the number of
# num_sim (simulation levels) and/or max_iter (maximum number of iterations) during
# training. 
threshold_epsilon=3.0
                    

# Build a model with only 2 hidden layers of 15 neurons each, using class BNN_ABCSS
model=abc(N=20000, mlp_neurons=[1,15,15,1], mlp_act_f =['tanh', 'tanh', 'linear'])

# Train the Bayesian neural network with 'train_loop' mode. Please note that the
# Illustrative Example 2 in the orignal manuscript uses 'train_levels', so the results
# may be slightly different, specially the quantification of the uncertainty.
model.train_loop(X_train=X, Y_train=Y, P0=0.1, num_sim=10, max_iter=1, start_prop_std=2.0,threshold=threshold_epsilon)

# Make predictions to verify how the model fit the training data
# The 'predict' function provides a tuple with a matrix containing the predictions from 
# all N samples, the median value of those predictions, the 95 precentile,
# the 5 precentile and the average.
results=model.predict(Xt)

# Plot the uncertainty bounds (5 and 95 percentile) in grey, median prediction in red,
# the true sine function in dashed grey line, and the training data with black crosses.
# This example shows how the uncertainty varies when making predictions within the 
# domain of the training data (interpolation) and outside of it (extrapolation).
fig, ax =plt.subplots(figsize=(1.61803398*4 , 4), dpi = 600, nrows = 1, ncols = 1)   
ax.plot(X, Y, color = 'black', marker='+', mfc='none', linestyle = 'None')
ax.plot(Xt, Yt, color = 'grey', mfc='none', linestyle = 'dashed')
ax.plot(Xt, results[1], color = 'darkred', linewidth=2.0)
ax.fill_between(Xt[:,0], results[2][:,0], results[3][:,0], facecolor='lightgrey',edgecolor='lightgrey',alpha=.5)
plt.show()

# Plot the PDF of a weight to see its distribution. These PDFs become much more 
# complex when the architecture of the neural network is larger with more neurons and layers.
model.plot_weight_pdf(weight_num=5)


# Plot the correlation between different weights of the neural network.
import seaborn as sns
data = {"w1":model.Ranked_ABCSamoACC[: , 0], "w2":model.Ranked_ABCSamoACC[: , 16]}
with sns.axes_style('white'):
    sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})
    g=sns.jointplot(data=data, x="w1", y="w2", kind='kde', space=0, fill=True, linewidth=5.0, color="grey")
    g.set_axis_labels(r"$w_{11}^{(1)}$", r"$w_{22}^{(2)}$")
    
# Plot a prediction inside the training data domain 
model.plot_pred_pdf(results=results, Y=Yt, data_point=50)

# Plot a prediction outside the training data domain, to evaluate how the uncertainty
# compares to the previous prediction inside the training data domain. 
model.plot_pred_pdf(results=results, Y=Yt, data_point=5)










