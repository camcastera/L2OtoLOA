#%%
from utils.essential_imports import *
## Functions to minimize
import utils.funcs_and_grads as objectives
## Baseline algorithms
import utils.true_algos as algos
## LOA model
import utils.models as models
## L2O baselines
from utils.coord_blackbox_lstm import CoordBlackboxLSTM
from utils.rnnprop import RNNprop
######################################################

import time
exec_times = {}

torch.manual_seed(46845712318) #Use a seed because dataset is generated at random


########################################################################


#########################################################################





## Dimension of the problem
P = 10
#%%
############ Create function to minimize ############
#generate 2 clusters, first by setting the labels
N_samples = 1
M = 50 #Number of points to classify
b = torch.ones(N_samples, M)
b[:, int(M/2):] *= -1 #fill half of b with -1 to get 2 classes
#choose parameter for generating the points in the cluster
A = 1/torch.sqrt(torch.tensor(P)) * torch.randn(N_samples, M, P)  #Samples matrices of size [m x n] from uni distrib
mu1 = torch.randn(N_samples, 1, 1) - 0.1
mu2 = torch.randn(N_samples, 1, 1) + 0.1
#Shift points from each class (to get means mu1 and mu2)
A[:,:int(M/2)] += mu1
A[:,int(M/2):] += mu2
# IMPORTANT: because it is affine regression, the first column of A should always contain ones (see https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf p120)
A[:, :, 0] = 1.

f = objectives.logistic
xstar = None
lambdamin = 1/M #smallest eigenvalue (comes from the regularization)
lambdamax = 1/2 * torch.sum(A**2, dim=(-1, -2)) + 1/M
func = objectives.function_to_minimize(f=f, A=A, b=b, xstar=xstar, lambdamax=lambdamax)
#%%
########## Prepare initialization #########
xm1 = torch.randn(N_samples, P)
gradxm1 = func.evalgradf(xm1.clone().requires_grad_(True))
gamma_1ststep = 1/(2*func.lambdamax)
if gamma_1ststep.shape[0]>1:
    gamma_1ststep = gamma_1ststep.unsqueeze(-1) #add dim for broadcasting to size P
x0 = xm1 - gamma_1ststep * gradxm1
gradx0 = func.evalgradf(x0.clone().requires_grad_(True))
dm1_0 = x0.clone().detach() - xm1.clone().detach()
DG = gradx0 - gradxm1
BB_stepsize = torch.sum((DG * dm1_0), axis=-1) / torch.sum( (DG * DG), axis=-1)
#Initialize BFGS matrix using the Barzilai-Borwein step-size
first_guess = 0.8 * BB_stepsize[:, None, None] * torch.eye(P)

# %%
########### Initialize the LOA algorithm ###########
gamma = 1.
layers_out_dim_factors = [2]  #only a single linear layer
safe_init = 'BFGS_like'

LAO_BFGS = models.LOA_BFGS_Model(P=P, layers_out_dim_factors=layers_out_dim_factors, gamma=gamma)

LAO_BFGS.load_state_dict(torch.load("Data/pretrained_LOA_BFGS_Model.npy"))



#%%
########## Run the algorithm #########
niter_max = P + 300
Wolfe_LS = True

LAO_BFGS.reset_mat(n_traj=N_samples, first_guess=first_guess)
x, gradm1, grad, dm1 = x0.clone(), gradxm1.clone(), gradx0.clone(), dm1_0.clone() 
list_values_LOA = torch.zeros(N_samples, niter_max + 1)
start = time.time()
for iteration in range(niter_max):
    current_value = func.evalf(x)  #Compute current value for this func and point
    grad = func.evalgradf(x.clone().detach().squeeze().requires_grad_(True))
    list_values_LOA[:, iteration] = current_value
    previous_stepsize = torch.clone(gamma) if torch.is_tensor(gamma) else gamma
    #set current step-size
    stepsize = gamma #might be overwritten by line-search
    next_step = LAO_BFGS(grad, gradm1, dm1, previous_gamma=previous_stepsize)  #Predict all trajectories at once.
    ## Optionally do line-search
    if Wolfe_LS:
        #Start LS, it should not be part of the graph
        with torch.no_grad():
            c1 = 1e-2  #Params of the linesearch
            step = next_step.clone().detach()
            stepXgrad = torch.sum(step * grad, dim=-1)
            # Do a line search for each element
            n_LS = 0 ; 
            stepsize = torch.ones(N_samples, 1)  #If line-search, start with unit step-size
            Wolfecrit = torch.ones(N_samples, dtype=bool) #mask for successful LS
            temp_values = current_value.clone()
            while n_LS < 5 and torch.sum(Wolfecrit)>0:  #Try 5 times the line-search
                n_LS += 1
                y = x + stepsize * step  #copy x
                temp_values[Wolfecrit] = func.evalf(y)[Wolfecrit] #Not optimized at all!
                Wolfecrit = (temp_values - current_value - c1 * stepsize.squeeze() * stepXgrad) > 0   #this criterion must be negative to exist the LS loop
                stepsize[Wolfecrit] *= 0.1 #Update stepsizes for which it failed
        #Once a new step-size is chosen, it should be part of the graph
        #########################
    #Store step-size for next iter (very important)
    previous_stepsize = stepsize  #store step-size for next iter
    ##update all variables
    dm1 = stepsize * next_step #step between xk and xk+1
    gradm1 = grad.clone().detach() #current grad is stored for next iter
    x = x + dm1 # compute xk+1
list_values_LOA[:, -1] = func.evalf(x) #compute final value
list_values_LOA = list_values_LOA.clone().detach()
exec_times['LOA_BFGS'] = (time.time() - start) / niter_max


#%%
########## Compute baselines ##########
start = time.time()
list_values_BFGS = algos.BFGS(x0, xm1, func, gamma_BFGS=gamma, first_guess=None, niter_max=niter_max, BB_init=True, Wolfe_LS=Wolfe_LS)
exec_times['BFGS'] = (time.time() - start) / niter_max

niter_Newton = 20
start = time.time()
list_values_Newton = algos.Newton(x0, func, gamma_Newton=gamma, niter_max=niter_Newton, Wolfe_LS=Wolfe_LS)
exec_times['Newton'] = (time.time() - start) / niter_Newton


start = time.time()
list_values_GD = algos.GD(x0, func, gamma=1.99*gamma_1ststep, niter_max=niter_max)
exec_times['GD'] = (time.time() - start) / niter_max

start = time.time()
list_values_Nesterov = algos.Nesterov(x0, func, gamma=1.99*gamma_1ststep, alpha=3.01, niter_max=niter_max)
exec_times['Nesterov'] = (time.time() - start) / niter_max

start = time.time()
list_values_HB = algos.HeavyBall(x0, func, gamma=1.99*gamma_1ststep, alpha=2*torch.sqrt(torch.tensor(lambdamin)), niter_max=niter_max)
exec_times['HB'] = (time.time() - start) / niter_max

######################### L2O BASELINES ####################


#%%

L2ORNN_optim = RNNprop(
         input_size  = 2, #Input size of the Network
         output_size = 1, #Output size of the LSTM
         hidden_size = 20, #Hidden size of the LSTM
         layers = 2, #Number of LSTM layers
         beta1 = 0.95, #Parameter of the RNNprop features
         beta2 = 0.95  #Parameter of the RNNprop features
        )

L2ORNN_optim.load_state_dict(torch.load('Data/my_RNNprop.pth')) #Load pretrained model
L2ORNN_optim.training = False
#%%

L2ORNN_optim.reset_state(P, step_size=1e-1)

x = x0.clone().detach()
list_values_RNNprop = torch.zeros(N_samples, niter_max + 1)
start = time.time()
for iteration in range(niter_max):
    current_value = func.evalf(x)  #Compute current value for this func and point
    grad = func.evalgradf(x.clone().detach().squeeze().requires_grad_(True))
    list_values_RNNprop[:, iteration] = current_value
    #Process each coordinate in the LSTM in parallel
    #grad = grad.unsqueeze(-1)
    next_step = L2ORNN_optim(grad)
    with torch.no_grad():
        x = x + next_step.squeeze(-1)

list_values_RNNprop[:, -1] = func.evalf(x) #compute final value
exec_times['RNNprop'] = (time.time() - start) / niter_max



## RNNprop trained as in the original paper ##


L2ORNN_optim = RNNprop(
         input_size  = 2, #Input size of the Network
         output_size = 1, #Output size of the LSTM
         hidden_size = 20, #Hidden size of the LSTM
         layers = 2, #Number of LSTM layers
         beta1 = 0.95, #Parameter of the RNNprop features
         beta2 = 0.95  #Parameter of the RNNprop features
        )

L2ORNN_optim.load_state_dict(torch.load('Data/Baseline_RNNprop_0_03.pth')) #Load pretrained model
L2ORNN_optim.training = False
#%%

L2ORNN_optim.reset_state(P, step_size=1e-1)

x = x0.clone().detach()
list_values_baseline_RNNprop = torch.zeros(N_samples, niter_max + 1)
start = time.time()
for iteration in range(niter_max):
    current_value = func.evalf(x)  #Compute current value for this func and point
    grad = func.evalgradf(x.clone().detach().squeeze().requires_grad_(True))
    list_values_baseline_RNNprop[:, iteration] = current_value
    #Process each coordinate in the LSTM in parallel
    #grad = grad.unsqueeze(-1)
    next_step = L2ORNN_optim(grad)
    with torch.no_grad():
        x = x + next_step.squeeze(-1)

list_values_baseline_RNNprop[:, -1] = func.evalf(x) #compute final value
exec_times['Baseline_RNNprop'] = (time.time() - start) / niter_max


## LLGD from the same repo with same training setting as RNNprop ##
#%%

LLGD_optim = CoordBlackboxLSTM(
         input_size  = 1, #Input size of the Network
         output_size = 1, #Output size of the LSTM
         hidden_size = 20, #Hidden size of the LSTM
         layers = 2, #Number of LSTM layers
        )

LLGD_optim.load_state_dict(torch.load('Data/my_CoordBlackboxLSTM.pth')) #Load pretrained model
LLGD_optim.training = False

LLGD_optim.reset_state(P, step_size=1e-1)

x = x0.clone().detach()
list_values_LLGD = torch.zeros(N_samples, niter_max + 1)
start = time.time()
for iteration in range(niter_max):
    current_value = func.evalf(x)  #Compute current value for this func and point
    grad = func.evalgradf(x.clone().detach().squeeze().requires_grad_(True))
    list_values_LLGD[:, iteration] = current_value
    #Process each coordinate in the LSTM in parallel
    #grad = grad.unsqueeze(-1)
    next_step = LLGD_optim(grad)
    with torch.no_grad():
        x = x + next_step.squeeze(-1)

list_values_LLGD[:, -1] = func.evalf(x) #compute final value
exec_times['LLGD'] = (time.time() - start) / niter_max

#%%

### Run the same algo but trained in the framework from the literatture 

LLGD_optim = CoordBlackboxLSTM(
         input_size  = 1, #Input size of the Network
         output_size = 1, #Output size of the LSTM
         hidden_size = 20, #Hidden size of the LSTM
         layers = 2, #Number of LSTM layers
        )

LLGD_optim.load_state_dict(torch.load('Data/Baseline_CoordBlackboxLSTM_initlr_0_01.pth')) #Load pretrained model
LLGD_optim.training = False

LLGD_optim.reset_state(P, step_size=1e-1)

x = x0.clone().detach()
list_values_baseline_LLGD = torch.zeros(N_samples, niter_max + 1)
start = time.time()
for iteration in range(niter_max):
    current_value = func.evalf(x)  #Compute current value for this func and point
    grad = func.evalgradf(x.clone().detach().squeeze().requires_grad_(True))
    list_values_baseline_LLGD[:, iteration] = current_value
    #Process each coordinate in the LSTM in parallel
    #grad = grad.unsqueeze(-1)
    next_step = LLGD_optim(grad)
    with torch.no_grad():
        x = x + next_step.squeeze(-1)

list_values_baseline_LLGD[:, -1] = func.evalf(x) #compute final value
exec_times['baseline_LLGD'] = (time.time() - start) / niter_max

#%%

############################################################
relative_sorted_exec_times = {k: v / exec_times['GD'] for k, v in sorted(exec_times.items(), key=lambda x: x[1])}
for key in relative_sorted_exec_times.keys():
    print(key, relative_sorted_exec_times[key])


#%%
########### Plot the result ############

def plot_area(xarray, values, ax, label, color=None, ls='-', zorder=5): 
    below, _ = torch.min(values, dim=0) 
    above, _ = torch.max(values, dim=0)
    ax.plot(xarray, torch.median(values, dim=0)[0], lw=3.5, color=color, zorder=zorder, ls=ls, label=label)
    ax.fill_between(xarray, below, above, lw=3.5, color=color, zorder=zorder, alpha=0.2)
    pass


fig, ax = plt.subplots(figsize=(6,5))
## Newton and QN
fstar = torch.min(list_values_Newton, dim=-1)[0].unsqueeze(-1)
initial_gap = func.evalf(x0).unsqueeze(-1) - fstar



plot_area(np.arange(niter_max+1), (list_values_LOA-fstar)/initial_gap, ax=ax, color='dodgerblue', label='LOA BFGS', zorder=10)
plot_area(np.arange(niter_max+1), (list_values_BFGS-fstar)/initial_gap, ax=ax, color='magenta', label='vanilla BFGS', ls='--', zorder=4)
plot_area(np.arange(niter_Newton+1), (list_values_Newton-fstar)/initial_gap, ax=ax, color='black', label='Newton', ls='--', zorder=4)
## First order
plot_area(np.arange(niter_max+1), (list_values_GD-fstar)/initial_gap, ax=ax, color='limegreen', label='Gradient Descent', ls='--', zorder=4)
plot_area(np.arange(niter_max+1), (list_values_Nesterov-fstar)/initial_gap, ax=ax, color='red', label="Nesterov's Accelerated Gradient", ls='--', zorder=4)
plot_area(np.arange(niter_max+1), (list_values_HB-fstar)/initial_gap, ax=ax, color='orange', label="Heavy-Ball", ls='--', zorder=4)
# L2O baselines
plot_area(np.arange(niter_max+1), (list_values_LLGD-fstar)/initial_gap, ax=ax, color='purple', label="LLGD our training setting", zorder=3)
plot_area(np.arange(niter_max+1), (list_values_baseline_LLGD-fstar)/initial_gap, ax=ax, color='purple', label="LLGD", zorder=3, ls='--')
plot_area(np.arange(niter_max+1), (list_values_RNNprop-fstar)/initial_gap, ax=ax, color='mediumorchid', label="RNNprop our training setting", zorder=3)
plot_area(np.arange(niter_max+1), (list_values_baseline_RNNprop-fstar)/initial_gap, ax=ax, color='mediumorchid', label="RNNprop", ls ='--', zorder=3)

# parameters of the plot
ax.set_xlim(0, 310)

ax.set_xlabel(r'iteration $k$')
#ax.set_ylabel(r'relative suboptimality $\frac{f(x_k) - f^\star}{f(x_0)-f^\star}$',fontsize=14)
ax.set_title(r'Logistic regression on synthetic dataset', fontsize=14)
ax.set_yscale('log')
if ax.get_ylim()[0]<1e-12:
    ax.set_ylim(ymin=1e-12)

fig.tight_layout()

fig.savefig('Figures/logistic_synthetic.pdf')
fig.show()
# %%

# ## Export legend ##

# fig2, ax2 = plt.subplots(figsize=(14,1))
# ax2.axis('off')
# ax2.legend(*ax.get_legend_handles_labels(), ncol=5, handlelength=3)
# fig2.savefig('Figures/EXP_legend_horizontal.pdf')


# fig2, ax2 = plt.subplots(figsize=(8,8))
# ax2.axis('off')
# ax2.legend(*ax.get_legend_handles_labels(), ncol=1, handlelength=3)
# fig2.savefig('Figures/EXP_legend_vertical.pdf')
# %%
