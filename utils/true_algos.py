import warnings
from .essential_imports import *
import torch

def BFGS(x0, xm1, f, gamma_BFGS=1., first_guess=None, niter_max=1000, BB_init=False, Wolfe_LS=False):
    #Initialize first iterations
    x = x0.clone().detach().requires_grad_(True)  #Initialize at x0 with require_grad
    if first_guess is None:
        first_guess = torch.eye(x0.shape[-1])
    elif BB_init == True: #check if first_guess and BB_init are both sets
        warnings.warn('Warning: BB_init overwrites the first_guess argument')
    gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)
    gradgm1 = f.evalgradf(xm1.clone().requires_grad_(True))
    if BB_init==True: #Initialize first_guess with BBstepsize
        dm1 = x0.clone().detach() - xm1.clone().detach()
        DGK = gradg - gradgm1
        BB_stepsize = torch.sum((DGK * dm1), axis=-1) / torch.sum( (DGK * DGK), axis=-1)
        first_guess = 0.8 * BB_stepsize[:, None, None] * torch.eye(x0.shape[-1])
    ## Compute Lipschitz Constant for optimal gamma
    Bk = first_guess.clone().detach() #approx of inverse hessian
    if len(Bk.shape)<3: #if no batch dim add it in front
        Bk = Bk.unsqueeze(0)
    #list_x = [x.clone().detach()] ; list_g = [g0.item()]
    N_samples = x0.shape[0] if len(x0.shape)>0 else 1
    list_g = torch.zeros(N_samples, niter_max + 1)
    list_g[:, 0] = g.clone().detach()
    #list_x = [x] ; list_g = [g.item()]
    # Iterate
    for n in range(niter_max):
        ## Update the matrix Bk
        with torch.no_grad():
            if torch.is_tensor(gradg): #while gradg has not hit numerical precision and has not become 0, update Bk
                #Update Bk (for next update)
                dm1 = x.clone().detach() - xm1 
                DGK = gradg.clone().detach() - gradgm1
                if len(dm1.shape)<2: #add batch dim in front
                    dm1 = dm1.unsqueeze(0)
                if len(DGK.shape)<2: #add batch dim in front
                    DGK = DGK.unsqueeze(0)
                Bdg = torch.bmm(Bk, DGK.unsqueeze(-1)).squeeze(-1) # Bk Delta g
                rho = 1 / torch.sum(dm1 * DGK, dim=-1).unsqueeze(-1) #1/<dk,Delta gk>
                Bk = Bk + (rho * (1. + rho*torch.sum(DGK * Bdg, dim=-1).unsqueeze(-1))).unsqueeze(-1) * dm1[:,:,None] * dm1[:,None,:] - rho.unsqueeze(-1) * Bdg[:,:,None] * dm1[:,None,:] - rho.unsqueeze(-1) * dm1[:,:,None] * Bdg[:,None,:]
                v = - gradg.clone().detach()
                if len(v.shape)<2: #add batch dim in front
                    v = v.unsqueeze(0)
                next_step = torch.bmm(Bk, v.unsqueeze(-1)).squeeze()
        ## Fixed step-size or line-search
            if Wolfe_LS: #Check 1st Wolfe condition
                c1 = 1e-2 ; n_LS = 0
                step = next_step.clone().detach() 
                stepXgrad = torch.sum(step * gradg, dim=-1)
                stepsize = torch.ones(N_samples, 1)  #If line-search, start with unit step-size
                Wolfecrit = torch.ones(N_samples, dtype=bool) #mask for successful LS
                temp_values = g.clone().detach()
                while n_LS < 5 and torch.sum(Wolfecrit)>0: #Try 5 times the line-search
                    n_LS += 1
                    y = x.clone().detach() + stepsize * step #Try this step-size
                    temp_values[Wolfecrit] = f.evalf(y)[Wolfecrit] #Not optimized at all!
                    Wolfecrit = (temp_values - g.clone().detach() - c1*stepsize.squeeze()*stepXgrad) > 0 #this criterion must be negative to exist the LS loop
                    stepsize[Wolfecrit] *= 0.1 #Update stepsizes for which it failed
            else:#Use constant step-size (no line-search)
                stepsize = gamma_BFGS
            ## Prepare next step
            xm1 = x.clone().detach()
            gradgm1 = gradg.clone().detach()
            ## Update x
            x.add_(stepsize * next_step)
        #Compute grad for next step
        gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)   
        list_g[:, n+1] = g.clone().detach()
    return list_g

def Newton(x0, f, gamma_Newton=1., niter_max=1000, Wolfe_LS=False):
    #Initialize first iterations
    x = x0.clone().detach().requires_grad_(True)  #Initialize at x0 with require_grad
    #f = function_to_minimize(f) # Turn the function into a class
    gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)
    #list_x = [x.clone().detach()] ; list_g = [g0.item()]
    N_samples = x0.shape[0] if len(x0.shape)>0 else 1
    list_g = torch.zeros(N_samples, niter_max + 1)
    list_g[:, 0] = g.clone().detach()
    # Iterate
    for n in range(niter_max):
        Hg = f.evalHessf(x) # Compute Hessian
        with torch.no_grad(): #This is important since x has require_grad=True
            next_step = torch.linalg.solve(Hg, -gradg)
            if Wolfe_LS: #Check 1st Wolfe condition
                c1 = 1e-2 ; n_LS = 0
                step = next_step.clone().detach() 
                stepXgrad = torch.sum(step * gradg, dim=-1)
                stepsize = torch.ones(N_samples, 1)  #If line-search, start with unit step-size
                Wolfecrit = torch.ones(N_samples, dtype=bool) #mask for successful LS
                temp_values = g.clone().detach()
                while n_LS < 5 and torch.sum(Wolfecrit)>0: #Try 5 times the line-search
                    n_LS += 1
                    y = x.clone().detach() + stepsize * step #Try this step-size
                    temp_values[Wolfecrit] = f.evalf(y)[Wolfecrit] #Not optimized at all!
                    Wolfecrit = (temp_values - g.clone().detach() - c1*stepsize.squeeze()*stepXgrad) > 0 #this criterion must be negative to exist the LS loop
                    stepsize[Wolfecrit] *= 0.1 #Update stepsizes for which it failed
            else: 
                stepsize = gamma_Newton
            x.add_(stepsize * next_step)
    #Store last value and last point
        gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)
        list_g[:, n+1] = g.clone().detach()
    return list_g

def GD(x0, f, gamma=0.01, niter_max=1000):
    #Initialize first iterations
    x = x0.clone().detach().requires_grad_(True)  #Initialize at x0 with require_grad
    #f = function_to_minimize(f) # Turn the function into a class
    gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)
    ## Compute Lipschitz Constant for optimal gamma
    if gamma=='semibest':
        Hg = f.evalHessf(x) # Compute Hessian
        Lip = torch.max( torch.real(torch.linalg.eigvals(Hg)) ) #get Lipschitz constant
        gamma = 1./(2*Lip) #hald of optimal step-size
    if gamma=='best':
        Hg = f.evalHessf(x) # Compute Hessian
        Lip = torch.max( torch.real(torch.linalg.eigvals(Hg)) ) #get Lipschitz constant
        gamma = 1./Lip #hald of optimal step-size
    ##
    #list_x = [x.clone().detach()] ; list_g = [g0.item()]
    N_samples = x0.shape[0] if len(x0.shape)>0 else 1
    list_g = torch.zeros(N_samples, niter_max + 1)
    list_g[:, 0] = g.clone().detach()
    # Iterate
    for n in range(niter_max):
        gradg = f.evalgradf(x) # Get also the value (was computed while evaluating gradient)
        with torch.no_grad(): #This is important since x has require_grad=True
            x.add_(-gamma*gradg)
        g = f.evalf(x)
        list_g[:, n+1] = g.clone().detach()
    return list_g



def Nesterov(x0, f, gamma=0.01, alpha=3.1, niter_max=1000):
    #Initialize first iterations
    x = x0.clone().detach().requires_grad_(True) #Initialize at x0 with require_grad
    speed = torch.zeros_like(x)
    #f = function_to_minimize(f) # Turn the function into a class
    gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)
    ## Compute Lipschitz Constant for optimal gamma
    if gamma=='semibest':
        Hg = f.evalHessf(x) # Compute Hessian
        Lip = torch.max( torch.real(torch.linalg.eigvals(Hg)) ) #get Lipschitz constant
        gamma = 1./(2*Lip) #hald of optimal step-size
    if gamma=='best':
        Hg = f.evalHessf(x) # Compute Hessian
        Lip = torch.max( torch.real(torch.linalg.eigvals(Hg)) ) #get Lipschitz constant
        gamma = 1./Lip #hald of optimal step-size
    ##
    #list_x = [x.clone().detach()] ; list_g = [g0.item()]
    N_samples = x0.shape[0] if len(x0.shape)>0 else 1
    list_g = torch.zeros(N_samples, niter_max + 1)
    list_g[:, 0] = g.clone().detach()
    # Iterate
    for n in range(niter_max):
        gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)
        with torch.no_grad(): #This is important since x has require_grad=True
            step = -gamma*gradg + (n+1-alpha)/(n+1) * speed
            x.add_(step)
            speed = step.clone() #speed is diff between two steps
        #Store last value and last point
        g = f.evalf(x)
        list_g[:, n+1] = g.clone().detach()
    return list_g



def HeavyBall(x0, f, gamma=0.01, alpha=3.1, niter_max=1000):
    #Initialize first iterations
    x = x0.clone().detach().requires_grad_(True) #Initialize at x0 with require_grad
    speed = torch.zeros_like(x)
    #f = function_to_minimize(f) # Turn the function into a class
    gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)
    ## Compute Lipschitz Constant for optimal gamma
    if gamma=='semibest':
        Hg = f.evalHessf(x) # Compute Hessian
        Lip = torch.max( torch.real(torch.linalg.eigvals(Hg)) ) #get Lipschitz constant
        gamma = 1./(2*Lip) #hald of optimal step-size
    if gamma=='best':
        Hg = f.evalHessf(x) # Compute Hessian
        Lip = torch.max( torch.real(torch.linalg.eigvals(Hg)) ) #get Lipschitz constant
        gamma = 1./Lip #hald of optimal step-size
    ##
    #list_x = [x.clone().detach()] ; list_g = [g0.item()]
    N_samples = x0.shape[0] if len(x0.shape)>0 else 1
    list_g = torch.zeros(N_samples, niter_max + 1)
    list_g[:, 0] = g.clone().detach()
    if N_samples>1 and len(alpha.shape)<2:
        alpha = alpha.unsqueeze(-1)
    if N_samples>1 and len(gamma.shape)<2:
        gamma = gamma.unsqueeze(-1)
    # Iterate
    for n in range(niter_max):
        gradg = f.evalgradf(x) ; g = f.fx # Get also the value (was computed while evaluating gradient)
        with torch.no_grad(): #This is important since x has require_grad=True
            #list_x.append(x.clone().detach()) ; 
            step = - gamma * gradg + (1 - alpha*torch.sqrt(torch.tensor(gamma))) *  speed
            x.add_(step)
            speed = step.clone() #speed is diff between two steps
        g = f.evalf(x)
        list_g[:, n+1] = g.clone().detach()
    return list_g