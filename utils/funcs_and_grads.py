# Define a generic class for functions f
# It takes the formula for f and has functions that evaluate gf and Hf via autodiff
from .essential_imports import *
import torch


class function_to_minimize:
    # This class implements a function f to minimize, 
    #but also its gradient and hessian matrix
    def __init__(self, f, A, b, xstar=None, lambdamax=None):
        self.f = f #takes x returns f(x)
        self.fx = None
        self.gradfx = None
        self.Hessfx = None
        self.A = torch.tensor(A, device=device)
        self.b = torch.tensor(b, device=device)
        self.lambdamax = lambdamax
        self.xstar = torch.tensor(xstar, device=device) if xstar is not None else None
        if xstar is not None:
            self.fstar = self.f(self.xstar, A=self.A, b=self.b)
        else:
            self.fstar = torch.tensor(0.)
        self.overflow = False

    def evalf(self, x, sum=False): #sum is used only for computing grads and hessians
        #if x has batch dimension, add it to A and b
        if sum ==True:
            return torch.sum(self.f(x, A=self.A, b=self.b), dim=0)
        return self.f(x, A=self.A, b=self.b)

    def evalgradf(self, x):
        # maybe torch.autograd.functional.jacobian would be easier for unrolling
        if type(x) is not torch.Tensor:
            x.zero_grad()
            fx = self.f(x, A=self.A, b=self.b) ; self.fx = fx
        else:
         #if neural network is passed
            if x.grad is not None:
                x.grad.zero_()
            fx = self.f(x, A=self.A, b=self.b) ; self.fx = fx
        if torch.sum(fx != fx) == 0:  #check for NaNs
            ##backward version
            #fx.backward()
            #self.gradfx = torch.clone(x.grad)
            ## torch.autograd.grad version
            if type(x) is not torch.Tensor:
                fx.backward()
                grad = torch.cat([p.grad.view(-1) for p in x.parameters()])
                self.gradfx = grad
            else:
                grad = torch.autograd.grad(outputs=torch.sum(fx, dim=0), inputs=x)
                self.gradfx = grad[0]  #returns a tuple, grad is the first element
        else:
            if not self.overflow:  #NaN warning only once
                print('encountered NaN in grad computation')
                self.overflow = True
            self.gradfx = torch.zeros_like(x)
        return self.gradfx

    def evalHessf(self, x):
        if x.grad is not None:
            x.grad.zero_()
        if x.shape[0]==1 and len(x.shape)>1: #if only one sample
            self.Hessfx = torch.autograd.functional.hessian(self.evalf, x.squeeze()).unsqueeze(0) #remove and add batchdim
        else:
            #Not optimized at all, overwill but works
            Hg = torch.autograd.functional.hessian(lambda x: self.evalf(x, sum=True), x) #this creates a large matrix with interaction between ALL problems 
            self.Hessfx = torch.zeros(x.shape[0], x.shape[-1], x.shape[-1])
            for i in range(x.shape[0]):
                self.Hessfx[i] = Hg[i,:,i]
        return self.Hessfx


#########Some functions###########
def quad(X, A, b):
    # Check if X has batch dimension but not A and b
    if len(X.shape) > len(A.shape)-1:
        A = A[None]
    if len(X.shape) > len(b.shape):
        b = b[None]
    #Need to add and remove last dim of X for mat vec product
    g = 1/2 * torch.sum(((A @ X.unsqueeze(-1)).squeeze(-1) - b)**2, dim=-1)
    return g


def logistic(X, A, b):
    if len(X.shape) > len(A.shape)-1:
        A = A[None]
    if len(X.shape) > len(b.shape):
        b = b[None]
    correl = (A @ X.unsqueeze(-1)).squeeze()  #Linear model times x in logistic regression
    if torch.sum(b==0)>0: #check if b contains zeros
        warnings.warn('Warning: this implementation expects the b in {-1,1} formulation')
    
    loglike = torch.sum(torch.log(1. + torch.exp(-b * correl)), dim=-1) #vector of all the log likelihood b is binary \in {0,1}
    #regul = 1e-3/2. * torch.sum(X**2, dim=0)  #add a quadratic regularizer to make the problem str cvx
    Y = X.clone()  #Used to avoid problems with backpropagating at multiple iterations
    regul = 1 / (2.*b.shape[-1]) * torch.sum(Y**2, dim=-1) #Small L2 regul
    #g = 1 / A.shape[0] * torch.sum(loglike + regul, dim=0)   #Average of all likelihoods (sum along batch dim)
    g = loglike + regul #returns one value per data sample
    return g