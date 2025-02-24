from .essential_imports import *
import torch

from torch import nn


################################################################################
#Define the possible normalizations#
def row_softmax(X, **kwargs):
    Softmax = nn.Softmax(dim=-1)
    return Softmax(X)


def row_squaredl2norm(X, **kwargs):
    norm2 = torch.sum(X**2, dim=-1)  #compute the l2 norm over rows
    return X**2/norm2[:, None]  #make positive and normalize each row


def row_ReLUnorm(X, **kwargs):
    ReLU = torch.nn.ReLU()
    reluX = ReLU(X)  #Apply ReLU (to keep only positive correlations in attention)
    return reluX/torch.sum(reluX, dim=-1)[:, None]  #normalize each row

################################################################################


class LOA_BFGS_Model(nn.Module):
    def __init__(self, P, gamma=1., layers_out_dim_factors=None, use_fullskipco=True, use_gammaFeature=False, safe_init=False):
        super().__init__()  #If class inherits from another
        ### Save the parameters of the model (to then load or save easily)
        self.param_dict = locals().copy()   #save the parameters used for the model.
        self.param_dict['model'] = self.__class__.__name__   #save the name of the model
        [self.param_dict.pop(key) for key in ['self', '__class__']]   #removes some keys
        ###
        self.had_hidden = False  #Whether model has hidden states or not
        self.is_lstm = False  #Tells if model is of LSTM type (hence requires cellstate)
        self.is_QN = True  #used in "network_iter_loop" to reset QN mat
        self.P = P
        self.gamma = gamma  #If the original step-size is known then we'd better use it. 

        self.use_gammaFeature = use_gammaFeature  #Whether we use gamma as a feature in the input or not.


        self.n_features = 3
        #Set the out dimensions of each layers (in proportion to the previous layer)
        ### Feedforward layers part ###
        self.sigma = torch.nn.ReLU()
        #self.sigma = torch.nn.GELU()
        #Initialize the linear layers
        self.use_fullskipco = use_fullskipco  #Whether to use a skip-connection from beginning to end
        if use_fullskipco:
            self.fullskip_layer = nn.Linear(2*self.n_features, 1, bias=False)  #A layer directly from output to input, always used
        self.layers = nn.ModuleList()  #self.layers_keys = []
        #If we have other layers, create them below
        if layers_out_dim_factors is not None:
            input_dim = 2*self.n_features
            for layer_num, out_dim_factors in enumerate(layers_out_dim_factors):  #Do all layers except the last
                out_dim = int(input_dim*out_dim_factors)  #how many times the output is larger or smaller than the input
                self.layers.append(nn.Linear(input_dim, out_dim, bias=False))
                input_dim = out_dim  #next layer's input dim is previous layer's output dim
            #Do the last layer
            self.layers.append(nn.Linear(input_dim, 1, bias=False))
        ###

        ### Outer FF part ###
        self.outer_FF = nn.ModuleList()  #self.layers_keys = []
        input_dim = self.n_features
        for layer_num, out_dim_factors in enumerate([2, 2]):  #Do all layers except the last
            out_dim = int(input_dim*out_dim_factors)  #how many times the output is larger or smaller than the input
            self.outer_FF.append(nn.Linear(input_dim, out_dim, bias=False))
            input_dim = out_dim  #next layer's input dim is previous layer's output dim
        #Do the last layer
        self.outer_FF.append(nn.Linear(input_dim, self.n_features, bias=False))  #Get back n_features at the end (to then sum)

        
        #To initialize the network to be exactly BFGS
        if safe_init == 'BFGS_like':
            with torch.no_grad():
                ##Fullskip
                if self.use_fullskipco == True:
                    self.fullskip_layer.weight.mul_(0.) ;
                    self.fullskip_layer.weight[:, 1].add_(1.)  #linear layer outputs $d_{k-1}$ which is what BFGS uses
                else:
                    print('Warning, BFGS_like init needs full skip-connections')
                #If other layers exist set their values to zero
                if layers_out_dim_factors is not None:
                    self.layers[-1].weight.mul_(0.)

    #Reset the learned matrix to the identity
    def reset_mat(self, n_traj, factor=1, first_guess=None):  #initialize matrices to predict with identity
        if n_traj == 1:
            self.predicted_mat = self.diag(factor*torch.ones(P)) if first_guess is None else first_guess.clone()
            if len(self.predicted_mat.shape)<3:
                self.predicted_mat = self.predicted_mat.unsqueeze(0)
        else:  #more than 1 traj
            if first_guess is None:  #Unspecified initialization
                self.predicted_mat = torch.diag_embed(factor*torch.ones(n_traj, self.P))
            elif len(first_guess.shape) == 2:  #Same first guess for all matrices
                self.predicted_mat = torch.ones(n_traj, 1, 1)*first_guess[None, :, :]  #duplicate N times first_guess to make a N*P*P mat
            else:  #A different first_guess for each traj
                self.predicted_mat = first_guess.clone()

    #Constructs the input of the NN
    def construct_features(self, grad, gradm1, dm1, previous_gamma):
        #Construct the input x of the network that contains all the features
        with torch.no_grad():  #Do not backprop on this (Treat predmat just as a previous guess coming out of somewhere else)
            if torch.is_tensor(previous_gamma) and len(previous_gamma.shape) < 2:  #add a fake dim at the end of gamma to multiply with
                previous_gamma = previous_gamma.unsqueeze(-1)
            QNDG = torch.bmm(self.predicted_mat, (grad-gradm1).unsqueeze(-1)).squeeze(-1)  #DG in the current QN metric
            Bkm1timesGrad = - previous_gamma * torch.bmm(self.predicted_mat, (grad).unsqueeze(-1)).squeeze(-1)
            #gradnorm = (torch.norm(grad, dim=-1).unsqueeze(-1) * torch.ones(grad.shape[0], grad.shape[1])).squeeze(-1)  #norm of each vector times a Ones vector
            Features = (QNDG, dm1, Bkm1timesGrad)
            x = torch.cat(Features, dim=1)  #concatenate all the features into a big tensor of size batchsize*features*P
        return x

    def freeze(self, layer, unfreeze=False):
        #Freeze or unfreeze a given layer
        if unfreeze:
            layer.weight.requires_grad = True
            if layer.bias is not None:
                layer.bias.requires_grad = True
        else:
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False

    def forward(self, grad, gradm1, dm1, previous_gamma=None):  #use_dm1 makes use of it in the network (not only for v)
        previous_gamma = self.gamma if previous_gamma is None else previous_gamma  #Either step-size is specified or not
        #if we have only one element in the minibatch add fake dim **in front**
        if len(grad.shape) < 2:
            grad = grad.unsqueeze(0)
        if len(gradm1.shape) < 2:
            gradm1 = gradm1.unsqueeze(0) 
        if len(dm1.shape) < 2:
            dm1 = dm1.unsqueeze(0)


        ###Prepare input###
        N = grad.shape[0] #batchsize
        #Add features depending on the current predmat in the input
        x = self.construct_features(grad, gradm1, dm1, previous_gamma=previous_gamma)
        # Here we need a tensor of size batchsize*P*n_features (to operate along the feature dim)
        x = x.reshape(N, self.n_features, self.P).transpose(-2, -1)  #reshape then transpose to get batchsize*P*n_features (with correct placement of the elements) # direct reshaping would not do that

        ### Compute outer features (summed over all coordinates) ###
        outer_features = x.clone()  #create ones vector
        for layer in self.outer_FF[:-1]:  #loop over all layers except the last one
            outer_features = layer(outer_features)  #should be batchsize*P
            outer_features = self.sigma(outer_features)  #activation function
        outer_features = self.outer_FF[-1](outer_features)  #result should be of size batchsize*P*n_features
        outer_features = 1/outer_features.shape[1] * torch.sum(outer_features, dim=1)  #average over coordinates (size batchsize*n_features)
        outer_features = outer_features[:, None] * torch.ones_like(x)  #add the middle dimension again and make of same size as x
        # Finally concatenate with the rest of the features
        x = torch.cat((x, outer_features), dim=-1)
        
        ##FF layers loop
        fullskip = self.fullskip_layer(x).squeeze(-1) if self.use_fullskipco is True else torch.zeros_like(weighted).squeeze(-1)
        out = x  #Ready to pass in the FF layers
        #If there are more layers, process them
        if len(self.layers) > 0:
            for layer in self.layers[:-1]:  #loop over all layers except the last one
                out = layer(out)  #should be batchsize*P
                out = self.sigma(out)  #activation function
            out = self.layers[-1](out).squeeze(-1)  #last layer (no activation) and drop fake dim
        out = out + fullskip  #add the FF part of the NN to the skipco
        ##

        #BFGS inspired
        DGK = grad-gradm1  #Delta g_k (diff of grads in BFGS)
        secant = x[:, :, 1] - x[:, :, 0]  # dm1 - BkDeltaGk
        normalization = 1. / (torch.sum(out*DGK, dim=-1).unsqueeze(-1))  #From the BFGS formula
        coef = torch.sum(secant*DGK, dim=-1).unsqueeze(-1) / torch.sum(out*DGK, dim=-1).unsqueeze(-1)  #a scalar needed below
        # Below add fake dims (to get same dimension as the rest and use * sign)
        normalization = normalization.unsqueeze(-1) ; coef = coef.unsqueeze(-1)
        #Update QN matrix (approx of inverse Hessian)
        self.predicted_mat += normalization * (secant[:, :, None]*out[:, None, :] + out[:, :, None]*secant[:, None, :]
                                               - coef * out[:, :, None]*out[:, None, :])  # Comes from the generalized BFGS formula 
        ###
        #Finally compute next step of the algo (the QN step)
        v = - grad.clone().detach()
        d = torch.bmm(self.predicted_mat, v[:, :, None]).squeeze(-1)  #add fake dim and remove it after matmul (bmm=batchmatmul)
        return d


################################################################################