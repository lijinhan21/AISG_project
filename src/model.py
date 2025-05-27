from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from register import Register

class Output(nn.Module):
    def __init__(self, input_dim, output_dim, init=1, bias=False):
        super(Output, self).__init__()
        assert (input_dim >= output_dim)
        data = torch.zeros(output_dim, input_dim)
        for i in range(output_dim):
            data[i, i] = init
        self.weight = nn.Parameter(data)
        self.is_bias = bias
        if self.is_bias:
            data = torch.zeros(output_dim)
            self.bias = nn.Parameter(data)
        
    def forward(self, x):
        if self.is_bias:
            out = torch.mm(x, self.weight.t()) + self.bias
        else:
            out = torch.mm(x, self.weight.t())
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, drop_rate=0, batchnorm=False, output_features=1, bias=True):
        '''
        Args:
            hidden_size: list of hidden unit dimensions, the number of elements equal the humber of hidden layers.
        '''
        super(MLP, self).__init__()
        self.hidden_size = [input_dim] + hidden_size # self.hidden_size: [input dim, hidden dims, ...]
        self.hidden_layers = []
        # input layer and hidden layers
        for i in range(len(self.hidden_size) - 1):
            input_dim = self.hidden_size[i]
            output_dim = self.hidden_size[i + 1]
            self.hidden_layers.append((f'linear{i+1}', nn.Linear(
                            in_features = input_dim,
                            out_features = output_dim,
                            bias=bias
                        )))
            if batchnorm:
                self.hidden_layers.append((f'batchnorm{i+1}', nn.BatchNorm1d(num_features=output_dim)))
            self.hidden_layers.append((f'relu{i+1}', nn.ReLU()))
            self.hidden_layers.append((f'dropout{i+1}', nn.Dropout(p=drop_rate)))
        self.hidden_layers.append((f'linear{len(self.hidden_size)}', nn.Linear(
                            in_features = self.hidden_size[-1],
                            out_features = output_features,
                            bias=bias
                        )))
        # output layer
        self.output_layer = Output(output_features, output_features, init=1)
        print (self.hidden_layers)
        self.fc = nn.Sequential(OrderedDict(self.hidden_layers))


    def forward(self, x):
        phi = self.fc(x)
        y = self.output_layer(phi)
        return y
    
    def model(self):
        return self.fc

    def head(self):
        return self.output_layer
     
    
def reset_parameters(module:nn.Module, method='default'):
    if method=='default':
        for layer in module.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                reset_parameters(layer, method)
    elif method=='normal':
        for p in module.parameters():
            nn.init.normal_(p)
    elif method=='constant':
        for p in module.parameters():
            nn.init.constant_(p, 1)
        
    else:
        raise NotImplementedError


global loss_register
loss_register = Register('loss_register')


class Loss:
    def __init__(self, weight=None):
        self.weight = weight
    
    def update_weight(self, weight):
        '''
        Args:
            weight: tensor or None
        '''
        self.weight = weight

    def __call__(self, predict, target, env, reduction='mean'):
        raise NotImplementedError

@loss_register.register
class bce_loss(Loss):
    def __call__(self, predict, target, env, reduction='mean'):
        loss = nn.BCEWithLogitsLoss(reduction='none')(predict, target.float())
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            total_loss = 0
            env_list = env.unique()
            for env_id in env_list:
                total_loss += loss[env==env_id].mean()
            total_loss /= len(env_list)
            return total_loss
        else:
            raise NotImplementedError
      
@loss_register.register
class groupDRO(Loss):
    def __init__(self, risk:Loss, device, n_env=2, eta=0.05):
        super(groupDRO, self).__init__()
        self.risk = risk                    # bce_loss: Loss. Call self.risk(predict, target, env, reduction='mean' or 'none') to calculate loss
        self.device = device
        self.n_env = n_env                  # two environments
        self.eta = eta                      # hyperparameter of groupDRO
        self.prob = np.ones(n_env) / n_env  # weight of each environment, initialized as uniform
    def __call__(self, predict, target, env, reduction='mean'):
        """
        Calculate GroupDRO loss
        Parameters:
            predict: (batch_size) model output
            target: (batch_size)  label
            env: (batch_size) environment id
            reduction: 'mean' or 'none'
                if reduction is 'none', return loss for each sample (batch_size)
                if reduction is 'mean', return GroupDRO loss (scalar)
        An example implementation method:
             Step 1: calculate loss for each environment.
             Step 2: update the weight of each environment.
             Step 3: calculate the weighted average of the loss. 
        """
        # print("env=", env)
        loss = self.risk(predict, target, env, reduction='none') # batchsize
        
        env_loss = np.zeros(self.n_env)
        for i in range(self.n_env):
            env_loss[i] = loss[env==i].mean()
        # print("shape of env_loss", env_loss.shape, "shape of self.prob", self.prob.shape)
        assert env_loss.shape == self.prob.shape
        
        self.prob = self.eta * self.prob * np.exp(env_loss)
        self.prob = self.prob / self.prob.sum()
        
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            total_loss = 0
            for i in range(self.n_env):
                total_loss += self.prob[i] * loss[env==i].mean()
            return total_loss
        

def flatten_and_concat_variables(vs):
    """Flatten and concat variables to make a single flat vector variable."""
    flatten_vs = [torch.flatten(v) for v in vs]
    return torch.cat(flatten_vs, axis=0)

@loss_register.register
class IRM(Loss):
    def __call__(self, predict, target, env, network:nn.Module, risk:Loss):
        """
        Calculate IRM loss
        Parameters:
            predict: (batch_size) model output
            target: (batch_size)  label
            env: (batch_size) environment id
            network: last linear layer(classifer) of the model, the parameter of which is fixed to (1.0)
            risk: loss function 
        An example implementation method:
            Step 1: calculate loss for each environment.
            Step 2: calculate gradient of loss for each environment.
                        Hint: using torch.autograd.grad()
            Step 3: the average l2 norm of the gradients for every environment is the IRM loss.
        """
        
        loss = risk(predict, target, env, reduction='none')
        
        irm_loss = 0
        env_list = env.unique()
        for env_id in env_list:
            loss_env = loss[env==env_id].mean()
            grad = torch.autograd.grad(loss_env, network.parameters(), create_graph=True, allow_unused=True)[0]
            irm_loss += torch.norm(grad, p=2) ** 2
        irm_loss /= len(env_list)
        
        return irm_loss

@loss_register.register
class REx(Loss):
    def __call__(self, predict, target, env, network:nn.Module, risk:Loss):
        """
        Calculate REx loss
        Parameters:
            predict: (batch_size) model output
            target: (batch_size)  label
            env: (batch_size) environment id
            network: last linear layer(classifer) of the model, the parameter of which is fixed to (1.0)
            risk: loss function 
        An example implementation method:
            Step 1: calculate loss for each environment.
            Step 2: calculate the Var of the loss for each environment.
            Step 3: the average l2 norm of the gradients for every environment is the REx loss.
        """
        
        loss = risk(predict, target, env, reduction='none')
        
        env_list = env.unique()
        loss_env_list = []
        for env_id in env_list:
            loss_env = loss[env==env_id].mean()
            loss_env_list.append(loss_env)
        mean_losses_tensor = torch.stack(loss_env_list)
        rex_loss = torch.var(mean_losses_tensor)
        # print("rex_loss:", rex_loss)
        
        return rex_loss

@loss_register.register
class InvRat(Loss):
    def __call__(self, predict, env_predict, target, env, network:nn.Module, risk:Loss):
        """
        TODO: calculate InvRat loss
        Parameters:
            predict: (batch_size) model output
            target: (batch_size)  label
            env: (batch_size) environment id
            network: last linear layer(classifer) of the model, the parameter of which is fixed to (1.0)
            risk: loss function 
        An example implementation method:
            Step 1: calculate loss for each environment.
            Step 2: calculate the optimal loss for each environment.
            Step 3: calculate the maximum of the difference between the optimal loss and the loss for each environment.
        """
        
        loss = risk(predict, target, env, reduction='none')
        env_enable_loss = risk(env_predict, target, env, reduction='none')
        
        invrat_loss = None
        env_list = env.unique()
        for env_id in env_list:
            loss_env = loss[env==env_id].mean()
            optimal_loss_env = env_enable_loss[env==env_id].mean()
            
            if invrat_loss is None:
                invrat_loss = loss_env - optimal_loss_env
            else:
                invrat_loss = max(invrat_loss, loss_env - optimal_loss_env)
        
        return invrat_loss