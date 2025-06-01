from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Net(nn.Module):
    """
    Simple MLP for Colored MNIST experiments.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 3 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x).flatten()
        return logits


class ConvNet(nn.Module):
    """
    Convolutional neural network for Colored MNIST experiments.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x).flatten()
        return logits
     
    
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

@loss_register.register
class ChiSquareDRO(Loss):
    def __init__(self, risk:Loss, device, n_env=2, eta=0.05, rho=10, beta=0.01):
        super(ChiSquareDRO, self).__init__()
        self.risk = risk                    # Base loss function
        self.device = device
        self.n_env = n_env                  # Number of environments
        self.rho = rho                      # Chi-square radius parameter
        self.beta = beta                    # Smoothing parameter
        self.env_weights = torch.ones(n_env, device=device) / n_env  # Initial weights
        
    def __call__(self, predict, target, env, reduction='mean'):
        """
        Calculate Chi-Square DRO loss
        Parameters:
            predict: (batch_size) model output
            target: (batch_size) label
            env: (batch_size) environment id
            reduction: 'mean' or 'none'
                if reduction is 'none', return loss for each sample
                if reduction is 'mean', return Chi-Square DRO loss (scalar)
                
        Based on: Namkoong & Duchi, "Stochastic Gradient Methods for 
        Distributionally Robust Optimization with f-divergences"
        """
        # Get per-sample losses
        sample_losses = self.risk(predict, target, env, reduction='none')
        
        if reduction == 'none':
            return sample_losses
            
        # Calculate environment-specific losses
        env_losses = []
        env_sizes = []
        
        for i in range(self.n_env):
            env_mask = (env == i)
            if env_mask.sum() > 0:
                env_losses.append(sample_losses[env_mask].mean())
                env_sizes.append(env_mask.sum().item())
            else:
                env_losses.append(torch.tensor(0.0, device=self.device))
                env_sizes.append(0)
                
        env_losses = torch.stack(env_losses)
        env_sizes = torch.tensor(env_sizes, device=self.device).float()
        valid_envs = (env_sizes > 0)
        
        # Calculate the optimal weights using chi-square formulation
        if valid_envs.sum() > 1:  # Need at least two environments
            # Normalize environment sizes
            env_probs = env_sizes[valid_envs] / env_sizes[valid_envs].sum()
            
            # Center the losses (optional but stabilizes optimization)
            centered_losses = env_losses[valid_envs] - env_losses[valid_envs].mean()
            
            # Compute chi-square weights
            delta = torch.clamp(1 + self.rho * centered_losses, min=self.beta)
            weights = env_probs * delta
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Update environment weights for tracking
            new_weights = torch.zeros_like(self.env_weights)
            new_weights[valid_envs] = weights
            self.env_weights = new_weights
            
            # Compute weighted loss
            total_loss = (env_losses[valid_envs] * weights).sum()
        else:
            # If only one environment has data, use standard average
            total_loss = env_losses[valid_envs].mean() if valid_envs.sum() > 0 else 0
            
        return total_loss
    
    def get_weights(self):
        """Return the current environment weights for analysis"""
        return self.env_weights.detach().cpu().numpy()

# Variational Autoencoder for generating latent space with values in [0,1]
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=[128, 64], latent_dim=5, drop_rate=0, batchnorm=False):
        super(VAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dim:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(p=drop_rate))
            prev_dim = dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dim[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dim):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if batchnorm:
                decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(p=drop_rate))
            prev_dim = dim
            
        decoder_layers.append(nn.Linear(hidden_dim[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.latent_dim = latent_dim
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Constrain latent space to [0,1] with sigmoid
        z = torch.sigmoid(z)
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var, z
    
    def get_latent(self, x):
        """Get latent representation for input"""
        with torch.no_grad():
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
        return z

def categorize_latent_samples(z):
    """
    Categorize samples based on 5D latent space into 32 categories
    Each dimension has 2 conditions: > 0.5 or <= 0.5
    
    Args:
        z: Tensor of shape (batch_size, 5) with values in [0,1]
    
    Returns:
        categories: Tensor of shape (batch_size,) with values 0-31
    """
    batch_size = z.shape[0]
    binary_encoding = (z > 0.5).int()  # Convert to binary: 1 if > 0.5, 0 if <= 0.5
    
    # Calculate category indices (0-31)
    # Each sample's category is determined by treating the 5 binary values as a 5-bit number
    categories = torch.zeros(batch_size, dtype=torch.long, device=z.device)
    for i in range(5):
        categories += binary_encoding[:, i] * (2 ** i)
        
    return categories

@loss_register.register
class VAELoss(Loss):
    def __init__(self, reconstruction_weight=1.0, kl_weight=0.1):
        super(VAELoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        
    def __call__(self, x, x_reconstructed, mu, log_var, reduction='mean'):
        # Reconstruction loss (MSE)
        recon_loss = torch.nn.functional.mse_loss(x_reconstructed, x, reduction='none').sum(dim=1)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        # Total loss
        total_loss = self.reconstruction_weight * recon_loss + self.kl_weight * kl_loss
        
        if reduction == 'none':
            return total_loss
        elif reduction == 'mean':
            return total_loss.mean()
        else:
            raise NotImplementedError