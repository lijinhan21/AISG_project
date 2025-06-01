import torch
from tqdm import tqdm
from model import *
from metric import compute_metrics
from register import Register
from dataset import *

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

global trainer_register
trainer_register = Register('trainer_register')

@trainer_register.register
class ERM(object):
    def __init__(self, device, model, optimizer, dataset:TensorLoader, loss_fn:Loss, regularizer:Loss, reset_model=True, **kwargs):
        self._device = device
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._dataset = dataset
        self._loss_fn = loss_fn
        self._regularizer = regularizer
        self._reg_lambda = kwargs['reg_lambda']
        
        if reset_model:
            reset_parameters(self._model, kwargs['model_init'])

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(num_training_updates), desc='Training'):
            self._model.train()
            input_batch, label_batch, env_batch, ids = next(iterator)
            input_batch = input_batch.to(self._device)
            label_batch = label_batch.to(self._device)
            env_batch = env_batch.to(self._device)
            
            predict = self._model(input_batch)
            predict = predict.squeeze()
            loss = self._loss_fn(predict, label_batch, env_batch, reduction='mean')
            if self._regularizer is not None:
                regularization = self._regularizer(predict, label_batch, env_batch, network=self._model.head(), risk=self._loss_fn)     
            else:
                regularization = 0
            # print(predict[0], label_batch[0], env_batch[0])
            # print("loss:", loss, "regularization:", regularization, "lambda:", self._reg_lambda, "reg loss:", self._reg_lambda * regularization)
            # input('Press Enter to continue...')
            total_loss = loss + self._reg_lambda * regularization
            total_loss.backward()
            self._optimizer.step()
            self._model.zero_grad()

            if eval_steps > 0 and (i+1) % eval_steps == 0:
                val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
                print (f'train loss:{loss} val loss:{val_loss}')
                loss_inspect = self._loss_fn(predict, label_batch, env_batch, reduction='none')
                for env in env_batch.unique():
                    print (f'Loss of env {env}:', loss_inspect[env_batch==env].mean().detach().cpu().numpy())
                print ('Validation metrics:', metric_dict)

            if logging_steps > 0 and (i+1) % logging_steps == 0:
                print (f'train loss:{loss} regularization:{regularization}') 
            
        return None


    def evaluate(self, dataloader, metrics=[], loss_reduction=True, return_loss=True):
        self._model.eval()
        sample = 0
        loss_list = []
        loss = 0
        predict_list = []
        label_list = []
        env_list = []
        with torch.no_grad():
            for bundle_batch in tqdm(dataloader, desc='Evaluating'):
                input_batch, label_batch = bundle_batch[0], bundle_batch[1]
                batch_size = input_batch.shape[0]
                input_batch = input_batch.to(self._device)
                label_batch = label_batch.to(self._device)
                sample += batch_size
                predict = self._model(input_batch) # [batch_size]
                predict = predict.squeeze()
                
                predict_list.append(predict)
                label_list.append(label_batch)
                
                if return_loss or len(bundle_batch) > 2:
                    env_batch = bundle_batch[2]
                    env_batch = env_batch.to(self._device)
                    env_list.append(env_batch)
                    
                    if return_loss:
                        # calculate loss
                        if loss_reduction:
                            loss += self._loss_fn(predict, label_batch, env_batch) * batch_size
                        else:
                            loss_list.append(self._loss_fn(predict, label_batch, env_batch, reduction='none'))
                    
            predict_matrix = torch.cat(predict_list, dim=0).cpu().numpy()
            label_array = torch.cat(label_list, dim=0).cpu().numpy()
            
            # Collect environment information if available
            env_matrix = None
            if len(env_list) > 0:
                env_matrix = torch.cat(env_list, dim=0).cpu().numpy()

            if return_loss:
                if loss_reduction:
                    loss /= sample
                else:
                    loss = torch.cat(loss_list)
            
        if len(metrics)>0:
            metric_dict = compute_metrics(predict_matrix, label_array, metrics=metrics, env_matrix=env_matrix)
            if return_loss:
                return loss, metric_dict
            else:
                return metric_dict
        else:  
            return loss

@trainer_register.register
class InvRat(ERM):
    def __init__(self, device, model, env_model, optimizer, dataset:TensorLoader, loss_fn:Loss, regularizer:Loss, reset_model=True, **kwargs):
        super(InvRat, self).__init__(device, model, optimizer, dataset, loss_fn, regularizer, reset_model=reset_model, **kwargs)
        self._env_model = env_model.to(self._device)
        self._env_optimizer = kwargs['env_optimizer']
        self._env_loss_fn = kwargs['env_loss_fn']
        
        if reset_model:
            reset_parameters(self._env_model, kwargs['model_init'])

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(num_training_updates), desc='Training'):
            
            input_batch, label_batch, env_batch, ids = next(iterator)
            input_batch = input_batch.to(self._device)
            label_batch = label_batch.to(self._device)
            env_batch = env_batch.to(self._device)
            
            # Train env-specific model
            self._env_model.train()
            # Concatenate env_ids with input for env-specific model
            env_input = torch.cat([input_batch, env_batch.float().unsqueeze(1)], dim=1)
            env_predict = self._env_model(env_input)
            env_predict = env_predict.squeeze()
            env_loss = self._env_loss_fn(env_predict, label_batch, env_batch, reduction='mean')
            env_loss.backward()
            self._env_optimizer.step()
            self._env_model.zero_grad()
            
            # Train env-agnostic model
            self._model.train()
            
            # Forward pass for env-agnostic model
            predict = self._model(input_batch)
            predict = predict.squeeze()
            loss = self._loss_fn(predict, label_batch, env_batch, reduction='mean')
            if self._regularizer is not None:
                regularization = self._regularizer(predict, env_predict.detach(), label_batch, env_batch, network=self._model.head(), risk=self._loss_fn)     
            else:
                regularization = 0
            # print("loss:", loss, "regularization:", regularization, "lambda:", self._reg_lambda, "reg loss:", self._reg_lambda * regularization)
            # input('Press Enter to continue...')
            total_loss = loss + self._reg_lambda * regularization
            total_loss.backward()
            self._optimizer.step()
            self._model.zero_grad()

            if eval_steps > 0 and (i+1) % eval_steps == 0:
                val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
                print(f'train loss:{total_loss} env_loss:{env_loss} val loss:{val_loss}')
                loss_inspect = self._loss_fn(predict, label_batch, env_batch, reduction='none')
                for env in env_batch.unique():
                    print(f'Loss of env {env}:', loss_inspect[env_batch==env].mean().detach().cpu().numpy())
                print('Validation metrics:', metric_dict)

            if logging_steps > 0 and (i+1) % logging_steps == 0:
                print(f'train loss:{loss} env_loss:{env_loss} regularization:{regularization}') 
            
        return None

@trainer_register.register
class VAETrainer(object):
    def __init__(self, device, model, optimizer, dataset:TensorLoader, loss_fn:VAELoss, reset_model=True, **kwargs):
        self._device = device
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._dataset = dataset
        self._loss_fn = loss_fn
        
        if reset_model:
            reset_parameters(self._model, kwargs['model_init'])

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(num_training_updates), desc='Training VAE'):
            self._model.train()
            input_batch, label_batch, env_batch, ids = next(iterator)
            input_batch = input_batch.to(self._device)
            
            # Forward pass
            x_reconstructed, mu, log_var, z = self._model(input_batch)
            
            # Calculate loss
            loss = self._loss_fn(input_batch, x_reconstructed, mu, log_var)
            
            # Backward pass
            loss.backward()
            self._optimizer.step()
            self._model.zero_grad()

            if eval_steps > 0 and (i+1) % eval_steps == 0:
                val_loss = self.evaluate(self._dataset.validation_loader)
                print(f'Train loss: {loss.item():.4f}, Val loss: {val_loss:.4f}')
                
                # Get latent representations and categories for a batch
                with torch.no_grad():
                    z = self._model.get_latent(input_batch)
                    categories = categorize_latent_samples(z)
                    category_counts = torch.bincount(categories, minlength=32)
                    print(f"Category distribution (sample): {category_counts}")

            if logging_steps > 0 and (i+1) % logging_steps == 0:
                print(f'Train loss: {loss.item():.4f}')
            
        return None

    def evaluate(self, dataloader):
        self._model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for bundle_batch in tqdm(dataloader, desc='Evaluating VAE'):
                input_batch = bundle_batch[0].to(self._device)
                batch_size = input_batch.shape[0]
                
                # Forward pass
                x_reconstructed, mu, log_var, z = self._model(input_batch)
                
                # Calculate loss
                loss = self._loss_fn(input_batch, x_reconstructed, mu, log_var)
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
        return total_loss / total_samples
        
    def get_latent_representations(self, dataloader):
        """
        Generate latent representations for all samples in the dataloader
        
        Returns:
            latent_vectors: Tensor of shape (n_samples, latent_dim)
            categories: Tensor of shape (n_samples,) with values 0-31
            labels: Original labels
            ids: Sample IDs
        """
        self._model.eval()
        latent_vectors = []
        all_labels = []
        all_ids = []
        
        with torch.no_grad():
            for bundle_batch in tqdm(dataloader, desc='Generating latent representations'):
                input_batch, label_batch = bundle_batch[0], bundle_batch[1]
                ids = bundle_batch[3] if len(bundle_batch) > 3 else None
                
                input_batch = input_batch.to(self._device)
                
                # Get latent vectors
                z = self._model.get_latent(input_batch)
                
                latent_vectors.append(z.cpu())
                all_labels.append(label_batch.cpu())
                if ids is not None:
                    all_ids.append(ids.cpu())
        
        latent_vectors = torch.cat(latent_vectors, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate categories
        categories = categorize_latent_samples(latent_vectors)
        
        if len(all_ids) > 0:
            all_ids = torch.cat(all_ids, dim=0)
            return latent_vectors, categories, all_labels #, all_ids
        else:
            return latent_vectors, categories, all_labels
    
    def analyze_categories(self, dataloader):
        """
        Analyze the distribution of samples across the 32 categories
        
        Returns:
            category_stats: Dict with stats about each category
        """
        latent_vectors, categories, labels = self.get_latent_representations(dataloader)
        
        # Count samples per category
        category_counts = torch.bincount(categories, minlength=32)
        
        # Calculate label distribution per category
        category_stats = {}
        for i in range(32):
            cat_mask = (categories == i)
            if cat_mask.sum() > 0:
                cat_labels = labels[cat_mask]
                label_counts = torch.bincount(cat_labels.long(), minlength=2)
                label_ratio = label_counts.float() / cat_mask.sum()
                
                # Create binary representation of category
                binary = [(i >> bit) & 1 for bit in range(5)]
                
                category_stats[i] = {
                    'count': cat_mask.sum().item(),
                    'percentage': (cat_mask.sum().float() / len(categories)).item() * 100,
                    'label_counts': label_counts.tolist(),
                    'label_ratio': label_ratio.tolist(),
                    'binary': binary
                }
        
        return category_stats

@trainer_register.register
class CategoryReweightedERM(ERM):
    def __init__(self, device, model, optimizer, dataset:TensorLoader, loss_fn:Loss, regularizer:Loss, reset_model=True, **kwargs):
        super(CategoryReweightedERM, self).__init__(device, model, optimizer, dataset, loss_fn, regularizer, reset_model=reset_model, **kwargs)
        
        # Calculate category weights during initialization
        self.category_weights = self._calculate_category_weights()
        print(f"Calculated category weights for {len(self.category_weights)} categories")
        print(f"Weight statistics - Min: {self.category_weights.min():.6f}, Max: {self.category_weights.max():.6f}, Mean: {self.category_weights.mean():.6f}")
        print(f"Category weights: {self.category_weights}")

    def _calculate_category_weights(self):
        """
        Calculate category weights based on the formula:
        weight_i = (1/n_i) / sum_j(1/n_j)
        where n_i is the number of samples in category i
        """
        # Count samples in each category from training data
        category_counts = {}
        
        # Iterate through training dataset to count categories
        for input_batch, label_batch, env_batch, ids in self._dataset.training_loader_sequential:
            for env_id in env_batch:
                env_id = env_id.item()
                if env_id in category_counts:
                    category_counts[env_id] += 1
                else:
                    category_counts[env_id] = 1
        
        print(f"Found {len(category_counts)} categories with counts: {category_counts}")
        
        # Calculate weights according to the formula
        # weight_i = (1/n_i) / sum_j(1/n_j)
        inverse_counts = {cat: 1.0 / count for cat, count in category_counts.items()}
        normalization_factor = sum(inverse_counts.values())
        
        category_weights = {}
        for cat, inv_count in inverse_counts.items():
            category_weights[cat] = inv_count / normalization_factor
        
        # Convert to tensor for efficient lookup during training
        # We'll create a tensor where index corresponds to category ID
        max_category = max(category_weights.keys())
        weight_tensor = torch.ones(max_category + 1, device=self._device)
        
        for cat, weight in category_weights.items():
            weight_tensor[cat] = weight
            
        return weight_tensor

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, gamma=50, **kwargs):
        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(num_training_updates), desc='Training with Category Reweighting'):
            self._model.train()
            input_batch, label_batch, env_batch, ids = next(iterator)
            input_batch = input_batch.to(self._device)
            label_batch = label_batch.to(self._device)
            env_batch = env_batch.to(self._device)
            
            predict = self._model(input_batch)
            predict = predict.squeeze()
            
            # Calculate loss for each sample
            loss_per_sample = self._loss_fn(predict, label_batch, env_batch, reduction='none')
            
            # Apply category-based weights
            sample_weights = self.category_weights[env_batch]
            weighted_loss = loss_per_sample * sample_weights * gamma
            loss = weighted_loss.mean()
            
            # print(torch.unique(env_batch, return_counts=True))
            # print(self.category_weights)
            # print("Weighted loss per environment:", 
            #       {env.item(): weighted_loss[env_batch == env].mean().item() for env in env_batch.unique()})
            # input('Press Enter to continue...')
            
            if self._regularizer is not None:
                regularization = self._regularizer(predict, label_batch, env_batch, network=self._model.head(), risk=self._loss_fn)     
            else:
                regularization = 0
                
            total_loss = loss + self._reg_lambda * regularization
            
            total_loss.backward()
            self._optimizer.step()
            self._model.zero_grad()

            if eval_steps > 0 and (i+1) % eval_steps == 0:
                val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
                print (f'train loss:{loss} val loss:{val_loss}')
                loss_inspect = self._loss_fn(predict, label_batch, env_batch, reduction='none')
                for env in env_batch.unique():
                    print (f'Loss of env {env}:', loss_inspect[env_batch==env].mean().detach().cpu().numpy())
                print ('Validation metrics:', metric_dict)

            if logging_steps > 0 and (i+1) % logging_steps == 0:
                print (f'train loss:{loss} regularization:{regularization}') 
                # Print category distribution in current batch
                unique_envs, counts = torch.unique(env_batch, return_counts=True)
                category_dist = {env.item(): count.item() for env, count in zip(unique_envs, counts)}
                print (f'Batch category distribution: {category_dist}')
            
        return None