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
                
                if return_loss:
                    env_batch = bundle_batch[2]
                    env_batch = env_batch.to(self._device)
                    # calculate loss
                    if loss_reduction:
                        loss += self._loss_fn(predict, label_batch, env_batch) * batch_size
                    else:
                        loss_list.append(self._loss_fn(predict, label_batch, env_batch, reduction='none'))
                    
            predict_matrix = torch.cat(predict_list, dim=0).cpu().numpy()
            label_array = torch.cat(label_list, dim=0).cpu().numpy()

            if return_loss:
                if loss_reduction:
                    loss /= sample
                else:
                    loss = torch.cat(loss_list)
            
        if len(metrics)>0:
            metric_dict = compute_metrics(predict_matrix, label_array, metrics=metrics)
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