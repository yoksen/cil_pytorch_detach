import abc
import copy
import os

import numpy as np
import torch
from torch import nn, optim

from utils.center_loss import CenterLoss
from utils.toolkit import tensor2numpy

EPSILON = 1e-8

'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

# base is finetune with or without memory_bank
class BaseLearner(object):
    def __init__(self, logger, config):
        self._logger = logger
        self._config = copy.deepcopy(config)

        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._increment_steps = config.increment_steps
        self._nb_tasks = config.nb_tasks

        self._criterion_name = config.criterion
        self._method = config.method
        self._dataset = config.dataset
        self._use_valid = config.use_valid
        self._backbone = config.backbone
        self._seed = config.seed
        self._save_models = config.save_models

        self._gpu_num = config.gpu_num
        self._eval_metric = config.eval_metric
        self._logdir = config.logdir
        self._opt_type = config.opt_type

        # notice: if "init_epochs" is not assigned, it will follow the value of "epochs"
        # the same as the other "init_XXX"
        self._epochs = config.epochs
        self._batch_size = config.batch_size
        self._num_workers = config.num_workers

        self._criterion = None
        self._valid_loader = None

    @property
    def cur_taskID(self):
        return self._cur_task
    
    def set_save_models(self, choise:bool):
        self._save_models = choise

    @abc.abstractmethod
    def prepare_task_data(self, data_manager):
        '''
        prepare the dataloaders for the next stage training
        '''
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        self._train_loader = None
        self._valid_loader = None
        self._test_loader = None
        self._criterion = None
        # your code

    @abc.abstractmethod
    def prepare_model(self, checkpoint=None):
        '''
        prepare the model for the next stage training
        '''
        self._network = None
        # your code

    # need to be overwrite probably, base is finetune
    def incremental_train(self):
        if self._gpu_num > 1:
            self._network = nn.DataParallel(self._network, list(range(self._gpu_num)))
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config)
        scheduler = self._get_scheduler(optimizer, self._config)
        self._network = self._train_model(model=self._network, train_loader=self._train_loader, test_loader=self._test_loader, 
                optimizer=optimizer, scheduler=scheduler, epochs=self._epochs, valid_loader=self._valid_loader)
        if self._gpu_num > 1:
            self._network = self._network.module
    
    def store_samples(self):
        pass

    def _train_model(self, model, train_loader, test_loader, optimizer, scheduler, epochs=100, valid_loader=None):
        best_epoch_info = {'valid_acc': 0.}
        for epoch in range(epochs):
            model, train_acc, train_losses = self._epoch_train(model, train_loader, optimizer, scheduler)
            # update record_dict
            record_dict = {}

            info = ('Task {}, Epoch {}/{} => '.format(self._cur_task, epoch+1, epochs) + 
                ('{} {:.3f}, '* int(len(train_losses)/2)).format(*train_losses))
            
            for i in range(int(len(train_losses)/2)):
                record_dict['Train_'+train_losses[i*2]] = train_losses[i*2+1]
            
            if train_acc is not None:
                record_dict['Train_Acc'] = train_acc
                info = info + 'Train_accy {:.2f}, '.format(train_acc)          
            
            if 'pretrain' == self._config.method_type:
                if epoch+1 % 100 == 0:
                    self.save_checkpoint('{}_{}_{}_seed{}_epoch{}.pkl'.format(
                        self._config.mode, self._dataset, self._backbone, self._seed, epoch+1), 
                        self._network)
            else:
                if self._use_valid and valid_loader is not None:
                    valid_acc = self._epoch_test(model, valid_loader)
                    record_dict['Valid_Acc'] = valid_acc
                    info = info + 'Valid_accy {:.2f}'.format(valid_acc)
                    if best_epoch_info['valid_acc'] < valid_acc:
                        best_epoch_info['best_epoch'] = epoch
                        best_epoch_info['valid_acc'] = valid_acc
                        best_epoch_info['model_dict'] = model.state_dict()

                if self._config.test_epoch == None or (epoch+1) % self._config.test_epoch == 0:
                    test_acc = self._epoch_test(model, test_loader)
                    record_dict['Test_Acc'] = test_acc
                    info = info + 'Test_accy {:.2f}'.format(test_acc)

            self._logger.info(info)
            self._logger.visual_log('train', record_dict, step=epoch)
        
        if self._use_valid and valid_loader is not None:
            model.load_state_dict(best_epoch_info['model_dict'])
            self._logger.info('Reloaded model in epoch {}, with valid acc {}'.format(best_epoch_info['best_epoch'], 
                    best_epoch_info['valid_acc']))
        return model

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, feature_outputs = model(inputs)
            loss = self._criterion(logits, targets)
            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader)]
        return model, train_acc, train_loss
    
    def _epoch_test(self, model, test_loader, ret_pred_target=False):
        cnn_correct, total = 0, 0
        cnn_pred_all, target_all = [], []
        cnn_max_scores_all = []
        model.eval()
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feature_outputs = model(inputs)
            cnn_max_scores, cnn_preds = torch.max(torch.softmax(outputs, dim=-1), dim=-1)
            
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
            else:
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            target_all = np.concatenate(target_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            return cnn_pred_all, cnn_max_scores_all, target_all
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            return test_acc

    @abc.abstractmethod
    def eval_task(self):
        '''
        calculate evaluate metrics and print out
        '''
        # your code
        pass

    # need to be overwrite probably
    def after_task(self):
        self._known_classes = self._total_classes
        if self._save_models:
            self.save_checkpoint('seed{}_{}_{}_{}.pkl'.format(
                self._seed, self._method, self._dataset, self._backbone), 
                self._network)
    
    def release(self):
        self._network = self._network.cpu()
        self._network = None

    def save_checkpoint(self, filename, model=None, state_dict=None):
        save_path = os.path.join(self._logdir, filename)
        if state_dict != None:
            torch.save({'state_dict': state_dict, 'config':self._config.get_parameters_dict()}, save_path)
        else:
            torch.save({'state_dict': model.state_dict(), 'config':self._config.get_parameters_dict()}, save_path)
        self._logger.info('checkpoint saved at: {}'.format(save_path))
    

    def _get_optimizer(self, params, config, **kwargs):
        optimizer = None
        if config.opt_type == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=config.lrate, weight_decay=config.weight_decay)
        elif config.opt_type == 'adam':
            optimizer = optim.Adam(params, lr=config.lrate)
        else: 
            raise ValueError('No optimazer: {}'.format(config.opt_type))
        return optimizer
    
    def _get_scheduler(self, optimizer, config, **kwargs):
        '''config can be a dict'''
        scheduler = None
        if config.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.milestones, gamma=config.lrate_decay)
        elif config.scheduler == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs)
        elif config.scheduler == None:
            scheduler = None
        else: 
            raise ValueError('Unknown scheduler: {}'.format(config.scheduler))
        return scheduler
    
    def _get_criterion(self, loss_name):
        if loss_name == 'center_loss':
            return CenterLoss(num_classes=self._total_classes, feat_dim=self._network.feature_dim)
        else: # default option
            return nn.CrossEntropyLoss()

    