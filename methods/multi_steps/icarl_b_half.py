import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import target2onehot, tensor2numpy
from torch.utils.data import DataLoader

EPSILON = 1e-8

class iCaRL(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = config.T
        if self._incre_type != 'cil':
            raise ValueError('iCaRL is a class incremental method!')

                
    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        if self._cur_task > 0 and self._memory_bank != None:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), 
                    source='train', mode='train', appendent=self._memory_bank.get_memory())
        else:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        if self._cur_task == 0:
            self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
            self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')
        else:
            
        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')

    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()
    
    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, kd_losses = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            
            # # bce loss version implementation
            # onehots = target2onehot(targets, self._total_classes)
            # if self._old_network == None:
            #     loss = binary_cross_entropy_with_logits(logits[:,:task_end], onehots)
            # else:
            #     old_onehots = torch.sigmoid(self._old_network(inputs)[0].detach())
            #     new_onehots = onehots.clone()
            #     new_onehots[:, :task_begin] = old_onehots
            #     loss = binary_cross_entropy_with_logits(logits[:,:task_end], new_onehots)

            # ce loss version implementation
            ce_loss = cross_entropy(logits[:,:task_end], targets)
            if self._old_network is None:
                loss = ce_loss
                ce_losses += ce_loss.item()
            else:
                kd_loss = self._KD_loss(logits[:,:task_begin], self._old_network(inputs)[0], self._T)
                kd_losses += kd_loss.item()
                loss = ce_loss + kd_loss

            preds = torch.max(logits[:,:task_end], dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader), 'Loss_kd', kd_losses/len(train_loader)]
        return model, train_acc, train_loss

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        
        # random shuffle soft (teacher logits)
        # b, dim = soft.shape
        # max_idx = torch.argmax(soft, dim=1)
        # mask = torch.ones_like(soft, dtype=bool)
        # mask[torch.arange(mask.shape[0]) ,max_idx] = False
        # soft[mask] = soft[mask].view(b, dim-1)[:, torch.randperm(dim-1)].view(-1)

        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]