from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
class PixelContrastLoss(nn.Module, ABC):
    def __init__(self,temperature,n_view):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature 
        self.ignore_label = -1
        self.n_view=n_view 
  
    


    def _hard_anchor_sampling(self,X, y_hat, y):
        
        batch_size, feat_dim = X.shape[0], X.shape[-1]
      
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            classes.append(this_classes)
            total_classes += len(this_classes)
       
        n_view=self.n_view

        X_ = torch.zeros(size=[total_classes, n_view, feat_dim], dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        f=0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii] 
            this_y = y[ii]
            this_classes = classes[ii]
        
            for cls_id in this_classes:
                f=0

                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple=False)
                
                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
               
                if num_hard+num_easy>=n_view:
                  x_tmp=random.random()
                
                  x_tmp=int(x_tmp*n_view)
                
                  if num_hard >= x_tmp and num_easy >= n_view -x_tmp:
                    num_hard_keep = x_tmp
                    num_easy_keep = n_view - x_tmp
                  elif n_view>=num_hard:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                  elif n_view>=num_easy:
                     num_hard_keep=n_view-num_easy
                     num_easy_keep=num_easy
                  else:
                    f=1
                else:
                   f=1

                if f==1:
                    continue
                
               
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)
              
               
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        X_= X_[:X_ptr]
        y_=y_[:X_ptr]
        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits+ 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - self.temperature  * mean_log_prob_pos
        loss = loss.mean()

        return loss



    def forward(self, feats,labels, predict):

        batch_size = feats.shape[0]
        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
      
       
        feats_, labels_= self._hard_anchor_sampling(feats, labels, predict)
        
      

        loss = self._contrastive(feats_, labels_)
        return loss

