import torch
from collections import OrderedDict
import numpy as np
import os
import models.segloss as segloss

from .unet import *
from .gls import Gls
from .contrastive_loss import PixelContrastLoss

class Train_process():
    def __init__(self, opt,reloaddir=None,istest=None):
        super(Train_process, self).__init__()
        self.opt = opt
        self.n_cls = opt.nclass
        self.epoch=0

        if opt.model=='unet':
            self.netseg = Unet1(c=3, num_classes= self.n_cls)
            self.netseg=self.netseg.cuda()
            total_params = sum(p.numel() for p in self.netseg.parameters())
            print(f"Number of parameters: {total_params}")
        else:
            print("no this model")
        
        if istest == 1:
            print("reloaddir:",reloaddir)
            self.netseg.load_state_dict(torch.load(reloaddir))
           
                
    
        x=256
        projfunc= nn.Sequential(
                nn.Conv2d(x, x, kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace=True),
                nn.Conv2d(x, x, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace=True)
        )
        projfunc=projfunc.cuda()
        self.projfunc=projfunc
        
      
       
        self.glsfunc =Gls(alpha=self.opt.mixalpha,glsmix_f=self.opt.glsmix_f,out_channel = 3, n_layer =self.opt.gls_nlayer,interm_channel =self.opt.gls_interm,out_norm=self.opt.gls_outnorm).cuda()
        
        
        self.contrast_loss=PixelContrastLoss(self.opt.temperature,self.opt.n_view).cuda()

       
        self.criterionDice = segloss.SoftDiceLoss(self.n_cls).cuda() 
        self.ScoreDiceEval = segloss.Efficient_DiceScore(self.n_cls, ignore_chan0 = False).cuda() 
        self.criterionCE = segloss.My_CE(nclass = self.n_cls,batch_size = self.opt.batchSize, weight = torch.ones(self.n_cls,)).cuda()
        self.optimizer_seg = torch.optim.Adam( self.netseg.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay = 0.00003)



    def te_func(self,input):
        self.input_img_te = input['img'].float().cuda()
        self.input_mask_te = input['lb'].float().cuda()

        self.netseg.eval()

        with torch.no_grad():
            img,mask= self.input_img_te,self.input_mask_te
            seg, tmp= self.netseg(img,dropout_rate=0,fdropout=0)
            seg = torch.argmax(seg, 1)

        self.netseg.zero_grad()
        self.netseg.train()

        return mask, seg
    
    def forward_seg_train(self, input_img,fdropout,dropoutrate):
        pred,encf= self.netseg(input_img,dropout_rate=dropoutrate,fdropout=fdropout)

        loss_dice= self.criterionDice(input = pred, target = self.input_mask)
        loss_ce= self.criterionCE(inputs = pred, targets = self.input_mask.long() )
            
        self.seg_tr=pred.detach()

        return pred,encf,loss_dice,loss_ce
    

    def forward_con(self, pred_all,encf1):
            print("cal encoder feature consist")
            if self.opt.consist_f==1 or self.opt.contrast_f==1:
                index0=self.input_img1.shape[0]
                gt=self.input_mask
                size2,size3=gt.shape[2],gt.shape[3]
                loss_consist,loss_contrast=torch.zeros(1),torch.zeros(1)
                loss_consist,loss_contrast=loss_consist.float().cuda(),loss_contrast.float().cuda()
  
                encfaa,encfbb=encf1[:index0].detach(),encf1[index0:].detach()
                projf_tmp=self.projfunc(encf1)
                encfa,encfb=projf_tmp[:index0],projf_tmp[index0:]
                    
                
                encfaa=F.interpolate(encfaa,size=[size2,size3],mode="bilinear",align_corners=False)
                encfbb=F.interpolate(encfbb,size=[size2,size3],mode="bilinear",align_corners=False)
                encfa=F.interpolate(encfa,size=[size2,size3],mode="bilinear",align_corners=False)
                encfb=F.interpolate(encfb,size=[size2,size3],mode="bilinear",align_corners=False)
                
                pred1,pred2=pred_all[:index0].detach(),pred_all[index0:].detach()
                pred1,pred2=torch.argmax(pred1,dim=1),torch.argmax(pred2,dim=1)
                pred1,pred2=pred1.unsqueeze(1),pred2.unsqueeze(1)

                mask1,mask2=torch.zeros(size=gt.shape),torch.zeros(size=gt.shape)
                mask1,mask2=mask1.cuda(),mask2.cuda()
                mask1[(gt!=0) | (pred1!=0)]=1
                mask2[(gt!=0) | (pred2!=0)]=1
                mask1,mask2=mask1.squeeze(1),mask2.squeeze(1)

              
                if self.opt.consist_f==1:
                    tmp1=F.cosine_similarity(encfa, encfbb,dim=1) 
                    tmp2=F.cosine_similarity(encfaa, encfb,dim=1) 
                    if mask1.sum()!=0 and mask2.sum()!=0:
                        tmp1=((tmp1*mask1).sum()) / (mask1.sum())
                        tmp2=((tmp2*mask2).sum()) / (mask2.sum())
                        loss_consist=1-(tmp1+tmp2)/2
                    else:
                        if mask1.sum()!=0:
                           tmp1=((tmp1*mask1).sum()) / (mask1.sum())
                           loss_consist=1-tmp1
                        elif mask2.sum()!=0:
                           tmp2=((tmp2*mask2).sum()) / (mask2.sum())
                           loss_consist=1-tmp2
                        else:
                           loss_consist=torch.zeros(1).cuda() 
                else:
                    loss_consist=torch.zeros(1).cuda()
                    
               
                if self.opt.contrast_f==1:
                    print("contrast loss")
                    enca_c,encb_c=projf_tmp[:index0],projf_tmp[index0:]
                    size2_tmp,size3_tmp=enca_c.shape[2]*2,enca_c.shape[3]*2
                   
                    enca_c=F.interpolate(enca_c,size=[size2_tmp,size3_tmp],mode="bilinear",align_corners=False)
                    encb_c=F.interpolate(encb_c,size=[size2_tmp,size3_tmp],mode="bilinear",align_corners=False)

                    embedding=torch.cat((enca_c,encb_c),dim=0) 
                    embedding=F.normalize(embedding, dim=1)

                    predict,label,mask=torch.cat((pred1,pred2),dim=0),torch.cat((gt,gt),dim=0),torch.cat((mask1,mask2),dim=0)
                    predict,label=predict.float(),label.float()
                    mask=mask.unsqueeze(1)
                    predict2=F.interpolate(predict,size=[size2_tmp,size3_tmp],mode="nearest")
                    label2=F.interpolate(label,size=[size2_tmp,size3_tmp],mode="nearest")
                    mask2=F.interpolate(mask,size=[size2_tmp,size3_tmp],mode="nearest")
                    
                    self.contrast_pred,self.contrast_label,self.contrast_mask=predict2,label2,mask2
                    loss_contrast=self.contrast_loss(embedding,label2, predict2)
                    print("loss_contrast ori:",loss_contrast)
                else:
                    self.loss_contrast= torch.zeros(1).cuda()
                   
                self.loss_consist = self.opt.w_consist* loss_consist
                self.loss_contrast= self.opt.w_contrast*loss_contrast
            else:
                print("no consistency,no contrast")
                self.loss_consist=torch.zeros(1).cuda()
                self.loss_contrast= torch.zeros(1).cuda()
               
       
            print("loss_consist:",self.loss_consist)
            print("loss_contrast:",self.loss_contrast)

   
       
    def tr_func(self,train_batch,epoch):
        self.epoch=epoch

        for param in self.netseg.parameters():
            param.requires_grad =True

        w_seg  = self.opt.w_seg
        w_ce  = self.opt.w_ce
        w_dice = self.opt.w_dice

        if  self.opt.fmethod=='asymr' :
            self.input_img1= train_batch['img1'].float().cuda()
            self.input_img2= train_batch['img2'].float().cuda()
            self.input_mask= train_batch['lb'].float().cuda()
            self.input_ori= train_batch['ori'].float().cuda()
            
            if  self.opt.fmethod=='asymr' :
                print("img2 gls")
                self.input_img2 = self.glsfunc(self.input_img2)
            else:
                print("img2 no gls")
       
          
            print("self.opt.f_dropout1:",self.opt.f_dropout1,self.opt.dropout_rate1)
            pred_all1, encf1 ,loss_dice1 , loss_ce1 = self.forward_seg_train(self.input_img1,fdropout=self.opt.f_dropout1,dropoutrate=self.opt.dropout_rate1) 
            pred_all2, encf2 ,loss_dice2 , loss_ce2 = self.forward_seg_train(self.input_img2,fdropout=self.opt.f_dropout2,dropoutrate=self.opt.dropout_rate2)
             
            self.loss_seg = (loss_dice1*w_dice+loss_ce1*w_ce+loss_dice2*w_dice+loss_ce2*w_ce )/2*w_seg
            self.loss_dice  = (loss_dice1+loss_dice2)*w_dice/2
            self.loss_ce    = (loss_ce1+loss_ce2)*w_ce/2

            self.loss_seg1=loss_dice1*w_dice +loss_ce1*w_ce
            self.loss_seg2=loss_dice2*w_dice +loss_ce2*w_ce
            self.loss_dice1=loss_dice1
            self.loss_dice2=loss_dice2
            self.loss_ce1=loss_ce1
            self.loss_ce2=loss_ce2

            pred_all=torch.cat([pred_all1,pred_all2],dim=0)
            encf_all=torch.cat([encf1,encf2],dim=0)
            self.forward_con(pred_all,encf_all)
            self.loss_all=self.loss_seg + self.loss_consist+self.loss_contrast

            self.optimizer_seg.zero_grad()
            self.loss_all.backward()
            self.optimizer_seg.step()

            tr_log = [('dice',self.loss_dice),('ce', self.loss_ce),('seg',self.loss_seg),('consist', self.loss_consist),
                      ('contrast',self.loss_contrast),('lr',self.get_lr()),
                      ('loss',self.loss_all) ]

        for param in self.netseg.parameters():
            param.requires_grad =False

        tr_log = OrderedDict(tr_log)

        return tr_log
                

    def get_img_tr(self):
        img_tr1  = t2n( self.input_img1.detach())
        pred_tr = t2n(torch.argmax(self.seg_tr, dim =1, keepdim = True))
        gth_tr  = t2n(self.input_mask.detach())
        input_ori=t2n(self.input_ori.detach())
      
        

        ret_visuals = OrderedDict([('img_ori', input_ori),('img_tr1', img_tr1),('seg_tr',pred_tr),('gth_tr',gth_tr)])
        if self.opt.fmethod=='asymr':
            img_tr2=t2n( self.input_img2.data)
            ret_visuals['img_tr2']= img_tr2

        if self.opt.contrast_f==1:
            ret_visuals['contrast_pred']=t2n(self.contrast_pred.detach())
            ret_visuals['contrast_label']=t2n(self.contrast_label.detach())
            ret_visuals['contrast_mask']=t2n(self.contrast_mask.detach())

        return ret_visuals


    def save(self, snapshot_dir,label):
        save_filename = '%s_net_%s.pth' % (label, 'Seg')
        save_path = os.path.join(snapshot_dir, save_filename)
        print("save_path:",save_path)
        torch.save(self.netseg.state_dict(), save_path)


    def get_lr(self):
        lr = self.optimizer_seg.param_groups[0]['lr']
        x=[lr]
        x=torch.Tensor(x)
        return x

def t2n(x):
    if isinstance(x, np.ndarray):
        return x
    if x.is_cuda:
        x = x.data.cpu()
    else:
        x = x.data
    return np.float32(x.numpy())


