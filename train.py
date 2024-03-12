import random
import torch.backends.cudnn as cudnn
import time
import shutil
import torch
import numpy as np
import os
import argparse
import os.path as osp
from PIL import Image
from models.exp_trainer import *

from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import logging
from datetime import datetime
import dataloaders.AbdominalDataset as ABD
# import dataloaders.ProstateDataset as PROS
# import dataloaders.CardiacDataset as cardiac_cls
te_metric,val_metric=[],[]

def pre_labmap():
    labmap={}
    tmp1={'0':0,'1':255} 
    labmap['PROSTATE']=tmp1
    tmp2={'0':0,'1':63,'2':126,'3':189,'4':255}
    labmap['ABDOMINAL']=tmp2
    tmp3={'0':0,'1':85,'2':170,'3':255}
    labmap['CARDIAC']=tmp3
    return labmap

def deal_wit_lbvis(tmp_mp,x,ncls):
    y=torch.zeros(size=x.shape).cuda()
    x=torch.from_numpy(x)
    for i in range(ncls):
        y[x==i]=tmp_mp[str(i)]
    return y

def prediction_wrapper(tb_writer,type1,savdir,model, test_loader,  epoch, label_name, save_prediction ):
    with torch.no_grad():
        out_prediction_list = {} 
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0
                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['img1'].shape
              
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = torch.Tensor(np.zeros( [ nframe,nx, ny]  )).cuda()

            assert batch['lb'].shape[0] == 1 # enforce a batchsize of 1
            test_input = {'img': batch['img1'],'lb': batch['lb']}
            gth, pred = model.te_func(test_input)

            
            print("pred,gth:",pred.size(),gth.size())
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0,0,...]
            curr_img[slice_idx,...] = batch['img1'][0, 1,...]
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                out_prediction_list[scan_id_full]['img'] = curr_img

        print("Epoch {} test result on mode {} seg:".format(epoch, type1))

        eval_list_wrapper(tb_writer,epoch,type1,savdir,out_prediction_list,  model, label_name)

        if not save_prediction: 
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list

def eval_list_wrapper(tb_writer,epoch,type1,savdir,vol_list, model, label_name):
    nclass=len(label_name)
    out_count = len(vol_list)
    tables_by_domain = {} 
    dsc_table = np.ones([ out_count, nclass ] ) 

    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices = model.ScoreDiceEval(torch.unsqueeze(pred_, 1), gth_, dense_input = True).cpu().numpy() 
        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {} \n, std: {}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
        if type1=='testfinal' or type1=='tetrainfinal':
          with open(savdir+'/out.csv', 'a') as f:
            f.write("Organ"+label_name[organ] +"with dice: \n")
            f.write("mean:"+ str(mean_dc)+"\n")
            f.write("std:"+str(std_dc)+"\n")


    print("Overall mean dice by sample {}".format( dsc_table[:,1:].mean())) 
    error_dict['overall'] = dsc_table[:,1:].mean()

    if type1=='testfinal' or type1=='tetrainfinal':
        with open(savdir+'/out.csv', 'a') as f:
           f.write("Overall mean dice by sample:"+str(dsc_table[:,1:].mean())+" \n")

   
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)

    error_dict['overall_by_domain'] = np.mean(overall_by_domain)
    print("Overall mean dice by domain {}".format( error_dict['overall_by_domain'] ) )
    if type1=='testfinal' or type1=='tetrainfinal':
       with open(savdir+'/out.csv', 'a') as f:
           f.write("Overall mean dice by domain:"+str(error_dict['overall_by_domain'] )+" \n")
    # for prostate dataset, we use by-domain results to mitigate the differences in number of samples for each target domain

    if type1=='val':
       tmp=[str(epoch),str(error_dict['overall_by_domain'] )]
       val_metric.append(tmp)
       with open(osp.join(savdir+'/','valmetric.csv'), 'a') as f:
            f.write(str(epoch)+":"+str(error_dict['overall_by_domain'] )+" \n")
       tb_writer.add_scalar('val_metric',error_dict['overall_by_domain'], epoch)
    elif type1=='test' or type1=='testfinal':
        tmp=[str(epoch),str(error_dict['overall_by_domain'] )]
        te_metric.append(tmp)
        with open(osp.join(savdir+'/','temetric.csv'), 'a') as f:
            f.write(str(epoch)+":"+str(error_dict['overall_by_domain'] )+" \n")
        tb_writer.add_scalar('te_metric',error_dict['overall_by_domain'], epoch)


def convert_to_png(img,low_num,high_num):
    x = np.array([low_num*1.,high_num * 1.])
    newimg = (img-x[0])/(x[1]-x[0])  
    newimg = (newimg*255).astype('uint8')  
    return newimg

def save_teimgs(pred,img,gth,logdir,index,opt,tmp_mp):
    img1=img
    img1=convert_to_png(img1,img1.min(),img1.max())
    img1 = Image.fromarray(img1,mode='L')
    img1.save(logdir+'/'+str(index)+'img.png')
    
    img1=gth
    img1=deal_wit_lbvis(tmp_mp,img1,opt.nclass)
    img1=img1.cpu().numpy()
    img1=img1.astype('uint8')
    img1 = Image.fromarray(img1,mode='L')
    img1.save(logdir+'/'+str(index)+'gth.png')
    
    img1=pred
    img1=deal_wit_lbvis(tmp_mp,img1,opt.nclass)
    img1=img1.cpu().numpy()
    img1=img1.astype('uint8')
    img1 = Image.fromarray(img1,mode='L')
    img1.save(logdir+'/'+str(index)+'pred.png')
    

def save_trimgs(tr_viz,logdir,iternum,opt,tmp_mp):
    x=0
    size0=tr_viz['img_ori'].shape[0]
    for i in range(size0):
        if len(np.unique(tr_viz['gth_tr'][i]))>1  and len(np.unique(tr_viz['seg_tr'][i]))>1:
            x=i
            break
  
    img=tr_viz['img_ori']
    img=img[x]
    img=img[0] 
    img=convert_to_png(img,img.min(),img.max())
    img = Image.fromarray(img,mode='L')
    img.save(logdir+'/img/'+str(int(iternum))+'ori'+'.png')

    img=tr_viz['img_tr1']
    img=img[x]
    img=np.mean(img, axis=0) 
    img=convert_to_png(img,img.min(),img.max())
    img = Image.fromarray(img,mode='L')
    img.save(logdir+'/img/'+str(int(iternum))+'img1'+'.png')
    
    if opt.fmethod=='asymr' :
       img=tr_viz['img_tr2']
       img=img[x]
       img=np.mean(img, axis=0) 
       img=convert_to_png(img,img.min(),img.max())
       img = Image.fromarray(img,mode='L')
       img.save(logdir+'/img/'+str(int(iternum))+'img2'+'.png')

    img=tr_viz['seg_tr']
    img=img[x]
    img=img[0] 
    img=deal_wit_lbvis(tmp_mp,img,opt.nclass)
    img=img.cpu().numpy()
    img=img.astype('uint8')
    img = Image.fromarray(img,mode='L')
    img.save(logdir+'/img/'+str(int(iternum))+'seg'+'.png')

    img=tr_viz['gth_tr']
    img=img[x]
    img=img[0] 
    img=deal_wit_lbvis(tmp_mp,img,opt.nclass)
    img=img.cpu().numpy()
    img=img.astype('uint8')
    img = Image.fromarray(img,mode='L')
    img.save(logdir+'/img/'+str(int(iternum))+'gt'+'.png')

    
    if opt.contrast_f==1:
        img=tr_viz['contrast_pred']
        img=img[x]
        img=img[0]
        img=convert_to_png(img,img.min(),img.max())
        img = Image.fromarray(img,mode='L')
        img.save(logdir+'/img/'+str(int(iternum))+'cpred'+'.png')
    
        img=tr_viz['contrast_label']
        img=img[x]
        img=img[0]
        img=convert_to_png(img,img.min(),img.max())
        img = Image.fromarray(img,mode='L')
        img.save(logdir+'/img/'+str(int(iternum))+'clab'+'.png')
        
        img=tr_viz['contrast_mask']
        img=img[x]
        img=img[0]
        print("contrast img:",img.shape)
        img=convert_to_png(img,img.min(),img.max())
        img = Image.fromarray(img,mode='L')
        img.save(logdir+'/img/'+str(int(iternum))+'cmask'+'.png')


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def setup_default_logging(name, save_path, level=logging.INFO, 
                          format="[%(asctime)s][%(levelname)s] - %(message)s"):
    tmp_timestr = time_str()
    logger = logging.getLogger(name)
    logging.basicConfig(
        filename=os.path.join(save_path, f'seg_{tmp_timestr}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level)
    return logger


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--expname', type=str, default='1', help='expname')
    parser.add_argument('--phase', type=str, default='train', help='train or test')

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu')
    parser.add_argument('--f_seed', type=int, default=1, help='seed')
    parser.add_argument('--f_determin', type=int, default=1, help='determin')
    parser.add_argument('--lr', type=float, default=0.0005, help='lr')
    parser.add_argument('--model', type=str, default='unet', help='model')
    parser.add_argument('--batchSize', type=int, default=20, help='bs')
    parser.add_argument('--all_epoch', type=int, default=600, help='epochs')

    parser.add_argument('--data_name', type=str, default='PROSTATE', help='dataset')
    parser.add_argument('--nclass', type=int, default=2, help='nclass')
    parser.add_argument('--tr_domain', type=str, default='F', help='src_domain')
    parser.add_argument('--save_prediction', type=bool, default=True, help='save_pred')

    parser.add_argument('--validation_freq', type=int, default=50, help='valfreq')
    parser.add_argument('--testfreq', type=int, default=50, help='testfreq')
    parser.add_argument('--display_freq', type=int, default=500, help='imgfreq')

    parser.add_argument('--w_ce', type=float, default=1.0, help='w_ce')
    parser.add_argument('--w_dice', type=float, default=1.0, help='w_dice')
    parser.add_argument('--w_seg', type=float, default=1.0, help='w_seg')
    parser.add_argument('--w_consist', type=float, default=1.0, help='w_consist')
    parser.add_argument('--w_contrast', type=float, default=1.0, help='w_contrast')

    parser.add_argument('--consist_f', type=int, default=1, help='f_feature')
    parser.add_argument('--fmethod', type=str, default='asymr', help='fmethod')
    parser.add_argument('--contrast_f', type=int, default=1, help='contrast_f')
    
    
    parser.add_argument('--num_augs1', type=int, default=6, help='num_augs1')
    parser.add_argument('--augflag1', type=bool, default=True, help='augflag1')

    parser.add_argument('--f_dropout1', type=int, default=0, help='f_dropout1')
    parser.add_argument('--dropout_rate1', type=float, default=0.0, help='dropout_rate1')
    parser.add_argument('--f_dropout2', type=int, default=1, help='f_dropout2')
    parser.add_argument('--dropout_rate2', type=float, default=0.5, help='dropout_rate2')
    
    
    parser.add_argument('--gls_nlayer', type=int, default=4, help='nlayer')
    parser.add_argument('--gls_interm', type=int, default=2, help='interm')
    parser.add_argument('--gls_outnorm', type=str, default='frob', help='outnorm')
    parser.add_argument('--glsmix_f', type=int, default=1, help='mix_f')
    parser.add_argument('--mixalpha', type=float, default=0.2, help='mixalpha')

   
    parser.add_argument('--temperature', type=float, default=0.05, help='temperature')
    parser.add_argument('--n_view', type=int, default=10, help='n_view')

    args = parser.parse_args()
    return args
    
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    opt=get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(opt.f_seed)
    np.random.seed(opt.f_seed)
    torch.manual_seed(opt.f_seed)
    torch.cuda.manual_seed(opt.f_seed)

    ckpt_dir='../ckpts/'
    dn_dir='../ckpts/'+opt.tr_domain+'/'
    exp_dir=dn_dir+opt.expname+'/'
    snap_dir=exp_dir+'snapshots/'
    tbfile_dir =exp_dir+'tboard/'
    logdir=exp_dir+'log/'
    
    if opt.phase=='train':
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)  
        shutil.copytree('.',exp_dir+'code', shutil.ignore_patterns(['.git','__pycache__']))
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(dn_dir):
        os.makedirs(dn_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(snap_dir):
        os.mkdir(snap_dir)
    if not os.path.exists(tbfile_dir):
        os.mkdir(tbfile_dir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        os.mkdir(logdir+'train')
        os.mkdir(logdir+'img')
        os.mkdir(logdir+'pred')

    finalfile=logdir+'out.csv' 

    logger = logging.getLogger('log1')
    logging.basicConfig(filemode='a', level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.FileHandler(exp_dir+'log.txt',encoding='utf-8'))
    logging.info("config:"+str(opt))
    logging.info("name:"+opt.expname)
    logging.info("f_determin:"+str(opt.f_determin))
    logging.info("opt.f_seed:"+str(opt.f_seed))
    logging.info("data:"+opt.data_name)
    
    tb_writer = SummaryWriter( tbfile_dir  )

    with open(finalfile, 'a') as f:
        f.write(opt.expname+' '+opt.model+" \n")
    
    labmap=pre_labmap()
    labmap=labmap[opt.data_name]
    print("labmap:",labmap)
    
    if opt.data_name == 'ABDOMINAL':
        if opt.tr_domain=='SABSCT':
            tr_domain=['SABSCT']
            te_domain =['CHAOST2']
        else:
            tr_domain=['CHAOST2']
            te_domain =['SABSCT']

        train_set  = ABD.get_training(modality = tr_domain ,norm_func = None,opt = opt)
        tr_valset  = ABD.get_trval(modality = tr_domain, norm_func = train_set.normalize_op,opt = opt) 
        tr_teset   = ABD.get_trtest(modality = tr_domain, norm_func = train_set.normalize_op,opt = opt)
        test_set   = ABD.get_test(modality = te_domain, norm_func = None,opt = opt) 
       
        label_name          = ABD.LABEL_NAME

    # elif opt.data_name == 'PROSTATE':
    #     tr_domain=[opt.tr_domain]
    #     train_set  = PROS.get_training(modality = tr_domain , opt = opt)
    #     tr_valset  = PROS.get_trval(modality = tr_domain, opt = opt)
    #     tr_teset   = PROS.get_trtest(modality = tr_domain, opt = opt)
    #     test_set   = PROS.get_test(tr_modality = tr_domain, opt = opt)
      
    #     label_name      = PROS.LABEL_NAME
      
    # elif opt.data_name == 'CARDIAC':
    #     if opt.tr_domain=='LGE':
    #         tr_domain=['LGE']
    #         te_domain=['bSSFP']
    #     else:
    #         tr_domain=['bSSFP']
    #         te_domain=['LGE']
    #     train_set       = cardiac_cls.get_training(modality = tr_domain , opt = opt)
    #     tr_valset  = cardiac_cls.get_trval(modality = tr_domain , opt = opt)
    #     tr_teset  = cardiac_cls.get_trtest(modality = tr_domain , opt = opt)#as dataset split,cardiac didn't have this
    #     test_set        = cardiac_cls.get_test(modality = te_domain , opt = opt)
        
    #     label_name      = cardiac_cls.LABEL_NAME

    else:
        print('not implement this dataset',opt.data_name)


    train_loader = DataLoader(dataset = train_set, num_workers = 8,\
            batch_size = opt.batchSize, shuffle = True, drop_last = True, worker_init_fn = worker_init_fn, pin_memory = True)
    trval_loader=DataLoader(dataset = tr_valset, num_workers = 1,batch_size = 1, shuffle = False, pin_memory = True)
    trte_loader=DataLoader(dataset = tr_teset, num_workers = 1,batch_size = 1, shuffle = False, pin_memory = True)
    test_loader = DataLoader(dataset = test_set, num_workers = 1,batch_size = 1, shuffle = False, pin_memory = True)
    
    model = Train_process(opt,'0',0)
    
    total_steps,iternum = 0,0

    for epoch in range(1, opt.all_epoch + 1):
        print("------------------------epoch  "+str(epoch)+"-----------------------------")
        epoch_start_time = time.time()
      
        for i, train_batch in tqdm(enumerate(train_loader), total = train_loader.dataset.size // opt.batchSize - 1):
                print("iternum:",iternum)
                total_steps=total_steps+opt.batchSize
                iternum=iternum+1
                # avoid batchsize issues caused by fetching last training batch
                if train_batch["img1"].shape[0] != opt.batchSize:
                    continue
                
                tr_log=model.tr_func(train_batch,epoch)

                for key, x in tr_log.items():
                    tb_writer.add_scalar(key, x,iternum)
                    
                    with open(osp.join(logdir+'train/',key+'.csv'), 'a') as f:
                        log1=[[iternum] + [x.item()]]
                        log1=map(str, log1)
                        f.write(','.join(log1) + '\n')


                logger.info("Tr-Epoch:{},Iter:{},Lr:{:.5f}--loss:{:5f} con:{:5f} seg:{:5f} contra:{:5f} dc:{:5f} ce:{:5f}, seg1:{:5f} seg2:{:5f} dc1:{:5f} dc2:{:5f} ce1:{:5f} ce2:{:5f}".format(\
                        epoch, iternum,model.get_lr().item(),\
                        model.loss_seg.item()+model.loss_consist.item()+model.loss_contrast.item(),\
                        model.loss_consist.item(),model.loss_seg.item(),model.loss_contrast.item(),\
                        model.loss_dice.item(),model.loss_ce.item(),\
                        model.loss_seg1.item(),model.loss_seg2.item(),\
                        model.loss_dice1.item(),model.loss_dice2.item(),\
                        model.loss_ce1 .item(),model.loss_ce2 .item()))
                
    
                if iternum % opt.display_freq == 0:
                    tr_viz = model.get_img_tr()
                    save_trimgs(tr_viz,logdir,iternum,opt,labmap)
                   
     
 
        #val for tr_val
        if epoch % opt.validation_freq == 0:
            with torch.no_grad():
                type1='val'
                tmp= prediction_wrapper(tb_writer,type1,logdir,model, trval_loader,  epoch, label_name, save_prediction =False)

        
        if epoch == opt.all_epoch:
            print('saving the model at the end of epoch %d, iters %d' %(epoch, iternum))
            model.save(snap_dir,epoch)

        lr = opt.lr * (1 - epoch /opt.all_epoch )
        model.optimizer_seg.param_groups[0]['lr']=lr
        print('End of epoch %d / %d \t Time Taken: %d sec' %(epoch, opt.all_epoch, time.time() - epoch_start_time))

   
    if(total_steps>=0):
        print('final test epoch %d, iters %d' %(opt.all_epoch, total_steps))
        reload_model_fid=snap_dir+str(opt.all_epoch)+'_net_Seg.pth'
        print("reload_model_fid:",reload_model_fid)
        model1=Train_process(opt,reloaddir=reload_model_fid,istest=1)

        with torch.no_grad():
            with open(finalfile, 'a') as f:
                f.write("test testset final\n")
            type1='testfinal'
            preds= prediction_wrapper(tb_writer,type1,logdir,model1, test_loader,opt.all_epoch, label_name, save_prediction = opt.save_prediction)

            if opt.save_prediction==True:
                xx=0
                for scan_id, comp in preds.items():
                    _pred = comp['pred'].detach().data.cpu().numpy()
                    _img = comp['img'].detach().data.cpu().numpy()
                    _gt = comp['gth'].detach().data.cpu().numpy()   
                    for yy in range(_pred.shape[0]):
                       if len(np.unique(_gt[yy]))>1:
                           save_teimgs(_pred[yy],_img[yy],_gt[yy],logdir+'/pred',xx,opt,labmap)
                           xx=xx+1

                    if xx>=50:
                        break
                    
            print('\ntest for source domain')
            with open(finalfile, 'a') as f:
                 f.write("\n\ntest for source domain \n")          
            type1='tetrainfinal'  
            tmp= prediction_wrapper(tb_writer,type1,logdir,model1, trte_loader, opt.all_epoch, label_name, save_prediction = opt.save_prediction)
          
