import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loss_function import loss_func
from model import RCML
from utils import get_datasets, write_logs, loss_show, metrics,dice_show,linear_warmup
from datetime import datetime
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler


def main(args):
    train_dataset, val_dataset = get_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = RCML(args).float()

    optimizer = optim.RAdam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
    scaler = GradScaler()
    torch.cuda.set_device(args.gpu_id)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_losses,multi_epochs_dice = list(),list()
    metric = 0
    num_modality = len(args.modalities)
    # 训练
    for epoch in range(1, args.train_epochs + 1):
        torch.cuda.empty_cache()
        log_str = f'======>epoch: {epoch}/{args.train_epochs}'
        write_logs(args, log_str)
        train_loader_index = 0
        loss_list,dsc_list=list(),list()
        linear_warmup(optimizer, epoch,warmup_epochs=args.warmup_epochs)
        for X, Y, indexes in train_loader:
            train_loader_index = train_loader_index+1
            for n in range(num_modality):
                X[args.modalities[n]] = X[args.modalities[n]].to(device,dtype=torch.float32)
            Y = Y.to(device,dtype=torch.float32)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                evidences, evidence_a = model(X)            
                loss,l_conflict,CE_loss,KL_loss = loss_func(evidences,evidence_a,Y,args,epoch,device)
            _, Y_pre = torch.max(evidence_a, dim=2)
            metric_family= metrics(Y_pre, Y[:,:,1,:,:])
            dsc_list.append(metric_family[0])
            if train_loader_index % 50 == 1 or train_loader_index == len(train_loader):
                log_str = f"train_loader: {train_loader_index:<4}/{len(train_loader)} total loss: {loss:.4f};CE_loss: {CE_loss:.4f};conflict degree: {l_conflict:.4f};KL loss: {KL_loss:.4f}"
                write_logs(args, log_str)
            loss_list.append(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        metric = torch.stack(loss_list).mean()
        dice=sum(dsc_list) / len(dsc_list)
        log_str = f"average loss: {metric:.4f} DSC:{dice:.4f}"
        write_logs(args, log_str)
        train_losses.append(metric.item())
        multi_epochs_dice.append(dice)

    modal_path = f"{args.save_model_path}/{args.dataset_name}_lr-{args.lr}_train-epochs-{args.train_epochs}_gamma-{args.gamma}_{args.choice}_fold{args.fold}_time{args.time}.pth"
    torch.save(model.state_dict(), modal_path)
    log_str = f"training loss of {args.train_epochs}={train_losses}\ndsc of {args.train_epochs}={multi_epochs_dice}"
    write_logs(args, log_str)
    loss_show(train_losses,args.fold,args.time,args.dataset_name)
    dice_show(multi_epochs_dice,args.fold,args.time,args.dataset_name,figure_path='train')

    # 验证
    index_list = val_loader.dataset.samples_index
    log_str = f'val dataset index in original dataset: {index_list}'
    write_logs(args, log_str)
    model.eval()
    with torch.no_grad():
        eval_loader_index = 0
        dice_list, iou_list, acc_list, prec_list, sens_list, spec_list, hd95_list, asd_list = list(), list(), list(), list(), list(), list(), list(), list()
        for X, Y, indexes in val_loader:
            eval_loader_index = eval_loader_index+1  
            if eval_loader_index % 10 == 1 or eval_loader_index == len(val_loader):
                log_str = f"eval_loader:{eval_loader_index}/{len(val_loader)}"
                write_logs(args, log_str)
            
            for n in range(num_modality):
                X[val_dataset.modalities[n]] = X[val_dataset.modalities[n]].to(device,dtype=torch.float32)
            Y = Y.to(device,dtype=torch.float32)
            evidences, evidence_a = model(X)
            _, Y_pre = torch.max(evidence_a, dim=2)
            
            metric_family= metrics(Y_pre, Y[:,:,1,:,:])
            dice_list.append(metric_family[0])
            iou_list.append(metric_family[1])
            acc_list.append(metric_family[2])
            prec_list.append(metric_family[3])
            sens_list.append(metric_family[4])
            spec_list.append(metric_family[5])
            if not torch.all(Y[:,:,1,:,:]==0) and not torch.all(Y_pre==0):
                hd95_list.append(metric_family[6])  
                asd_list.append(metric_family[7])

    # 评估
    metric_dice = np.mean(dice_list)
    metric_iou = np.mean(iou_list)
    metric_acc = np.mean(acc_list)
    metric_prec = np.mean(prec_list)
    metric_sens = np.mean(sens_list)
    metric_spec = np.mean(spec_list)
    metric_hd95 = np.mean(hd95_list) 
    metric_asd = np.mean(asd_list)
    log_str = f'====> metric_dice: {metric_dice:.4f} metric_iou: {metric_iou:.4f}, \
    metric_acc: {metric_acc:.4f}, metric_prec: {metric_prec:.4f}, metric_sens: {metric_sens:.4f}, \
    metric_spec: {metric_spec:.4f}, metric_hd95: {metric_hd95:.4f}, metric_asd: {metric_asd:.4f}'
    write_logs(args, log_str)
    log_str = f'begin time at {args.time}, finish time at {datetime.now().strftime("%Y-%m%d-%H%M")}'
    write_logs(args, log_str)
    return metric_dice, metric_iou, metric_acc, metric_prec, metric_sens, metric_spec, metric_hd95, metric_asd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=2, help='GPU id')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',help='input batch size for training')
    parser.add_argument('--train_epochs', type=int, default=100, metavar='N',help='number of epochs to train')
    parser.add_argument('--annealing_step', type=int, default=50 , metavar='N',help='annealing step for training')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate')
    parser.add_argument('--dataset_dir',type=str, default='/media/admin1/yqh/ECSN/data/ISLES-2022',help='dataset absolute path')
    parser.add_argument('--modalities', default=["adc", "dwi", "flair"],help='modality,vary according to dataset')
    parser.add_argument('--save_model_path',type=str, default='/media/admin1/yqh/ECSN/model/isles22')
    parser.add_argument('--log_path', type=str, default='/media/admin1/yqh/ECSN/logs/isles22')
    parser.add_argument('--dataset_name', type=str, default='ISLES2022',choices=['ISLES2022', 'ISLES2024','APIS'])
    parser.add_argument('--gamma', type=float, default=0.1, help='weight for the conflict degree')
    parser.add_argument('--time', type=str, default=f'{datetime.now().strftime("%Y-%m%d-%H%M")}')
    parser.add_argument('--seed', type=int, default=42, help='random seed for training')
    parser.add_argument('--ablation', type=str, default='None', help='None, dis-KL, dis-conflict, dis-KL_conflict')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--use_checkpoint', action='store_false', help='whether use checkpoint')
    parser.add_argument('--align_img',default=(72,112,112), type=tuple, help='image size')

    args = parser.parse_args()

    command_line = ' '.join(sys.argv)
    write_logs(args, command_line)
    # fix random seed
    random.seed(args.seed) 
    os.environ['PYTHONHASHSEED'] = str(args.seed) 
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

    log_str = f'-----start the experiment at {args.time}-----'
    write_logs(args, log_str)
    single_dice, single_iou, single_acc, single_prec, single_sens, single_spec, single_hd95, single_asd = main(args)
    log_str = f'====> The {args.fold}th fold experimental measure in 5 fold cross validation:\n\
    DSC: {single_dice:.4f} iou: {single_iou:.4f} accuracy: {single_acc:.4f} precision: {single_prec:.4f}\
    sensitivity: {single_sens:.4f} specificity: {single_spec:.4f} hd95: {single_hd95:.4f} asd: {single_asd:.4f}'
    write_logs(args, log_str)
