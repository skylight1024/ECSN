import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1,1, num_classes,alpha.shape[-2],alpha.shape[-1]], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=2, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=2, keepdim=True)
        + torch.lgamma(ones).sum(dim=2, keepdim=True) 
        - torch.lgamma(ones.sum(dim=2, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=2, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def conflict_loss(evidences,evidence_fusion, modalities, device):
    num_modality = len(modalities)
    batch_size, num_slice, num_class, width, height = \
        (evidences[modalities[0]].shape[0], evidences[modalities[0]].shape[1],
         evidences[modalities[0]].shape[2], evidences[modalities[0]].shape[3],
         evidences[modalities[0]].shape[4])
    probability = dict()
    uncertainty = dict()
    c_sum,c_p = 0,0
    for modality in modalities:

        probability[modality] = torch.zeros((batch_size, num_slice, num_class, width, height)).to(device)
        uncertainty[modality] = torch.zeros((batch_size, num_slice, width, height)).to(device)

        alpha = evidences[modality] + 1
        S = torch.sum(alpha, dim=2, keepdim=True)
        probability[modality] = alpha / S
        uncertainty[modality] = torch.squeeze(num_class/S, dim=2)
    alpha_fusion = evidence_fusion + 1
    probability_fusion = alpha_fusion / torch.sum(alpha_fusion, dim=2, keepdim=True)
    
    
    if len(modalities) == 1:
        uncertainty_fusion = uncertainty[modalities[0]]
    else:
        uncertainty_fusion = uncertainty[modalities[0]]
        for i in range(1,len(modalities)):
            uncertainty_fusion = 2*uncertainty[modalities[i]] * uncertainty_fusion/(uncertainty[modalities[i]] + uncertainty_fusion)
    for mod in modalities:
        cos_sim = F.cosine_similarity(probability[mod], probability_fusion, dim=2)
        
        c_p = 0.5*(1 - cos_sim) 
        c_c = 0.5*(uncertainty[mod]+uncertainty_fusion)
        c = 0.5*(c_p+c_c)
        c_sum = c_sum+c

    c_sum = c_sum/num_modality
    loss_c = torch.mean(c_sum)

    return loss_c


def dice_coefficient(y_pred, y_label, eps=1e-6):
    """
    计算Dice系数
    """
    y_pred = y_pred.float()
    y_label = y_label.float()
    intersection = (y_pred * y_label).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_label.sum(dim=(2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

def get_DICE_loss(evidences, Y, modalities):

    K = Y.shape[2]
    result=0
    for modality in modalities:
        alpha = evidences[modality] + 1  
        S = torch.sum(alpha, dim=2, keepdim=True)
        term_numerator =torch.sum(Y*alpha / S, dim=(3,4))
        term_denominator = torch.sum((Y**2 + (alpha / S)**2 + (alpha * (S - alpha)) / (S**2 * (S + 1))), dim=(3, 4)) 
        sum1 = torch.sum(term_numerator / term_denominator, dim=2)
        result = result + 1 - (2 / K) * torch.mean(sum1, dim=(0,1))
    return result

def get_KL_loss(evidences,y,num_classes,modalities,epoch_num,annealing_step,device):

    kl_div=0
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32))
    for modality in modalities:
        alpha = evidences[modality] + 1
        S = torch.sum(alpha, dim=2, keepdim=True)
        kl_alpha = alpha * (1 - y) + y 
        kl_div = kl_div + annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    result = torch.mean(kl_div)
    return result

def loss_func(evidences,evidence_a,Y,args,epoch,device):
    l_conflict, CE_loss, KL_loss=torch.tensor(0.0,device=device,requires_grad=True),torch.tensor(0.0,device=device,requires_grad=True),torch.tensor(0.0,device=device,requires_grad=True)
    l_conflict = args.gamma*conflict_loss(evidences, evidence_a, args.modalities, device)
    CE_loss = get_DICE_loss(evidences, Y.float(), args.modalities)
    KL_loss = get_KL_loss(evidences,Y,args.num_classes,args.modalities,epoch,args.annealing_step,device)
    if args.ablation == 'None':
        loss = CE_loss + KL_loss + l_conflict
    elif args.ablation == 'dis-KL':
        loss = CE_loss + l_conflict
    elif args.ablation == 'dis-conflict':
        loss = CE_loss + KL_loss
    elif args.ablation == 'dis-KL_conflict':
        loss = CE_loss
    else:
        print('ablation error, please check the ablation name!')
    return loss, l_conflict, CE_loss, KL_loss

def dice_loss(inputs, targets, smooth=1e-6):
    _, Y_pre = torch.max(inputs, dim=2)
    Y_pre = Y_pre.view(-1)
    targets = targets.reshape(-1)
    
    intersection = (Y_pre * targets).sum()
    dice = (2. * intersection + smooth) / (Y_pre.sum() + targets.sum() + smooth)
    
    return 1 - dice
