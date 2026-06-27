import pathlib
import SimpleITK as sitk
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
import cv2
import os
from datetime import datetime
import matplotlib.pyplot as plt
from loss_function import dice_coefficient
from medpy.metric.binary import hd95 as hd95_v2
from medpy.metric.binary import asd as asd_v2

class Isles(Dataset):
    def __init__(self, data_dirs, samples_index, args, isval=False):
        super(Isles, self).__init__()
        self.data_dirs = data_dirs  # 数据的目录列表
        self.modalities = args.modalities+["seg"] # ["adc", "dwi", "flair", "seg"]
        self.num_modality = len(self.modalities)-1
        self.num_classes = 2
        self.num_channels = 1
        self.dataset_name = args.dataset_name # 'ISLES2022' or 'ISLES2024' 'APIS'
        self.X = []
        self.Y = []
        self.val = isval  # 是否为验证集
        self.niigzfile = dict()  # 读取到的nii.gz文件
        self.samples_index = samples_index
        self.niigzfile_attr = list()
        

        self.align_img=args.align_img
        for data_dir in data_dirs:
            data_id = data_dir.name
            if args.dataset_name == "ISLES2022":
                seg_dir = data_dir.parent.joinpath("derivatives", data_id)
                paths = [data_dir/"ses-0001"/"dwi"/f"{data_id}_ses-0001_adc_compress.nii.gz",
                        data_dir / "ses-0001" / "dwi" / f"{data_id}_ses-0001_dwi_compress.nii.gz",
                        data_dir / "ses-0001" / "anat" / f"{data_id}_ses-0001_FLAIR_compress.nii.gz",
                        seg_dir / "ses-0001" / f"{data_id}_ses-0001_msk.nii.gz"]

                patient = dict(id=data_id, adc=paths[0], dwi=paths[1], flair=paths[2], seg=paths[3])
            elif args.dataset_name =="ISLES2024":
                seg_dir = data_dir.parent.parent.joinpath("derivatives", data_id)
                paths = [
                         data_dir / "ses-02" / f"{data_id}_ses-02_adc_compress.nii.gz",
                         data_dir / "ses-02" / f"{data_id}_ses-02_dwi_compress.nii.gz",
                         seg_dir / "ses-02" / f"{data_id}_ses-02_lesion-msk_compress.nii.gz"]
                patient = dict(id=data_id, adc=paths[0], dwi=paths[1],seg=paths[2])
            elif args.dataset_name == "APIS":
                seg_dir = data_dir.joinpath('masks')
                paths =[
                        data_dir / f"{data_id}_adc_compress.nii.gz",
                        data_dir / f"{data_id}_ncct_compress.nii.gz",
                        seg_dir / f"{data_id}_r1_mask_compress.nii.gz"]
                patient = dict(id=data_id, adc=paths[0], ncct=paths[1],seg=paths[2])

            whole_image_path = patient
            whole_image = {key: self.load_nii(whole_image_path[key]) for key in whole_image_path if key not in ["id", "seg"]}
            patient_image = {key: min_max_preprocess(whole_image[key]) for key in whole_image}
            patient_mask = self.load_nii(whole_image_path["seg"])
            patient_X, patient_Y = self.preprocess(patient_image, patient_mask,align_img=self.align_img)

            for modality in self.modalities:  # 在（10，72，112，112）数据上增加一个维度，变成（10，72，1，112，112）
                if modality == "seg":
                    continue
                else:
                    patient_X[modality] = np.expand_dims(patient_X[modality], axis=1)  # 增加一个维度（1，72，112，112）->（1，72，1，112，112）
            patient_Y = np.expand_dims(patient_Y, axis=1) # 增加一个维度（1，72，112，112）->（1，72，1，112，112）
            patient_Y = self.exchange_label(patient_Y)  # 将标签的值，从一个维度扩展到2个维度 shape[0](第0个通道)是背景类别，shape[1]（第1个通道）是前景类别                   
            self.X.append(patient_X)
            self.Y.append(patient_Y)  # 两个ndarray类型的数据（1，112，112），拼接一起为（2，112，112）

        self.length = len(self.X)

    def __getitem__(self, index):
        patient_X = self.X[index]
        patient_Y = self.Y[index]
        return patient_X, patient_Y, index




    def __len__(self):
        return self.length

    def exchange_label(self, patient_Y):  # 将标签的值，从一个维度扩展到2个维度 shape[0](第0个通道)是背景类别，shape[1]（第1个通道）是前景类别
        temp0 = 1-patient_Y  # (72,1,112,112)
        temp1 = patient_Y  # (72,1,112,112)
        result = np.concatenate((temp0,temp1),axis=1)  # (72,2,112,112)
        return result


    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def preprocess(self, patient_image, patient_mask, align_img=(72, 112, 112)):
        """
        对齐数据维度
        :param patient_image:
        :param patient_mask:
        :param patient_image, patient_mask, align_img:
        :return:patient_image_aligned, patient_mask_aligned
        """
        patient_image_aligned = patient_image.copy()
        patient_mask_aligned = patient_mask.copy()
        patient_image["seg"] = patient_mask  # 将数据和掩码一起处理

        for modality in range(self.num_modality+1):
            img_0 = patient_image[self.modalities[modality]].shape[0]  # 数据单一模态的切片数量，第1个维度
            img_1 = patient_image[self.modalities[modality]].shape[1]  # 宽度，第2个维度
            img_2 = patient_image[self.modalities[modality]].shape[2]  # 高度，第3个维度
            temp = np.zeros(align_img)
            if img_1 <= align_img[1] and img_2 <= align_img[2] and (img_1 < align_img[1] or img_2 < align_img[2]):
            # if (img_1 < align_img[1] and img_2 < align_img[2]) or (img_1 < align_img[1] and img_2 <= align_img[2]) or (img_1 <= align_img[1] and img_2 < align_img[2]):  # 扩充宽度和高度
                (weight_l,weight_r) = (np.floor((align_img[1] - img_1)/2), np.ceil((align_img[1] - img_1)/2))
                (height_l,height_r) = (np.floor((align_img[2] - img_2)/2), np.ceil((align_img[2] - img_2)/2))

                padding = ((0,0), (weight_l,weight_r), (height_l,height_r))
                for i in range(img_0):
                    align_array = np.pad(patient_image[self.modalities[modality]][i], pad_width=padding, mode="constant", constant_values=0)
            elif img_1 >= align_img[1] and img_2 >= align_img[2] and (img_1 > align_img[1] or img_2 > align_img[2]):  # 剪切宽和高
                start_w = (img_1 - align_img[1])//2
                start_h = (img_2 - align_img[2])//2
                align_array = patient_image[self.modalities[modality]][:, start_w:start_w+align_img[1], start_h:start_h+align_img[2]]
            elif img_1 <= align_img[1] and img_2 >= align_img[2] and (img_1 < align_img[1] or img_2 > align_img[2]):  # 扩充宽，剪切高裁剪高
                (weight_l, weight_r) = (np.floor((align_img[1] - img_1) / 2), np.ceil((align_img[1] - img_1) / 2))
                padding = ((0, 0), (weight_l, weight_r), (0, 0))
                for i in range(img_0):
                    align_array0 = np.pad(patient_image[self.modalities[modality]][i], pad_width=padding, mode="constant", constant_values=0)
                start_h = (img_2 - align_img[2]) // 2
                align_array = align_array0[:, :, start_h:start_h+align_img[2]]
            elif img_1 >= align_img[1] and img_2 <= align_img[2] and (img_1 > align_img[1] or img_2 < align_img[2]):  # 裁剪宽，扩充高
                (height_l, height_r) = (np.floor((align_img[2] - img_2) / 2), np.ceil((align_img[2] - img_2) / 2))
                padding = ((0, 0), (0, 0), (height_l, height_r))
                for i in range(img_0):
                    align_array0 = np.pad(patient_image[self.modalities[modality]][i], pad_width=padding, mode="constant", constant_values=0)
                start_w = (img_1 - align_img[1]) // 2
                align_array = align_array0[:, start_w:start_w+align_img[1], :]
            else:
                align_array = patient_image[self.modalities[modality]]
            if self.modalities[modality] == 'seg':
                patient_mask_aligned = align_array
            else:
                patient_image_aligned[self.modalities[modality]] = align_array

        img_0 = patient_image[self.modalities[0]].shape[0]  # 数据单一模态的切片数量;假设不同模态的尺寸，切片数量相同
        temp = np.zeros(align_img)
        # 保证每个模态的随机插入和删除的切片位置相同
        if img_0 < align_img[0]:  # 扩充切片数量
            random_indices = np.sort(np.random.choice(range(align_img[0]), img_0, replace=False))
            for modality in range(self.num_modality+1):
                if self.modalities[modality] == 'seg':
                    align_array = patient_mask_aligned
                else:
                    align_array = patient_image_aligned[self.modalities[modality]]
                for index, value in zip(random_indices, align_array):  # 将align_array中的值替换到temp的随机位置
                    temp[index] = value
                remaining_indices = [i for i in range(align_img[0]) if i not in random_indices]
                for i in remaining_indices:  # 遍历列表，将剩余的0替换为前面不为0的值
                    if i != 0:
                        temp[i] = temp[i-1]
                    else:
                        temp[i] = np.zeros((align_img[1], align_img[2]))
                if self.modalities[modality] == 'seg':
                    patient_mask_aligned = temp.copy()
                else:
                    patient_image_aligned[self.modalities[modality]] = temp.copy()

        elif img_0 > align_img[0]:  # 缩减切片数量
            random_indices = np.sort(np.random.choice(range(img_0), align_img[0], replace=False))  # 随机选出固定数量的切片
            for modality in range(self.num_modality+1):
                if self.modalities[modality] == 'seg':
                    align_array = patient_mask_aligned
                    temp = align_array[random_indices]
                    patient_mask_aligned = temp.copy()
                else:
                    align_array = patient_image_aligned[self.modalities[modality]]
                    temp = align_array[random_indices]
                    patient_image_aligned[self.modalities[modality]] = temp.copy()


        return patient_image_aligned, patient_mask_aligned



    def load_nii(self, path_folder):
        # self.niigzfile = sitk.ReadImage(str(path_folder))  # 额外编写一个函数用来记录所有样本的nii.gz属性
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))
        return image_array
    
    def get_niigz_attr(self, sitk_image):
        result=dict()
        for modality in sitk_image:
            result[modality]=dict()
            result[modality]['space'] = sitk_image[modality].GetSpacing()
            result[modality]['origin'] = sitk_image[modality].GetOrigin()
            result[modality]['direction'] = sitk_image[modality].GetDirection()
        return result
        
# region get_datasets
def get_datasets(args): # dataset_dir相对路径
    """
    收集数据集中的数据
    :param dataset_dir:数据集目录

    :return:
    """
    # 读取目录下的所有文件名-》挑选出带有sub-strokecase的文件名-》读取每个文件名下的ses-001里的文件-》读取dwi文件和anat文件中的.nii.gz文件,并行存放成三种模态
    if args.dataset_name=="ISLES2022":
        base_folder = pathlib.Path(args.dataset_dir).resolve()
        patients_dir = sorted([x for x in base_folder.iterdir() if "sub-strokecase" in x.name])
    elif args.dataset_name =="ISLES2024":
        base_folder = pathlib.Path(args.dataset_dir,'raw_data').resolve()
        patients_dir = sorted([x for x in base_folder.iterdir() if "sub-stroke" in x.name])
    elif args.dataset_name =="APIS":
        base_folder = pathlib.Path(args.dataset_dir,'train').resolve()
        patients_dir = sorted([x for x in base_folder.iterdir() if "train" in x.name])
    else:
        raise ValueError("Unknown dataset name: {}".format(args.dataset_name))

    # 创建5折交叉验证对象
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    splits = list(kf.split(patients_dir))
    train_index, val_index = splits[args.fold]  # 生成训练集与验证集索引
    train = [patients_dir[i] for i in train_index]
    val = [patients_dir[i] for i in val_index]

    train_dataset = Isles(train, train_index,args,isval=False)
    val_dataset = Isles(val, val_index,args,isval=True)
    return train_dataset, val_dataset
# endregion
# region get_test_dataset
def get_test_dataset(args):
    """
    收集数据集中的测试数据
    :param dataset_dir:数据集目录

    :return:
    """
    # 读取目录下的所有文件名-》挑选出带有sub-strokecase的文件名-》读取每个文件名下的ses-001里的文件-》读取dwi文件和anat文件中的.nii.gz文件,并行存放成三种模态
    if args.dataset_name=="ISLES2022":
        base_folder = pathlib.Path(args.dataset_dir).resolve()
        patients_dir = sorted([x for x in base_folder.iterdir() if "sub-strokecase" in x.name])
    elif args.dataset_name =="ISLES2024":
        base_folder = pathlib.Path(args.dataset_dir,'raw_data').resolve()
        patients_dir = sorted([x for x in base_folder.iterdir() if "sub-stroke" in x.name])
    elif args.dataset_name =="APIS":
        base_folder = pathlib.Path(args.dataset_dir,'train').resolve()
        patients_dir = sorted([x for x in base_folder.iterdir() if "train" in x.name])
    else:
        raise ValueError("Unknown dataset name: {}".format(args.dataset_name))
    # 创建5折交叉验证对象
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    splits = list(kf.split(patients_dir))
    train_index, val_index = splits[args.fold]  # 生成训练集与验证集索引
    val = [patients_dir[i] for i in val_index]
    test_dataset = Isles(val, val_index,args,isval=True)

    return test_dataset
# endregion
def min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0  # 标记图像中大于0的像素
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])  # 返回指定百分位的值，这里用于确定图像中有效像素的范围
    image = np.clip(image, low, high)  # 将数组中小于下限的值设置为下限，将大于上限的值设置为上限
    image = normalize(image)
    return image  # 归一化后的图像

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image

def threshold(X_predict, mode='predict'):
    """
    将预测转换成伪标签
    """
    if mode == 'predict':

        y_pred, pseudo_label = dict(), dict()
        modalites = [x for x in X_predict]
        for modality in modalites:
            y_pred[modality] = torch.sigmoid(X_predict[modality])
            pseudo_label[modality] = (y_pred[modality][:,:,1,:,:] > y_pred[modality][:,:,0,:,:]).float()

        return pseudo_label 
    elif mode == 'label':
        y_gt = X_predict
        Y = (y_gt[:,:,1,:,:] > y_gt[:,:,0,:,:]).float()
        return Y
    else:
        raise ValueError('mode must be predict or label')


def img_show(X,Y,Y_pre):
    """
    显示验证集上的预测结果
    给出了一个样本的预测结果
    X:(112,112), Y:(112,112), X_pre:(112,112)
    """ 
    X = (X * 255).astype(np.uint8)
    # 标签与图像的合并
    X_rgb = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)
    indices = np.argwhere(Y == 1)
    for i in range(X_rgb.shape[-1]):
        if i == 2:
            X_rgb[indices[:,0],indices[:,1],i] = 255

        else:
            X_rgb[indices[:,0],indices[:,1],i] = 0
    # 预测值与图像和合并
    X_rgb_pre = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)
    indices = np.argwhere(Y_pre == 1)
    for i in range(X_rgb_pre.shape[-1]):
        if i == 2:
            X_rgb_pre[indices[:,0],indices[:,1],i] = 255

        else:
            X_rgb_pre[indices[:,0],indices[:,1],i] = 0
    cv2.imshow('原图', X)
    cv2.imshow('标签与图像的合并', X_rgb)
    cv2.imshow('预测值与图像和合并', X_rgb_pre)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_logs(args, log_str):
    # config = open(os.path.join(args.log_path,f'{args.dataset_name}_lr-{args.lr}_train-epochs-{args.train_epochs}_{datetime.now().strftime("%Y-%m%d-%H%M")}.txt'), 'w')
    config = open(os.path.join(args.log_path,f'{args.dataset_name}_lr-{args.lr}_train-epochs-{args.train_epochs}_gamma-{args.gamma}_fold{args.fold}_{args.time}.txt'), 'a')
    config.write(log_str + "\n")
    config.flush()
    print(log_str)
# region loss_show
def loss_show(losses,fold,time,dataset_name):
	# 绘制损失曲线
	# losses：列表
	plt.plot(losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training Loss over Epochs')
	plt.savefig(f'/media/admin1/yqh/ECSN/train_losses_figure/{dataset_name}/{time}losses_fold{fold}_{dataset_name}.png')
	plt.clf()
# endregion
# region dice_show
def dice_show(dice,fold,time,dataset_name,figure_path='train'):
    # 绘制dice曲线
    # dice：列表
    plt.plot(dice)
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title(f'{figure_path} dice over epochs')
    plt.savefig(f'/media/admin1/yqh/ECSN/{figure_path}_dice_figure/{dataset_name}/{time}{figure_path}_dice_fold{fold}_{dataset_name}.png')
    plt.clf() #  清除当前图形 ，即关闭当前的绘图窗口，以便重新开始绘制新的图形
# endregion
# TODO 
def iou_func(y_pred, y_true, eps=1e-6):
    """
    计算标签与预测之间的IOU
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    # result = list()
    intersection2=np.sum((y_pred * y_true),axis=(2,3))
    union2=np.sum(y_pred, axis=(2,3))+np.sum(y_true, axis=(2,3))
    result2 = (intersection2 + eps) / (union2 - intersection2 + eps)  # intersection和union2很关键
    result2 = np.mean(result2)
    return result2

def accuracy(y_pred, y_true, eps=1e-6):
    """
    计算标签与预测之间的准确率
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    result = list()
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            temp = (np.sum(y_true[i,j,:,:] == y_pred[i,j,:,:])+eps) / (y_true[i,j,:,:].size+eps)
            result.append(temp)
    return np.mean(result)
 
def precision(y_pred, y_true, eps=1e-6):
    """
    计算标签与预测之间的精确率
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    result = list()
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            intersection = np.logical_and(y_true[i,j,:,:], y_pred[i,j,:,:])
            temp = (np.sum(intersection)+eps) / (np.sum(y_pred[i,j,:,:])+eps)
            result.append(temp)
    return np.mean(result)

def sensitivity(y_pred, y_true, eps=1e-6):
    """
    计算标签与预测之间的召回率
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    result = list()
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            intersection = np.logical_and(y_true[i,j,:,:], y_pred[i,j,:,:])
            temp = (np.sum(intersection)+eps) / (np.sum(y_true[i,j,:,:])+eps)
            result.append(temp)
    return np.mean(result)

def specificity(y_pred, y_true, eps=1e-6):
    """
    计算标签与预测之间的特异度
    """
    y_true = 1-y_true.detach().cpu().numpy()  # 真实的反例
    y_pred = 1-y_pred.detach().cpu().numpy()  # 预测的反例
    result = list()
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            intersection = np.logical_and(y_true[i,j,:,:], y_pred[i,j,:,:])
            temp = (np.sum(intersection)+eps) / (np.sum(y_true[i,j,:,:])+eps)
            result.append(temp)
    return np.mean(result)


def hd95_func(y_pred, y_true):
    """
    计算标签与预测之间的HD95
    """
    y_true = y_true.detach().cpu().numpy()
    y_true = np.where(y_true > 0, 1, 0)
    y_pred = y_pred.detach().cpu().numpy()
    result = hd95_v2(y_true, y_pred)
    return result

def asd_func(y_pred, y_true, eps=1e-6):
    """
    计算标签与预测之间的平均表面距离
    """
    y_true = y_true.detach().cpu().numpy()
    y_true = np.where(y_true > 0, 1, 0)
    y_pred = y_pred.detach().cpu().numpy()
    result = asd_v2(y_true, y_pred)
    return result
# region metrics
def metrics(y_pred, y_true):
    """
    计算标签与预测之间的各项指标
    """
    dsc = dice_coefficient(y_pred, y_true)
    dsc = dsc.item()
    iou= iou_func(y_pred, y_true)
    acc = accuracy(y_pred, y_true)
    prec = precision(y_pred, y_true)
    sens = sensitivity(y_pred, y_true)
    spec = specificity(y_pred, y_true)
    # hd95, asd = None, None
    if not torch.all(y_true==0) and not torch.all(y_pred==0):
        hd95 = hd95_func(y_pred, y_true)
        asd = asd_func(y_pred, y_true)
        return dsc, iou, acc, prec, sens, spec, hd95, asd
    return dsc, iou, acc, prec, sens, spec
# endregion
# region uncertainty_calculate
def uncer_calculate(evidence_fs):
    batch_size, num_slice, num_class, width, height = \
        (evidence_fs.shape[0], evidence_fs.shape[1],
         evidence_fs.shape[2], evidence_fs.shape[3],
         evidence_fs.shape[4])
    alpha = evidence_fs + 1
    S = torch.sum(alpha, dim=2, keepdim=True)
    uncertainty = torch.squeeze(num_class/S, dim=2)
    return uncertainty.squeeze()  # 移除长度为1的维度
# endregion

def visualization(X_adc, Y_pre,index,tag='withmask', img_modality='adc'):
    plt.close('all')
    fig, axs = plt.subplots(8,5, figsize=(20, 40))
    fig.suptitle(f'img of {img_modality} {tag}',fontsize=32)
    for i in range(8):
        for j in range(5):
            idx = i*5+j
            #img = X_adc[idx].detach().cpu().numpy()
            img = X_adc[idx]
            if tag == 'withmask' or tag == 'withpredict':
                axs[i, j].imshow(img, cmap='gray') # gray,jet
                #mask = Y_pre[idx].detach().cpu().numpy()
                mask = Y_pre[idx]
                mask_rgba = np.zeros((*mask.shape, 4))
                mask_rgba[mask > 0, :] = [0,1,1, 1] # 红色：[1, 0, 0, 0.5] 原方案[0,1,1, 0.7]
                axs[i, j].imshow(mask_rgba)
            elif tag == 'uncertainty':
                axs[i, j].imshow(img, cmap='jet')
            else:
                axs[i, j].imshow(img, cmap='gray')
            axs[i, j].axis('off')

    plt.savefig(f'/media/admin1/apis/{datetime.now().strftime("%Y-%m%d-%H%M")}_index-{index}_img-{tag}-{img_modality}.png')
    pass


def linear_warmup(optimizer, epoch, warmup_epochs=20, initial_lr=0.00001, target_lr=0.0001):
    if epoch < warmup_epochs:
        lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr