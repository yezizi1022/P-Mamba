import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
import math
from thop import profile  # 需要安装thop
from ptflops import get_model_complexity_info
import time
from dataset import RemoteData, label_to_RGB, RGB_to_label
from seg_metric import SegmentationMetric
import logging
import warnings
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from scipy.spatial.distance import cdist

def hausdorff_distance(true_mask, pred_mask):
    true_points = np.column_stack(np.where(true_mask > 0))  # Ground truth points
    pred_points = np.column_stack(np.where(pred_mask > 0))  # Predicted points

    if len(true_points) == 0 or len(pred_points) == 0:
        return float('inf')  # Return infinity if one of the masks is empty

    # 计算两个点集之间的距离
    dists_true_to_pred = cdist(true_points, pred_points, metric='euclidean')
    dists_pred_to_true = cdist(pred_points, true_points, metric='euclidean')

    # Hausdorff距离为两个集合之间最大最小距离
    hd_true = np.max(np.min(dists_true_to_pred, axis=1))
    hd_pred = np.max(np.min(dists_pred_to_true, axis=1))

    return max(hd_true, hd_pred)


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--dataset", type=str, default='echo')
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--val_batchsize", type=int, default=1)
    parser.add_argument("--crop_size", type=int, nargs='+', default=[256, 256], help='H, W')
    parser.add_argument("--models", type=str, default='cswin',
                        choices=['cswin'])
    parser.add_argument("--head", type=str, default='seghead')
    parser.add_argument("--trans_cnn", type=str, nargs='+', default=['cswin_tiny', 'resnet50'], help='ttansformer, cnn')
    parser.add_argument("--use_edge", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default='/mnt/work_dir')
    parser.add_argument("--base_dir", type=str, default='./')
    parser.add_argument("--data_dir", type=str, default='/mnt/data/data')
    parser.add_argument("--information", type=str, default='MambaMoeV2_2222_psax_256_C128_ExtraAug2_seghead')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--save_gpu_memory", type=int, default=0)
    parser.add_argument("--val_full_img", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2

class FullModel(nn.Module):

    def __init__(self, model):
        super(FullModel, self).__init__()
        self.model = model

    def forward(self, input):

        return self.model(input)

def get_model():
    models = args2.models
    if models in ['swinT', 'resT', 'beit', 'cswin', 'volo']:
        print(models, args2.head)
    else:
        print(models)
    
    nclass = args2.nclass
    assert models in ['cswin']
    if models == 'cswin':
        from models.cswin import cswin_tiny as cswin
        model = cswin(nclass=nclass, img_size=args2.crop_size[0], pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)

    model = FullModel(model)
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args2.local_rank], output_device=args2.local_rank, find_unused_parameters=True)
    return model

def label_to_gray(label):
    return label * 255

def val(model, weight_path):
    nclasses = args2.nclass
    model.eval()
    metric = SegmentationMetric(numClass=nclasses)
    # 省略加载模型权重的代码
    total_time = 0
    total_samples = 0

    output_save_dir = os.path.join(object_path, 'segmentation_results')  # 输出图像保存目录
    if not os.path.exists(output_save_dir):
        os.makedirs(output_save_dir)

    hausdorff_distances = []  # 用于保存所有图像的Hausdorff距离

    with torch.no_grad():
        model_state_file = weight_path
        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            # checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            warnings.warn('weight is not existed !!!"')

        for i, sample in enumerate(dataloader_val):

            images, labels, names = sample['image'], sample['label'], sample['name']
            images = images.cuda()
            labels = labels.long().squeeze(1)
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            print("test:{}/{}".format(i, len(dataloader_val)))
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            metric.addBatch(logits, labels)

            # 计算Hausdorff距离并保存
            for j, pred in enumerate(logits):
                pred_mask = (pred > 0).astype(np.uint8)  # 将预测结果转为二值掩码
                true_mask = (labels[j] > 0).astype(np.uint8)  # 将真实标签转为二值掩码
                hd = hausdorff_distance(true_mask, pred_mask)
                hausdorff_distances.append(hd)

            for j, pred in enumerate(logits):
                # 假设pred是一个二维的numpy数组，其中包含的是每个像素点的类别标签（0或1）
                pred_img_gray = label_to_gray(pred)  # 使用label_to_gray函数将类别标签转换为灰度图像
                # 将灰度数组转换为PIL图像，确保数据类型是uint8
                pred_img_gray = Image.fromarray(pred_img_gray.astype(np.uint8))
                img_name = names[j]  # 获取原始图像的文件名
                save_path = os.path.join(output_save_dir, f"{img_name}")
                pred_img_gray.save(save_path)

        # 输出Hausdorff距离统计
        avg_hd = np.mean(hausdorff_distances)
        print(f"Average Hausdorff distance: {avg_hd:.4f}")
        logging.info(f"Average Hausdorff distance: {avg_hd:.4f}")

        result_count(metric)

        for i, sample in enumerate(dataloader_val):
            images = sample['image'].to(device)
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            total_time += end_time - start_time
            total_samples += images.size(0)

        avg_time_per_sample = (total_time / total_samples) * 1000  # 转换为毫秒
        print(f"Average inference time per sample: {avg_time_per_sample:.2f} ms")

def result_count(metric):
    iou = metric.IntersectionOverUnion()
    acc = metric.Accuracy()
    f1 = metric.F1()
    precision = metric.Precision()
    recall = metric.Recall()
    miou = np.nanmean(iou)
    mf1 = np.nanmean(f1)
    mprecision = np.nanmean(precision)
    mrecall = np.nanmean(recall)

    iou = reduce_tensor(torch.from_numpy(np.array(iou)).to(device) / get_world_size()).cpu().numpy()
    miou = reduce_tensor(torch.from_numpy(np.array(miou)).to(device) / get_world_size()).cpu().numpy()
    acc = reduce_tensor(torch.from_numpy(np.array(acc)).to(device) / get_world_size()).cpu().numpy()
    f1 = reduce_tensor(torch.from_numpy(np.array(f1)).to(device) / get_world_size()).cpu().numpy()
    mf1 = reduce_tensor(torch.from_numpy(np.array(mf1)).to(device) / get_world_size()).cpu().numpy()
    precision = reduce_tensor(torch.from_numpy(np.array(precision)).to(device) / get_world_size()).cpu().numpy()
    mprecision = reduce_tensor(torch.from_numpy(np.array(mprecision)).to(device) / get_world_size()).cpu().numpy()
    recall = reduce_tensor(torch.from_numpy(np.array(recall)).to(device) / get_world_size()).cpu().numpy()
    mrecall = reduce_tensor(torch.from_numpy(np.array(mrecall)).to(device) / get_world_size()).cpu().numpy()

    if args2.local_rank == 0:
        print('\n')
        logging.info('####################### full image val ###########################')
        print('|{}:{}{}{}{}|'.format(str('CLASSES').ljust(24),
                                     str('Precision').rjust(10), str('Recall').rjust(10),
                                     str('F1').rjust(10), str('IOU').rjust(10)))
        logging.info('|{}:{}{}{}{}|'.format(str('CLASSES').ljust(24),
                                            str('Precision').rjust(10), str('Recall').rjust(10),
                                            str('F1').rjust(10), str('IOU').rjust(10)))
        for i in range(len(iou)):
            print('|{}:{}{}{}{}|'.format(str(CLASSES[i]).ljust(24),
                                         str(round(precision[i], 4)).rjust(10), str(round(recall[i], 4)).rjust(10),
                                         str(round(f1[i], 4)).rjust(10), str(round(iou[i], 4)).rjust(10)))
            logging.info('|{}:{}{}{}{}|'.format(str(CLASSES[i]).ljust(24),
                                                str(round(precision[i], 4)).rjust(10),
                                                str(round(recall[i], 4)).rjust(10),
                                                str(round(f1[i], 4)).rjust(10), str(round(iou[i], 4)).rjust(10)))
        print('mIoU:{} ACC:{} mF1:{} mPrecision:{} mRecall:{}'.format(round(miou * 100, 2),
                                                                      round(acc * 100, 2), round(mf1 * 100, 2),
                                                                      round(mprecision * 100, 2),
                                                                      round(mrecall * 100, 2)))
        logging.info('mIoU:{} ACC:{} mF1:{} mPrecision:{} mRecall:{}'.format(round(miou * 100, 2),
                                                                             round(acc * 100, 2), round(mf1 * 100, 2),
                                                                             round(mprecision * 100, 2),
                                                                             round(mrecall * 100, 2)))
        print('\n')

def get_model_path(args2):
    object_path, weight_path = None, None
    file_dir = args2.save_dir #os.path.join(args2.base_dir, args2.save_dir)
    file_list = os.listdir(file_dir)
    for file in file_list:
        if args2.models in file and args2.information in file:
            weight_path = os.path.join(file_dir, file, 'weights', 'best_weight.pkl')
            object_path = os.path.join(file_dir, file)
    if object_path is None or weight_path is None:
        tmp_path = os.path.join(file_dir, 'tmp_save')
        output_path = os.path.join(tmp_path, 'outputs')
        weight_path = os.path.join(tmp_path, 'weights')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        object_path = tmp_path
        weight_path = weight_path + '/best_weight.pkl'
        warnings.warn('path is not defined, will be set as "./work_dir/tmp_save"')
    return object_path, weight_path

args2 = parse_args()

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True
distributed = False

device = torch.device(('cuda:{}').format(args2.local_rank))

if distributed:
    torch.cuda.set_device(args2.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
    )

remotedata_val = RemoteData(base_dir=args2.data_dir, mode="test",
                        dataset=args2.dataset, crop_szie=args2.crop_size)
if distributed:
    val_sampler = DistributedSampler(remotedata_val)
else:
    val_sampler = None
dataloader_val = DataLoader(
    remotedata_val,
    batch_size=args2.val_batchsize,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=val_sampler)

def main():
    model = get_model().to(device)  # 确保模型移到了正确的设备

    input_size = (3, *args2.crop_size)  # 根据模型实际输入调整

    macs, params = get_model_complexity_info(model, input_size, as_strings=False,
                                             print_per_layer_stat=False, verbose=True)
    flops = macs * 2

    # 转换为字符串形式，便于打印
    flops_str = f"{flops / 10**9:.2f} GFLOPs"  # 转换为GFLOPs单位
    params_str = f"{params / 10**6:.2f}M"  # 参数单位转换为百万

    print(f"GFLOPs: {flops_str}, Params: {params_str}")

if __name__ == '__main__':
    object_path, weight_path = get_model_path(args2)
    save_log = os.path.join(object_path, 'test.log')
    logging.basicConfig(filename=save_log, level=logging.INFO)
    CLASSES = ('Background', 'Echo')
    model = get_model()
    val(model, weight_path)


if __name__ == '__main__':
    main()
