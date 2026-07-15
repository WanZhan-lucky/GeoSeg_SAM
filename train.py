import argparse
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import datasets
import models
import utils
from statistics import mean
import torch
import os
import numpy as np
from PIL import Image
from eval_iou import SegmentationMetric
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
import torch.nn.functional as F
local_rank = 0  # 单卡默认主进程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测GPU

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict

def onehot_to_mask(mask, palette=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],[0, 0, 0]]):  #one-hot（C,H,W） → 彩色图（H,W,3），用来可视化预测/标签
    """
    Converts a mask (H, W, K) to (H, W, C)  一般形状是 (K, H, W) = (num_classes, H, W)
    """
    mask = mask.permute(1, 2, 0).numpy() #变成 (H, W, C)
    x = np.argmax(mask, axis=-1) #在最后一维（C）上取 argmax → 得到每个像素的类别索引图
    colour_codes = np.array(palette) #元素为 0 ~ num_classes-1
    x = np.uint8(colour_codes[x.astype(np.uint8)]) #把每个类别索引映射成对应的 RGB
    return x #one-hot → 类别索引 → RGB mask（用 palette 映射）
def onehot_to_index_label(mask):  # 将one-hot（C,H,W） → 索引图（H,W），用来算混淆矩阵、IoU 等指标。
    """
    将 one-hot mask (C, H, W) 转为索引标签 (H, W)
    """
    mask = mask.permute(1, 2, 0).numpy()
    x = np.argmax(mask, axis=-1)
    return x

def make_data_loader(spec, tag=''):  # 数据加载器创建
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))
    shuffle = tag == 'train'  # 训练集打乱，验证集不打乱
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=shuffle, num_workers=8, pin_memory=False, sampler=None)  # sampler设为None
    return loader
def make_data_loaders():  # 用 make_data_loader 来创建训练集和验证集的数据加载器。
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def eval_psnr(loader, model, config):
    model.eval()
    eval_type = config.get('eval_type') # 从配置中获取评估类型，比如 'f1', 'fmeasure', 'ber', 'cod', 'seg'
    class_num = config['model']['args']['num_classes'] # 读取类别数，用于分割 metric（SegmentationMetric）
    ignore_background = config['val_dataset']['dataset']['args']['ignore_bg']#是否忽略背景类（一般最后一类是 background）
    metric_seg = SegmentationMetric(class_num, ignore_background)  # 创建语义分割评价类，传入类别数和是否忽略背景
    if local_rank == 0:   # rank0 上显示进度条，其他卡不显示
        pbar = tqdm(total=len(loader), leave=True, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []# 存放所有 batch 的预测和 gt（现在是单卡版本）
    # {
    #     'inp': Tensor(B, C, H, W),  # 输入图像
    #     'gt': Tensor(B, C, H, W)  # 语义分割的 GT（也可能是 one-hot）
    # }
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda() # 将 batch 中每个 tensor 都搬到 GPU
        inp = batch['inp'] # 输入图像
        output_masks = model.infer(inp) #(1, num_classes, H, W) (1, 5, 1024, 1024)
        # pred = torch.sigmoid(output_masks) # Sigmoid 不会改变数值的大小顺序，所以最终生成的 Mask 索引图和使用 Softmax 是一样的
        pred = output_masks
        pred_cpu = pred.to('cpu')  # 把预测搬回 CPU 保存
        pred_list.extend([pred_cpu])  #   extend([pred_cpu])和append(pred_cpu)是一模一样的extend接受一个list
        # pred_list = [batch1_pred, batch2_pred, ...] 每个元素 shape 是 (B, num_classes, H, W) 按batch 存
        gt_cpu = batch['gt'].to('cpu')
        gt_list.extend([gt_cpu])

        if pbar is not None: # 更新进度条
            pbar.update(1)

        for i in range(len(gt_list[-1])): #len 看的是第 0 维，也就是 batch_size
            output_mask = pred_list[-1][i] #(5, 1024, 1024)
            mask_index_label = onehot_to_index_label(output_mask).flatten()  #从 5 个通道里选最大值的那个通道索引作为类别 -> (1024, 1024) (1048576,)
            gt_mask = gt_list[-1][i]  #(5, 1024, 1024) 已经成为了one_hot编码 [0, 0, 1, 0, 0] 某个像素是类别 2
            gt_index_label = onehot_to_index_label(gt_mask).flatten()   #gt_mask: (1, 5, H, W) -> (5, H, W) -> (H, W) -> flatten
            metric_seg.addBatch(mask_index_label, gt_index_label) #1D 向量去更新混淆矩阵，

    if pbar is not None:
        pbar.close()

    #segmentation 指标的汇总
    oa = metric_seg.overallAccuracy() #overall accuracy
    oa = np.around(oa, decimals=4)
    mIoU, IoU = metric_seg.meanIntersectionOverUnion()
    mIoU = np.around(mIoU, decimals=4) #mIoU: 平均 IoU，IoU: 每一类的 IoU 数组
    IoU = np.around(IoU, decimals=4)
    p = metric_seg.precision() #  每一类的 precision
    p = np.around(p, decimals=4)
    mp = np.nanmean(p) # mean precision（所有类的平均精度）
    mp = np.around(mp, decimals=4)
    r = metric_seg.recall()  # 每一类的 recall
    r = np.around(r, decimals=4)
    mr = np.nanmean(r)  # mean recall
    mr = np.around(mr, decimals=4)
    f1 = (2 * p * r) / (p + r) # 每一类的 F1 = 2pr / (p + r)，再做四舍五入
    f1 = np.around(f1, decimals=4)
    mf1 = np.nanmean(f1)   # mean F1
    mf1 = np.around(mf1, decimals=4)
    fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()  # Frequency Weighted IoU
    fwIOU = np.around(fwIOU, decimals=4)
    # 对混淆矩阵按列归一化（每一真值类的列和为 1）
    normed_confusionMatrix = metric_seg.confusionMatrix / metric_seg.confusionMatrix.sum(axis=0)
    normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)

    classes_list = config['train_dataset']['dataset']['args']['classes'] # 从 train_dataset 里取类别名列表
    if ignore_background:  # 忽略背景，则去掉最后一个 class 作为 axis label
        axis_labels = classes_list[:-1]
    else:
        axis_labels = classes_list

    # 下面构造一个 PrettyTable 表格用于打印展示指标
    # 每一行的内容先组织成 list
    IOU_row = ['mIOU', mIoU]
    IOU_row.extend(IoU.tolist()) #后面接上每一类的 IoU
    Precision_row = ['Precision', mp]
    Precision_row.extend(p.tolist())
    Recall_row = ['Recall', mr]
    Recall_row.extend(r.tolist())
    F1_row = ['F1', mf1]
    F1_row.extend(f1.tolist())
    # 表头：第一列是 'metrics'，第二列是 'average'，后面是各类别名
    title_row = ['metrics', 'average']
    title_row.extend(axis_labels)
    # OA 和 FWIOU 是单一标量，因此后面补空字符串占位
    OA_row = ['OA', oa]
    fwIOU_row = ['FWIOU', fwIOU]
    for i in range(len(axis_labels)):
        OA_row.append(' ')
        fwIOU_row.append(' ')
    # 创建 PrettyTable，并依次添加行
    table = PrettyTable(title_row)
    table.add_row(IOU_row)
    table.add_row(Precision_row)
    table.add_row(Recall_row)
    table.add_row(F1_row)
    table.add_row(OA_row)
    table.add_row(fwIOU_row)
    return table, normed_confusionMatrix, mIoU, axis_labels

def prepare_training():
    if config.get('resume') is not None:
        print("==========================resume start======================")
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
        # resume_model_path = os.path.join(config.get('work_dir'), 'model_epoch_' + str(config.get('resume')) + '.pth')
        # resume_optim_path = os.path.join(config.get('work_dir'), f'optim_epoch_{config.get("resume")}.pth')
        resume_model_path = os.path.join(config.get('work_dir'),'model_epoch_last.pth')
        resume_optim_path = os.path.join(config.get('work_dir'),'optim_epoch_last.pth')
        resume_checkpoint = torch.load(resume_model_path)
        model.load_state_dict(resume_checkpoint, strict=False)
        if os.path.exists(resume_optim_path):
            optimizer.load_state_dict(torch.load(resume_optim_path))
        else:
            print("⚠️ 找不到 optimizer checkpoint，优化器状态将从头开始")
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        # log("Model parameter names:")
        # for name, param in model.named_parameters():
        #     log(name)
        total_params = sum(p.numel() for p in model.parameters())
        log(f"Model Summary:")
        log(f"  Total params      : {total_params:,}")
        log(f"  Total Param size (MB)   : {total_params * 4 / 1024 ** 2:.2f} MB")  # float32

        # 打印 optimizer 信息
        log("Optimizer:")
        for i, group in enumerate(optimizer.param_groups):
            log(f"  Group {i}: lr={group['lr']}, weight_decay={group.get('weight_decay', 0)}")

        # 打印模型结构（可选，推荐调试时使用）
        # log("Model Architecture:")
        # log(str(model))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model):
    model.train()
    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=True, desc='train')
    else:
        pbar = None
    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = model.loss_G
        loss_list.append(batch_loss.item())
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    loss = mean(loss_list)
    return loss

def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f: #一个记录日志的函数
        yaml.dump(config, f, sort_keys=False) #TensorBoard 的 SummaryWriter，用来写标量、图等

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None: #可能后续 dataset 或 model 里会用到这个配置。
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    model, optimizer, epoch_start, lr_scheduler = prepare_training() #准备模型、优化器、LR 调度器
    model.optimizer = optimizer #把优化器挂到 model 上，当成一个属性：
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min')) #重新创建了一个 CosineAnnealingLR，覆盖了上面返回的 lr_scheduler
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    # model.load_state_dict(sam_checkpoint, strict=False) #名字对得上的层会加载 不对的层会跳过（比如你增加了新的 head）
    if config.get('resume') is None:
        try:
            model.load_state_dict(sam_checkpoint)
        except:
            try:
                new_dict = {}
                for key, value in sam_checkpoint.items():
                    new_key = key.replace("module.", "") #key无module， 不会发生任何事情
                    new_dict[new_key] = value
                model.load_state_dict(new_dict)
            except:
                print("new_state_dict")
                new_state_dict = load_from(model, sam_checkpoint, 256, 16) #兼容处理
                model.load_state_dict(new_state_dict)
    else:
        if local_rank == 0:
            print(f"resume = {config.get('resume')}, 跳过 SAM 预训练加载，使用 model_epoch_{config.get('resume')}.pth 的权重")
    for name, param in model.named_parameters():
        if "image_encoder" in name: #参数名包含 "image_encoder"，且不包含 "prompt_generator"：
            param.requires_grad_(False)
        if "image_encoder" in name and "Adapter" in name:
            param.requires_grad_(True)
        if any(k in name for k in [
            "down_proj",
            "d_convs",
            "p2t_attn",
            "p2t_mlp",
            "up_proj",
            "attn_0",
            "attn_1",
        ]) and "image_encoder" in name:
            param.requires_grad_(True)
        if any(k in name for k in ["norm3", "norm4", "norm5", "norm6"]) and "image_encoder" in name:
            param.requires_grad_(True)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("TRAIN:", name)
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        total_size = total_params * 4 / 1024 ** 2
        trainable_size = trainable_params * 4 / 1024 ** 2
        frozen_size = frozen_params * 4 / 1024 ** 2
        log("\n========== Model Parameter Statistics ==========")
        log(f"Total Params      : {total_params:,}")
        log(f"Trainable Params  : {trainable_params:,}")
        log(f"Frozen Params     : {frozen_params:,}")
        log("-----------------------------------------------")
        log(f"Total Size (MB)      : {total_size:.2f} MB")
        log(f"Trainable Size (MB)  : {trainable_size:.2f} MB")
        log(f"Frozen Size (MB)     : {frozen_size:.2f} MB")
        log("===============================================")

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    # max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    best_mIoU = 0.0
    for epoch in range(epoch_start, epoch_max + 1):
        # train_loader.sampler.set_epoch(epoch) # 注释分布式采样器epoch设置
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model) #进行一轮训练，拿到平均 loss
        lr_scheduler.step() #更新学习率

        if local_rank == 0:
            log_info = [
                '\n ############################ epoch {}/{} ############################'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            save(config, model, save_path, 'last')
            torch.save(optimizer.state_dict(), os.path.join(save_path, f"optim_epoch_last.pth"))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            with torch.no_grad():
                seg_eval_table, normed_confusionMatrix, mIoU, axis_labels = eval_psnr(val_loader, model, config)
            log_info.append('mIoU = {:.4f}'.format(mIoU))
            writer.add_scalars('mIoU', {'mIoU': mIoU}, epoch)
            if local_rank == 0:
                if mIoU > best_mIoU:
                    best_mIoU = mIoU
                    log_info.append(f"✅ New best mIoU: {best_mIoU}, saving confusion matrix...")
                    sns.heatmap(normed_confusionMatrix, annot=True, cmap='Blues', yticklabels=axis_labels, xticklabels=axis_labels)

                    plt.ylabel('True labels')
                    plt.xlabel('Predicted labels')
                    heatmap_dir = os.path.join(args.path, "heatmap")
                    os.makedirs(heatmap_dir, exist_ok=True)
                    filename = f"heatmap_{epoch}.jpg"  # 示例：confusion_epoch10.jpg
                    save_heatmap = os.path.join(heatmap_dir, filename)
                    plt.savefig(save_heatmap, dpi=150, bbox_inches='tight')
                    plt.close()
                    save(config, model, save_path, 'best')
                    if best_mIoU > 0.61:
                        log_info.append(f"✅NEWANZHAN_mIoU: {best_mIoU}, =============")
                        save(config, model, save_path,  f"best_mIoU_{best_mIoU:.4f}")
                t = timer.t() #前总耗时
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1) #训练进度（0~1）
                t_epoch = utils.time_text(t - t_epoch_start) #本轮训练+验证时间
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog) #已经用的总时间
                log_info.append('epoch train + val time: {} {}/{}'.format(t_epoch, t_elapsed, t_all)) #预计总时间（按目前进度估算）

                log_info.append(str(seg_eval_table))
                log_info.append('Confusion Matrix:')
                log_info.append(str(normed_confusionMatrix))

                log('\n'.join(log_info))
                writer.flush() #刷新 TensorBoard 的 writer 缓冲区


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},  #只保存：prompt_generator 的参数 decode_head 的参数
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth")) #保存整个模型 state_dict() 成：
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth")) #统一直接 torch.save(model.state_dict(), ...)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/wlk-256input.yaml")
    parser.add_argument('--path', default="../../../autodl-tmp/[graduation2]606paceAdapter+MLPAdapter(series)+WMA_PRA_four")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    # 保留local_rank参数但不使用（兼容原有命令行）
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    path = args.path
    save_path = os.path.join(path, save_name)
    os.makedirs(save_path, exist_ok=True)  # 显式创建模型保存目录，exist_ok=True兼容已存在情况
    main(config, save_path, args=args)