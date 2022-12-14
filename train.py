from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

import warnings
warnings.filterwarnings("ignore")

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

#"""
#--data_config config/coco.data  
#--pretrained_weights weights/darknet53.conv.74
#"""

if __name__ == "__main__":
    
    #---------------------------------------------------
    #action1
    #---------------------------------------------------
    #配置所有运行启动的参数，需要在lunch.json中的args中填写
    #---------------------------------------------------
    #需要预先配置两个参数
    #"--data_config","config/coco.data",
    #"--pretrained_weights","weights/darknet53.conv.74"    
    #---------------------------------------------------
    
    parser = argparse.ArgumentParser()
    #读取数据是的迭代次数
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")                    
    parser.add_argument("--batch_size", type=int, default=5, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="D:/PyTorch-YOLOv3/PyTorch-YOLOv3/config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="D:/PyTorch-YOLOv3/PyTorch-YOLOv3/config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str,default='D:/PyTorch-YOLOv3/PyTorch-YOLOv3/weights/darknet53.conv.74', help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    #-------------------
    #（detail_picture_1）
    #-------------------
    #parse_data_config函数用于读取coco.date，并以字典形式返回
    data_config = parse_data_config(opt.data_config)
    #读取训练集合测试集图片路径文件路径
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    #load_classes函数是读取data_config["names"]获得路径的文件（分类信息的文件），并转化为list
    class_names = load_classes(data_config["names"])
    
    #---------------------------------------------------
    #action2
    #---------------------------------------------------
    # 搭建网络
    #---------------------------------------------------
    #创建网络模型对象（detail_model_1）
    print(opt.model_def)
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    #---------------------------------------------------
    #action3
    #---------------------------------------------------
    # 读入数据：在训练的时候分批次读入
    #---------------------------------------------------
    #创建datasets.py中的ListDataset对象（detail_picture_2）
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    #创建torch.utils.data.DataLoader类用于分批次读取数据(detail_picture_3)
    #注意是随机选择的一个大小
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,    #每个batch加载多少个样本【将数据几个几个一打包输出】
        shuffle=True,                 #在每个epoch重新打乱数据
        num_workers=opt.n_cpu,        #用多少个子进程加载数据。0表示数据将在主进程中加载
        pin_memory=True,
        collate_fn=dataset.collate_fn,#将一个batch的数据和标签进行合并操作（重写了方法将图片进行缩放到416*416）
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    #进行epochs次循环迭代(detail_picture_3)
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        #进行以batch_size为一捆的数据照片的迭代(detail_picture_3)
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            #此时的image已经带batch号了
            batches_done = len(dataloader) * epoch + batch_i
            #将照片copy到GPU上，再GU上运行
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            print ('imgs',imgs.shape)
            print ('targets',targets.shape)
            #训练传入参数，进入到forward函数中
            loss, outputs = model(imgs, targets)
            #进行反向传播
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
