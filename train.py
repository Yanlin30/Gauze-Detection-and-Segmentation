from Model.Detection import *
import torch.optim
from Model.SematicSegment import fcn_resnet, Unet, Network_Res2Net_GRA_NCD
from Model.Detection import faster_rcnn, r_fcn, SSD
from torch.utils.data import DataLoader
from data_process.dataset import *
from metric_function.seg_metric import SegmentationMetric
from metric_function.detection_metric import *
import torch.nn as nn
import torchvision.models as models
from log import *
from test import *
import torch.nn.functional as F


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def seg_train(model_name, fold, device, model_type=None, continue_train=False):
    """

    :param model_name:UNet or fcn_resnet(50 or 101)
    :param fold:
    :param model_type:
    :return:
    """

    
    record_log = "log/seg_log.txt"
    logger = logger_config(log_path=record_log, logging_name=f"{model_name} train")

    # 数据集路径
    image_root = "data/images"
    mask_root = "data/mask"

    # 设置参数
    learning_rate = 1e-4
    loss_fn = nn.CrossEntropyLoss().to(device)
    epoch = 80
    metric = SegmentationMetric(numClass=2)
    train_value = []
    test_value = []
    batchsize = 8

    for i in range(fold):
          if model_name == "fcn_resnet":
            model = fcn_resnet.fcn(num_classes=2, model_type=model_type)
            model_name = "{}{}".format(model_name, model_type)
        elif model_name == "UNet":
            model = Unet.UNet(in_channels=3, num_classes=2)
        elif model_name == "COD":
            model = Network_Res2Net_GRA_NCD.Network(imagenet_pretrained=True)
            # model = Network_Res2Net_GRA_NCD.Network(imagenet_pretrained=False)
            # model.load_state_dict(torch.load("weights/Net_epoch_best.pth"))
            
        model = model.to(device)

        train_str = "log/{}_fold_train.txt".format(i)
        val_str = "log/{}_fold_val.txt".format(i)
        train_img, train_d = get_data(log_file=train_str, mask_root=mask_root, task="seg")
        val_img, val_d = get_data(log_file=val_str, mask_root=mask_root, task="seg")
        train_data = seg_dataset(data_image=train_img, data_mask=train_d,image_root=image_root, mask_root=mask_root)
        val_data = seg_dataset(data_image=val_img, data_mask=val_d,image_root=image_root, mask_root=mask_root)
        train_dataloader = DataLoader(train_data, batch_size=batchsize)
        val_dataloader = DataLoader(val_data, batch_size=batchsize)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logger.info("---------------------{}th fold train------------------\n"
                    "the number of pictures of train set: {}\n"
                    "the number of pictures of validation set: {}\n".format(i, len(train_img), len(val_img)))
 
        train_step = 0
        best_train_iou = 0
        best_test_iou = 0
        for n in range(epoch):
            model.train()
            for data in train_dataloader:
                imgs, masks = data
                # masks = masks.squeeze(dim=1).type(torch.long).to(device)
                predicts = model(imgs.to(device))

                optimizer.zero_grad()
                if model_name == "COD":
                    masks = masks.unsqueeze(dim=1).to(device)
                    loss_init = structure_loss(predicts[0], masks) + structure_loss(predicts[1], masks) + structure_loss(predicts[2],masks)
                    loss_final = structure_loss(predicts[3], masks)
                    loss = loss_final + loss_init
                else:
                    masks = masks.squeeze(dim=1).type(torch.long).to(device)
                    loss = loss_fn(predicts, masks)
                loss.backward()
                clip_gradient(optimizer, 0.45)
                optimizer.step()
                if train_step % 10 == 0:
                    logger.info("after {} steps training, the loss is {:.4f}".format(train_step, loss))
                train_step += 1


            model.eval()
            metric.reset()
            with torch.no_grad():
                for data in val_dataloader:
                    imgs, masks = data
                    predicts = model(imgs.to(device))
                    if model_name == "COD":
                        predicts = predicts[3].sigmoid().data.squeeze()
                        predicts = (predicts - predicts.min()) / (predicts.max() - predicts.min() + 1e-8)
                    else:
                        predicts = torch.argmax(predicts, dim=1)
                    if predicts.shape != masks.shape:
                        predicts = predicts.unsqueeze(dim=0)
                    # print(predicts.shape)
                    # print(masks.shape)
                    metric.addBatch(predicts.cpu(), masks.cpu())

                acc = metric.meanPixelAccuracy()
                iou = metric.meanIntersectionOverUnion()
                dice = metric.Dice_score()
                logger.info("val: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(acc, iou, dice))

            if iou > best_train_iou:
                best_train_iou = iou
                train_best = [acc, iou, dice]
             
                save_path = "weights/{}_{}fold_{}epoch_{:.4f}ACC_{:.4f}IOU_{:.4f}DICE.pth".format(model_name, i, n, acc, iou, dice)
                torch.save(model, save_path)
                logger.info("save_path: {}".format(save_path))
                
                temp_test = seg_test(read_log="log/inter_set.txt", model_pth=save_path, write_log=record_log, device=device)
                logger.info("test: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(*temp_test))
                # seg_test(read_log="log/inter_set.txt", model_pth=save_path, write_log=record_log, device=device)
                if temp_test[1] > best_test_iou:
                    best_test_iou = temp_test[1]
                    test_best = temp_test

                
    
        train_value.append(train_best)
        test_value.append(test_best)
        # extest_value.append(extest_best)
        logger.info(f"{'-' * 20} {model_name} best  performance {'-' * 20}")
        logger.info("VAL: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*train_best))
        logger.info("TEST: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*test_best))
        # logger.info("EXTEST: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*extest_best))
             

    val_mean = np.mean(np.array(train_value), axis=0)
    test_mean = np.mean(np.array(test_value), axis=0)
    # extest_mean = np.mean(np.array(extest_value), axis=0)
    logger.info(f"{'-' * 20} {model_name}  avg performance {'-' * 20}")
    logger.info("VAL: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*val_mean))
    logger.info("TEST: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*test_mean))
    # logger.info("EXTEST: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*extest_mean))
    logger.info("-" * 50)


def detection_train(model_name, fold,device, model_type=None, continue_train=False ):
    """

    :param model_name: vgg16, faster_rcnn, r_fcn,ssd
    :param fold:
    :param model_type:
    :return:
    """
  
    image_root = "data/images"
    mask_root = "data/mask"
    json_root = "data/json"
    checkpoint_path = "detection_checkpoint.pth"
    record_log = "log/detection_log.txt"
    logger = logger_config(log_path=record_log, logging_name=f"{model_name}train")
    learning_rate = 0.0001
    epoch = 80
    batchsize = 8
    metric = DetectionMetrics()
    # [mAP50 precision recall]
    train_value = []
    test_value = []
    extest_value = []

    for i in range(fold):
        if model_name == "r_fcn":
          model = r_fcn.r_fcn().to(device)
          model_name = f"{model_name}_{model_type}"
      elif model_name == "faster_rcnn":
          model = faster_rcnn.Faster_RCNN(model_type=model_type).to(device)
          model_name = f"{model_name}_{model_type}"
      elif model_name == "SSD":
          model = SSD.ssd().to(device)

        train_str = "log/{}_fold_train.txt".format(i)
        val_str = "log/{}_fold_val.txt".format(i)
        train_img, train_d = get_data(log_file=train_str, mask_root=mask_root, task="detection")
        val_img, val_d = get_data(log_file=val_str, mask_root=mask_root, task="detection")
        train_data = detection_dataset(data_image=train_img, data_json=train_d,
                                       image_root=image_root, json_root=json_root)
        val_data = detection_dataset(data_image=val_img, data_json=val_d,
                                     image_root=image_root, json_root=json_root)
        train_dataloader = DataLoader(train_data, batch_size=batchsize, collate_fn=my_collate)
        val_dataloader = DataLoader(val_data, batch_size=batchsize, collate_fn=my_collate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logger.info("---------------------{}th fold train------------------\n"
                    "the number of pictures of train set: {}\n"
                    "the number of pictures of validation set: {}\n".format(i, len(train_img), len(val_img)))

        train_step = 0
        best_train_precision = 0
        best_test_precision = 0
        best_extest_precision = 0
        for n in range(epoch):
            model.train()
            if continue_train and os.path.exists(checkpoint_path):
                model, optimizer, i, n, losses = load_checkpoint_model(checkpoint_path, model, optimizer)
                continue_train = False
            for imgs, targets in train_dataloader:
                loss_dict = model(imgs.to(device), targets)
                # print(targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                clip_gradient(optimizer, 0.45)
                optimizer.step()


                if train_step % 10 == 0:
                    logger.info("after {} steps training, the loss is {:.4f}".format(train_step, losses))
                train_step += 1

         
            model.eval()
            metric.reset()
            with torch.no_grad():
                for imgs, targets in val_dataloader:
                    predicts = model(imgs.to(device))
                    pred_boxes, true_boxes = get_TrueBox_and_predBoex(targets, predicts)
                    metric.addBatch(pred_boxes, true_boxes)

            mAP50 = metric.calculate_map(iou_threshold=0.5)
            precision, recall = metric.calculate_precision_recall(iou_threshold=0.4)
            # train_value.append([mAP50, precision, recall])
            logger.info("val: mAP50_{:.4f} precision_{:.4f} recall_{:.4f}".format(mAP50, precision, recall))

            if precision > best_train_precision:
                best_train_precision = precision
                train_best = [mAP50, precision, recall]
               
                save_path = "weights/{}_{}fold_{}epoch_{:.4f}mAP50_{:.4f}precision_{:.4f}recall.pth".format(model_name, i, n, mAP50, precision, recall)
                torch.save(model, save_path)
                logger.info("save_path: {}".format(save_path))
                
                test_temp = detection_test(read_log="log/inter_set.txt", model_pth=save_path, write_log=record_log, device=device)
                if test_temp[1] > best_test_precision:
                    best_test_precision = test_temp[1]
                    test_best = test_temp
                
                # extest_temp = detection_test(read_log="log/exter_set.txt", model_pth=save_path, write_log=record_log, device=device)
                # if extest_temp[1] > best_extest_precision:
                #     best_extest_precision = extest_temp[1]
                #     extest_best = test_temp
            save_checkpoint_model(i, n, model, optimizer, losses, None, checkpoint_path)
                    
       
        train_value.append(train_best)
        test_value.append(test_best)
        # extest_value.append(extest_best)
        logger.info(f"{'-'*20} {model_name} best performance {'-'*20}")
        logger.info("VAL:   mAP50:{:.4f}    precision:{:.4f}    recall:{:.4f}".format(*train_best))
        logger.info("TEST:   mAP50:{:.4f}    precision:{:.4f}    recall:{:.4f}".format(*test_best))
        # logger.info("EXTEST:   mAP50:{:.4f}    precision:{:.4f}    recall:{:.4f}".format(*extest_best))
          

    val_mean = np.mean(np.array(train_value), axis=0)
    test_mean = np.mean(np.array(test_value), axis=0)
    # extest_mean = np.mean(np.array(extest_value), axis=0)
    logger.info(f"{'-'*20} {model_name} avg performance {'-'*20}")
    logger.info("VAL:   mAP50:{:.4f}    precision:{:.4f}    recall:{:.4f}".format(*val_mean))
    logger.info("TEST:   mAP50:{:.4f}    precision:{:.4f}    recall:{:.4f}".format(*test_mean))
    # logger.info("EXTEST:   mAP50:{:.4f}    precision:{:.4f}    recall:{:.4f}".format(*extest_mean))


