import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
# print('using GPU %s' % ','.join(map(str, [1])))

os.environ['WANDB_API_KEY'] = 'd14367a70fe99f6d07256b084fcc49cf17bb01f4'

import time
import datetime
import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR
import json
import wandb
import logging
import gc
import operator
from functools import reduce
# from email import header
from bert.modeling_bert import BertModel
# from dataset.dataset_refer_bert import ReferDataset
# from dataset.RefSegRS_refer_bert import ReferDataset
from utils.loss import MixLoss
from utils import tools, evaluation
from utils import transforms
from args import get_parser


def get_dataset(image_set, transform, args):
    print("dataset"*10, args.dataset)

    if args.dataset == 'rrsisd':
        from dataset.dataset_refer_bert import ReferDataset
    elif args.dataset == 'RefSegRS':
        from dataset.RefSegRS_refer_bert import ReferDataset
    elif args.dataset == 'VaiRef':
        from dataset.ISPRS_VaiRef import ReferDataset
    elif args.dataset == 'PotsRef':
        from dataset.ISPRS_PotsRef import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch,
                    iterations, bert_model, metric_format, logger, args):
    model.train()
    metric_format.add_meter('lr', tools.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    wrapper_data = metric_format.log_every(data_loader, header, logger, args)
    train_loss = 0
    total_its = 0

    # for data in data_loader:
    for i, data in enumerate(wrapper_data):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, attentions)#, sentences_hidden_state)# [4,2,120,120]
        else:
            output = model(image, sentences, attentions)#, sentences_hidden_state)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_format.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if args.local_rank == 0:
        wandb.log({
            "Train Loss": train_loss / total_its,})


def main(args):
    dataset, num_classes = get_dataset("train",
                                       transforms.get_transform(args=args),
                                       args=args)

    dataset_test, _ = get_dataset("val",
                                  transforms.get_transform(args=args),
                                  args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {tools.get_rank()} successfully built train dataset.")
    num_tasks = tools.get_world_size()
    global_rank = tools.get_rank()
    train_sampler = data.distributed.DistributedSampler(dataset, 
                                                        num_replicas=num_tasks, 
                                                        rank=global_rank,
                                                        shuffle=True)
    test_sampler = data.SequentialSampler(dataset_test)

    # data loader
    data_loader = data.DataLoader(
                        dataset, batch_size=args.batch_size,
                        sampler=train_sampler, num_workers=args.workers, 
                        pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = data.DataLoader(
                        dataset_test, batch_size=1, 
                        sampler=test_sampler, num_workers=args.workers)

    # model initialization
    # print(args.model)
    if args.model == 'lavt_one' or args.model == 'lavt':
        from lib.LAVT import segmentation as lavt_seg
        # model = lavt_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, 
        #                                      args=args)
        model = getattr(lavt_seg, args.model)(pretrained=args.pretrained_swin_weights, 
                                              args=args)
    elif args.model == 'rmsin':
        from lib.RMSIN import segmentation as rmsin_seg
        # model = rmsin_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, 
        #                                       args=args)
        model = getattr(rmsin_seg, args.model)(pretrained=args.pretrained_swin_weights, 
                                               args=args)
    elif args.model == 'rrsis' or args.model == 'rrsis_one':
        from lib.RRSIS import segmentation as rrsis_seg
        # model = rrsis_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, 
        #                                       args=args)
        model = getattr(rrsis_seg, args.model)(pretrained=args.pretrained_swin_weights, 
                                               args=args)
    else:
        assert False, 'Unknown model: {}'.format(args.model)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    pure_model = model.module  # 剥离权重的 module，方便后续加载权重和保存权重

    # print(model)
    if args.model == 'lavt' or args.model == 'rrsis':
        bert_model = BertModel.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        pure_bert_model = bert_model.module
    else:
        bert_model = None
        pure_bert_model = None

    # resume training
    if args.resume:
        print('Resuming training from checkpoint: {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        pure_model.load_state_dict(checkpoint['model'], strict=False)
        if args.model == 'lavt' or args.model == 'rrsis':
            pure_bert_model.load_state_dict(checkpoint['bert_model'])
            print('Successfully loaded model and bert weights from checkpoint: {}'.format(args.resume))

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in pure_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    # if args.model != 'lavt_one' and args.model != 'rmsin':
    if args.model == 'lavt' or args.model == 'rrsis':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in pure_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in pure_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in pure_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in pure_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )
    criterion = MixLoss(weight=0.1)

    # learning rate scheduler
    lr_scheduler = LambdaLR(optimizer,
                            lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    metric_format = tools.MetricLogger(delimiter=" ")
    logger_train = logging.getLogger("train")
    logger_val = logging.getLogger("val")
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # training loops
    if args.local_rank == 0:
        wandb.watch(model, log="all")


    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        args.print_freq = 200  # after debugging, we can set print_freq to a large value, e.g., 50, for clartiy and less noisy logs
        train_one_epoch(model, criterion, optimizer, data_loader, \
                        lr_scheduler, epoch, iterations, bert_model, \
                        metric_format, logger_train, args)
        args.print_freq = 400
        iou, overallIoU = evaluation.evaluate(model, data_loader_test, bert_model, \
                                              criterion, tools.IoU, \
                                              metric_format, logger_val, args)
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        best = (best_oIoU < overallIoU)
        if pure_bert_model is not None:
            dict_to_save = {'model': pure_model.state_dict(), 
                            'bert_model': pure_bert_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 
                            'epoch': epoch, 
                            'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}
        else:
            dict_to_save = {'model': pure_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 
                            'epoch': epoch, 
                            'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

        if best:
            print('Better epoch: {}\n'.format(epoch))
            tools.save_on_master(dict_to_save, 
                                 os.path.join(args.output_dir,
                                            'model_best_{}.pth'.format(args.model)))
            best_oIoU = overallIoU
        tools.save_on_master(dict_to_save, 
                             os.path.join(args.output_dir,
                                        'model_last_{}.pth'.format(args.model)))
        if args.local_rank == 0:
            wandb.save('model.h5')

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    tools.seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    if args.local_rank == 0:
        wandb.init(project=args.model)

    # create rundir and copy args file
    run_id = datetime.datetime.now().strftime("%m%d-%H%M-") + args.model
    args.output_dir = os.path.join(args.output_dir, args.dataset + "_" + str(run_id))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    print("args.output_dir", args.output_dir)
    
    logging.basicConfig(level=logging.INFO, \
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", \
                        datefmt="%Y-%m-%d %H:%M:%S", \
                        handlers=[
                            logging.FileHandler(os.path.join(args.output_dir, "ref_seg.log"), mode="a"),  # 用于文件保存
                            logging.StreamHandler()   # 用于在 terminal 中的文件打印
                        ],
                        )
    
    # 下面这种写法无法在 terminal 中打印日志
    # logging.basicConfig(level=logging.INFO, \
    #                     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    #                     datefmt="%Y-%m-%d %H:%M:%S", 
    #                     filename=os.path.join(args.output_dir, "ref_seg.log"), 
    #                     filemode="a")
    # logger = logging.getLogger()

    # set up distributed learning
    tools.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 train.py --img_size 480 
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --img_size 480 
