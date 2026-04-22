import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
print('using GPU %s' % ','.join(map(str, [1])))

import torch
from torch.utils import data
import json
from argparse import Namespace
import logging
import numpy as np
from bert.modeling_bert import BertModel
# from lib import segmentation
from utils import tools, evaluation
from utils import transforms
# from dataset.dataset_refer_bert import ReferDataset
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
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, bert_model, logger, args):
    model.eval()
    metric_logger = tools.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, header, logger, args):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(args.device), \
                                                   target.to(args.device), \
                                                   sentences.to(args.device), \
                                                   attentions.to(args.device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], 
                                                    attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()

                output_mask = output.argmax(1).data.numpy()

                I, U = tools.computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)

                seg_total += 1

            del image, target, sentences, attentions, output,output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    # print('Final results:')
    # print('Mean IoU is %.2f\n' % (mIoU*100.))
    logger.info('Final results:')
    logger.info('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), \
                        seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    # print(results_str)
    logger.info(results_str)


def main(args):
    # device = torch.device(args.device)
    dataset_test, _ = get_dataset(
                                  args.split, 
                                #   'val',
                                  transforms.get_transform(args=args), 
                                  args)

    test_sampler = data.SequentialSampler(dataset_test)
    data_loader_test = data.DataLoader(dataset_test, batch_size=1,
                                    sampler=test_sampler, num_workers=args.workers)
    # print(args.model)
    if args.model == 'lavt_one' or args.model == 'lavt':
        from lib.LAVT import segmentation as lavt_seg
        # model = lavt_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
        model = getattr(lavt_seg, args.model)(pretrained='', args=args)
    elif args.model == 'rmsin':
        from lib.RMSIN import segmentation as rmsin_seg
        # model = rmsin_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
        model = getattr(rmsin_seg, args.model)(pretrained='', args=args)
    elif args.model == 'rrsis' or args.model == 'rrsis_one':
        from lib.RRSIS import segmentation as rrsis_seg
        # model = rrsis_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, 
        #                                       args=args)
        model = getattr(rrsis_seg, args.model)(pretrained=args.pretrained_swin_weights, 
                                               args=args)
    else:
        assert False, 'Unknown model: {}'.format(args.model)

    # single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    # print("checkpoint", checkpoint.keys())
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(args.device)

    if args.model == 'lavt' or args.model == 'rrsis':
        # bert_state_dict = checkpoint['bert_model'] # 或者 checkpoint['state_dict']
        # # 3. 创建一个新的字典，去掉 'module.' 前缀
        # new_state_dict = {}
        # for k, v in bert_state_dict.items():
        #     if k.startswith('module.'):
        #         name = k[7:] # 去掉前 7 个字符 'module.'
        #     else:
        #         name = k
        #     new_state_dict[name] = v

        single_bert_model = BertModel.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        single_bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        # if args.ddp_trained_weights:
        #     single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(args.device)
    else:
        bert_model = None

    args.print_freq = 1000
    evaluate(model, data_loader_test, bert_model, 
             logger=logging.getLogger("test"), args=args)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # args.dataset = 'VaiRef' # or rrsisd / RefSegRS / VaiRef

    # args.model = 'rrsis_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRs_0406-1231-rrsis_one'
    args.model = 'rmsin'
    model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0422-1054-rmsin'
    # args.model = 'lavt_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRs_0407-1109-lavt_one"
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0417-1026-lavt_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0421-2353-lavt_one'
    # args.model = 'lavt'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0417-1700-lavt'
    # args.model = 'rrsis'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0417-1323-rrsis'


    # args.model = 'rrsis_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/rrsisd_0421-0851-lavt_one'
    # args.model = 'rmsin'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RRSISD_0324-1831-rmsin'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/rrsisd_0420-1257-rmsin'
    # args.model = 'lavt_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RRSISD_0326-0951-lavt_one'
    # args.model = 'lavt'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RRSISD_LAVT'
    # args.model = 'rrsis'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/rrsisd_0419-1726-rrsis'
    # args.model = 'rrsis_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RRSISD_0413-2304-rrsis_one'
    # args.model = 'lavt'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0417-2155-lavt'


    # args.model = 'lavt'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0417-1842-lavt' 
    # args.model = 'lavt_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0418-2304-lavt_one' 
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0419-0945-lavt_one'
    # args.model = 'rmsin'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0417-2316-rmsin'
    # args.model = 'rrsis_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0418-1402-rrsis_one'


    # args.output_dir = os.path.split(model_path)[0]
    with open(os.path.join(model_path, 'args.json'), 'r') as f:
        arguments = json.load(f)
    args = Namespace(**arguments)
    args.resume = os.path.join(model_path, 'model_best_' + args.model + '.pth')
    args.VaiRef_version = 'complex' # standard or complex
    # args.dataset = 'VaiRef'
    # args.dataset = 'PotsRef'
    args.split = 'test'
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') 输出模式
    logging.basicConfig(level=logging.INFO, \
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", \
                        datefmt="%Y-%m-%d %H:%M:%S", \
                        handlers=[
                            logging.FileHandler(os.path.join(model_path, "results.log"), mode="a"),  # 用于文件保存
                            logging.StreamHandler()   # 用于在 terminal 中的文件打印
                        ],
                        )

    print('Weights: {}'.format(args.resume))
    # print('Image size: {}'.format(str(args.img_size)))


    main(args)


# python test.py 