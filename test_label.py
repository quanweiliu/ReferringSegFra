import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
print('using GPU %s' % ','.join(map(str, [1])))

import torch
from torch.utils import data
import json
from argparse import Namespace
import logging
import numpy as np
import cv2
from PIL import Image as PILImage
from tqdm import tqdm
from pathlib import Path
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


def labelling(model, dataset, bert_model, output_dir, args):
    """
    Generate pseudo masks from Referring Segmentation model for self-training.

    For each sample, runs model inference, upsamples the predicted mask
    to the original image resolution, and saves as PNG.

    Args:
        model: Referring segmentation model
        dataset: Dataset instance (must have imgs1 and labels attributes)
        bert_model: BERT model for text encoding (can be None)
        output_dir: Directory to save pseudo mask PNGs
        args: Arguments (needs device)
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    data_loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        sampler=data.SequentialSampler(dataset),
        num_workers=args.workers
    )

    save_count = 0
    skip_count = 0
    has_imgs_attr = hasattr(dataset, 'imgs1')
    has_labels_attr = hasattr(dataset, 'labels')

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Generating pseudo masks")):
            image, target, sentences, attentions = batch_data
            image = image.to(args.device)
            sentences = sentences.to(args.device)
            attentions = attentions.to(args.device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j],
                                                    attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output_mask = output.cpu().argmax(1).data.numpy()  # [1, H, W]
                # print("output_mask shape:", output_mask.shape)

                # # Skip samples where the predicted mask is all zeros
                # if output_mask.sum() == 0:
                #     skip_count += 1
                #     print(f"Skipping image {skip_count} with no valid predictions.")
                #     continue

                # Get original image size by opening the original image
                if has_imgs_attr:
                    img_path = dataset.imgs1[batch_idx]
                    with PILImage.open(img_path) as orig_img:
                        orig_w, orig_h = orig_img.size
                else:
                    orig_h, orig_w = output_mask.shape[1], output_mask.shape[2]

                # Upsample mask to original resolution
                mask_tensor = torch.from_numpy(output_mask[0]).float().unsqueeze(0).unsqueeze(0)
                mask_upsampled = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                pseudo_mask = (mask_upsampled > 0.5).cpu().numpy().astype(np.uint8) * 255

                # Resolve output filename from mask path
                if has_labels_attr:
                    mask_path = dataset.labels[batch_idx]
                    # filename = Path(mask_path).stem + '.tif'
                    filename = Path(mask_path).stem + '.png'
                else:
                    # filename = f"{batch_idx:06d}.tif"
                    filename = f"{batch_idx:06d}.png"

                cv2.imwrite(os.path.join(output_dir, filename), pseudo_mask)
                save_count += 1

            del image, target, sentences, attentions, output

    print(f"\nPseudo masks saved to {output_dir}/")
    print(f"  Total saved: {save_count}, skipped (all-zero): {skip_count}")


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

    if getattr(args, 'do_label', False):
        labelling(model, dataset_test, bert_model, args.pseudo_dir, args)
    else:
        evaluate(model, data_loader_test, bert_model,
                 logger=logging.getLogger("test"), args=args)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # args.dataset = 'VaiRef' # or rrsisd / RefSegRS / VaiRef

    # # other version
    # args.model = 'rrsis_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRs_0406-1231-rrsis_one'
    # args.model = 'rmsin'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0422-1054-rmsin'
    # args.model = 'lavt_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_complex/VaiRef_0418-2304-lavt_one' 
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRs_0407-1109-lavt_one"
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0417-1026-lavt_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0421-2353-lavt_one'
    # args.model = 'lavt'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0417-1700-lavt'
    # args.model = 'rrsis'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRS_0417-1323-rrsis'


    # complex version
    # args.model = 'lavt'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_complex/VaiRef_0417-2155-lavt'
    # args.model = 'lavt_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_complex/VaiRef_0418-2304-lavt_one'
    # args.model = 'rmsin'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_complex/VaiRef_0418-1503-rmsin'
    # args.model = 'rrsis'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_complex/VaiRef_0418-1046-rrsis'
    # args.model = 'rrsis_one'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_complex/VaiRef_0418-2354-rrsis_one'


    # # standard version
    # args.model = 'lavt'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_standard/VaiRef_0417-1842-lavt' 
    args.model = 'lavt_one'
    model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_standard/VaiRef_0419-0945-lavt_one'
    # args.model = 'rmsin'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_standard/VaiRef_0417-2316-rmsin'
    # # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_standard/VaiRef_0503-1457-rmsin'
    # args.model = 'rrsis'
    # model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_standard/VaiRef_0419-0943-rrsis'
    args.model = 'rrsis_one'
    model_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_standard/VaiRef_0418-1402-rrsis_one'


    # simple version
    # args.model = 'lavt'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_simple/VaiRef_0501-1658-lavt"
    # args.model = 'lavt_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_simple/VaiRef_0427-2315-lavt_one"
    # args.model = 'rmsin'
    ############## model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_simple/VaiRef_0427-1553-rmsin"
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_simple/VaiRef_0502-2219-rmsin"
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_simple/VaiRef_0522-0039-rmsin"
    # args.model = 'rrsis'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_simple/VaiRef_0428-1421-rrsis"
    # args.model = 'rrsis_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_simple/VaiRef_0429-1102-rrsis_one"


    # concept version
    # args.model = 'rmsin'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_concept/VaiRef_0505-0900-rmsin"
    # args.model = 'lavt_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_concept/VaiRef_0505-1623-lavt_one"
    # args.model = 'rrsis_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints_concept/VaiRef_0506-0029-rrsis_one"

    # ss concept version
    # args.model = 'rmsin'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0519-0040-rmsin"
    # args.model = 'lavt_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0519-2159-lavt_one"
    # args.model = 'rrsis_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0517-2220-rrsis_one"


    # ss standard version
    # args.model = 'rmsin'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0522-2319-rmsin"
    # args.model = 'lavt_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0524-0003-lavt_one"
    # args.model = 'rrsis_one'
    # model_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/VaiRef_0523-1320-rrsis_one"


    # Save labelling-related args before JSON overwrites them
    _do_label = getattr(args, 'do_label', False)
    _pseudo_dir = getattr(args, 'pseudo_dir', '')

    # args.output_dir = os.path.split(model_path)[0]
    with open(os.path.join(model_path, 'args.json'), 'r') as f:
        arguments = json.load(f)
    args = Namespace(**arguments)
    args.resume = os.path.join(model_path, 'model_best_' + args.model + '.pth')

    # Restore labelling-related args
    args.do_label = _do_label
    args.pseudo_dir = _pseudo_dir
    args.VaiRef_version = 'standard' # simple, standard or complex, concept
    args.dataset = 'VaiRef'
    # args.dataset = 'PotsRef'
    args.split = 'test'
    # args.split = 'val'
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


# 评估模式（原有功能，默认行为）
# python test_label.py

# 伪标签生成模式
# python test_label.py --do-label --pseudo-dir /home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/assets/rrsis_one
