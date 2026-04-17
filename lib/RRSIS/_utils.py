import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from bert.modeling_bert import BertModel


# def load_weights(model, load_path):
#     dict_trained = torch.load(load_path)['model']
#     dict_new = model.state_dict().copy()
#     for key in dict_new.keys():
#         if key in dict_trained.keys():
#             dict_new[key] = dict_trained[key]
#     model.load_state_dict(dict_new)
#     del dict_new
#     del dict_trained
#     torch.cuda.empty_cache()
#     print('load weights from {}'.format(load_path))
#     return model


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    """
    完全替代 mmengine.runner.checkpoint.load_checkpoint
    """
    # 1. 加载文件
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"找不到权重文件: {filename}")
        
    checkpoint = torch.load(filename, map_location=map_location)

    # 2. 提取真正的参数字典 (处理 mmengine 保存的嵌套格式)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 3. 移除分布式训练可能产生的 'module.' 前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # 4. 加载到模型中
    load_info = model.load_state_dict(state_dict, strict=strict)
    
    # 5. 打印日志 (如果有传入 logger)
    if logger is not None:
        if load_info.missing_keys:
            logger.warning(f"缺失的权重键值: {load_info.missing_keys}")
        if load_info.unexpected_keys:
            logger.warning(f"冗余的权重键值: {load_info.unexpected_keys}")
            
    return state_dict # 模仿 mmengine 返回 state_dict 的习惯


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        #self.gtoken = torch.nn.Parameter(torch.randn([1,768,20]))
        #self.gtoken = torch.nn.Parameter(torch.ones([1,768,20]))

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        # B = x.shape[0]
        #gtoken = repeat(self.gtoken, '1 c d -> b c d', b = B)
        gtoken = l_feats
        x = self.classifier(x_c4, x_c3, x_c2, x_c1, gtoken)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None
        #self.gtoken = torch.Parameter(torch.randn(l_feats.shape))

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1, l_feats)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass
