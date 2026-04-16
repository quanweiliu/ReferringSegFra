import os
import random
import numpy as np
import torch
from torch.utils import data
from PIL import Image
# from refer import REFER
from refer.refer import REFER
from bert.tokenization_bert import BertTokenizer

# import sys
# sys.path.append('/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra')
# from bert.tokenization_bert import BertTokenizer
# from args import get_parser
# from utils import transforms

# from args import get_parser

def add_random_boxes(img, min_num=20, max_num=60, size=32):
    ''''
        Random Erasing
    '''
    h,w = size, size
    img = np.asarray(img).copy()
    img_size = img.shape[1]
    boxes = []
    num = random.randint(min_num, max_num)
    for k in range(num):
        y, x = random.randint(0, img_size-w), random.randint(0, img_size-h)
        img[y:y+h, x: x+w] = 0
        boxes. append((x,y,h,w) )
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img

refer_data_root = "/home/icclab/Documents/lqw/DatasetMMF/RRSISD/"

class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        num_images_to_mask = int(len(ref_ids) * 0.2)
        self.images_to_mask = random.sample(ref_ids, num_images_to_mask)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids
        # print("Reference IDs (first 10):", self.ref_ids[:10], "Total:", len(self.ref_ids))
        # [3600, 20960, 11348, 3697, 18573, 12702, 17423, 14856, 9311, 9120]

        self.input_ids = []
        self.attention_masks = []

        # 将人类能看懂的字符串文本，转换成计算机（BERT 模型）能处理的数字矩阵
        # Tokenizer 完成以下三件事：分词 (Tokenization)； 映射 (Mapping)； 填充 (Padding)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]
            # print("Processing reference ID {}: image ID {}, number of sentences {}".format(r, ref[0]['image_id'], len(ref[0]['sentences'])))
            # print(ref)
            # {'image_id': 17260, 
            # 'split': 'test', 
            # 'sentences': [{'tokens': ['A', 'vehicle', 'at', 'the', 'bottom'], 
                            # 'raw': 'A vehicle at the bottom', 
                            # 'sent_id': 17260, 
                            # 'sent': 'A vehicle at the bottom'}], 
            # 'file_name': '17260.jpg', 
            # 'category_id': 18, 
            # 'ann_id': 17260, 
            # 'sent_ids': [17260], 
            # 'ref_id': 17260}

            sentences_for_ref = []
            attentions_for_ref = []
            # input_ids 告诉模型“是什么词”，而 attention_mask 告诉模型“哪个词是真的”
            # 如果不给 attention_mask，模型会把后面的 0 也当成有意义的词去计算。

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                # print("Original sentence:", sentence_raw)  
                # # Original sentence: A vehicle at the bottom
                # print("Tokenized input IDs:", input_ids)
                # Tokenized input IDs: [101, 1037, 4316, 2012, 1996, 3953, 102], 
                # where 101 is [CLS], 102 is [SEP], and the rest are token IDs for the words in the sentence.    

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))
        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = add_random_boxes(img)

        ref = self.refer.loadRefs(this_ref_id)
        # print("what is ref", ref)
        # [{'image_id': 3600, 
        # 'split': 'test', 
        # 'sentences': [{'tokens': ['The', 'gray', 'small', 'windmill'], 
                        # 'raw': 'The gray small windmill', 
                        # 'sent_id': 3600, 
                        # 'sent': 'The gray small windmill'}], 
        # 'file_name': '03600.jpg', 
        # 'category_id': 19, 
        # 'ann_id': 3600, 
        # 'sent_ids': [3600], 
        # 'ref_id': 3600}]
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        # “把这个矩阵变成一张‘调色板’模式的索引图
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            # 通过自定义实现的 transforms，保证图像和对应的分割掩码做同样的变换
            # target = annot
            img, target = self.image_transforms(img, annot)

        # print("Number of sentences for reference {}: {}".format(this_ref_id, len(self.input_ids[index])))
        # 其实全部都是 1

        # 如何处理同一个物体的多个不同描述。
        if self.eval_mode:
            # 把该样本对应的所有句子全部提取出来，并且把它们的 BERT embedding 都计算出来，作为模型的输入。
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            # 从该样本拥有的所有描述句子中，随机挑一句。
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask


# if __name__ == "__main__":

#     # Dataset configuration initialization
#     parser = get_parser()
#     args = parser.parse_args()
#     transform = transforms.get_transform(args=args)

#     # dataset = ReferDataset(args, split='train', image_transforms=transform, eval_mode=False)
#     dataset = ReferDataset(args, split='test', image_transforms=transform, eval_mode=True)
#     print(len(dataset))  # 12181 / 1740  / 3481
#     for i in range(100):
#         img, target, tensor_embeddings, attention_mask = dataset[i]

#         # train [3, 480, 480] [480, 480] [1, 20] [1, 20]
#         # test [3, 480, 480] [480, 480] [1, 20, 1] [1, 20, 1]
#         # print(img.shape, target.shape, tensor_embeddings.shape, attention_mask.shape)



