import os
import torch
import torch.utils.data as data
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from bert.tokenization_bert import BertTokenizer


# import sys
# sys.path.append('/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra')
# from bert.tokenization_bert import BertTokenizer
# from args import get_parser
# from utils import transforms

# Dataset configuration initialization
data_root = "/home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/"


suffix = {
    "Impervious_surface": "0",
    "Building": "1",
    "Car": "2",
    "Tree": "3",
    "LowVeg": "4"
}

reverse_suffix = {v: k for k, v in suffix.items()}


def build_rsris_batches(setname, \
                        args):
    im_dir1 = f'{data_root}/images/'
    seg_label_dir = f'{data_root}/binary_masks/'
    if setname == 'train':
        set_prefix = 'train'
    elif setname == 'val':
        set_prefix = 'val'
    elif setname == 'test':
        set_prefix = 'test'

    versions = ['simple', 'standard', 'complex']

    # load all four versions, aligned by mask filename
    version_sentences = {}  # mask_name -> [sent_concept, sent_simple, sent_standard, sent_complex]
    image_map = {}          # mask_name -> (image_path, mask_path)

    for vi, version in enumerate(versions):
        setfile = f'output_phrase_{set_prefix}_{version}.txt'
        tf = os.path.join(data_root, setfile)

        with open(tf, 'r') as rf:
            for idx, line in enumerate(rf.readlines()):
                lsplit = line.split(' ')
                mask_name = lsplit[0]
                image_split = mask_name.split('_')
                image_name = image_split[0] + '_' + image_split[1]
                seg_label_name = reverse_suffix.get(image_split[2].split('.')[0])
                # print("image_split", lsplit[0],
                #       "image_name", image_name, 
                #       "seg_label_name", seg_label_name)

                # print("lsplit", lsplit)
                im_name1 = os.path.join(im_dir1, image_name + '.tif')
                seg = os.path.join(seg_label_dir, seg_label_name, mask_name)
                # print("im_name1", im_name1)
                # print("seg", seg)
                del(lsplit[0])
                sentence = ' '.join(lsplit).strip()

                if mask_name not in version_sentences:
                    version_sentences[mask_name] = [''] * len(versions)
                    image_map[mask_name] = (im_name1, seg)

                version_sentences[mask_name][vi] = sentence

    all_imgs1 = []
    all_labels = []
    all_sentences = []  # each element: [concept, simple, standard, complex]

    for mask_name in sorted(image_map.keys()):
        img_path, mask_path = image_map[mask_name]
        all_imgs1.append(img_path)
        all_labels.append(mask_path)
        all_sentences.append(version_sentences[mask_name])

    print(f"Dataset Loaded: {len(all_imgs1)} samples, each with {len(versions)} text versions.")
    return all_imgs1, all_labels, all_sentences

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
        self.max_tokens = 20

        all_imgs1, all_labels, all_sentences = build_rsris_batches(self.split, args)
        self.sentences = all_sentences
        self.imgs1 = all_imgs1
        self.labels = all_labels

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in range(len(self.imgs1)):
            img_sentences = self.sentences[r]
            sentences_for_ref = []
            attentions_for_ref = []

            for i, el in enumerate(img_sentences):
                sentence_raw = el
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

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
        return len(self.imgs1)

    def __getitem__(self, index):
        this_img1 = self.imgs1[index]

        img1 = Image.open(this_img1).convert("RGB")
        #img1 = cv2.imread(this_img1)
        # print("label_mask", self.labels)
        label_mask = cv2.imread(self.labels[index], 2)
        # print("label_mask", label_mask)
        #label_mask = Image.open(self.labels[index]).convert('L')

        ref_mask = np.array(label_mask) > 50
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img1, target = self.image_transforms(img1, annot)

        if self.eval_mode:
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
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img1, target, tensor_embeddings, attention_mask



if __name__ == "__main__":

    # Dataset configuration initialization
    parser = get_parser()
    args = parser.parse_args()
    transform = transforms.get_transform(args=args)

    dataset = ReferDataset(args, 
                           split='train', 
                           image_transforms=transform, 
                           eval_mode=False)
    # dataset = ReferDataset(args, split='test', image_transforms=transform, eval_mode=True)
    print(len(dataset))  # 12181 / 1740  / 3481
    for i in range(100):
        img, target, tensor_embeddings, attention_mask = dataset[i]

        # train [3, 480, 480] [480, 480] [1, 20] [1, 20]
        # test [3, 480, 480] [480, 480] [1, 20, 1] [1, 20, 1]
        print(img.shape, target.shape, tensor_embeddings.shape, attention_mask.shape)
        break



