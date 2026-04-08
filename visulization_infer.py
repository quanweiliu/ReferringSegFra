# pre-process the input image
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from bert.tokenization_bert import BertTokenizer
# initialize model and load weights
from bert.modeling_bert import BertModel
# from lib.RRSIS import segmentation
# from lib.RRSIS import segmentation
from lib.LAVT import segmentation
from utils.tools import overlay_davis


# construct a mini args class; like from a config file
class args:
    # model = "rrsis"
    model = "lavt_one"
    # model = "lavt"
    swin_type = 'base'
    pretrained_swin_weights = './pretrained_weights/swin/swin_base_patch4_window12_384_22k.pth'
    ck_bert = './pretrained_weights/bert-base-uncased/'
    window12 = True
    mha = ''
    fusion_drop = 0.0


# sentence = 'train'
# sentence = 'the most handsome guy'
sentence = 'car'
# sentence = 'all people'
# sentence = 'all chair'
# sentence = 'window'
# sentence = 'wall'

image_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/assets/images/1.tif'
weights = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/checkpoints/RefSegRs_0407-1109-lavt_one/model_best_lavt_one.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



img = Image.open(image_path).convert("RGB")
img_ndarray = np.array(img)  # (orig_h, orig_w, 3); for visualization
original_w, original_h = img.size  # PIL .size returns width first and height second

# plt.figure(figsize=(10, 10))
# plt.imshow(img_ndarray)
# plt.axis('off')
# plt.show()


image_transforms = T.Compose(
    [
     T.Resize(480),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

img = image_transforms(img).unsqueeze(0)  # (1, 3, 480, 480)
img = img.to(device)  # for inference (input)

# pre-process the raw sentence
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
sentence_tokenized = sentence_tokenized[:20]  # if the sentence is longer than 20, then this truncates it to 20 words
# pad the tokenized sentence
padded_sent_toks = [0] * 20
padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
# create a sentence token mask: 1 for real words; 0 for padded tokens
attention_mask = [0] * 20
attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)
# convert lists to tensors
padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)  # (1, 20)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 20)
padded_sent_toks = padded_sent_toks.to(device)  # for inference (input)
attention_mask = attention_mask.to(device)  # for inference (input)

# print(args.model)
if args.model == 'lavt_one' or args.model == 'lavt':
    from lib.LAVT import segmentation as lavt_seg
    # model = lavt_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    model = getattr(lavt_seg, args.model)(pretrained='', args=args)
elif args.model == 'rmsin':
    from lib.RMSIN import segmentation as rmsin_seg
    # model = rmsin_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    model = getattr(rmsin_seg, args.model)(pretrained='', args=args)
elif args.model == 'rrsis':
    from lib.RRSIS import segmentation as rrsis_seg
    # model = rrsis_seg.__dict__[args.model](pretrained=args.pretrained_swin_weights, 
    #                                       args=args)
    model = getattr(rrsis_seg, args.model)(pretrained=args.pretrained_swin_weights, 
                                            args=args)
else:
    assert False, 'Unknown model: {}'.format(args.model)

checkpoint = torch.load(weights, map_location='cpu')
single_model = model
single_model.to(device)

if args.model == 'lavt':
    single_bert_model = BertModel.from_pretrained('./pretrained_weights/bert-base-uncased/')
    single_bert_model.pooler = None
    single_bert_model.load_state_dict(checkpoint['bert_model'])
    bert_model = single_bert_model.to(device)
else:
    bert_model = None

single_model.load_state_dict(checkpoint['model'])
model = single_model.to(device)


# # inference
# print("Running inference...")
# print("Input image shape (after transforms):", img.shape, \
#       "embedding shape:", embedding.shape, \
#       "Attention mask shape:", attention_mask.shape)

if bert_model is not None:
      last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
      embedding = last_hidden_states.permute(0, 2, 1)
      output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))
else:
      output = model(img, padded_sent_toks, l_mask=attention_mask)

output = output.argmax(1, keepdim=True)  # (1, 1, 480, 480)
output = F.interpolate(output.float(), (original_h, original_w))  # 'nearest'; resize to the original image size
output = output.squeeze()  # (orig_h, orig_w)
output = output.cpu().data.numpy()  # (orig_h, orig_w)


output = output.astype(np.uint8)  # (orig_h, orig_w), np.uint8
print("Output mask shape:", output.shape, "dtype:", output.dtype)
# plt.figure(figsize=(10, 10))
# plt.imshow(output, cmap='gray')
# plt.axis('off')
# plt.show()

# Overlay the mask on the image
visualization = overlay_davis(img_ndarray, output)  # red
visualization = Image.fromarray(visualization)

# show the visualization
# visualization.show()

# Save the visualization
visualization.save('./assets/demo_result.jpg')



























