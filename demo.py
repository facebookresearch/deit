from PIL import Image

import torch
import timm
import torchvision
import torchvision.transforms as T

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import gradio as gr

torch.set_grad_enabled(False);

with open("imagenet_classes.txt", "r") as f:
    imagenet_categories = [s.strip() for s in f.readlines()]

transform = T.Compose([
    T.Resize(256, interpolation=3),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

def detr(im):
  img = transform(im).unsqueeze(0)

  # compute the predictions
  out = model(img)

  # and convert them into probabilities
  scores = torch.nn.functional.softmax(out, dim=-1)[0]

  # finally get the index of the prediction with highest score
  topk_scores, topk_label = torch.topk(scores, k=5, dim=-1)

  
  d = {}
  for i in range(5):
      pred_name = imagenet_categories[topk_label[i]]
      pred_name = f"{pred_name:<25}"
      score = topk_scores[i]
      score = f"{score:.3f}"
      d[pred_name] = score
  return d

inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Label(type="confidences",num_top_classes=5)

title = "Deit"
description = "demo for Facebook DeiT: Data-efficient Image Transformers. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2012.12877'>Training data-efficient image transformers & distillation through attention</a> | <a href='https://github.com/facebookresearch/deit'>Github Repo</a></p>"

examples = [
    ['deer.jpg'],
    ['cat.jpg']
]

gr.Interface(detr, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()