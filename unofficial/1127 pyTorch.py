# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:28:37 2023

@author: Jasic
"""

import os
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights # Importing specific model
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

current_fdrfc=os.path.dirname(os.path.realpath('C:\\Users\\Jasic\\OneDrive - SNU\\2023-2\\01 AI2\\44 Codes\\Image20231127'))
plt.rcParams["savefig.bbox"] = 'tight'


# image source:
#       https://curiosity.gr/index.php/2018/06/29/how-do-birds-fly-and-why-cant-humans/\
img = read_image(os.path.join(r'C:\Users\Jasic\OneDrive - SNU\2023-2\01 AI2\44 Codes', 'roadview_bongcheon.png'))


# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

## image segmentation with pretrained FCN_ResNet50 (20 classes)

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
print(normalized_masks.shape)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
print(class_to_idx)
mask = normalized_masks[0, class_to_idx["bird"]]
to_pil_image(mask).show()



## object detection with pretrained FasterRCNN_ResNet50 (90 classes)

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
im = to_pil_image(box.detach())
im.show()


## Instance segmentation with pretrained MaskRCNN_ResNet50 (90 classes)

# Step 1: Initialize model with the best available weights
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
output = model(batch)
print(output)
bird1_output = output[0]
bird1_masks = bird1_output['masks']
print(f"shape = {bird1_masks.shape}, dtype = {bird1_masks.dtype}, "
      f"min = {bird1_masks.min()}, max = {bird1_masks.max()}")
print("For the first bird, the following instances were detected:")
print([weights.meta["categories"][label] for label in bird1_output['labels']])
# print(bird1_output['scores'])

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

proba_threshold = 0.5