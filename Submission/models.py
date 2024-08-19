import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation


class CustomModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic",
                                                                                 num_labels=num_classes,ignore_mismatched_sizes=True,
                                                                                 ignore_value=23)
    def forward(self,inputs):
        outputs = self.model(**inputs)
        return outputs