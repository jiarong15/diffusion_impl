from transformers import ConvNextV2ForImageClassification
import torch.nn as nn


class ConvNextV2ForImageClassificationWithAttributes(nn.Module):
    def __init__(self, num_new_attributes):
        super().__init__()
        self.convnextv2 = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")
        self.new_attribute_head = nn.Linear(self.convnextv2.config.hidden_sizes[-1], num_new_attributes)
    
    def forward(self, images, labels, new_attributes):
        outputs = self.convnextv2(pixel_values=images, labels=labels)
        new_attribute_logits = self.new_attribute_head(outputs.logits)
        return outputs, new_attribute_logits