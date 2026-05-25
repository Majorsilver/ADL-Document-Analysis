import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def build_model(num_classes: int) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model