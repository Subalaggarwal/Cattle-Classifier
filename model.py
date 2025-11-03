# resnet50_model.py
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

def create_resnet50(num_classes, pretrained=True):
    """
    Create a ResNet50 model for cattle breed classification.
    
    Args:
        num_classes (int): Number of output classes (breeds)
        pretrained (bool): If True, load pretrained ImageNet weights
    
    Returns:
        model (torch.nn.Module): Customized ResNet50 model
    """
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
