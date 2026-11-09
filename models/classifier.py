import torch
import torch.nn as nn
from torchvision import models


def create_classifier():

    model = models.densenet121(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        1
    )

    return model


def load_classifier(model_path, device):

    model = create_classifier()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model