import torch
from monai.networks.nets import UNet


def create_segmentation_model():

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )

    return model


def load_segmentation_model(model_path, device):

    model = create_segmentation_model()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model