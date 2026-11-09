import torch
from monai.networks.nets import UNet


def create_3d_segmentation_model():

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
        dropout=0.1
    )

    return model


def load_3d_segmentation_model(model_path, device):

    model = create_3d_segmentation_model()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    
    state_dict = checkpoint["model"]

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(
        f"3D U-Net loaded successfully. "
        f"Epoch: {checkpoint.get('epoch', 'unknown')}"
    )

    return model