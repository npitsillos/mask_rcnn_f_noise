from typing import DefaultDict
from collections import defaultdict
from dataset import CocoDataset
from detection.utils import collate_fn

import torch
import torch.nn.functional as F
import detection.utils as utils

from resnet import resnet50
from mask_rcnn import MaskRCNN
from backbone import _validate_trainable_layers, _resnet_fpn_extractor

from torch.optim import Adam
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection._utils import overwrite_eps
from torchvision._internally_replaced_utils import load_state_dict_from_url

MASK_RCNN_URL = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"


def maskrcnn_resnet50_fpn(
    pretrained=False, progress=True, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, use_f_noise: bool = False, **kwargs
):
    """
    Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.
    Reference: `"Mask R-CNN" <https://arxiv.org/abs/1703.06870>`_.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - masks (``UInt8Tensor[N, H, W]``): the segmentation binary masks for each instance
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - masks (``UInt8Tensor[N, 1, H, W]``): the predicted masks for each instance, in ``0-1`` range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (``mask >= 0.5``)
    For more details on the output and on how to plot the masks, you may refer to :ref:`instance_seg_output`.
    Mask R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "mask_rcnn.onnx", opset_version = 11)
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = resnet50(pretrained=pretrained_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, use_f_noise=use_f_noise)
    model = MaskRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(MASK_RCNN_URL, progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # cuDNN reproducibility
    torch.backends.cudnn.benchmark = False

    # Algorithm reproducibility
    torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    model = maskrcnn_resnet50_fpn(pretrained=True, use_f_noise=True)
    
    IMAGE_SHAPE = (520, 520)

    import os
    import numpy as np
    import detection.transforms as T

    from detection.engine import train_one_epoch, evaluate
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import Subset, DataLoader
    from detection.coco_utils import get_coco
    seed_everything(2021)

    exp = "f_noise"

    device = torch.device("cpu")
    coco_val_2014 = get_coco('./', "val", transforms=T.ToTensor())
    epoch = 0
    if os.path.exists(f"./weights/{exp}"):
        weights_files = next(os.walk(f"./weights/{exp}"))[2]
        if len(weights_files) > 0:
            latest_weights = sorted(weights_files)[-1]
            model.load_state_dict(torch.load(os.path.join(f"./weights/{exp}", latest_weights)))
        epoch = len(weights_files)
    else:
        os.makedirs(f"./weights/{exp}")
    # coco_val_2014 = CocoDataset("./val2014", "./annotations/instances_val2014.json", T.Compose([T.ToTensor(), T.Resize(IMAGE_SHAPE)]))
    # We only want a portion of the data
    indices = torch.randperm(len(coco_val_2014)).tolist()
    
    dataloaders = {}
    subsets = ["train", "val", "test"]
    subset_lens = [0.7, 0.2, 0.1]
    start_idx = 0
    for subset, subset_len in zip(subsets, subset_lens):
        idx = int(len(indices) * subset_len)
        coco_subset = Subset(coco_val_2014, indices[start_idx:start_idx+idx])
        if subset == "train":
            batch_size = 10
            shuffle = True
        else:
            batch_size = 1
            shuffle = False
        dataloaders[subset] = DataLoader(coco_subset, batch_size, shuffle=shuffle, num_workers=0, collate_fn=utils.collate_fn)
        start_idx += idx

    model.to(device)
    optimiser = Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    writer = SummaryWriter("./experiments/f_noise")
    for epoch in range(epoch, num_epochs):
        train_one_epoch(model, optimiser, dataloaders["train"], device, epoch, writer, print_freq=1, use_f_noise=True)
        torch.save(model.state_dict(), f"./weights/{exp}/mask_rcnn_w_f_noise_{epoch}.pth")
        evaluate(model, dataloaders["val"], device, epoch, use_f_noise=True, writer=writer)