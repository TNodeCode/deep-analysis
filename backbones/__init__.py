from backbones.ae import Autoencoder_Backbone
from backbones.densenet121 import DenseNet121
from backbones.densenet161 import DenseNet161
from backbones.densenet169 import DenseNet169
from backbones.densenet201 import DenseNet201
from backbones.fasterrcnn_resnet50_v2 import FasterRCNNResnet50V2
from backbones.fcos import FCOS
from backbones.inceptionv3 import InceptionV3
from backbones.keypoint_rcnn import KeyPointRCNN
from backbones.maskrcnnv2 import MaskRCNNV2
from backbones.resnet18 import ResNet18
from backbones.resnet34 import ResNet34
from backbones.resnet50 import ResNet50
from backbones.resnet101 import ResNet101
from backbones.resnet152 import ResNet152
from backbones.retinanetv2 import RetinaNetV2
from backbones.tripletenc import TripletEncoder
from backbones.unet import UNet
from backbones.vgg11 import VGG11
from backbones.vgg13 import VGG13
from backbones.vgg16 import VGG16
from backbones.vgg19 import VGG19
from backbones.vitb16 import ViTB16
from backbones.wae_mmd import WAE_MMD_Backbone

models = {
    "autoencoder": Autoencoder_Backbone,
    "densenet121": DenseNet121,
    "densenet161": DenseNet161,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "fasterrcnn_resnet50_v2": FasterRCNNResnet50V2,
    "fcos": FCOS,
    "inceptionv3": InceptionV3,
    "keypoint_rcnn": KeyPointRCNN,
    "maskrcnnv2": MaskRCNNV2,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "retinanetv2": RetinaNetV2,
    "unet": UNet,
    "vgg11": VGG11,
    "vgg13": VGG13,
    "vgg16": VGG16,
    "vgg19": VGG19,
    "vitb16": ViTB16,
    "wae_mmd": WAE_MMD_Backbone,
}

def load_backbone(name: str, filepath: str = None):
    return models[name](filepath=filepath)

def get_available_backbones():
    return models.keys()