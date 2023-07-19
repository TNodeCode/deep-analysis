import torch
import torchvision.models as models
from backbones.models.unet import OutConv, UNet as UNetModel
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from backbones.backbone import BackboneModel

class UNet(BackboneModel):
    def __init__(self, filepath=None, n_channels=3, n_classes=2, bilinear=False):
        super().__init__(filepath=filepath)
        self.name = "unet"
        self.feature_keys = [f"d{i}" for i in range(1,5)] + [f"u{i}" for i in range(1,5)] + ["cls"]
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.model = UNetModel(n_channels, n_classes=n_classes, bilinear=bilinear)
        self.classifier = self.model.outc
        self.classifier_input_size = 2048
        self.return_nodes = return_nodes={
            "down1.maxpool_conv.1.double_conv.5": self.feature_keys[0],
            "down2.maxpool_conv.1.double_conv.5": self.feature_keys[1],
            "down3.maxpool_conv.1.double_conv.5": self.feature_keys[2],
            "down4.maxpool_conv.1.double_conv.5": self.feature_keys[3],
            "up1.conv.double_conv.5": self.feature_keys[4],
            "up2.conv.double_conv.5": self.feature_keys[5],
            "up3.conv.double_conv.5": self.feature_keys[6],
            "up4.conv.double_conv.5": self.feature_keys[7],
            "outc.conv": self.feature_keys[8],
        }
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.outc = classifier
        
    def create_classifier_network(self, options):
        """
        Create a classification layer
        """
        # Creating the Feedforward Classifier
        classifier = OutConv(in_channels=64, out_channels=self.n_classes)
        return classifier
    
    def load_state_dict(self, options):
        """
        Load weights of a trained model instance
        """
        self.replace_classifier(options)
        mask_values = options.pop('mask_values', [0, 1])
        self.model.load_state_dict(options)