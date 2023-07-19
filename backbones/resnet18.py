import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from backbones.backbone import BackboneModel

class ResNet18(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "resnet18"
        self.feature_keys = [f"f{i}" for i in range(1,5)]
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.classifier = self.model.fc
        self.return_nodes = return_nodes={
            "layer1.1.relu_1": self.feature_keys[0],
            "layer2.1.relu_1": self.feature_keys[1],
            "layer3.1.relu_1": self.feature_keys[2],
            "layer4.1.relu_1": self.feature_keys[3],
        }
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.fc = classifier