import torch
import torchvision.models as models
from backbones.models.ae import Autoencoder
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from backbones.backbone import BackboneModel

class Autoencoder_Backbone(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "autoencoder"
        self.feature_keys = [f"e{i}" for i in range(1,4)]
        self.model = Autoencoder()
        self.return_nodes = return_nodes={
            "encoder.0.2": self.feature_keys[0],
            "encoder.1.2": self.feature_keys[1],
            "encoder.2.2": self.feature_keys[2],
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
        self.model.load_state_dict(options['state_dict'], strict=False)