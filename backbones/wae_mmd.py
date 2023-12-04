import torch
import torchvision.models as models
from backbones.models.wae_mmd import WAE_MMD
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from backbones.backbone import BackboneModel

class WAE_MMD_Backbone(BackboneModel):
    def __init__(self, filepath=None, in_channels=3, latent_dim=128, patch_size=256, kernel_type='rbf', hidden_dims=[32, 64, 128, 256, 512, 1024]):
        super().__init__(filepath=filepath)
        self.name = "wae_mmd"
        self.feature_keys = [f"e{i}" for i in range(1,7)]
        self.in_channels = in_channels
        self.hidden_dims=hidden_dims
        self.model = WAE_MMD(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            patch_size=patch_size,
            kernel_type=kernel_type,
        )
        self.return_nodes = return_nodes={
            "encoder.0.2": self.feature_keys[0],
            "encoder.1.2": self.feature_keys[1],
            "encoder.2.2": self.feature_keys[2],
            "encoder.3.2": self.feature_keys[3],
            "encoder.4.2": self.feature_keys[4],
            #"encoder.5.2": self.feature_keys[5],
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