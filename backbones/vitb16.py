import torch
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from backbones.backbone import BackboneModel

class ViTB16(BackboneModel):
    def __init__(self, filepath=None):
        super().__init__(filepath=filepath)
        self.name = "vitb16"
        self.feature_keys = [f"f{i}" for i in range(1,13)]
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.classifier = self.model.heads
        self.classifier_input_size = 2048
        self.return_nodes = return_nodes={
            "encoder.layers.encoder_layer_0.self_attention": self.feature_keys[0],
            "encoder.layers.encoder_layer_1.self_attention": self.feature_keys[1],
            "encoder.layers.encoder_layer_2.self_attention": self.feature_keys[2],
            "encoder.layers.encoder_layer_3.self_attention": self.feature_keys[3],
            "encoder.layers.encoder_layer_4.self_attention": self.feature_keys[4],
            "encoder.layers.encoder_layer_5.self_attention": self.feature_keys[5],
            "encoder.layers.encoder_layer_6.self_attention": self.feature_keys[6],
            "encoder.layers.encoder_layer_7.self_attention": self.feature_keys[7],
            "encoder.layers.encoder_layer_8.self_attention": self.feature_keys[8],
            "encoder.layers.encoder_layer_9.self_attention": self.feature_keys[9],
            "encoder.layers.encoder_layer_10.self_attention": self.feature_keys[10],
            "encoder.layers.encoder_layer_11.self_attention": self.feature_keys[11],
        }
        self.feature_extractor = create_feature_extractor(
            self.model,
            self.return_nodes,
        )
        self.load_weights()

    def set_classifier(self, classifier):
        self.model.heads = classifier