import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, BitImageProcessor

# DINOv2 : 细粒度结构专家
# facebook/dinov2-large ViT-Large / Patch 14
class DinoV2Encoder(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-large", freeze: bool = True):
        super().__init__()
        print(f"Loading DINOv2 Model: {model_name}")
        self.vision_tower = AutoModel.from_pretrained(model_name)
        
        # DINOv2 可能使用 BitImageProcessor (Bit = Big Transfer, similar pipeline)
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        except:
            print("AutoImageProcessor failed, trying BitImageProcessor...")
            self.image_processor = BitImageProcessor.from_pretrained(model_name)
            
        self.hidden_size = self.vision_tower.config.hidden_size
        
        if freeze:
            print("Freezing DINOv2 Model...")
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            self.vision_tower.eval()

    def forward(self, images):
        """
        Args:
            images: Tensor [B, 3, H, W]
        Returns:
            features: [B, Seq_Len, Hidden_Size]
        """
        outputs = self.vision_tower(images)
        
        # DINOv2 输出同样包含 last_hidden_state
        # Shape: [B, Seq_Len, Hidden_Size]
        last_hidden_state = outputs.last_hidden_state
        
        return last_hidden_state
