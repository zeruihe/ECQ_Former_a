import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModel

# BioMedClip : 医学领域专家
class BioMedCLIPEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", freeze: bool = True):
        super().__init__()
        print(f"Loading BioMedCLIP Vision Model: {model_name}")
        
        # BioMedCLIP uses a similar architecture to CLIP
        try:
            self.vision_tower = CLIPVisionModel.from_pretrained(model_name)
            self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading as standard CLIP: {e}")
            print("Trying AutoModel...")
            self.vision_tower = AutoModel.from_pretrained(model_name)
            try:
                self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
            except:
                from transformers import AutoImageProcessor
                self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Record Hidden Size for Fusion modules
        self.hidden_size = self.vision_tower.config.hidden_size
        
        if hasattr(self.vision_tower.config, "patch_size"):
            self.patch_size = self.vision_tower.config.patch_size
        else:
            self.patch_size = 16 # BioMedCLIP base usually has patch size 16

        if freeze:
            print("Freezing BioMedCLIP Vision Model...")
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
        outputs = self.vision_tower(images, output_hidden_states=True)
        
        # Use last_hidden_state for fine-grained features
        # Shape: [B, Num_Patches + 1, D]
        last_hidden_state = outputs.last_hidden_state
        
        return last_hidden_state
