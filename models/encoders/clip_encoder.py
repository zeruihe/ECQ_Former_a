import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

# CLIP(VIT-L/14) : 通用语义锚点
# openai/clip-vit-large-patch14 
class CLIPEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", freeze: bool = True):
        super().__init__()
        print(f"Loading CLIP Vision Model: {model_name}")
        self.vision_tower = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # 记录 Hidden Size，方便后续 Fusion 模块自动获取
        self.hidden_size = self.vision_tower.config.hidden_size
        self.patch_size = self.vision_tower.config.patch_size
        
        if freeze:
            print("Freezing CLIP Vision Model...")
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            self.vision_tower.eval()

    def forward(self, images):
        """
        Args:
            images: Tensor [B, 3, H, W]
        Returns:
            features: [B, Seq_Len, Hidden_Size] (Excluding CLS token if needed, or keeping it)
        """
        # CLIP 输出: pooler_output (CLS), last_hidden_state [B, L, D]
        outputs = self.vision_tower(images, output_hidden_states=True)
        
        # 我们通常使用 last_hidden_state 作为细粒度特征
        # Shape: [B, (H/P * W/P) + 1, D]
        last_hidden_state = outputs.last_hidden_state
        
        # 可选：你可以选择去掉第一个 CLS token，只保留 spatial tokens
        # return last_hidden_state[:, 1:, :]
        
        return last_hidden_state
