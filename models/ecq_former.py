import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertLMHeadModel
from transformers.models.bert.modeling_bert import BertEncoder

class ECQFormer(nn.Module):
    """
    ECQ-Former (Extended & Efficient Q-Former).
    
    This module acts as the bridge between multiple visual encoders and the LLM.
    It uses a set of learnable query tokens to extract relevant visual information
    from multiple sources (e.g., CLIP + DINOv2) through Cross-Attention.
    """
    def __init__(
        self, 
        num_query_tokens: int = 32, 
        cross_attention_freq: int = 2,
        vision_width_list: list = [1024, 768], # Dims for CLIP and DINOv2
        hidden_size: int = 768,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        
        # 1. Input Projections
        # Map different visual encoder dimensions to Q-Former hidden size
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, hidden_size) for dim in vision_width_list
        ])
        
        # 2. Q-Former Config (Leveraging BERT architecture for robustness)
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = 12
        self.config.add_cross_attention = True
        self.config.cross_attention_freq = cross_attention_freq
        
        # 3. Learnable Query Tokens
        # These are the "buckets" that will gather visual information
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        # 4. The Transformer Encoder (Standard BERT-like)
        # It has Self-Attention (Query-Query) and Cross-Attention (Query-Image)
        self.bert = BertEncoder(self.config)
        
        # Optional: Output projection if LLM hidden size != Q-Former hidden size
        # This is usually handled in the MainFlow or external Projector.

    def forward(self, visual_features_list: list):
        """
        Args:
            visual_features_list: List of tensors from different encoders.
                                  e.g., [Clip_Feats, Dino_Feats]
        Return:
            query_output: [Batch, Num_Query_Tokens, Hidden_Size]
        """
        batch_size = visual_features_list[0].shape[0]
        
        # 1. Project and Concatenate Visual Features
        projected_feats = []
        for i, feat in enumerate(visual_features_list):
            # feat shape: [B, Seq_Len, Dim] -> [B, Seq_Len, Hidden_Size]
            projected_feats.append(self.input_projs[i](feat))
            
        # Combine all visual clues into one context sequence
        # Shape: [B, Total_Visual_Seq_Len, Hidden_Size]
        image_embeds = torch.cat(projected_feats, dim=1)
        
        # Create attention mask for image embeds (assume all valid for now)
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        
        # 2. Expand Query Tokens for Batch
        # Shape: [B, Num_Queries, Hidden_Size]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # 3. Pass through BERT Encoder
        # The 'encoder_hidden_states' argument triggers Cross-Attention in BertEncoder
        encoder_outputs = self.bert(
            hidden_states=query_tokens,
            attention_mask=None,           # Self-attention mask (all see all)
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True
        )
        
        # Output: [B, Num_Query_Tokens, Hidden_Size]
        return encoder_outputs.last_hidden_state
