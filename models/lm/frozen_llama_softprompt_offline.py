# models/lm/frozen_llama_softprompt_offline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class LMOutputs:
    loss: torch.Tensor
    logits: torch.Tensor

class FrozenLlamaWithSoftPromptOffline(nn.Module):
    def __init__(self, local_dir: str, torch_dtype=torch.bfloat16, trust_remote_code: bool = True):
        super().__init__()
        self.local_dir = local_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            local_files_only=True,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @property
    def embed_dim(self) -> int:
        return self.model.get_input_embeddings().embedding_dim

    def build_prompt_and_labels(
        self,
        prompts: List[str],
        targets: List[str],
        device: torch.device,
        max_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full = [p + t for p, t in zip(prompts, targets)]
        tok_full = self.tokenizer(full, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tok_prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        input_ids = tok_full["input_ids"].to(device)
        attn_mask = tok_full["attention_mask"].to(device)
        labels = input_ids.clone()

        prompt_lens = tok_prompt["attention_mask"].sum(dim=1).to(device)
        for i in range(len(prompts)):
            labels[i, : prompt_lens[i]] = -100
        labels[attn_mask == 0] = -100
        return input_ids, attn_mask, labels

    def forward_with_soft_prompt(
        self,
        soft_prompt: torch.Tensor,  # (B, M, D_lm)
        prompts: List[str],
        targets: List[str],
        device: torch.device,
        max_length: int = 256,
    ) -> LMOutputs:
        input_ids, attn_mask, labels = self.build_prompt_and_labels(prompts, targets, device, max_length=max_length)

        text_emb = self.model.get_input_embeddings()(input_ids)     # (B, L, D)
        inputs_embeds = torch.cat([soft_prompt, text_emb], dim=1)   # (B, M+L, D)

        B, M, _ = soft_prompt.shape
        soft_mask = torch.ones((B, M), dtype=attn_mask.dtype, device=device)
        attn_mask2 = torch.cat([soft_mask, attn_mask], dim=1)

        pad_labels = torch.full((B, M), -100, dtype=labels.dtype, device=device)
        labels2 = torch.cat([pad_labels, labels], dim=1)

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask2, labels=labels2)
        return LMOutputs(loss=out.loss, logits=out.logits)

    @torch.no_grad()
    def generate_with_soft_prompt(
        self,
        soft_prompt: torch.Tensor,
        prompts: List[str],
        device: torch.device,
        max_new_tokens: int = 64,
        num_beams: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = False,
    ) -> List[str]:
        tok = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_emb = self.model.get_input_embeddings()(tok["input_ids"])
        inputs_embeds = torch.cat([soft_prompt, text_emb], dim=1)

        B, M, _ = soft_prompt.shape
        soft_mask = torch.ones((B, M), dtype=tok["attention_mask"].dtype, device=device)
        attn_mask = torch.cat([soft_mask, tok["attention_mask"]], dim=1)

        # 根据参数决定采样策略
        do_sample = (num_beams == 1)
        
        gen = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping if num_beams > 1 else False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # 解码并去除 prompt 前缀
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        outs = []
        for i, t in enumerate(texts):
            prompt_prefix = prompts[i] if i < len(prompts) else ""
            if prompt_prefix and prompt_prefix in t:
                t = t.split(prompt_prefix, 1)[-1].strip()
            outs.append(t.strip())
        return outs

    @torch.no_grad()
    def generate_with_soft_prompt_and_count(
        self,
        soft_prompt: torch.Tensor,
        prompts: List[str],
        device: torch.device,
        max_new_tokens: int = 64,
        num_beams: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = False,
    ) -> Tuple[List[str], int]:
        """
        与 generate_with_soft_prompt 相同，但额外返回生成的 token 数量。
        Returns: (captions, num_generated_tokens)
        """
        tok = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_emb = self.model.get_input_embeddings()(tok["input_ids"])
        inputs_embeds = torch.cat([soft_prompt, text_emb], dim=1)

        B, M, _ = soft_prompt.shape
        soft_mask = torch.ones((B, M), dtype=tok["attention_mask"].dtype, device=device)
        attn_mask = torch.cat([soft_mask, tok["attention_mask"]], dim=1)

        do_sample = (num_beams == 1)
        
        gen = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping if num_beams > 1 else False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # 计算生成的 token 数
        # 注意：使用 inputs_embeds 时, generate 可能返回包含生成序列的张量
        # 需要排除 pad token 来计算实际生成的 token 数
        # gen shape: (B, seq_len)
        gen_len = gen.shape[1]
        
        # 计算非 pad token 的数量作为生成 token 数
        # 排除 eos/pad token
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        non_pad_mask = (gen != pad_token_id)
        num_generated_tokens = int(non_pad_mask.sum().item() // gen.shape[0])  # 平均每个样本的 token 数
        
        # 如果仍为 0，使用总长度作为备选
        if num_generated_tokens == 0:
            num_generated_tokens = gen_len
        
        # 解码
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        outs = []
        for i, t in enumerate(texts):
            prompt_prefix = prompts[i] if i < len(prompts) else ""
            if prompt_prefix and prompt_prefix in t:
                t = t.split(prompt_prefix, 1)[-1].strip()
            outs.append(t.strip())
        
        return outs, num_generated_tokens
