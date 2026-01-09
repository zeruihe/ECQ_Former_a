# scripts/cache_models.py
## 下载huggingface 中的模型
import os
from huggingface_hub import snapshot_download

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
print("HF_ENDPOINT =", HF_ENDPOINT)

targets = [
    ("TsinghuaC3I/Llama-3.1-8B-UltraMedical",
     "/root/autodl-tmp/hf_models/TsinghuaC3I/Llama-3.1-8B-UltraMedical"),
    ("openai/clip-vit-large-patch14",
     "/root/autodl-tmp/hf_models/openai/clip-vit-large-patch14"),
    ("facebook/dinov2-large",
     "/root/autodl-tmp/hf_models/facebook/dinov2-large"),
    ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
     "/root/autodl-tmp/hf_models/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
]

for repo_id, local_dir in targets:
    os.makedirs(local_dir, exist_ok=True)
    print(f"\n==> Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,   # 直接落盘实体文件，便于迁移/备份
        resume_download=True,
    )
    print("   done.")
