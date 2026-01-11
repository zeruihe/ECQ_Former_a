import os
from huggingface_hub import snapshot_download

# 获取环境变量（如果有设置镜像站）
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
print("HF_ENDPOINT =", HF_ENDPOINT)

# 目标：只下载 roberta-large
# 存放路径保持与你其他模型一致：/root/autodl-tmp/hf_models/roberta-large
targets = [
    ("roberta-large",
     "/root/autodl-tmp/hf_models/roberta-large"),
]

for repo_id, local_dir in targets:
    os.makedirs(local_dir, exist_ok=True)
    print(f"\n==> Downloading {repo_id} -> {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 实体文件落盘
            resume_download=True,
        )
        print(f"    [Success] {repo_id} downloaded successfully.")
    except Exception as e:
        print(f"    [Error] Failed to download {repo_id}: {e}")

print("\nAll downloads finished.")