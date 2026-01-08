import argparse
from dataclasses import dataclass
import yaml

from data.schema import DataSourceCfg, AnnSplitCfg
from data.builder import build_caption_loader

def load_sources_from_yaml(yaml_path: str):
    raw = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
    data = raw["data"]
    sources = []
    for s in data["sources"]:
        ann = AnnSplitCfg(**s["ann"])
        sources.append(DataSourceCfg(
            name=s["name"],
            root=s["root"],
            ann=ann,
            image_key=s.get("image_key", "image"),
            text_key=s.get("text_key", "caption"),
            id_key=s.get("id_key", None),
        ))
    return data, sources

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train","valid","test"])
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    data_cfg, sources = load_sources_from_yaml(args.config)

    loader = build_caption_loader(
        split=args.split,
        sources=sources,
        batch_size=data_cfg.get("batch_size", 2),
        num_workers=data_cfg.get("num_workers", 2),
        return_pil=True,
        pin_memory=data_cfg.get("pin_memory", True),
        shuffle=False,
    )

    it = iter(loader)
    for i in range(args.n):
        batch = next(it)
        print(f"Batch {i}:")
        print("  captions[0]:", batch["captions"][0][:120])
        print("  meta[0]:", batch["meta"][0])
        print("  image size:", batch["images_pil"][0].size)

if __name__ == "__main__":
    main()
