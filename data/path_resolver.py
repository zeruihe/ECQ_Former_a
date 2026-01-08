from __future__ import annotations
import os

## 解析图片路径
def resolve_image_path(root: str, rel: str, ann_path: str) -> str:
    """
    Robust resolver for cases like:
      root=/data
      rel=mimic_cxr/valid/xxx.jpg  -> /data/mimic_cxr/valid/xxx.jpg

    Also handles accidental duplication:
      root=/data/mimic_cxr
      rel=mimic_cxr/valid/xxx.jpg  -> /data/mimic_cxr/valid/xxx.jpg (dedup)

    Finally tries ann_dir + rel.
    """
    rel = rel.strip()
    if not rel:
        return rel

    # absolute path
    if os.path.isabs(rel) and os.path.exists(rel):
        return rel

    rel2 = rel.lstrip("/")

    cand = os.path.join(root, rel2)
    if os.path.exists(cand):
        return cand

    # dedup if root basename equals rel prefix
    root_base = os.path.basename(os.path.normpath(root))
    parts = rel2.split(os.sep)
    if parts and parts[0] == root_base:
        cand2 = os.path.join(root, os.path.join(*parts[1:]))
        if os.path.exists(cand2):
            return cand2

    # ann directory + rel
    ann_dir = os.path.dirname(os.path.abspath(ann_path))
    cand3 = os.path.join(ann_dir, rel2)
    if os.path.exists(cand3):
        return cand3

    # return default candidate for easier debugging
    return cand
