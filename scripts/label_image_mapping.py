import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np

try:
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
except Exception:
    pd = None

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_model(device: str = None, model_name: str = "ViT-B/32"):   # same model used in the cliptext code is used
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


def list_label_images(images_root: str, label: str) -> List[str]:
    folder = Path(images_root) / label
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(str(folder / f"*{ext}")))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No images found for label '{label}' in {folder}")
    return paths


# feature encoding for texts
@torch.no_grad()
def encode_texts(model, device, texts: List[str], batch_size: int = 64) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats)
    return torch.cat(out, dim=0)


# feature encoding for images
@torch.no_grad()
def encode_images(model, preprocess, device, image_paths: List[str], batch_size: int = 64) -> torch.Tensor:
    out = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        imgs = torch.stack(imgs, dim=0).to(device)
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats)
    return torch.cat(out, dim=0)


# Initial label image mapping
# For each label, select the candidate image with the highest similarity to the label text as in the paper
def init_mapping_by_text_similarity(model, preprocess, device,
                                    labels: List[str],
                                    images_root: str) -> Dict[str, str]:
    
    print("\n[Step A] Initialization by text–image similarity (label name vs candidate images)")

    # Prepare label texts
    label_texts = labels

    label_text_emb = encode_texts(model, device, label_texts)  # shape [L, D]
    mapping = {}

    for li, label in enumerate(labels):
        candidates = list_label_images(images_root, label)
        img_emb = encode_images(model, preprocess, device, candidates)  # [M, D]
        # similarity: (M x D) · (D) -> (M,)
        sims = (img_emb @ label_text_emb[li].unsqueeze(1)).squeeze(1)
        best_idx = int(torch.argmax(sims).item())
        mapping[label] = candidates[best_idx]
        print(f"  - {label}: picked {Path(candidates[best_idx]).name} (init)")
    return mapping


# as the paper fine-tuned images selection using a dev set to pick the image giving best validation accuracy
def load_dev_csv(dev_csv: str) -> Tuple[List[str], List[str]]:
    if pd is None:
        raise RuntimeError("pandas & scikit-learn are required for dev-set validation. Please install them.")
    df = pd.read_csv(dev_csv)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dev CSV must have columns: text,label")
    return df["text"].astype(str).tolist(), df["label"].astype(str).tolist()


# dev evalution
# Given a fixed mapping, evaluate classification on dev set and predict label with highest dot-product similarity
@torch.no_grad()
def evaluate_mapping_on_dev(model, preprocess, device,
                            mapping: Dict[str, str],
                            labels: List[str],
                            dev_texts: List[str],
                            dev_labels: List[str],
                            metric: str = "accuracy") -> float:
    
    # Encode label images
    label_order = labels  # fixed order
    label_img_paths = [mapping[lbl] for lbl in label_order]
    label_img_emb = encode_images(model, preprocess, device, label_img_paths)  # [L, D]

    texts = dev_texts

    text_emb = encode_texts(model, device, texts)  # [N, D]
    
    sims = text_emb @ label_img_emb.t()                
    pred_idx = torch.argmax(sims, dim=1).cpu().numpy()
    pred_labels = [label_order[i] for i in pred_idx]

    if metric == "accuracy":
        score = accuracy_score(dev_labels, pred_labels)
    elif metric == "f1_weighted":
        score = f1_score(dev_labels, pred_labels, average="weighted")
    elif metric == "f1_macro":
        score = f1_score(dev_labels, pred_labels, average="macro")
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return float(score)


# choosing the image with the best validation score
# as described in paper, keep others fiexed, swap candidate images for target label and pick the one with best dev score
def argmax_over_candidates_for_label(model, preprocess, device,
                                     base_mapping: Dict[str, str],
                                     target_label: str,
                                     candidates: List[str],
                                     labels: List[str],
                                     dev_texts: List[str],
                                     dev_labels: List[str],
                                     metric: str) -> Tuple[str, float]:
    
    best_path = base_mapping[target_label]
    best_score = -1.0

    for cand_path in candidates:
        test_mapping = dict(base_mapping)
        test_mapping[target_label] = cand_path
        score = evaluate_mapping_on_dev(model, preprocess, device,
                                        test_mapping, labels, dev_texts, dev_labels,
                                        metric=metric)
        if score > best_score:
            best_score = score
            best_path = cand_path
    return best_path, best_score


def main():
    parser = argparse.ArgumentParser(description="CLIPTEXT-style label→image mapping with dev validation.")
    parser.add_argument("--images_root", type=str, required=True, help="Root folder with subfolders per label.")
    parser.add_argument("--labels", type=str, nargs="*", default=None, help="Label names (space-separated).")
    parser.add_argument("--dev_csv", type=str, default=None, help="Optional dev CSV with columns text,label.")
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "f1_weighted", "f1_macro"],
                        help="Evaluation metric for dev-set selection.")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="CLIP model variant.")
    parser.add_argument("--out_dir", type=str, default="./mapping_out", help="Output directory.")
    args = parser.parse_args()

    # Collect labels
    labels = []
    if args.labels:
        labels.extend(args.labels)
    labels = [l.strip() for l in labels if l.strip()]
    if not labels:
        raise ValueError("No labels provided. Use --labels")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model, preprocess, device = load_model(model_name=args.model_name)
    model.eval()

    # Step A: initialization by text–image similarity
    init_map = init_mapping_by_text_similarity(model, preprocess, device,
                                               labels=labels,
                                               images_root=args.images_root)

    # If no dev set, save and finish
    if not args.dev_csv:
        out_path = Path(args.out_dir) / "label_image_mapping_init_only.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(init_map, f, indent=2, ensure_ascii=False)
        print(f"\nSaved initial mapping (no dev refinement) → {out_path}")
        return

    # Load dev set
    dev_texts, dev_labels = load_dev_csv(args.dev_csv)

    # Evaluate initial mapping
    init_score = evaluate_mapping_on_dev(model, preprocess, device,
                                         init_map, labels, dev_texts, dev_labels,
                                         metric=args.metric)
    print(f"\n[Step B] Dev-set score with initial mapping: {args.metric} = {init_score:.4f}")

    # Step C: paper-style per-label refinement with dev selection
    print("\n[Step C] Refining each label by dev-set search (fix others, try all candidates for the target label)")
    refined = dict(init_map)
    for label in labels:
        candidates = list_label_images(args.images_root, label)
        best_path, best_score = argmax_over_candidates_for_label(
            model, preprocess, device,
            base_mapping=refined,
            target_label=label,
            candidates=candidates,
            labels=labels,
            dev_texts=dev_texts,
            dev_labels=dev_labels,
            metric=args.metric
        )
        refined[label] = best_path
        print(f"  - {label}: selected {Path(best_path).name} (dev {args.metric}={best_score:.4f})")

    final_score = evaluate_mapping_on_dev(model, preprocess, device,
                                        refined, labels, dev_texts, dev_labels,
                                        metric=args.metric)
    print(f"\n[Result] Dev-set score with refined mapping: {args.metric} = {final_score:.4f}")

    # Save outputs
    with open(Path(args.out_dir) / "label_image_mapping_refined.json", "w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2, ensure_ascii=False)
    summary = {
        "model_name": args.model_name,
        "metric": args.metric,
        "init_score": init_score,
        "final_score": final_score,
        "labels": labels,
    }
    with open(Path(args.out_dir) / "mapping_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved refined mapping in {Path(args.out_dir) / 'label_image_mapping_refined.json'}")


if __name__ == "__main__":
    main()