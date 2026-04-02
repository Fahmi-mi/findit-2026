from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TestImageDataset(Dataset):
    def __init__(self, file_paths: list[str], transform: A.Compose):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        file_path = self.file_paths[index]
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {file_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img)["image"]
        return tensor, Path(file_path).name


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


@torch.no_grad()
def predict_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
    tta_hflip: bool,
    tta_light_bc: bool,
    tta_bc_contrast: float,
    tta_bc_brightness: float,
    mixed_precision: bool,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    probs_all: list[np.ndarray] = []
    names_all: list[str] = []

    for images, file_names in loader:
        images = images.to(device, non_blocking=True)

        def _tta_views(x: torch.Tensor) -> list[torch.Tensor]:
            views = [x]
            if tta_hflip:
                views.append(torch.flip(x, dims=[3]))
            if tta_light_bc:
                bc = x * tta_bc_contrast + tta_bc_brightness
                bc = torch.clamp(bc, -3.0, 3.0)
                views.append(bc)
                if tta_hflip:
                    views.append(torch.flip(bc, dims=[3]))
            return views

        if use_tta:
            views = _tta_views(images)
            probs_acc = None
            for view in views:
                with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
                    logits = model(view)
                probs = torch.softmax(logits, dim=1)
                probs_acc = probs if probs_acc is None else (probs_acc + probs)
            probs = (probs_acc / len(views)).detach().cpu().numpy()
        else:
            with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
                logits = model(images)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        probs_all.append(probs)
        names_all.extend(list(file_names))

    return np.concatenate(probs_all, axis=0), names_all


def main() -> None:
    project_root = Path.cwd()
    oof_dir = project_root / "experiments" / "oof_predictions"
    ckpt_base = project_root / "experiments" / "checkpoints" / "baseline"
    out_dir = project_root / "output" / "test_probabilities"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_meta_path = project_root / "data" / "processed" / "clean" / "test_metadata_with_hash.csv"
    sample_sub_path = project_root / "data" / "raw" / "samplesubmission.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_oof_runs = sorted([p.stem.replace("oof_", "") for p in oof_dir.glob("oof_baseline_*.csv")])
    all_test_prob_runs = sorted([p.stem.replace("test_probs_", "") for p in out_dir.glob("test_probs_baseline_*.csv")])
    missing_runs = [rn for rn in all_oof_runs if rn not in set(all_test_prob_runs)]

    print(f"OOF runs: {len(all_oof_runs)} | Existing test_probs: {len(all_test_prob_runs)} | Missing: {len(missing_runs)}")
    if not missing_runs:
        print("No missing runs. Nothing to do.")
        return

    test_df = pd.read_csv(test_meta_path)
    sample_sub = pd.read_csv(sample_sub_path)
    test_paths = test_df["file_path"].tolist()

    for i, run_name in enumerate(missing_runs, start=1):
        print("=" * 80)
        print(f"[{i}/{len(missing_runs)}] Generating test_probs for: {run_name}")

        oof_path = oof_dir / f"oof_{run_name}.csv"
        oof_df = pd.read_csv(oof_path)
        prob_cols = sorted([c for c in oof_df.columns if c.startswith("prob_")])
        label_names = [c.replace("prob_", "") for c in prob_cols]
        num_classes = len(label_names)

        run_ckpt_dir = ckpt_base / run_name
        cfg_path = run_ckpt_dir / "config_used.json"
        if not cfg_path.exists():
            print(f"Skip: missing config file {cfg_path}")
            continue

        cfg = json.loads(cfg_path.read_text())
        model_name = cfg["model_name"]
        img_size = int(cfg["img_size"])
        batch_size = int(cfg.get("batch_size", 16))
        num_workers = int(cfg.get("num_workers_test", 0))
        mixed_precision = bool(cfg.get("mixed_precision", True))

        use_tta = bool(cfg.get("use_tta", True))
        tta_hflip = bool(cfg.get("tta_hflip", True))
        tta_light_bc = bool(cfg.get("tta_light_bc", False))
        tta_bc_contrast = float(cfg.get("tta_bc_contrast", 1.05))
        tta_bc_brightness = float(cfg.get("tta_bc_brightness", 0.02))

        tfm = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

        ds = TestImageDataset(test_paths, tfm)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        fold_paths = sorted(run_ckpt_dir.glob("fold*_best.pt"))
        if not fold_paths:
            print(f"Skip: no fold checkpoints in {run_ckpt_dir}")
            continue

        probs_sum = np.zeros((len(test_df), num_classes), dtype=np.float32)
        file_names_ref = None

        for fold_path in fold_paths:
            model = build_model(model_name=model_name, num_classes=num_classes).to(device)
            ckpt = torch.load(fold_path, map_location=device)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            model.load_state_dict(state)

            fold_probs, file_names = predict_test(
                model=model,
                loader=loader,
                device=device,
                use_tta=use_tta,
                tta_hflip=tta_hflip,
                tta_light_bc=tta_light_bc,
                tta_bc_contrast=tta_bc_contrast,
                tta_bc_brightness=tta_bc_brightness,
                mixed_precision=mixed_precision,
            )

            probs_sum += fold_probs
            if file_names_ref is None:
                file_names_ref = file_names

        probs_avg = probs_sum / float(len(fold_paths))

        prob_df = pd.DataFrame(probs_avg, columns=prob_cols)
        prob_df["id"] = [Path(n).stem for n in file_names_ref]
        prob_df = sample_sub[["id"]].merge(prob_df, on="id", how="left")

        if prob_df[prob_cols].isna().any().any():
            raise ValueError(f"Missing probabilities after id alignment for run: {run_name}")

        out_path = out_dir / f"test_probs_{run_name}.csv"
        prob_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
