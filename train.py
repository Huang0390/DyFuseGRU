"""Train and evaluate the original DyFuseGRU on disjoint ground-motion groups."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import DyFuseGRU


def read_manifest(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))["records"]
    if not rows:
        raise ValueError("Manifest has no records")
    groups, ids, array_owners = {}, set(), {}
    records = []
    for row in rows:
        record = dict(row)
        for key in ("id", "group_id", "split", "input_path", "target_path", "append_path", "floors"):
            if key not in record:
                raise ValueError(f"Missing field {key}")
        if record["id"] in ids:
            raise ValueError(f"Duplicate record id: {record['id']}")
        ids.add(record["id"])
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        group = str(record["group_id"])
        if group in groups and groups[group] != split:
            raise ValueError(f"Ground-motion group crosses splits: {group}")
        groups[group] = split
        record["floors"] = int(record["floors"])
        record["prefix_samples"] = int(record.get("prefix_samples", 0))
        if record["floors"] < 1 or record["prefix_samples"] < 0:
            raise ValueError("Invalid floors or prefix_samples")
        for key in ("input_path", "target_path", "append_path"):
            source = (path.parent / record[key]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            # A reused array across partitions is a direct leakage error.
            if source in array_owners and array_owners[source] != split:
                raise ValueError(f"Array crosses splits: {source}")
            array_owners[source] = split
            record[key] = source
        records.append(record)
    if not any(r["split"] == "train" for r in records) or not any(r["split"] == "val" for r in records):
        raise ValueError("Both train and val records are required")
    return records


def load_record(record, model_floors):
    arrays = []
    for key in ("input_path", "target_path", "append_path"):
        arr = np.load(record[key], allow_pickle=False, mmap_mode="r")
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3 or arr.shape[-1] != 1 or not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Expected numerical [floors,time,1]: {record[key]}")
        arrays.append(arr)
    if len({a.shape for a in arrays}) != 1 or arrays[0].shape[0] != model_floors:
        raise ValueError(f"Inconsistent input shapes for {record['id']}")
    if record["floors"] > model_floors:
        raise ValueError(f"Too many real floors: {record['id']}")
    prefix = record["prefix_samples"]
    if prefix >= arrays[0].shape[1]:
        raise ValueError(f"Prefix consumes whole record: {record['id']}")
    if prefix and any(np.any(a[:, :prefix] != 0) for a in arrays):
        raise ValueError(f"Declared prefix contains nonzero data: {record['id']}")
    data = tuple(np.array(a[:, prefix:], dtype=np.float32, copy=True) for a in arrays)
    if not all(np.isfinite(a).all() for a in data):
        raise ValueError(f"Nonfinite data: {record['id']}")
    if any(np.any(a[record["floors"]:] != 0) for a in data):
        raise ValueError(f"Padded floor contains data: {record['id']}")
    return data


def window(array, start, length):
    """Slice physical time; fill samples before time zero and after the end with zero."""
    result = np.zeros((array.shape[0], length, 1), dtype=np.float32)
    lo, hi = max(start, 0), min(start + length, array.shape[1])
    if hi > lo:
        result[:, lo - start:hi - start] = array[:, lo:hi]
    return result


class WaveWindows(Dataset):
    def __init__(self, records, floors, history, horizon, stride, response_scale, input_scale):
        if min(history, horizon, stride) < 1 or horizon > history:
            raise ValueError("Require positive history/stride and horizon <= history")
        self.data = [(r, load_record(r, floors)) for r in records]
        self.history, self.horizon = history, horizon
        self.response_scale, self.input_scale = response_scale, input_scale
        self.indices = [(i, start) for i, (_, arrays) in enumerate(self.data)
                        for start in range(0, arrays[1].shape[1], stride)]
        if not self.indices:
            raise ValueError("No training windows")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record_index, start = self.indices[index]
        record, (earthquake, target, append) = self.data[record_index]
        history, horizon = self.history, self.horizon
        past = window(target, start - history, history) * self.response_scale
        known_earthquake = window(earthquake, start - history, history + horizon) * self.input_scale
        known_append = window(append, start - history, history + horizon) * self.response_scale
        expected = window(target, start, horizon) * self.response_scale
        mask = np.zeros_like(expected)
        mask[:record["floors"], :min(horizon, target.shape[1] - start)] = 1
        return tuple(torch.from_numpy(x) for x in (past, known_earthquake, known_append, expected, mask))


def masked_mse(prediction, target, mask):
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp_min(1)


@torch.inference_mode()
def predict_event(model, earthquake, append, history, horizon, input_scale, response_scale, device):
    """Predict from physical time zero; never receives a target response."""
    if earthquake.shape != append.shape:
        raise ValueError("Known inputs have different shapes")
    floors, n, _ = earthquake.shape
    past = torch.zeros((1, floors, history, 1), device=device)
    zeros = torch.zeros((1, floors, horizon, 1), device=device)
    blocks = []
    model.eval()
    for start in range(0, n, horizon):
        eq = torch.from_numpy(window(earthquake, start - history, history + horizon))[None].to(device) * input_scale
        ap = torch.from_numpy(window(append, start - history, history + horizon))[None].to(device) * response_scale
        forecast = model(torch.cat((past, zeros), dim=2), eq, ap)
        if not torch.isfinite(forecast).all():
            raise FloatingPointError("Nonfinite model prediction")
        blocks.append(forecast[0].cpu().numpy())
        past = torch.cat((past[:, :, horizon:], forecast), dim=2)
    return np.concatenate(blocks, axis=1)[:, :n] / response_scale


def evaluate(model, records, args, device):
    nmre_sum, nrmse_sum, r_sum = 0.0, 0.0, 0.0
    error_count, correlation_count = 0, 0
    for record in records:
        earthquake, target, append = load_record(record, args.floors)
        forecast = predict_event(model, earthquake, append, args.history, args.horizon,
                                 args.input_scale, args.response_scale, device)
        for floor in range(record["floors"]):
            actual = target[floor, :, 0].astype(np.float64)
            predicted = forecast[floor, :, 0].astype(np.float64)
            peak = float(np.max(np.abs(actual)))
            if peak == 0:
                continue
            difference = predicted - actual
            nmre_sum += float(np.mean(np.abs(difference))) / peak
            nrmse_sum += math.sqrt(float(np.mean(np.square(difference)))) / peak
            error_count += 1

            actual_centered = actual - actual.mean()
            predicted_centered = predicted - predicted.mean()
            denominator = math.sqrt(float(np.dot(actual_centered, actual_centered)) *
                                    float(np.dot(predicted_centered, predicted_centered)))
            if denominator > 0:
                r_sum += float(np.dot(actual_centered, predicted_centered)) / denominator
                correlation_count += 1
    if error_count == 0:
        raise ValueError("Evaluation split has no nonzero target response")
    return {"NMRE": nmre_sum / error_count,
            "NRMSE": nrmse_sum / error_count,
            "R": r_sum / correlation_count if correlation_count else None}


def make_model(args):
    return DyFuseGRU(floor_num=args.floors, time_len=args.history + args.horizon,
                     pred_lenth=args.horizon, dim=args.dim, n_layers=args.layers,
                     dropout=args.dropout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/dyfusegru"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--floors", type=int, help="Padded model floor count; required for training")
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--response-scale", type=float, default=1.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = read_manifest(args.manifest)
    if args.command == "evaluate":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for evaluation")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for key, value in saved["model_config"].items():
            setattr(args, key, value)
        args.input_scale = saved["input_scale"]
        args.response_scale = saved["response_scale"]
        model = make_model(args).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        print(json.dumps(evaluate(model, [r for r in records if r["split"] == args.split], args, device)))
        return
    if args.epochs < 1 or args.lr <= 0:
        parser.error("Require positive epochs and learning rate")
    if args.floors is None:
        parser.error("--floors is required for training")
    if min(args.floors, args.history, args.horizon, args.dim, args.layers, args.batch_size) < 1:
        parser.error("Model and batch dimensions must be positive")
    if args.dim < 2 or args.horizon > args.history or not 0 <= args.dropout < 1:
        parser.error("Require dim >= 2, horizon <= history and 0 <= dropout < 1")
    if args.response_scale <= 0 or args.input_scale <= 0:
        parser.error("Scales must be positive")
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    model = make_model(args).to(device)
    dataset = WaveWindows(train_records, args.floors, args.history, args.horizon,
                          args.stride, args.response_scale, args.input_scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        for past, earthquake, append, target, mask in loader:
            past, earthquake, append, target, mask = (
                x.to(device) for x in (past, earthquake, append, target, mask)
            )
            tokens = torch.cat((past, torch.zeros_like(target)), dim=2)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tokens, earthquake, append)
            loss = masked_mse(prediction, target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_records, args, device)
        row = {"epoch": epoch, "val": metrics}
        print(json.dumps(row), flush=True)
        if metrics["NRMSE"] < best:
            best = metrics["NRMSE"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "manifest_sha256": manifest_sha256,
                        "model_config": {"floors": args.floors, "history": args.history,
                                         "horizon": args.horizon, "dim": args.dim,
                                         "layers": args.layers, "dropout": args.dropout},
                        "input_scale": args.input_scale,
                        "response_scale": args.response_scale}, args.output / "best.pt")


if __name__ == "__main__":
    main()
