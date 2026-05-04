#!/usr/bin/env python3
"""
Evaluate EgoExOR model on MISS test split using exocentric RGB frames only.

Usage:
    # After setup.sh and download.py
    python evaluate.py \
        --model_path data/model/llava-v1.5-7b-task-lora_hybridor_qlora_4perm_EgoExOR \
        --test_json data/test_1perm_Falsetemp_Falsetempaug_EgoExOR_5k_samples_drophistory0.5.json \
        --hdf5_path data/egoexor_miss.h5 \
        --output_csv eval_miss_exo_results.csv
"""
import os
import sys
import re
import csv
import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
EGOEXOR_DIR = SCRIPT_DIR / "EgoExOR"
sys.path.insert(0, str(EGOEXOR_DIR))
sys.path.insert(0, str(EGOEXOR_DIR / "scene_graph_generation" / "LLaVA"))

EXO_SOURCES = {"or_light", "microscope", "external_1", "external_2", "external_3", "external_4", "external_5", "simstation"}
EGO_SOURCES = {"head_surgeon", "assistant", "circulator", "anesthetist"}

SOURCES = {
    "head_surgeon": 1, "assistant": 2, "circulator": 3, "anesthetist": 4,
    "or_light": 5, "microscope": 6, "external_1": 7, "external_2": 8,
    "external_3": 9, "external_4": 10, "external_5": 11, "simstation": 12,
    "ultrasound": 13, "blank": -1,
}
reversed_sources = {v: k for k, v in SOURCES.items()}

scene_graph_name_to_vocab_idx = {
    "anesthetist": 0, "anesthesia_equipment": 1, "antiseptic": 2, "assistant": 3,
    "bin": 4, "body_marker": 5, "circulator": 6, "cotton": 7, "curette": 8,
    "dressing_material": 9, "forceps": 10, "gloves": 11, "head_surgeon": 12,
    "health_monitor": 13, "herbal_disk": 14, "instrument_table": 15, "instruments": 16,
    "microscope": 17, "microscope_controller": 18, "microscope_eye": 19,
    "microscope_screen": 20, "needle": 21, "operating_room": 22, "operating_table": 23,
    "patient": 24, "scalpel": 25, "scissors": 26, "syringe": 27, "tissue_mark": 28,
    "tissue_paper": 29, "ultrasound_gel": 30, "ultrasound_machine": 31,
    "ultrasound_probe": 32, "ultrasound_screen": 33, "unsterile_instruments": 34,
    "vertebrae": 35,
    "anaesthetising": 36, "applying": 37, "aspirating": 38, "looking": 39,
    "closeto": 40, "controlling": 41, "cutting": 42, "disinfection": 43, "dressing": 44,
    "dropping": 45, "entering": 46, "holding": 47, "injecting": 48, "inserting": 49,
    "lyingon": 50, "manipulating": 51, "positioning": 52, "preparing": 53, "removing": 54,
    "scanning": 55, "touching": 56, "wearing": 57,
}
vocab_idx_to_scene_graph_name = {v: k for k, v in scene_graph_name_to_vocab_idx.items()}

entity_synonyms = {
    "operating_table": ["operation_table", "operating_table"],
    "anesthetist": ["anaesthetist"],
    "microscope": ["microcope"],
}
relation_synonyms = {
    "closeto": ["closeTo"],
    "lyingon": ["lyingOn"],
    "looking": ["looking", "checking"],
    "anaesthetising": ["anesthasing"],
}


def reverse_synonym_mapping(synonyms_dict):
    reversed_dict = {}
    for key, synonyms_list in synonyms_dict.items():
        for synonym in synonyms_list:
            reversed_dict[synonym] = key
    return reversed_dict


reversed_entity_synonyms = reverse_synonym_mapping(entity_synonyms)
reversed_relation_synonyms = reverse_synonym_mapping(relation_synonyms)


def map_scene_graph_name_to_vocab_idx(name):
    name = name.lower()
    if name in reversed_relation_synonyms:
        name = reversed_relation_synonyms[name]
    if name in reversed_entity_synonyms:
        name = reversed_entity_synonyms[name]
    return scene_graph_name_to_vocab_idx[name]


def parse_triplets(text):
    triplets = []
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    if "<SG>" in text and "</SG>" in text and text.index("<SG>") < text.index("</SG>"):
        triplet_str = text.split("<SG>")[1].split("</SG>")[0].strip().split(";")
    else:
        triplet_str = text.split(";")

    for triplet in triplet_str:
        triplet = triplet.replace(".", "").replace("</s>", "").replace("<s>", "").strip()
        if not triplet:
            continue
        parts = [p.strip() for p in triplet.split(",")]
        if len(parts) != 3:
            continue
        sub, obj, pred = parts
        if sub in reversed_entity_synonyms:
            sub = reversed_entity_synonyms[sub]
        if obj in reversed_entity_synonyms:
            obj = reversed_entity_synonyms[obj]
        if pred in reversed_relation_synonyms:
            pred = reversed_relation_synonyms[pred]
        triplets.append((sub, pred, obj))
    return triplets


def map_triplets_to_indices(triplets):
    mapped = []
    for sub, pred, obj in triplets:
        try:
            s = map_scene_graph_name_to_vocab_idx(sub.replace(" ", "_"))
            o = map_scene_graph_name_to_vocab_idx(obj.replace(" ", "_"))
            p = map_scene_graph_name_to_vocab_idx(pred)
            mapped.append((s, p, o))
        except Exception:
            continue
    return mapped


def get_exo_cameras_for_take(h5_path, take_path):
    """Get exocentric camera indices and names for a given take."""
    exo_ids, exo_names = [], []
    with h5py.File(h5_path, "r") as f:
        src_path = f"{take_path}/sources"
        if src_path not in f:
            return exo_ids, exo_names
        src_grp = f[src_path]
        count = src_grp.attrs.get("source_count", 0)
        for i in range(count):
            key = f"source_{i}"
            if key in src_grp.attrs:
                name = src_grp.attrs[key]
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                if name in EXO_SOURCES:
                    exo_ids.append(i)
                    exo_names.append(name)
    return exo_ids, exo_names


def adjust_prompt_for_exo_only(conversation_value, exo_count):
    """Replace image tokens in prompt to match exocentric camera count."""
    num_tokens = conversation_value.count("<image>")
    if num_tokens == exo_count:
        return conversation_value
    prefix = " ".join(["<image>"] * exo_count)
    prompt_text = re.sub(r"(<image>\s*)+", "", conversation_value).strip()
    return f"{prefix} {prompt_text}"


class FrameTransform:
    def __init__(self, processor, pad_to_square=True):
        self.processor = processor
        self.pad_to_square = pad_to_square

    @staticmethod
    def expand2square(pil_img, background_color):
        w, h = pil_img.size
        if w == h:
            return pil_img
        elif w > h:
            result = Image.new(pil_img.mode, (w, w), background_color)
            result.paste(pil_img, (0, (w - h) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (h, h), background_color)
            result.paste(pil_img, ((h - w) // 2, 0))
            return result

    def __call__(self, data):
        if isinstance(data, Image.Image):
            if self.pad_to_square:
                bg = tuple(int(x * 255) for x in self.processor.image_mean)
                data = self.expand2square(data, bg)
            processed = self.processor.preprocess(data, return_tensors="pt")["pixel_values"]
            return processed.squeeze(0).to(dtype=torch.bfloat16)

        elif isinstance(data, torch.Tensor):
            shape = data.shape
            if len(shape) == 3:
                frame = data.cpu().numpy()
                if frame.max() == 0.0:
                    return None
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                pil = Image.fromarray(frame).convert("RGB")
                return self.__call__(pil)
            elif len(shape) == 4:
                results = []
                for i in range(shape[0]):
                    proc = self.__call__(data[i])
                    if proc is not None:
                        results.append(proc)
                return torch.stack(results) if results else None
        return None


class MISSExoRGBDataset(torch.utils.data.Dataset):
    """Dataset providing MISS test samples with exocentric RGB frames only."""

    def __init__(self, test_json_path: str, hdf5_path: str):
        import json

        with open(test_json_path) as f:
            all_samples = json.load(f)

        self.samples = []
        self.hdf5_path = hdf5_path

        self._exo_cache = {}

        for sample in all_samples:
            st = sample["hdf5_indices"]["surgery_type"]
            if st != "MISS":
                continue
            sample = deepcopy(sample)
            sample["hdf5_indices"]["available_modalities"] = ["exo_frames"]
            self.samples.append(sample)

        print(f"Filtered {len(self.samples)} MISS test samples from {len(all_samples)} total")

    def _get_exo_cameras(self, take_path):
        if take_path not in self._exo_cache:
            self._exo_cache[take_path] = get_exo_cameras_for_take(self.hdf5_path, take_path)
        return self._exo_cache[take_path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        h = sample["hdf5_indices"]
        take_path = f"data/{h['surgery_type']}/{h['procedure_id']}/take/{h['take_id']}"
        exo_ids, exo_names = self._get_exo_cameras(take_path)

        sample["conversations"][0]["value"] = adjust_prompt_for_exo_only(
            sample["conversations"][0]["value"], len(exo_ids)
        )

        sample["sample_id"] = f"{h['surgery_type']}_{h['procedure_id']}_{h['take_id']}_{h['frame_idx']}"

        return {
            "sample": sample,
            "ego_source_ids": [],
            "ego_source_names": [],
            "exo_source_ids": exo_ids,
            "exo_source_names": exo_names,
        }


class ExoCollator:
    def __call__(self, instances):
        return {
            "sample": [inst["sample"] for inst in instances],
            "ego_source_names": [inst["ego_source_names"] for inst in instances],
            "ego_source_ids": [inst["ego_source_ids"] for inst in instances],
            "exo_source_names": [inst["exo_source_names"] for inst in instances],
            "exo_source_ids": [inst["exo_source_ids"] for inst in instances],
        }


def run_evaluation(args):
    from LLaVA.llava.constants import IMAGE_TOKEN_INDEX
    from LLaVA.llava.conversation import default_conversation, SeparatorStyle
    from LLaVA.llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
    from LLaVA.llava.model.builder import load_pretrained_model

    print(f"Loading model from {args.model_path}...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        "liuhaotian/llava-v1.5-7b",
        model_name,
        False,
        True,
        device_map="auto",
    )
    model.config.mv_type = "learned"
    model.config.tokenizer_padding_side = "left"

    frame_transform = FrameTransform(image_processor)

    print(f"Loading test data from {args.test_json}...")
    dataset = MISSExoRGBDataset(args.test_json, args.hdf5_path)
    collator = ExoCollator()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    print(f"Running inference on {len(dataset)} MISS test frames (exocentric RGB only)...")
    csv_rows = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch_size = len(batch["sample"])
        outputs_data = []

        with h5py.File(args.hdf5_path, "r") as f:
            for bidx in range(batch_size):
                metadata = batch["sample"][bidx]["hdf5_indices"]
                exo_source_ids = batch["exo_source_ids"][bidx]

                path = f"data/{metadata['surgery_type']}/{metadata['procedure_id']}/take/{metadata['take_id']}"
                frame_idx = metadata["frame_idx"]
                frame_rgb = torch.from_numpy(f[f"{path}/frames/rgb"][frame_idx]).float()

                exo_images = []
                for sid in exo_source_ids:
                    processed = frame_transform(frame_rgb[sid])
                    if processed is not None:
                        exo_images.append(processed)

                conv = deepcopy(default_conversation)
                convo = batch["sample"][bidx]["conversations"]
                conv.append_message(convo[0]["from"], convo[0]["value"])
                conv.append_message(convo[1]["from"], None)
                prompt = conv.get_prompt()

                data_dict = {"prompt": prompt}

                if exo_images:
                    data_dict["exo_frames"] = torch.stack(exo_images)
                    data_dict["exo_source_ids"] = exo_source_ids
                    data_dict["exo_source_names"] = batch["exo_source_names"][bidx]
                else:
                    data_dict["exo_frames"] = torch.zeros(1, 3, 336, 336)
                    data_dict["exo_source_ids"] = [0]
                    data_dict["exo_source_names"] = []

                data_dict["ego_frames"] = None
                data_dict["ego_source_ids"] = []
                data_dict["ego_source_names"] = []

                outputs_data.append(data_dict)

        all_prompts = [x["prompt"] for x in outputs_data]

        if batch_size == 1:
            input_ids = tokenizer_image_token(
                all_prompts[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
        else:
            ids_list = [
                tokenizer_image_token(p, tokenizer, return_tensors="pt")
                for p in all_prompts
            ]
            inverted = [torch.flip(ids, dims=[0]) for ids in ids_list]
            padded = torch.nn.utils.rnn.pad_sequence(inverted, batch_first=True, padding_value=tokenizer.pad_token_id)
            input_ids = torch.flip(padded, dims=[1]).to(model.device)

        def collect(key):
            vals = [x[key] for x in outputs_data if key in x and x[key] is not None]
            if all(isinstance(v, torch.Tensor) for v in vals):
                try:
                    return torch.stack(vals)
                except RuntimeError:
                    return vals
            return vals

        forward_kwargs = {
            "input_ids": input_ids,
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": 300,
        }

        for key in ["ego_frames", "exo_frames", "ego_source_ids", "exo_source_ids",
                     "ego_source_names", "exo_source_names"]:
            collected = collect(key)
            if collected:
                forward_kwargs[key] = collected

        device = next(model.parameters()).device
        for k, v in forward_kwargs.items():
            if torch.is_tensor(v):
                forward_kwargs[k] = v.to(device)
            elif isinstance(v, list):
                forward_kwargs[k] = [
                    vi.to(device) if torch.is_tensor(vi) else vi for vi in v
                ]

        with torch.inference_mode():
            output_ids = model.generate(**forward_kwargs)

        if batch_size == 1:
            text_outputs = [tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()]
        else:
            text_outputs = tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1:].tolist()],
                skip_special_tokens=True,
            )

        for idx, output in enumerate(text_outputs):
            sample = batch["sample"][idx]
            gt_text = sample["conversations"][1]["value"]
            h = sample["hdf5_indices"]

            pred_triplets = parse_triplets(output)
            gt_triplets = parse_triplets(gt_text)

            pred_mapped = map_triplets_to_indices(pred_triplets)
            gt_mapped = map_triplets_to_indices(gt_triplets)

            gt_set = set(gt_mapped)
            pred_set = set(pred_mapped)

            if len(gt_set) == 0 and len(pred_set) == 0:
                frame_f1 = 1.0
            elif len(gt_set) == 0:
                frame_f1 = 0.0
            else:
                tp = len(gt_set & pred_set)
                precision = tp / len(pred_set) if pred_set else 0.0
                recall = tp / len(gt_set)
                frame_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            csv_rows.append({
                "sample_id": sample.get("sample_id", sample.get("id", "")),
                "procedure": h["surgery_type"],
                "phase": h["procedure_id"],
                "take": h["take_id"],
                "frame_idx": h["frame_idx"],
                "ground_truth": "; ".join(f"{s},{o},{p}" for s, p, o in gt_triplets),
                "prediction": "; ".join(f"{s},{o},{p}" for s, p, o in pred_triplets),
                "gt_count": len(gt_mapped),
                "pred_count": len(pred_mapped),
                "true_positives": len(gt_set & pred_set),
                "frame_f1": round(frame_f1, 4),
                "raw_output": output[:1000],
            })

    output_csv = Path(args.output_csv)
    if not csv_rows:
        print("No results to save.")
        return

    fieldnames = list(csv_rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    total_tp = sum(r["true_positives"] for r in csv_rows)
    total_gt = sum(r["gt_count"] for r in csv_rows)
    total_pred = sum(r["pred_count"] for r in csv_rows)
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    overall_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_frame_f1 = np.mean([r["frame_f1"] for r in csv_rows])

    print(f"\n{'=' * 60}")
    print(f"  MISS Exocentric RGB-Only Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Total frames evaluated: {len(csv_rows)}")
    print(f"  Precision:              {precision:.4f}")
    print(f"  Recall:                 {recall:.4f}")
    print(f"  Overall F1 (triplet):   {overall_f1:.4f}")
    print(f"  Average Frame F1:       {avg_frame_f1:.4f}")
    print(f"{'=' * 60}")
    print(f"  Results saved to: {output_csv}")

    summary_path = output_csv.with_suffix(".summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"MISS Exocentric RGB-Only Evaluation Summary\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Total frames: {len(csv_rows)}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Overall F1: {overall_f1:.4f}\n")
        f.write(f"Average Frame F1: {avg_frame_f1:.4f}\n")
    print(f"  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EgoExOR on MISS test split (exocentric RGB only)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test samples JSON")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to merged MISS HDF5 file")
    parser.add_argument("--output_csv", type=str, default="eval_miss_exo_results.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
