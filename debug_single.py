import h5py
import json
import torch
from evaluate import get_exo_cameras, process_frame, register_llava_model

register_llava_model()
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.conversation import default_conversation
from llava.constants import IMAGE_TOKEN_INDEX
from transformers import CLIPImageProcessor

model_path = "data/model/model"
model_name = get_model_name_from_path(model_path)
tokenizer, model, _, ctx = load_pretrained_model(
    model_path, "liuhaotian/llava-v1.5-7b", model_name,
    load_8bit=False, load_4bit=False, device_map="auto",
)
model.config.tokenizer_padding_side = "left"

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device="cuda", dtype=torch.float16)

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
device = next(model.parameters()).device

with open("data/test_1perm_Falsetemp_Falsetempaug_EgoExOR_5k_samples_drophistory0.5.json") as f:
    samples = json.load(f)

sample = None
for s in samples:
    if s["hdf5_indices"]["surgery_type"] == "MISS":
        sample = s
        break

h = sample["hdf5_indices"]
take = f"data/{h['surgery_type']}/{h['procedure_id']}/take/{h['take_id']}"
exo_ids, exo_names = get_exo_cameras("data/hdf5/data/egoexor_miss.h5", take)
print(f"Take: {take}")
print(f"Exo cameras: {exo_names} (ids: {exo_ids})")

# Pick external_1
chosen = exo_ids[0]
for k, name in zip(exo_ids, exo_names):
    if name == "external_1":
        chosen = k
        break
print(f"Using camera index {chosen}")

with h5py.File("data/hdf5/data/egoexor_miss.h5", "r") as f:
    frame_rgb = f[f"{take}/frames/rgb"][h["frame_idx"]]

img = process_frame(torch.from_numpy(frame_rgb[chosen]).float(), image_processor)
print(f"Image shape: {img.shape}")

# Use ORIGINAL prompt from test JSON (has single <image> token)
conv = default_conversation.copy()
conv.append_message(conv.roles[0], sample["conversations"][0]["value"])
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
print(f"input_ids shape: {input_ids.shape}")
print(f"Num <image> tokens: {(input_ids == IMAGE_TOKEN_INDEX).sum().item()}")

images = [img.unsqueeze(0).to(device, dtype=torch.float16)]

with torch.inference_mode():
    output_ids = model.generate(
        inputs=input_ids,
        images=images,
        do_sample=False,
        use_cache=True,
        max_new_tokens=300,
    )

output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
print(f"\nRaw output: [{output[:500]}]")
print(f"\nGT: [{sample['conversations'][1]['value'][:300]}]")
