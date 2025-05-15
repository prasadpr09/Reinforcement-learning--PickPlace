import torch

ckpt_path = "/home/anu/mimicgen/training_results/core/pick_place_d0/image/trained_models/core_pick_place_d0_image/multimulti/models/model_epoch_600/data.pkl"

ckpt = torch.load(ckpt_path, map_location="cpu")
print(type(ckpt))
if isinstance(ckpt, dict):
    print(ckpt.keys())
