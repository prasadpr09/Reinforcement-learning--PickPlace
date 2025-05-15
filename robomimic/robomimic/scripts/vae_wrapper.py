import torch
from robomimic.utils.file_utils import maybe_dict_from_checkpoint
from robomimic.models.policy_nets import VAEActor

class VAEWrapper:
    def __init__(self, vae_path, device="cuda"):
        checkpoint = maybe_dict_from_checkpoint(vae_path)
        # Initialize VAEActor with parameters matching checkpoint
        self.vae = VAEActor(
            obs_shapes=checkpoint["shape_meta"]["all_shapes"],
            goal_shapes={},  # No goals in this config
            ac_dim=checkpoint["shape_meta"]["ac_dim"],
            device=device,
            encoder_kwargs={},  # Add if needed from checkpoint["config"]
            latent_dim=14,  # Matches bc_rnn.json
            kl_weight=1.0
        )
        self.vae.load_state_dict(checkpoint["model"])  # Load state_dict
        self.device = torch.device(device)
        self.vae.eval()
    
    def __call__(self, batch):
        with torch.no_grad():
            # Handle RNN sequence data: [B, T, ...]
            obs_dict = batch["obs"]
            B, T = obs_dict[list(obs_dict.keys())[0]].shape[:2]
            # Reshape to [B*T, ...] for VAE processing
            flat_obs = {k: v.view(B*T, *v.shape[2:]) for k, v in obs_dict.items()}
            vae_out = self.vae.forward_train(
                actions=None,
                obs_dict=flat_obs,
                goal_dict=batch.get("goal_obs", None),
                freeze_encoder=False
            )
            # Reshape latent_vae back to [B, T, latent_dim]
            batch["obs"]["latent_vae"] = vae_out["encoder_z"].view(B, T, -1)
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(self.device)
                elif isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
        return batch