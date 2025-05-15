import torch
from robomimic.utils.file_utils import maybe_dict_from_checkpoint, get_shape_metadata_from_dataset
from robomimic.models.policy_nets import VAEActor
import numpy as np
import logging
import inspect

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VAEWrapper:
    def __init__(self, vae_path, dataset_path, device="cuda"):
        checkpoint = maybe_dict_from_checkpoint(vae_path)
        logger.debug(f"Checkpoint shape_meta: {checkpoint.get('shape_meta', {})}")
        # Load shape_meta from dataset if not in checkpoint
        if "shape_meta" not in checkpoint:
            obs_keys = [
                "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                "robot0_joint_pos", "robot0_joint_vel"
            ]
            shape_meta = get_shape_metadata_from_dataset(
                dataset_path=dataset_path,
                all_obs_keys=obs_keys,
                verbose=True
            )
        else:
            shape_meta = checkpoint["shape_meta"]
        
        # Initialize VAEActor with low-dimensional observations
        self.vae = VAEActor(
            obs_shapes=shape_meta["all_shapes"],
            goal_shapes={},
            ac_dim=shape_meta["ac_dim"],
            encoder_layer_dims=[300, 400],
            decoder_layer_dims=[300, 400],
            latent_dim=14,
            device=device
        )
        
        # Load state_dict with proper key handling
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Remove 'policy._vae.' prefix from keys if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('policy._vae.'):
                new_key = k[len('policy._vae.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

        # Load the modified state_dict
        self.vae.load_state_dict(new_state_dict, strict=False)
        
        # Move the entire VAE model to the specified device
        self.device = torch.device(device)
        self.vae.to(self.device)
        
        # Debug: Inspect VAE structure, device, and dtype
        logger.debug(f"VAE attributes: {dir(self.vae)}")
        if hasattr(self.vae, '_vae'):
            logger.debug(f"_vae attributes: {dir(self.vae._vae)}")
            # Log device and dtype of a sample parameter
            for name, param in self.vae._vae.named_parameters():
                logger.debug(f"Parameter {name} device: {param.device}, dtype: {param.dtype}")
                break  # Log only the first parameter for brevity
        
        self.vae.eval()
    
    def __call__(self, batch_list):
        
        logger.debug(f"Processing batch_list with {len(batch_list)} samples")
        
        # Input validation
        if not batch_list:
            raise ValueError("Batch list is empty")
        expected_keys = {"actions", "rewards", "dones", "obs"}
        for i, sample in enumerate(batch_list):
            logger.debug(f"Sample {i} keys: {list(sample.keys())}")
            if not all(k in sample for k in expected_keys):
                raise ValueError(f"Sample {i} missing keys: {set(expected_keys) - set(sample.keys())}")
            logger.debug(f"Sample {i} obs keys: {list(sample['obs'].keys())}")

        # Initialize batch dictionary
        batch = {}
        for k in batch_list[0].keys():
            if k == "obs":
                batch["obs"] = {}
                for obs_k in batch_list[0]["obs"].keys():
                    # Convert each observation modality to tensor, ensure float32
                    obs_list = [b["obs"][obs_k] for b in batch_list]
                    if isinstance(obs_list[0], np.ndarray):
                        obs_list = [torch.from_numpy(x).to(self.device, dtype=torch.float) for x in obs_list]
                    batch["obs"][obs_k] = torch.stack(obs_list, dim=0)
            else:
                # Convert non-obs fields (e.g., actions, rewards) to tensor, ensure float32
                item_list = [b[k] for b in batch_list]
                if isinstance(item_list[0], np.ndarray):
                    item_list = [torch.from_numpy(x).to(self.device, dtype=torch.float) for x in item_list]
                batch[k] = torch.stack(item_list, dim=0)
        
        logger.debug(f"Batch obs shapes: { {k: v.shape for k, v in batch['obs'].items()} }")
        
        # Flatten observations to [B * T, ...]
        flat_obs = {}
        for k, v in batch["obs"].items():
            flat_obs[k] = v.reshape(-1, *v.shape[2:])  # e.g., [160, 3] for robot0_eef_pos
            logger.debug(f"Observation {k} device: {flat_obs[k].device}, dtype: {flat_obs[k].dtype}")
        
        # Flatten actions
        flat_actions = batch["actions"].reshape(-1, *batch["actions"].shape[2:])
        logger.debug(f"Flat obs shapes: { {k: v.shape for k, v in flat_obs.items()} }")
        logger.debug(f"Flat actions shape: {flat_actions.shape}, device: {flat_actions.device}, dtype: {flat_actions.dtype}")
        
        # Debug: Print VAEActor.forward_train signature
        logger.debug(f"VAEActor.forward_train signature: {inspect.signature(self.vae.forward_train)}")
        
        try:
            vae_out = self.vae.forward_train(
                actions=flat_actions,
                obs_dict=flat_obs,
            )
            logger.debug(f"VAE output keys: {list(vae_out.keys())}")
        except Exception as e:
            logger.error(f"Error in VAE forward pass:")
            logger.error(f"Input shapes: { {k: v.shape for k, v in flat_obs.items()} }")
            logger.error(f"Action shape: {flat_actions.shape}")
            logger.error(f"Exception: {str(e)}")
            raise
        
        # Extract latent representation (z) from vae_out - use "encoder_z" key
        latent_vae = vae_out["encoder_z"]
        
        # Reshape latent_vae to match batch dimensions [B, T, latent_dim]
        batch_size = batch["actions"].shape[0]
        seq_length = batch["actions"].shape[1]
        latent_vae = latent_vae.reshape(batch_size, seq_length, -1)
        
        # Add latent_vae to batch["obs"]
        batch["obs"]["latent_vae"] = latent_vae
        
        logger.debug(f"Added latent_vae to batch with shape: {latent_vae.shape}, device: {latent_vae.device}, dtype: {latent_vae.dtype}")
        
        return batch