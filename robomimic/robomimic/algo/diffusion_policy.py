import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import contextlib
from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.utils.obs_utils import ObsNets

# Utility functions to replace TorchUtils
def maybe_no_grad(no_grad=False):
    """Context manager for conditionally disabling gradients"""
    return torch.no_grad() if no_grad else contextlib.nullcontext()

def backprop_for_loss(net, optim, loss, max_grad_norm=None):
    """Simplified backpropagation"""
    optim.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    optim.step()
    return None  # Skip grad norm calculation for simplicity

class SimpleFiLMGenerator(nn.Module):
    """Simplified FiLM generator for conditioning"""
    def __init__(self, input_dim, output_dim, layer_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in layer_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class SimpleVisualEncoder(nn.Module):
    """Simplified visual encoder"""
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape  # (C, H, W)
        
        # Define convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        # Calculate the output size of the conv layers
        def conv_output_size(size, kernel_size=3, stride=2, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        # Compute feature map size after convolutions
        h = conv_output_size(input_shape[1])  # After first conv
        h = conv_output_size(h)  # After second conv
        w = conv_output_size(input_shape[2])  # After first conv
        w = conv_output_size(w)  # After second conv
        conv_output_dim = 64 * h * w
        
        # Add flatten and linear layer
        self.conv.add_module("flatten", nn.Flatten())
        self.conv.add_module("linear", nn.Linear(conv_output_dim, output_shape[0]))
    
    def forward(self, x):
        return self.conv(x)

class SimpleConv1dSequence(nn.Module):
    """Simplified 1D convolutional sequence network"""
    def __init__(self, input_dim, output_dim, horizon, layer_dims, kernel_size=5, obs_dim=128):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in layer_dims:
            conv = nn.Conv1d(prev_dim, dim, kernel_size, padding=kernel_size//2)
            self.layers.append(conv)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(obs_dim, dim))  # Scale
            self.layers.append(nn.Linear(obs_dim, dim))  # Shift
            prev_dim = dim
        self.layers.append(nn.Conv1d(prev_dim, output_dim, kernel_size, padding=kernel_size//2))
    
    def forward(self, x, obs_features):
        h = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv1d):
                h = layer(h)
            elif isinstance(layer, nn.ReLU):
                h = layer(h)
            elif i % 4 == 2:  # Scale
                scale = layer(obs_features).unsqueeze(-1)
                h = h * scale
            elif i % 4 == 3:  # Shift
                shift = layer(obs_features).unsqueeze(-1)
                h = h + shift
        return h

@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    return DiffusionPolicy, {}

class DiffusionPolicy(PolicyAlgo):
    def __init__(self, algo_config, obs_config, **kwargs):
        # Store algo_config as an instance variable
        
        self.algo_config = algo_config
        self.obs_config = obs_config
        # Get device from kwargs or parent class
        self.device = kwargs.get('device', getattr(self, 'device', 'cpu'))  # Default to 'cpu' if not found
        # Get action bounds from kwargs
        self.action_low = kwargs.get('action_low', None)
        self.action_high = kwargs.get('action_high', None)
        # Move to device if not None
        self.action_low = self.action_low.to(self.device) if self.action_low is not None else None
        self.action_high = self.action_high.to(self.device) if self.action_high is not None else None
        # Get global_config, obs_key_shapes, and ac_dim from kwargs
        self.global_config = kwargs.get('global_config', None)
        self.obs_key_shapes = kwargs.get('obs_key_shapes', None)
        self.ac_dim = kwargs.get('ac_dim', None)
        # Initialize parent class with required arguments
        super().__init__(
            algo_config=algo_config,
            obs_config=obs_config,
            global_config=self.global_config,
            obs_key_shapes=self.obs_key_shapes,
            ac_dim=self.ac_dim,
            device=self.device
        )
        # Rest of initialization...
        optimizer_params = {
            "lr": float(self.algo_config.optim_params.policy["learning_rate"]["initial"]),  # Ensure float
            "weight_decay": float(self.algo_config.optim_params.policy["regularization"]["L2"]),  # Ensure float
            "betas": [float(b) for b in self.algo_config.optim_params.policy.get("betas", [0.9, 0.999])]  # Convert to floats
        }
        # Create networks
        self._create_networks()
        # Manually create optimizer
        self.optimizers = {
            "noise_predictor": torch.optim.Adam(
                self.nets["noise_predictor"].parameters(),
                **optimizer_params
            )
        }
        # Rest of initialization
        self._setup_noise_schedule()
        self.num_train_steps = 0
        self.action_counter = 0
        self.current_action_sequence = None
        self.position_control = getattr(self.algo_config, "position_control", True)
    


    # def _setup_optimizers(self):
    #     """Programmatically set up optimizers to avoid config issues"""
    #     self.optimizers = {}
        
    #     optimizer_params = {
    #         "lr": self.algo_config.optim_params.policy["learning_rate"]["initial"],
    #         "weight_decay": self.algo_config.optim_params.policy["regularization"]["L2"],
    #         "betas": [0.9, 0.999]
    #     }
        
    #     self.optimizers["noise_predictor"] = torch.optim.Adam(
    #         self.nets["noise_predictor"].parameters(),
    #         **optimizer_params
    #     )

    def _safe_config_update(self, config, key, value):
        """Helper method to safely update locked configs"""
        was_locked = config.is_locked()
        if was_locked:
            config.unlock()
        
        setattr(config, key, value)
        
        if was_locked:
            config.lock()
            
    def _setup_noise_schedule(self):
        """Cosine noise schedule"""
        self.num_diffusion_steps = self.algo_config.diffusion["steps"]
        self.inference_steps = self.algo_config.diffusion["inference_steps"]
        
        self.betas = self._cosine_beta_schedule(self.num_diffusion_steps).to(self.device)
        self.alphas = (1. - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)
        
        self.ddim_timesteps = torch.linspace(
            0, self.num_diffusion_steps-1, 
            self.inference_steps, dtype=torch.long
        ).to(self.device)
        
    def _cosine_beta_schedule(self, num_steps, s=0.008):
        """Cosine schedule"""
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
        
    def _create_networks(self):
        """Create simplified networks"""
        
        self.nets = nn.ModuleDict()
    
        # Check if RGB modalities are present in observation config
        if len(self.obs_config.modalities.obs.rgb) > 0:
            # Create a visual encoder for each RGB modality
            self.nets["visual_encoder"] = nn.ModuleDict()
            for rgb_key in self.obs_config.modalities.obs.rgb:
                # Get the expected input shape (C, H, W)
                input_shape = self.obs_shapes[rgb_key]
                # Ensure the shape is in CHW format
                if len(input_shape) == 3:  # Already CHW
                    pass
                elif len(input_shape) == 4:  # NHWC format
                    input_shape = (input_shape[3], input_shape[1], input_shape[2])  # Convert to CHW
                else:
                    raise ValueError(f"Unexpected input shape for {rgb_key}: {input_shape}")
                    
                self.nets["visual_encoder"][rgb_key] = SimpleVisualEncoder(
                    input_shape=input_shape,
                    output_shape=(64,)
                )
        
        # Rest of your network creation code...
        if self.algo_config.network["type"] == "cnn":
            self.nets["noise_predictor"] = SimpleConv1dSequence(
                input_dim=self.ac_dim,
                output_dim=self.ac_dim,
                horizon=self.algo_config.diffusion["horizon"],
                layer_dims=[self.algo_config.network["hidden_dim"]] * 2,
                kernel_size=self.algo_config.network["cnn"]["kernel_size"]
            )
        
        self.nets = self.nets.float().to(self.device)


    def train_on_batch(self, batch, epoch, validate=False):
        """Training on a single batch of data"""
        info = OrderedDict()
        with maybe_no_grad(no_grad=validate):
            seq_batch = self._process_batch_into_sequences(batch)
            diffusion_steps = torch.randint(0, self.num_diffusion_steps, (seq_batch["actions"].shape[0],)).to(self.device)
            noisy_actions, noise = self._add_noise(seq_batch["actions"], diffusion_steps)
            obs_features = self._get_obs_features(seq_batch["obs"])
            pred_noise = self.nets["noise_predictor"](noisy_actions.permute(0, 2, 1), obs_features).permute(0, 2, 1)
            loss = F.mse_loss(pred_noise, noise)
            info["diffusion/loss"] = loss.item()
            if not validate:
                backprop_for_loss(
                    net=self.nets["noise_predictor"],
                    optim=self.optimizers["noise_predictor"],
                    loss=loss,
                    max_grad_norm=self.algo_config.network["max_gradient_norm"],
                )
                self.num_train_steps += 1
        # Log actions during rollout (if applicable)
        if not validate and epoch % 50 == 0:
            actions = self._unnormalize_actions(seq_batch["actions"][0])
            info["action_mean"] = actions.mean().item()
            info["action_std"] = actions.std().item()
        return info
        
    def _get_obs_features(self, obs):
        """Extract observation features with visual encoder if needed"""
        if "visual_encoder" in self.nets:
        # Process image observations
            visual_features = []
            for rgb_key in self.obs_config.modalities.obs.rgb:
                # Get the image tensor
                img = obs[rgb_key]
                
                # Ensure correct shape: (B, C, H, W)
                if img.dim() == 4:  # Already (B, C, H, W)
                    pass
                elif img.dim() == 5:  # (B, T, C, H, W)
                    B, T = img.shape[:2]
                    img = img.reshape(-1, *img.shape[2:])  # (B*T, C, H, W)
                else:
                    raise ValueError(f"Unexpected image dimensions for {rgb_key}: {img.shape}")
                
                # Process through visual encoder
                encoded = self.nets["visual_encoder"][rgb_key](img)
                
                # Reshape back if we had temporal dimensions
                if img.dim() == 5:
                    encoded = encoded.reshape(B, T, -1)
                
                visual_features.append(encoded)
                
            obs_features = torch.cat(visual_features, dim=-1)
        else:
            # Use state observations directly
            state_features = []
            for state_key in self.obs_config.modalities.obs.low_dim:
                state_features.append(obs[state_key])
            obs_features = torch.cat(state_features, dim=-1)
            
        return obs_features

        
    def _add_noise(self, actions, t):
        """Add noise to actions according to diffusion step"""
        # Ensure actions are on the same device as other tensors
        actions = actions.to(self.device)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        noise = torch.randn_like(actions)
        noisy_actions = sqrt_alpha * actions + sqrt_one_minus_alpha * noise
        
        return noisy_actions, noise
        
    def _process_batch_into_sequences(self, batch):
        """Process batch into observation and action sequences"""
        actions = batch["actions"].to(self.device)
    
        # Ensure actions have the correct shape: (B, T, D)
        if actions.dim() == 2:  # Shape (B, D)
            actions = actions.unsqueeze(1)  # Add time dimension: (B, 1, D)
        elif actions.dim() == 3:  # Shape (B, T, D)
            pass  # Already in correct shape
        else:
            raise ValueError(f"Unexpected action tensor shape: {actions.shape}")

        seq_batch = {
            "obs": {
                k: v.to(self.device)  # Move to device
                for k, v in batch["obs"].items()
            },
            "actions": actions  # (B, T, D)
        }
        return seq_batch
        
    def get_action(self, obs_dict, goal_dict=None):
        """Generate actions through denoising diffusion process"""
        assert not self.nets.training
  
        # Handle receding horizon control
        if (self.current_action_sequence is None or 
            self.action_counter % self.algo_config.diffusion["action_horizon"] == 0):
            # Generate new action sequence
            self.current_action_sequence = self._generate_action_sequence(
                obs_dict, goal_dict
            )
            self.action_counter = 0
            
        # Get next action
        action = self.current_action_sequence[self.action_counter]
        self.action_counter += 1
        
        # Unnormalize action
        unnormalized_action = self._unnormalize_actions(action.unsqueeze(0)).squeeze(0)
        
        # Debug: Print action info
        print(f"Unnormalized action: {unnormalized_action}, shape: {unnormalized_action.shape}")
        
        # Return action as a tuple to match robomimic's expectations
        action = unnormalized_action.detach()  # Return unnormalized action
        return (action,)
            
    def _generate_action_sequence(self, obs_dict, goal_dict):
        # Get observation features
        obs_features = self._get_obs_features(obs_dict)
        
        # Initialize with noise on the correct device
        noisy_actions = torch.randn(
            (1, self.algo_config.diffusion["horizon"], self.ac_dim),
            device=self.device
        )
        
        # DDIM sampling
        for t in reversed(self.ddim_timesteps):
            # Predict noise
            pred_noise = self.nets["noise_predictor"](
                noisy_actions.permute(0, 2, 1)  # (1,T,D) -> (1,D,T)
            ).permute(0, 2, 1)  # (1,D,T) -> (1,T,D)
            
            # Compute predicted x0
            sqrt_alpha = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
            pred_x0 = (noisy_actions - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
            
            # Clip to action range
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1. - self.alphas_cumprod[t-1]) * pred_noise
            
            # Update noisy actions (DDIM update)
            noisy_actions = (
                torch.sqrt(self.alphas_cumprod[t-1]) * pred_x0 + 
                dir_xt
            )
        
        # Return denoised action sequence
        action_sequence = noisy_actions.squeeze(0)
        
        # Convert to position control if needed
        if self.position_control:
            # Convert from normalized [-1,1] to actual position range
            action_sequence = self._unnormalize_actions(action_sequence)
            
        return action_sequence
        
    def _unnormalize_actions(self, actions):
        """Convert normalized actions back to original range"""
        if self.action_low is None or self.action_high is None:
            # Fallback to default bounds for PickPlace_D0
            action_low = torch.tensor([-0.3, -0.3, 0.0, -1.0, -1.0, -1.0, -1.0], device=self.device)
            action_high = torch.tensor([0.3, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0], device=self.device)
            print("Warning: Using default action bounds for PickPlace_D0 in _unnormalize_actions")
        else:
            action_low = self.action_low
            action_high = self.action_high
        
        actions = 0.5 * (actions + 1.0) * (action_high - action_low) + action_low
        return actions.clamp(action_low, action_high)