from robomimic.config.base_config import BaseConfig

class DiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy"

    def experiment_config(self):
        """
        Sets default experiment configuration.
        """
        super(DiffusionPolicyConfig, self).experiment_config()
        
        # Experiment settings
        self.experiment.name = "diffusion_policy_example"
        self.experiment.validate = True
        self.experiment.logging.terminal_output_to_txt = True
        self.experiment.logging.log_tb = True
        self.experiment.logging.log_wandb = False
        
        # Saving settings
        self.experiment.save.enabled = True
        self.experiment.save.every_n_epochs = 50
        self.experiment.save.on_best_rollout_success_rate = True
        
        # Rollout settings
        self.experiment.rollout.enabled = True
        self.experiment.rollout.n = 50
        self.experiment.rollout.horizon = 400
        self.experiment.rollout.rate = 50

    def train_config(self):
        """
        Sets default training configuration.
        """
        super(DiffusionPolicyConfig, self).train_config()
        
        # Training settings
        self.train.batch_size = 100
        self.train.num_epochs = 2000
        self.train.seed = 1
        self.train.hdf5_normalize_obs = False
        self.train.hdf5_cache_mode = "all"
        self.train.seq_length = 1
        self.train.frame_stack = 1

    def algo_config(self):
        """
        Sets algorithm-specific configuration.
        """
        super(DiffusionPolicyConfig, self).algo_config()
        
        # Optimization parameters
        self.algo.optim_params.policy = {
            "learning_rate": {
                "initial": 1e-4,
                "decay_factor": 0.0,
                "epoch_schedule": []
            },
            "regularization": {
                "L2": 0.0
            }
        }

        # Diffusion process parameters
        self.algo.diffusion = {
            "steps": 100,
            "noise_schedule": "cosine",
            "inference_steps": 16,
            "horizon": 16,
            "action_horizon": 8,
            "obs_horizon": 2
        }

        # Network architecture
        self.algo.network = {
            "type": "cnn",
            "hidden_dim": 256,
            "max_gradient_norm": 1.0,
            "cnn": {
                "layer_dims": [256, 256, 256],
                "kernel_size": 5
            }
        }

        # Position control
        self.algo.position_control = True

    def observation_config(self):
        """
        Sets default observation configuration.
        """
        super(DiffusionPolicyConfig, self).observation_config()
        
        # Observation modalities
        self.observation.modalities.obs.low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat", 
            "robot0_gripper_qpos",
            "object"
        ]
        self.observation.modalities.obs.rgb = []
        self.observation.modalities.obs.depth = []
        self.observation.modalities.obs.scan = []
        
        # Goal modalities
        self.observation.modalities.goal.low_dim = []
        self.observation.modalities.goal.rgb = []
        self.observation.modalities.goal.depth = []
        self.observation.modalities.goal.scan = []