from robomimic.config.base_config import BaseConfig


class CSILConfig(BaseConfig):
    ALGO_NAME = "csil"

    def algo_config(self):
        # Behavioral Cloning (BC) pretraining
        self.algo.bc_learning_rate = 1e-3
        self.algo.bc_num_epochs = 100
        self.algo.bc_batch_size = 256

        # Reward shaping parameters
        self.algo.alpha = 1.0  # Temperature for BC policy (pseudo-posterior)
        self.algo.beta = 0.5   # Reduced temperature for fine-tuning
        self.algo.prior_policy = "uniform"  # Policy prior (uniform or learned)

        # SAC fine-tuning
        self.algo.sac_learning_rate = 3e-4
        self.algo.sac_discount = 0.99  # Gamma
        self.algo.sac_target_update_tau = 0.005
        self.algo.sac_batch_size = 256
        self.algo.sac_num_steps = 100000

        # Offline or online mode
        self.algo.offline = True  # Set to False for online fine-tuning