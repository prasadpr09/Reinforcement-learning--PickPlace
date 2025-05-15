import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from robomimic.algo import Algo, register_algo_factory_func
from copy import deepcopy
import torch.nn as ptnn  # Dummy for compatibility
import torch

# Custom MLP implementation
class MLP(nn.Module):
    hidden_dims: list
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

@register_algo_factory_func("csil")
def algo_config_to_class(algo_config):
    return CSIL, {}

class CSIL(Algo):
    def __init__(self, global_config, algo_config, obs_config, obs_key_shapes, ac_dim, device="cpu"):
        if not hasattr(algo_config, 'optim_params'):
            algo_config.optim_params = {
                "policy": {"learning_rate": algo_config.bc_learning_rate, "regularization": 0.0}
            }

        super().__init__(
            global_config=global_config,
            algo_config=algo_config,
            obs_config=obs_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

        self.observation_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]

        self.bc_policy = MLP(hidden_dims=[256, 256], output_dim=self._get_action_dim())
        self.sac_actor = MLP(hidden_dims=[256, 256], output_dim=self._get_action_dim())
        self.sac_critic = MLP(hidden_dims=[256, 256], output_dim=1)
        self.sac_value = MLP(hidden_dims=[256, 256], output_dim=1)

        self.rng = jax.random.PRNGKey(0)
        obs_dim = self._get_obs_dim()
        critic_input_dim = obs_dim + self._get_action_dim()
        self.rng, bc_key, actor_key, critic_key, value_key = jax.random.split(self.rng, 5)

        self.bc_params = self.bc_policy.init(bc_key, jnp.ones((1, obs_dim)))
        self.sac_actor_params = self.sac_actor.init(actor_key, jnp.ones((1, obs_dim)))
        self.sac_critic_params = self.sac_critic.init(critic_key, jnp.ones((1, critic_input_dim)))
        self.sac_value_params = self.sac_value.init(value_key, jnp.ones((1, obs_dim)))
        self.sac_target_critic_params = deepcopy(self.sac_critic_params)

        self.bc_optimizer = optax.adam(self.algo_config.bc_learning_rate)
        self.sac_actor_optimizer = optax.adam(self.algo_config.sac_learning_rate)
        self.sac_critic_optimizer = optax.adam(self.algo_config.sac_learning_rate)
        self.sac_value_optimizer = optax.adam(self.algo_config.sac_learning_rate)

        self.bc_opt_state = self.bc_optimizer.init(self.bc_params)
        self.sac_actor_opt_state = self.sac_actor_optimizer.init(self.sac_actor_params)
        self.sac_critic_opt_state = self.sac_critic_optimizer.init(self.sac_critic_params)
        self.sac_value_opt_state = self.sac_value_optimizer.init(self.sac_value_params)

        self.training = True
        self.current_epoch = 0
        self.device = "cpu"

    def _create_networks(self):
        self.nets = ptnn.ModuleDict()

    def _create_optimizers(self):
        self.optimizers = {}
        self.lr_schedulers = {}

    def _get_obs_dim(self):
        dims = self.obs_config.dims
        total_dim = 0
        for k in self.observation_keys:
            total_dim += 6 if k == "object" else max(6, dims[k])
        return total_dim

    def _get_action_dim(self):
        return self.ac_dim

    def process_batch_for_training(self, batch):
        obs_dict = {}
        max_dim = 6
        for k in self.observation_keys:
            data = batch["obs"][k]
            data = jnp.asarray(data, dtype=jnp.float32)
            if data.ndim == 3 and data.shape[1] == 1:
                data = data.squeeze(1)
            if k == "object":
                data = data[:, :6]
            if data.shape[1] < max_dim:
                padding = jnp.zeros((data.shape[0], max_dim - data.shape[1]), dtype=jnp.float32)
                data = jnp.concatenate([data, padding], axis=1)
            obs_dict[k] = data

        return {
            "obs": obs_dict,
            "actions": jnp.asarray(batch["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(batch["rewards"], dtype=jnp.float32),
            "dones": jnp.asarray(batch["dones"], dtype=jnp.float32),
            "next_obs": obs_dict
        }

    def train_on_batch(self, batch, epoch, validate=False):
        self.current_epoch = epoch
        batch = self.process_batch_for_training(batch)

        flat_obs = jnp.concatenate([batch["obs"][k] for k in self.observation_keys], axis=-1)
        actions = batch["actions"]
        rewards = batch["rewards"]

        if epoch < self.algo_config.bc_num_epochs:
            def bc_loss_fn(params):
                pred_actions = self.bc_policy.apply(params, flat_obs)
                return jnp.mean(jnp.sum((pred_actions - actions) ** 2, axis=-1))

            loss, grads = jax.value_and_grad(bc_loss_fn)(self.bc_params)
            updates, self.bc_opt_state = self.bc_optimizer.update(grads, self.bc_opt_state)
            self.bc_params = optax.apply_updates(self.bc_params, updates)

            return {"bc_loss": np.array(loss), "epoch": epoch}

        # SAC phase
        def critic_loss_fn(params):
            q_pred = self.sac_critic.apply(params, jnp.concatenate([flat_obs, actions], axis=-1))
            v_target = self.sac_value.apply(self.sac_value_params, flat_obs)
            return jnp.mean((q_pred - rewards - self.algo_config.sac_discount * v_target) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(self.sac_critic_params)
        updates, self.sac_critic_opt_state = self.sac_critic_optimizer.update(critic_grads, self.sac_critic_opt_state)
        self.sac_critic_params = optax.apply_updates(self.sac_critic_params, updates)

        def value_loss_fn(params):
            q_est = self.sac_critic.apply(self.sac_critic_params, jnp.concatenate([flat_obs, self.sac_actor.apply(self.sac_actor_params, flat_obs)], axis=-1))
            v_pred = self.sac_value.apply(params, flat_obs)
            return jnp.mean((v_pred - q_est) ** 2)

        value_loss, value_grads = jax.value_and_grad(value_loss_fn)(self.sac_value_params)
        updates, self.sac_value_opt_state = self.sac_value_optimizer.update(value_grads, self.sac_value_opt_state)
        self.sac_value_params = optax.apply_updates(self.sac_value_params, updates)

        def actor_loss_fn(params):
            actions_pred = self.sac_actor.apply(params, flat_obs)
            q_val = self.sac_critic.apply(self.sac_critic_params, jnp.concatenate([flat_obs, actions_pred], axis=-1))
            return -jnp.mean(q_val)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(self.sac_actor_params)
        updates, self.sac_actor_opt_state = self.sac_actor_optimizer.update(actor_grads, self.sac_actor_opt_state)
        self.sac_actor_params = optax.apply_updates(self.sac_actor_params, updates)

        return {
            "sac_actor_loss": np.array(actor_loss),
            "sac_critic_loss": np.array(critic_loss),
            "sac_value_loss": np.array(value_loss),
            "epoch": epoch
        }

    def get_action(self, obs_dict, goal_dict=None, deterministic=False):
        obs_data = []
        for k in self.observation_keys:
            data = obs_dict[k]
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            if isinstance(data, np.ndarray):
                data = jnp.array(data)
            data = jnp.ravel(data)
            if k == "object":
                data = data[:6]
            if data.shape[0] < 6:
                padding = jnp.zeros(6 - data.shape[0])
                data = jnp.concatenate([data, padding])
            obs_data.append(data)

        obs = jnp.concatenate(obs_data).reshape(1, -1)

        if self.training and self.current_epoch < self.algo_config.bc_num_epochs:
            action = self.bc_policy.apply(self.bc_params, obs)
        else:
            action = self.sac_actor.apply(self.sac_actor_params, obs)

        return np.asarray(action).astype(np.float32)

    def set_train(self):
        self.training = True

    def set_eval(self):
        self.training = False

    def serialize(self):
        return {
            "bc_params": self.bc_params,
            "sac_actor_params": self.sac_actor_params,
            "sac_critic_params": self.sac_critic_params,
            "sac_value_params": self.sac_value_params,
            "sac_target_critic_params": self.sac_target_critic_params,
            "current_epoch": self.current_epoch
        }

    def deserialize(self, model_dict):
        self.bc_params = model_dict["bc_params"]
        self.sac_actor_params = model_dict["sac_actor_params"]
        self.sac_critic_params = model_dict["sac_critic_params"]
        self.sac_value_params = model_dict["sac_value_params"]
        self.sac_target_critic_params = model_dict["sac_target_critic_params"]
        self.current_epoch = model_dict["current_epoch"]