# LLM-Guided Pick-and-Place with Robomimic

An LLM interprets a natural-language goal and determines where objects should be placed on a table, while a pre-trained Robomimic policy performs the physical pick-and-place.

## Here is a Youtube video link to see it working.
---

## Background

Robotic pick-and-place is a fundamental task in household, warehouse, and industrial settings. This project explores **goal-conditioned planning** using natural language and **learned policies** trained via behavioral cloning. The high-level goal is:

> Use an LLM to generate spatial targets based on a human-written goal, and run a trained policy to place the corresponding objects at those locations.

This approach combines:
- **Language understanding** via GPT-4 or GPT-3.5
- **Policy learning** via Robomimic
- **Simulation** on Robosuite environments (e.g., pick-place on a tabletop)

---

## Setup Instructions

> **Important:** Before proceeding, follow the [MimicGen installation instructions](https://mimicgen.github.io/docs/introduction/installation.html) exactly as described in their docs. Then activate the `mimicgen` conda environment.

### 1. Clone This Repository

```bash
git clone https://github.com/prasadpr09/Reinforcement-learning--PickPlace.git
cd Reinforcement-learning--PickPlace
```
### 2. Set Up Environment and Install Dependencies
```
conda activate mimicgen

# Install Robomimic dependencies- should've already done it if you followed mimicgen's docummentation
pip install -r robomimic/requirements.txt

# Install OpenAI API client
pip install openai
```
### 3. Download or Train a Robomimic Policy

You can use your own pre-trained model from Robomimic or use our trained model. The code expects a .pth checkpoint trained on the PickPlace environment.

Update this path in main.py:

```
agent_path = "/your/path"
```

### 4. Set Up OpenAI API/.-* 

You must export your OpenAI API key:
```
export OPENAI_API_KEY="your-key-here"

```

### 5. Run the LLM + Robomimic Pipeline
```
python main.py
```
When prompted try entering something like:
Pick and place the cereal and milk


### To Train:
1. Edit the json file e.g, bc.json
2. Run this command:
```
python /path/to/robomimic/scripts/train.py --config /path/to/mimicgen/exps/paper/core/coffee_d0/image/bc_rnn.json
```

### To load checkpoints:
Run:
```
python run_trained_agent.py   --agent ~/mimicgen/training_results/core/pick_place_d0/image/trained_models/core_pick_place_d0_image/multimulti/models/model_epoch_600/data.pkl   --n_rollouts 50   --horizon 400   --seed 0   --video_path /home/output.mp4   --camera_names agentview robot0_eye_in_hand

```
## Training on MultiModal Data 

1. We used Behavior Cloning algorithm with increased modality as mentioned below (these were our settings in the JSON file):

```
"low_dim": [
    "robot0_eef_pos",          
    "robot0_eef_quat",         
    "robot0_gripper_qpos",     
    "robot0_joint_pos",
    "robot0_joint_vel",
    ]
    
"rgb": ["agentview_image",
        "robot0_eye_in_hand_image"
    ]
```

### Results:

Training loss graph:

![Pick and Place](output_images/bc_increased_modalities/Screenshot%20from%202025-05-15%2001-49-36.png)

Pick and Place Success Rollout Graph:

![Rollout](output_images/bc_increased_modalities/Screenshot%20from%202025-05-15%2001-48-42.png)


## Benchmark: We compare all our results with Behavior Cloning + Reduced Modalities as it outperformed other algorithms till now:

```
"low_dim": [
    "robot0_eef_pos",          
    "robot0_eef_quat",         
    "robot0_gripper_qpos",     
    ]
    
"rgb": ["agentview_image",
        "robot0_eye_in_hand_image"
    ]
```
Training loss:

![Training loss](output_images/bc_reduced%20modalities/Screenshot%20from%202025-05-15%2001-50-40.png)

Pick and Place Success Rollout Graph:

![Success_rollouts](output_images/bc_reduced%20modalities/Screenshot%20from%202025-05-15%2001-50-20.png)



## We have implemented Algorithms not already present in Robomimic:

## 1. Behavior Cloning Baseline (with Increased Modalities) 
Custom Implementation of Behaviour Cloning Algorithm with both Proprioceptive (Added new modalities for our version of implementation) and Visual data for baseline performance.
```
    "low_dim": [
        "robot0_eef_pos",          
        "robot0_eef_quat",         
        "robot0_gripper_qpos",     
        "robot0_joint_pos",
        "robot0_joint_vel",
        "object"       
    ],
    "rgb": [
        "agentview_image",
        "robot0_eye_in_hand_image"
    ],
```


## 2. Behavior Cloning with Proprioception Only (Reduced Modalities)
Measuring the effect of removing high-dimensional vision input, using only robot state information (proprioception) and ground truth environment data building on our custom implementation.
```
    "low_dim": [
        "robot0_eef_pos",          
        "robot0_eef_quat",         
        "robot0_gripper_qpos",     
        "robot0_joint_pos",
        "robot0_joint_vel",
        "object"       
    ],
    "rgb": [   # removed this modality 
    ],
```
To Train:
```
python train_vae.py --config /home/anu/robomimic/robomimic/exps/templates/bc_noimgs_2.json --dataset /home/anu/mimicgen/datasets/core/pick_place_d0.hdf5

```
Observations:

Training loss graph:  
![Training loss](output_images/bc_with_proprioception_only/Screenshot%20from%202025-05-15%2014-35-55.png)

Pick and Place Success Rollout Graph:  
![Success_rollouts](output_images/bc_with_proprioception_only/Screenshot%20from%202025-05-15%2014-34-57.png)



## 3. Behavior Cloning with VAE Enabled Training on Reduced Modalities
Uses a VAE enabled integration on our custom Implementation for BC training to extract latent trajectory features and store them as weights, on our reduced set.

```
    "low_dim": [
        "robot0_eef_pos",          
        "robot0_eef_quat",         
        "robot0_gripper_qpos",     
        "robot0_joint_pos",
        "robot0_joint_vel",
        "object"       
    ],
    "rgb": [ # removed this modality 
    ],
```

Observations:

Training loss graph:
Pick and Place Success Rollout Graph:


## 4. Behavior Cloning with VAE Latent on Reduced Modalities
Utilizes the pre-trained latent representations obtained from a VAE to enhance proprioceptive-only input, aiming to recover performance lost due to the absence of raw visual data. 

```
    "low_dim": [
        "robot0_eef_pos",          
        "robot0_eef_quat",         
        "robot0_gripper_qpos",     
        "robot0_joint_pos",
        "robot0_joint_vel",
        "object"
        "latent_vae"       
    ],
    "rgb": [ # removed this modality 
    ],
```


Observations:

Training loss graph:
Pick and Place Success Rollout Graph:


4. [Coherent Soft Imitation Learning](https://github.com/google-deepmind/csil)
You can find it in [robomimic/robomimic/models/csil.py](robomimic/robomimic/algo/csil.py)

Training loss graph:



5. [Diffusion Policy:](https://github.com/real-stanford/diffusion_policy)
You can find it in [robomimic/robomimic/models/diffusion_policy.py](robomimic/robomimic/algo/diffusion_policy_config.py)

Training loss graph:

