# LLM-Guided Pick-and-Place with Robomimic

An LLM interprets a natural-language goal and determines where objects should be placed on a table, while a pre-trained Robomimic policy performs the physical pick-and-place.

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

> 💡 **Important:** Before proceeding, follow the [MimicGen installation instructions](https://mimicgen.github.io/docs/introduction/installation.html) exactly as described in their docs. Then activate the `mimicgen` conda environment.

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
### Training on MultiModal Data 

1. 


### We have implemented Algorithms not already present in Robomimic:
1. [Coherent Soft Imitation Learning](https://github.com/google-deepmind/csil)
You can find it in robomimic/robomimic/models/csil.py 
To run it, 

2. BC with VAE and Multimodal Data

