import openai
import os
import json
from run_trained_agent import run_trained_agent, build_args, get_parser  # Assuming get_parser exists


# OpenAI API key'
openai.api_key = "REMOVED_SECRET_KEYproj-lpgTz5rGKAgsX3YVPlKVRkk1R8WDCWTIQzeKmHKN6DSQq11xMN9aUdAJuPZkYgf6lOegnRTNIbT3BlbkFJO50BfmdRDrFtRbamc47PpoLAc3CqdXPLxOQEZA64DzXPNs1eEDvFCzu5xgcU3zvhgBmz2jsLYA"

# Robosuite table bounds
X_MIN, X_MAX = -0.145, 0.145
Y_MIN, Y_MAX = -0.195, 0.195
Z_MIN, Z_MAX = 0.80, 0.90

OBJECTS = ["Milk", "Bread", "Cereal", "Can"]


def generate_prompt(goal_description: str):
    prompt = (
        f"You are a planning assistant for a robot working on a table.\n"
        f"The table bounds are:\n"
        f"- x-axis: {X_MIN} to {X_MAX} meters\n"
        f"- y-axis: {Y_MIN} to {Y_MAX} meters\n"
        f"- z-axis: {Z_MIN} to {Z_MAX} meters (for object placement)\n\n"
        f"Please avoid placing objects directly at the minimum or maximum bounds.\n"
        f"Prefer placing objects comfortably within the table, at least 5 cm away from edges.\n"
        f"Recommended safe ranges:\n"
        f"- x-axis: -0.12 to 0.12 meters\n"
        f"- y-axis: -0.17 to 0.17 meters\n"
        f"- z-axis: 0.81 to 0.89 meters\n\n"
        f"Ensure that objects are at least 6 cm apart from each other in the x-y plane to avoid collisions.\n"
        f"Make the x and y coordinates somewhat random (within the safe range), so that different calls give different values.\n\n"
        f"Available objects: {', '.join(OBJECTS)}\n\n"
        f"Goal: \"{goal_description}\"\n\n"
        f"Decide which objects are needed and assign each one a target 3D position within the safe bounds.\n"
        f"Make sure:\n"
        f"- Objects are well spaced\n"
        f"- Coordinates are within the allowed safe ranges\n"
        f"- Output is valid JSON, no extra text\n\n"
        f"Output format:\n"
        f"{{\n"
        f"  \"active_objects\": [\"object_name1\", \"object_name2\"],\n"
        f"  \"targets\": {{\n"
        f"    \"object_name1\": [x, y, z],\n"
        f"    \"object_name2\": [x, y, z]\n"
        f"  }}\n"
        f"}}"
    )
    return prompt


def get_llm_pickplace_plan(goal_description: str, model="gpt-4"):
    prompt = generate_prompt(goal_description)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You help generate pick-and-place goals for robot simulations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    output = response["choices"][0]["message"]["content"].strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw model output:")
        print(output)
        raise

def run_agent_for_targets(targets, agent_path="/home/anu/mimicgen/training_results/core/pick_place_d0/image/trained_models/core_pick_place_d0_image/multimulti/models/model_epoch_600.pth"):
    for obj_name, goal_coords in targets.items():
        print(f"Running agent for {obj_name} → goal: {goal_coords}")

        # Build command-line-style args
        cmd_args = [
            "--agent", agent_path,
            "--goal", *map(str, goal_coords),
            "--n_rollouts", "1",
            "--video_path", f"/home/anu/robomimic/robomimic/scripts/outputs/output4.mp4",
            "--camera_names", "agentview"
        ]

        # Parse them using the parser
        parser = get_parser()
        parsed_args = parser.parse_args(cmd_args)

        run_trained_agent(parsed_args)

# Main script
if __name__ == "__main__":
    goal = input("Enter your pick-place goal: ")
    llm_plan = get_llm_pickplace_plan(goal)
    print(json.dumps(llm_plan, indent=2))
    print("Performing pick and place....")

    # Run trained agent for each object-target pair
    run_agent_for_targets(llm_plan["targets"])
