# FlowCycle
The Pytorch Implementation of ''FlowCycle: Pursuing Cycle-Consistent Flows for Text-based Editing'' 


# Device Requirments

It requires GPU with at least 16GB memory.

# Environment installation

```
conda env create -f environment.yaml
```
# Activate the Environment

```
conda activate flowcycle
```

# Access Permission of SD-3-medium on Huggingface
```
pip install --upgrade huggingface_hub
hf auth login
```
Enter your access token.

# Edit the Example Image with SD-3-medium
```
python demo.py
```
#### Source Prompt
A blue-gray Audi car parked in a grassy area. A white dog sitting on the grass, next to the car. A cat laying on the hood of the car.

#### Target Prompt
A blue-gray Audi car parked in a grassy area. A Husky dog sitting on the grass, next to the car. A tiger cab laying on the hood of the car.
