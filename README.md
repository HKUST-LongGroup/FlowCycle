# FlowCycle

**FlowCycle is a target-aware image editing method with high source consistency.**

![](teaser.png)

> [**FlowCycle: Pursuing Cycle-Consistent Flows for Text-based Editing**](https://arxiv.org/pdf/2508.11330)
> 
> Yanghao Wang, Zhen Wang, Long Chen  

## Device Requirements

It requires a GPU with at least 16GB of memory.

## Environment installation

```
conda env create -f environment.yaml
```
## Activate the Environment

```
conda activate flowcycle
```

## Access Permission of SD-3-medium on Huggingface
```
pip install --upgrade huggingface_hub
hf auth login
```
Enter your access token.

## Edit the Example Image with SD-3-medium
```
python demo.py
```
## Source Prompt
```
A blue-gray Audi car parked in a grassy area. A white dog sitting on the grass, next to the car. A cat laying on the hood of the car.
```

## Target Prompt
```
A blue-gray Audi car parked in a grassy area. A Husky dog sitting on the grass, next to the car. A tiger cab laying on the hood of the car.
```

# Citing NoOp

If you use FlowCycle in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX

```
