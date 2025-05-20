# heterogeneous_swarm

Repository for [Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](https://arxiv.org/abs/2502.04510).

## Quick Start

#### Initialization

Create a conda environment for Heterogeneous Swarms.
```
conda env create -f h_swarm.yml
conda activate h_swarm
```

Log into huggingface (for model access).
```
huggingface-cli login
```

Download initial experts.
```
cd initial_experts
python initial_experts.py
cd ..
```

#### Execute your first Heterogeneous Swarms search.

Let's run Heterogeneous Swarms on the [NLGraph](https://arxiv.org/abs/2305.10037) dataset focusing on LLM graph reasoning. `nlgraph.sh` provides the starter script for this.

Before running, how many GPUs do you have (and what are the GPU ids?). Change `-g` in line 23 of `search_nlgraph.sh`: by default five GPUs with ids `0,1,2,3,4`, but you can change to `0`, `0,1`, `0,2,4,5,6` or any combination you'd like. It is recommended to have ~40GB memory per GPU and the code will auto double batch size for ~80GB GPUs.

Run it!
```
bash nlgraph.sh
```

You might be prompted to log into WandB and it is highly recommended. There will be a directory in `search/` that starts with `nlgraph_...`, all the logs, models, and results will be stored there. You can check out `search/nlgraph_.../log.txt` to see current progress. Check out the logs on the wandb website too. Yes, it might be slow.

#### Other Objectives

Change `(-e, -d)` pairs to `(multiple_choice, abkg)`, `(multiple_choice, com2)`, `(exact_match, gaia)`, `(exact_match, gsm8k)`, `(multiple_choice, knowledge_crosswords)`, `(gemini_match, ltp)` (need to configure gcloud api), `(multiple_choice, mmlu_pro)`, `(abstainqa, mmlu)`, `(exact_match, nlgraph)`, `(multiple_choice, normad)`, `(gemini_match, qasper)`, `(multiple_choice, wow)`.

Adding your data: just follow the format of existing data files in `data/eval/` and then change `-n` and use a respective `-e`.

Adding your models: look at `initial_experts/`: it is essentially a folder of 10 LoRA adapters of Gemma-7B. Create your own folder of models with the same architecture like `initial_experts` and change add the argument `-i <path_to_folder>` (see search.py). If the models are adapters/only have 1 shard, you don't need to change anything else. If they are full models/multiple shards, change `--fast_merge` from `1` to `0`.

Adding your evaluation: well that will be a bit more workload. Go to `evaluate.py`: there's a `evaluate()`, essentially you specify path to a pool of models and a graph and these functions give you a scalar score. In both of them there are `if eval_type == ...` clauses: name your evaluation type, open a new if clause `if eval_type == <your_eval>`, implement how you load the data and get a scalar score. `batch_generate()` and `batch_generate_chat_template()` provide two helper functions to generate text from a model and please use them for model output. You could refer to `eval_type == multiple_choice` or `eval_type == exact_match` as examples.

## Changing Hyperparameters and Settings

Do `python search.py -h` to see a list of all possible hyperparameters and settings. Additionally look at the comments for hyperparameters in `search.py`. We already included the default settings in the four `nlgraph.sh` starter scripts, but feel free to play around different settings.

## Citation

If Heterogeneous Swarms is helpful to you:

```
@article{feng2025heterogeneous,
  title={Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems},
  author={Feng, Shangbin and Wang, Zifeng and Goyal, Palash and Wang, Yike and Shi, Weijia and Xia, Huang and Palangi, Hamid and Zettlemoyer, Luke and Tsvetkov, Yulia and Lee, Chen-Yu and others},
  journal={arXiv preprint arXiv:2502.04510},
  year={2025}
}
```