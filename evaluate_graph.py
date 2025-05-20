import os
import json
import math
import time
import torch
import random
import datetime
import numpy as np
import graph_decode
from tqdm import tqdm
from datasets import load_dataset
from tenacity import retry, wait_random_exponential, stop_after_attempt
from googleapiclient import discovery
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
from evaluate import evaluate

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def evaluate_best_graph(search_pass_name, dataset, eval_type, gpus, split="dev"):

    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_path_list = []
    for i in range(len(os.listdir(os.path.join("search", search_pass_name, "all_time_best", "models")))):
        model_path_list.append(os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))

    assignment = json.load(open(os.path.join("search", search_pass_name, "all_time_best", "assignment.json")))

    if split == "dev":
        best_dev = evaluate(os.path.join("search", search_pass_name, "all_time_best", "graph"), assignment, model_path_list, eval_type, dataset, "dev", gpus[0])
        return best_dev
    else:
        best_test = evaluate(os.path.join("search", search_pass_name, "all_time_best", "graph"), assignment, model_path_list, eval_type, dataset, "test", gpus[0])
        return best_test

# search_pass_name = "NAME_OF_YOUR_SEARCH_PASS"
# dataset = "nlgraph"
# eval_type = "exact_match"
# gpus = [0,1,2,3,4]

# best_dev = evaluate_best_graph(search_pass_name, dataset, eval_type, gpus, "dev")
# print("All time best dev: " + str(best_dev))