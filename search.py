import os
import math
import json
import torch
import shutil
import socket
import argparse
import random
import logging
import datetime
import wandb
import numpy as np
from merge import lora_merge
from evaluate import evaluate
from multiprocessing import Pool
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from evaluate_graph import evaluate_best_graph

def log_with_flush(message, level=logging.INFO):
  logging.log(level, message)
  logging.getLogger().handlers[0].flush()

def curret_time_string():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def assign_gpu(num_gpus, process_idx, total_processes):
    process_per_gpu = math.ceil(total_processes / num_gpus)
    gpu_idx = math.floor(process_idx / process_per_gpu)
    return gpu_idx

def node_evaluate(graph_path, model_path_list, eval_type, dataset, dev_test_split, gpus, trial_num=10):
    adjacency = np.load(os.path.join(graph_path, "adjacency.npy"))
    assert len(adjacency) == len(model_path_list)
    assignments = []
    for i in range(trial_num):
        this_assignment = []
        for j in range(len(adjacency)):
            this_assignment.append(random.randint(0, len(model_path_list)-1))
        assignments.append(this_assignment)
    
    graph_evaluate_args = []
    for i in range(len(assignments)):
        graph_evaluate_args.append((graph_path, assignments[i], model_path_list, eval_type, dataset, dev_test_split, gpus[assign_gpu(len(gpus), i, len(assignments))]))
    
    pool = Pool(processes=len(gpus))
    graph_scores = pool.starmap(evaluate, graph_evaluate_args, chunksize=math.ceil(len(assignments)/len(gpus)))
    pool.close()
    pool.join()

    # figure out best assignments
    best_assignment_idx = graph_scores.index(max(graph_scores))
    best_assignment = assignments[best_assignment_idx]

    # node_score is the weighted average of graph scores
    node_frequency = [0] * len(model_path_list)
    node_score = [0] * len(model_path_list)

    for i in range(len(graph_scores)):
        for j in range(len(assignments[i])):
            node_frequency[assignments[i][j]] += 1
            node_score[assignments[i][j]] += graph_scores[i]
    
    for i in range(len(node_score)):
        node_score[i] /= node_frequency[i]
    
    return node_score, {"best_score": max(graph_scores), "best_assignment": best_assignment}

def initialize_search_records(search_pass_name, particle_paths, graph_num, eval_type, dataset, gpus, base_model = "google/gemma-7b-it", fast_merge = True, starting_velocity_mode = "random"):

    # os.mkdir(os.path.join("search", search_pass_name))
    os.mkdir(os.path.join("search", search_pass_name, "models"))
    os.mkdir(os.path.join("search", search_pass_name, "graphs"))

    dev_test_split = "dev" # default dev split

    # initialize the model directory
    for i in range(len(particle_paths)):
        os.mkdir(os.path.join("search", search_pass_name, "models", "model_"+str(i)))
        for checkpoint_type in ["personal_best", "now", "velocity"]:
            os.mkdir(os.path.join("search", search_pass_name, "models", "model_"+str(i), checkpoint_type))
    os.mkdir(os.path.join("search", search_pass_name, "models", "global_best"))
    os.mkdir(os.path.join("search", search_pass_name, "models", "global_worst"))

    # initialize the graph directory
    for i in range(graph_num):
        os.mkdir(os.path.join("search", search_pass_name, "graphs", "graph_"+str(i)))
        for checkpoint_type in ["personal_best", "now", "velocity"]:
            os.mkdir(os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), checkpoint_type))
    os.mkdir(os.path.join("search", search_pass_name, "graphs", "global_best"))
    os.mkdir(os.path.join("search", search_pass_name, "graphs", "global_worst"))

    # initialize model directory content

    # moving models to now and personal_best
    for i in range(len(particle_paths)):
        shutil.copytree(particle_paths[i], os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"), dirs_exist_ok=True)
        shutil.copytree(particle_paths[i], os.path.join("search", search_pass_name, "models", "model_"+str(i), "personal_best"), dirs_exist_ok=True)
    
    # initializing model velocity
    if starting_velocity_mode == "zero":
        raise NotImplementedError("Zero velocity initialization is not implemented yet.")
    elif starting_velocity_mode == "best":
        raise NotImplementedError("Best velocity initialization is not implemented yet.")
    elif starting_velocity_mode == "random":
        merge_args = []
        for i in range(len(particle_paths)):
            secret_lover_idx = random.randint(0, len(particle_paths)-1)
            while secret_lover_idx == i:
                secret_lover_idx = random.randint(0, len(particle_paths)-1)
            merge_args.append(([-1, 1], [os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"), os.path.join("search", search_pass_name, "models", "model_"+str(secret_lover_idx), "now")], os.path.join("search", search_pass_name, "models", "model_"+str(i), "velocity"), gpus[assign_gpu(len(gpus), i, len(particle_paths))], fast_merge))

        pool = Pool(processes=len(gpus))
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()
    
    log_with_flush("Model initialization done.")
    
    # initializing graph directory content
    for i in range(graph_num):
        random_adjacency_matrix = np.random.rand(len(particle_paths), len(particle_paths))
        np.fill_diagonal(random_adjacency_matrix, 0)
        np.save(os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), "now", "adjacency.npy"), random_adjacency_matrix)
        np.save(os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), "personal_best", "adjacency.npy"), random_adjacency_matrix)
    
    # initializing graph velocity
    if starting_velocity_mode == "zero":
        raise NotImplementedError("Zero velocity initialization is not implemented yet.")
    elif starting_velocity_mode == "best":
        raise NotImplementedError("Best velocity initialization is not implemented yet.")
    elif starting_velocity_mode == "random":
        for i in range(graph_num):
            this_adjacency = np.load(os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), "now", "adjacency.npy"))
            secret_lover_idx = random.randint(0, graph_num-1)
            while secret_lover_idx == i:
                secret_lover_idx = random.randint(0, graph_num-1)
            secret_lover_adjacency = np.load(os.path.join("search", search_pass_name, "graphs", "graph_"+str(secret_lover_idx), "now", "adjacency.npy"))
            velocity_adjacency = (secret_lover_adjacency - this_adjacency)
            np.save(os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), "velocity", "adjacency.npy"), velocity_adjacency)
    
    log_with_flush("Graph initialization done.")

    # evaluating the utility of starting graphs
    graph_evaluate_args = []
    assignment = [i for i in range(len(particle_paths))] # default assignment
    model_path_list = []
    for i in range(len(particle_paths)):
        model_path_list.append(os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"))
    for i in range(graph_num):
        graph_evaluate_args.append((os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), "now"), assignment, model_path_list, eval_type, dataset, dev_test_split, gpus[assign_gpu(len(gpus), i, graph_num)]))

    pool = Pool(processes=len(gpus))
    graph_scores = pool.starmap(evaluate, graph_evaluate_args, chunksize=math.ceil(graph_num/len(gpus)))
    pool.close()
    pool.join()

    # graph_scores = [np.random.rand() for i in range(graph_num)]

    # updating the global best and worst graphs
    global_best_score = -1
    global_best_idx = -1
    global_worst_score = 101
    global_worst_idx = -1
    for i in range(graph_num):
        if graph_scores[i] > global_best_score:
            global_best_score = graph_scores[i]
            global_best_idx = i
        if graph_scores[i] < global_worst_score:
            global_worst_score = graph_scores[i]
            global_worst_idx = i
    
    shutil.copytree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(global_best_idx), "now"), os.path.join("search", search_pass_name, "graphs", "global_best"), dirs_exist_ok=True)
    shutil.copytree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(global_worst_idx), "now"), os.path.join("search", search_pass_name, "graphs", "global_worst"), dirs_exist_ok=True)

    best_found_initial_graph_index = global_best_idx

    log_with_flush("Graph evaluation done.")

    utility_scratchpad = {}

    # all time best dev, graph side only in initialization
    utility_scratchpad["all_time_best_dev"] = max(graph_scores)
    # save a copy of its current graph, models, assignment to all_time_best/
    os.mkdir(os.path.join("search", search_pass_name, "all_time_best"))
    # os.mkdir(os.path.join("search", search_pass_name, "all_time_best", "graph"))
    shutil.copytree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(best_found_initial_graph_index), "now"), os.path.join("search", search_pass_name, "all_time_best", "graph"))
    os.mkdir(os.path.join("search", search_pass_name, "all_time_best", "models"))
    for i in range(len(particle_paths)):
        # os.mkdir(os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))
        shutil.copytree(os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"), os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))
    assignment = [i for i in range(len(particle_paths))]
    with open(os.path.join("search", search_pass_name, "all_time_best", "assignment.json"), "w") as f:
        json.dump(assignment, f, indent=4)

    # evaluating the utility of starting models
    model_evaluate_args = []
    model_path_list = []
    for i in range(len(particle_paths)):
        model_path_list.append(os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"))

    model_scores, _ = node_evaluate(os.path.join("search", search_pass_name, "graphs", "graph_"+str(global_best_idx), "now"), model_path_list, eval_type, dataset, dev_test_split, gpus)

    # model_scores = [np.random.rand() for i in range(len(particle_paths))]

    # updating the global best and worst models
    global_best_score = -1
    global_best_idx = -1
    global_worst_score = 101
    global_worst_idx = -1

    for i in range(len(particle_paths)):
        if model_scores[i] > global_best_score:
            global_best_score = model_scores[i]
            global_best_idx = i
        if model_scores[i] < global_worst_score:
            global_worst_score = model_scores[i]
            global_worst_idx = i
    
    shutil.copytree(os.path.join("search", search_pass_name, "models", "model_"+str(global_best_idx), "now"), os.path.join("search", search_pass_name, "models", "global_best"), dirs_exist_ok=True)
    shutil.copytree(os.path.join("search", search_pass_name, "models", "model_"+str(global_worst_idx), "now"), os.path.join("search", search_pass_name, "models", "global_worst"), dirs_exist_ok=True)

    log_with_flush("Model evaluation done.")

    # logging
    utility_scratchpad["model_g"] = max(model_scores)
    utility_scratchpad["model_g_worst"] = min(model_scores)
    utility_scratchpad["model_g_history"] = [max(model_scores)]
    for i in range(len(model_scores)):
        utility_scratchpad["model_"+str(i)+"_now"] = model_scores[i]
        utility_scratchpad["model_"+str(i)+"_best"] = model_scores[i]
        utility_scratchpad["model_"+str(i)+"_history"] = [model_scores[i]]
    utility_scratchpad["graph_g"] = max(graph_scores)
    utility_scratchpad["graph_g_worst"] = min(graph_scores)
    utility_scratchpad["graph_g_history"] = [max(graph_scores)]
    for i in range(len(graph_scores)):
        utility_scratchpad["graph_"+str(i)+"_now"] = graph_scores[i]
        utility_scratchpad["graph_"+str(i)+"_best"] = graph_scores[i]
        utility_scratchpad["graph_"+str(i)+"_history"] = [graph_scores[i]]

    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)
    
    wandb_log = {
        "model_g": max(model_scores),
        "model_g_worst": min(model_scores),
        "graph_g": max(graph_scores),
        "graph_g_worst": min(graph_scores)
    }
    for i in range(len(model_scores)):
        wandb_log["model_"+str(i)+"_now"] = model_scores[i]
    for i in range(len(graph_scores)):
        wandb_log["graph_"+str(i)+"_now"] = graph_scores[i]
    
    wandb.log(wandb_log)

    log_with_flush("Initialization done.")

def model_update(i, gpu_id, search_pass_name, weight_randomness, inertia, cognitive_coeff, social_coeff, repel_coeff, fast_merge, step_length, repel_term, restart_flag):
    particle_path = os.path.join("search", search_pass_name, "models", "model_"+str(i))
    now_path = os.path.join(particle_path, "now")
    best_path = os.path.join(particle_path, "personal_best")
    velocity_path = os.path.join(particle_path, "velocity")

    if restart_flag:
        shutil.copytree(best_path, now_path, dirs_exist_ok=True)
        lora_merge([0], [now_path], velocity_path, gpu_id, fast_merge)
    
    # weight randomness
    if weight_randomness == 1:
        r_w = random.uniform(0, 1)
        r_p = random.uniform(0, 1)
        r_s = random.uniform(0, 1)
        r_b = random.uniform(0, 1) # b for bad, repel term weight
    else:
        r_w = 1
        r_p = 1
        r_s = 1
        r_b = 1
    
    # weight normalize (probably we don't need to? bc the weighted average of three E[X]=0 stuff will again be E[X]=0 regardless of weights)
    self_weight = r_w * inertia
    cognitive_weight = r_p * cognitive_coeff
    social_weight = r_s * social_coeff
    repel_weight = r_b * repel_coeff if repel_term else 0
    weight_sum = self_weight + cognitive_weight + social_weight + repel_weight

    # normalize weights
    self_weight = self_weight / weight_sum
    cognitive_weight = cognitive_weight / weight_sum
    social_weight = social_weight / weight_sum
    repel_weight = repel_weight / weight_sum

     # p_i-x_i task vector
    lora_merge(
        weights = [1, -1],
        lora_name_list = [os.path.join("search", search_pass_name, "models", "model_"+str(i), "personal_best"), os.path.join("search", search_pass_name, "models", "model_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "models", "model_"+str(i), "p_x"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # g-x_i task vector
    lora_merge(
        weights = [1, -1],
        lora_name_list = [os.path.join("search", search_pass_name, "models", "global_best"), os.path.join("search", search_pass_name, "models", "model_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "models", "model_"+str(i), "g_x"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # x_i - w task vector
    lora_merge(
        weights = [-1, 1],
        lora_name_list = [os.path.join("search", search_pass_name, "models", "global_worst"), os.path.join("search", search_pass_name, "models", "model_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "models", "model_"+str(i), "x_w"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # update velocity
    lora_merge(
        weights = [self_weight, cognitive_weight, social_weight, repel_weight],
        lora_name_list = [os.path.join("search", search_pass_name, "models", "model_"+str(i), "velocity"),
                            os.path.join("search", search_pass_name, "models", "model_"+str(i), "p_x"),
                            os.path.join("search", search_pass_name, "models", "model_"+str(i), "g_x"),
                            os.path.join("search", search_pass_name, "models", "model_"+str(i), "x_w")],
        output_name = os.path.join("search", search_pass_name, "models", "model_"+str(i), "velocity"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # update current position
    lora_merge(
        weights = [1, step_length],
        lora_name_list = [os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"), os.path.join("search", search_pass_name, "models", "model_"+str(i), "velocity")],
        output_name = os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

def graph_update(i, gpu_id, search_pass_name, weight_randomness, inertia, cognitive_coeff, social_coeff, repel_coeff, fast_merge, step_length, repel_term, restart_flag):
    graph_path = os.path.join("search", search_pass_name, "graphs", "graph_"+str(i))
    now_path = os.path.join(graph_path, "now")
    best_path = os.path.join(graph_path, "personal_best")
    velocity_path = os.path.join(graph_path, "velocity")
    graph_size = len(np.load(os.path.join(now_path, "adjacency.npy")))

    if restart_flag:
        shutil.copytree(best_path, now_path, dirs_exist_ok=True)
        np.save(os.path.join(velocity_path, "adjacency.npy"), np.zeros((graph_size, graph_size)))
    
    # weight randomness
    if weight_randomness == 1:
        r_w = random.uniform(0, 1)
        r_p = random.uniform(0, 1)
        r_s = random.uniform(0, 1)
        r_b = random.uniform(0, 1) # b for bad, repel term weight
    else:
        r_w = 1
        r_p = 1
        r_s = 1
        r_b = 1
    
    # weight normalize (probably we don't need to? bc the weighted average of three E[X]=0 stuff will again be E[X]=0 regardless of weights)
    self_weight = r_w * inertia
    cognitive_weight = r_p * cognitive_coeff
    social_weight = r_s * social_coeff
    repel_weight = r_b * repel_coeff if repel_term else 0
    weight_sum = self_weight + cognitive_weight + social_weight + repel_weight

    # normalize weights
    self_weight = self_weight / weight_sum
    cognitive_weight = cognitive_weight / weight_sum
    social_weight = social_weight / weight_sum
    repel_weight = repel_weight / weight_sum

    # p_i-x_i task vector
    p_x = np.load(os.path.join(best_path, "adjacency.npy")) - np.load(os.path.join(now_path, "adjacency.npy"))
    # g-x_i task vector
    g_x = np.load(os.path.join("search", search_pass_name, "graphs", "global_best", "adjacency.npy")) - np.load(os.path.join(now_path, "adjacency.npy"))
    # x_i - w task vector
    x_w = np.load(os.path.join(now_path, "adjacency.npy")) - np.load(os.path.join("search", search_pass_name, "graphs", "global_worst", "adjacency.npy"))
    # update velocity
    old_velocity = np.load(os.path.join(velocity_path, "adjacency.npy"))
    new_velocity = self_weight * old_velocity + cognitive_weight * p_x + social_weight * g_x + repel_weight * x_w
    np.save(os.path.join(velocity_path, "adjacency.npy"), new_velocity)
    # update current position
    old_adjacency = np.load(os.path.join(now_path, "adjacency.npy"))
    new_adjacency = old_adjacency + step_length * new_velocity
    np.save(os.path.join(now_path, "adjacency.npy"), new_adjacency)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of this model swarms search, also save directory in search/")
    argParser.add_argument("-e", "--eval_type", help="evaluation types") # multiple_choice
    argParser.add_argument("-d", "--dataset", help="dataset as the search objective/evaluation") # mmlu
    argParser.add_argument("-g", "--gpus", help="available gpu ids in a string") # 0,1,2,3,4,10,11
    argParser.add_argument("--graph_num", default=10, help="number of graphs in the search")
    argParser.add_argument("--num_cpu_when_merging", default=1, help="number of cpu cores when merging")
    argParser.add_argument("--inertia", default = 0.4, help="inertia of particle weight update")
    argParser.add_argument("--cognitive_coeff", default = 0.3, help="cognitive coefficient of particle weight update")
    argParser.add_argument("--social_coeff", default = 0.3, help="social coefficient of particle weight update")
    argParser.add_argument("--repel_coeff", default = 0.3, help="repel coefficient of particle weight update")
    argParser.add_argument("--step_length", default = 1, help="step length of the search in the direction of velocity")
    argParser.add_argument("-p", "--patience", default = 10, help="patience of the search")
    argParser.add_argument("-m", "--max_iteration", default = 200, help="max iteration of the search")
    argParser.add_argument("--weight_randomness", default = 1, help="whether to use weight randomess") # 0, 1
    argParser.add_argument("-i", "--initial_expert_directory", default="./initial_experts", help="initial expert directory")
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument("--starting_test_set_eval", default=0, help="starting test set evaluation") # 0, 1
    argParser.add_argument("--fast_merge", default=0, help="whether to use fast merge by only loading the safetensor file") # 0, 1
    argParser.add_argument("--project_name_wb", default="search", help="wandb project name") # as you wish
    argParser.add_argument("--populate_initial_experts", default=0, help="whether to populate initial experts") # 0, 1
    argParser.add_argument("--initial_experts_num", default=None, help="number of initial experts to populate, when populate flag is 1")
    argParser.add_argument("--starting_velocity_mode", default="zero", help="starting velocity mode: zero, random, best") # zero, random, best
    argParser.add_argument("--repel_term", default=0, help="whether to incorporate a repel term with global_worst") # 0, 1
    argParser.add_argument("--step_length_factor", default=1, help="step length *= step_length_factor every iteration") # 1 for no scheduling, 0.95 maybe?
    argParser.add_argument("--minimum_step_length", default=0.1, help="minimum step length")
    argParser.add_argument("--restart_stray_particles", default=0, help="whether to restart stray particles") # 0, 1
    argParser.add_argument("--restart_patience", default=0.5, help="restart patience * patience = when to restart particles")
    argParser.add_argument("--clean_up_on_end", default=1, help="whether to clean up on end") # 0, 1
    argParser.add_argument("--to_visualize", default=False, help="whether to visualize the search process") # 0, 1
    argParser.add_argument("--correctness_emergence", default=False, help="whether to track correctness changes wrt iteration") # 0, 1
    argParser.add_argument("--dropK", default=0, help="dropout-K, 0-1")
    argParser.add_argument("--dropN", default=0, help="dropout-N, 0-1")

    args = argParser.parse_args()
    search_pass_name = args.name
    eval_type = args.eval_type
    dataset = args.dataset
    gpus = args.gpus
    graph_num = int(args.graph_num)
    num_cpu_when_merging = int(args.num_cpu_when_merging)
    inertia = float(args.inertia)
    cognitive_coeff = float(args.cognitive_coeff)
    social_coeff = float(args.social_coeff)
    repel_coeff = float(args.repel_coeff)
    patience = int(args.patience)
    step_length = float(args.step_length)
    max_iteration = int(args.max_iteration)
    weight_randomness = int(args.weight_randomness)
    initial_expert_directory = args.initial_expert_directory
    base_model = args.base_model
    starting_test_set_eval = int(args.starting_test_set_eval)
    fast_merge = int(args.fast_merge)
    project_name_wb = args.project_name_wb
    populate_initial_experts = int(args.populate_initial_experts)
    try:
        initial_experts_num = int(args.initial_experts_num)
    except:
        initial_experts_num = None
    starting_velocity_mode = args.starting_velocity_mode
    repel_term = int(args.repel_term)
    step_length_factor = float(args.step_length_factor)
    minimum_step_length = float(args.minimum_step_length)
    restart_stray_particles = int(args.restart_stray_particles)
    restart_patience = float(args.restart_patience)
    clean_up_on_end = int(args.clean_up_on_end)
    to_visualize_flag = args.to_visualize
    correctness_emergence = args.correctness_emergence
    dropK = float(args.dropK)
    dropN = float(args.dropN)

    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    search_pass_name += ("_" + socket.gethostname())
    if os.path.exists(os.path.join("search", search_pass_name)):
        search_pass_name += curret_time_string().replace(" ", "_")
    args.name = search_pass_name

    os.mkdir(os.path.join("search", search_pass_name))

    # write args to file
    with open(os.path.join("search", search_pass_name, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    run = wandb.init(name=search_pass_name, project=project_name_wb)
    run.config.update(args)
    torch.multiprocessing.set_start_method('spawn')
    random.seed(42)
    # Configure logging to write to a file
    logging.basicConfig(filename=os.path.join("search", search_pass_name, "log.txt"), level=logging.DEBUG)
    gpus = [int(gpu) for gpu in gpus.split(",")]
    particle_paths = []
    for particle_path in os.listdir(initial_expert_directory):
        if os.path.isdir(os.path.join(initial_expert_directory, particle_path)):
            particle_paths.append(os.path.join(initial_expert_directory, particle_path))
    particle_paths = sorted(particle_paths)

    print(gpus)

    log_with_flush("initializing search... "+curret_time_string())
    initialize_search_records(search_pass_name, particle_paths, graph_num, eval_type, dataset, gpus, base_model, fast_merge, starting_velocity_mode)
    log_with_flush("search initialized... "+curret_time_string())
    for i in range(len(particle_paths)):
        log_with_flush("expert " + str(i) + ": " + particle_paths[i])

    if os.path.exists(os.path.join("search", search_pass_name, "tmp")):
        shutil.rmtree(os.path.join("search", search_pass_name, "tmp"))
    
    log_with_flush("starting search... "+curret_time_string())

    graph_no_restart_count = [0] * graph_num
    model_no_restart_count = [0] * len(particle_paths)

    # main search iteration
    iter_count = 0
    while iter_count < max_iteration:
        iter_count += 1
        log_with_flush("--------------------------")
        log_with_flush("iteration "+str(iter_count)+"! "+curret_time_string())
        log_with_flush("updating particles...")

        # patience and ending condition
        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
            utility_scratchpad = json.load(f)
        g_best = utility_scratchpad["graph_g"]
        g_history = utility_scratchpad["graph_g_history"]
        if len(g_history) > patience:
            g_history = g_history[-patience:]
            # if g_history hasn't changed
            if max(g_history) == min(g_history):
                log_with_flush("patience reached!")
                break
        
        # update graphs
        graph_update_args = []
        for i in range(graph_num):
            if restart_stray_particles:
                particle_history = utility_scratchpad["graph_"+str(i)+"_history"]
                particle_best_so_far = utility_scratchpad["graph_"+str(i)+"_best"]
                first_time_best_idx = particle_history.index(particle_best_so_far)
                if len(particle_history) - first_time_best_idx >= restart_patience * patience and graph_no_restart_count[i] == 0:
                    restart_flag = True
                    log_with_flush("particle_"+str(i)+" restarted!")
                    graph_no_restart_count[i] = int(restart_patience * patience)
                else:
                    restart_flag = False
                    graph_no_restart_count[i] = max(0, graph_no_restart_count[i] - 1)
            else:
                restart_flag = False
            
            graph_update_args.append((i, gpus[assign_gpu(len(gpus), i, graph_num)], search_pass_name, weight_randomness, inertia, cognitive_coeff, social_coeff, repel_coeff, fast_merge, step_length, repel_term, restart_flag))

        pool = Pool(processes=num_cpu_when_merging)
        pool.starmap(graph_update, graph_update_args, chunksize=math.ceil(graph_num/num_cpu_when_merging))
        pool.close()
        pool.join()

        # evaluate new graphs
        log_with_flush("evaluating graphs...")
        graph_evaluate_args = []
        model_path_list = []
        assignment = [i for i in range(len(particle_paths))]
        for i in range(len(particle_paths)):
            model_path_list.append(os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"))
        for i in range(graph_num):
            graph_evaluate_args.append((os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), "now"), assignment, model_path_list, eval_type, dataset, "dev", gpus[assign_gpu(len(gpus), i, graph_num)]))
        
        pool = Pool(processes=len(gpus))
        graph_scores = pool.starmap(evaluate, graph_evaluate_args, chunksize=math.ceil(graph_num/len(gpus)))
        pool.close()
        pool.join()

        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
            utility_scratchpad = json.load(f)

        # all time best dev update (if needed)
        if max(graph_scores) > utility_scratchpad["all_time_best_dev"]:
            log_with_flush("all time best dev updated on the graph side!")
            # for i in range(graph_num):
            #     log_with_flush("graph_"+str(i)+"_now: "+str(graph_scores[i]))
            log_with_flush(str(graph_scores.index(max(graph_scores))))
            # change graph
            shutil.rmtree(os.path.join("search", search_pass_name, "all_time_best", "graph"))
            shutil.copytree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(graph_scores.index(max(graph_scores))), "now"), os.path.join("search", search_pass_name, "all_time_best", "graph"))
            # change models
            for i in range(len(particle_paths)):
                shutil.rmtree(os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))
                shutil.copytree(os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"), os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))
            # change assignment
            assignment = [i for i in range(len(particle_paths))]
            with open(os.path.join("search", search_pass_name, "all_time_best", "assignment.json"), "w") as f:
                json.dump(assignment, f, indent=4)
            # change all_time_best_dev
            utility_scratchpad["all_time_best_dev"] = max(graph_scores)

            log_with_flush("changing all time best dev to " + str(max(graph_scores)))
            # log_with_flush("immediately re-evaluating current best")

            # model_path_list = []
            # for i in range(len(particle_paths)):
            #     model_path_list.append(os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))
            # assignment = json.load(open(os.path.join("search", search_pass_name, "all_time_best", "assignment.json")))
            # # all_time_best_dev_score = evaluate(os.path.join("search", search_pass_name, "all_time_best", "graph"), assignment, model_path_list, eval_type, dataset, "dev", gpus[0])

            # all_time_best_dev_score = evaluate_best_graph(search_pass_name, dataset, eval_type, gpus, "dev")

            # log_with_flush("immediately re-evaluating current best dev: " + str(all_time_best_dev_score))
        
        # personal bests update
        for i in range(graph_num):
            if graph_scores[i] > utility_scratchpad["graph_"+str(i)+"_best"]:
                shutil.copytree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), "now"), os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), "personal_best"), dirs_exist_ok=True)
                utility_scratchpad["graph_"+str(i)+"_best"] = graph_scores[i]
            utility_scratchpad["graph_"+str(i)+"_now"] = graph_scores[i]
            utility_scratchpad["graph_"+str(i)+"_history"].append(graph_scores[i])
        
        # global best and worst update
        global_best_score = -1
        global_best_idx = -1
        global_worst_score = 101
        global_worst_idx = -1
        for i in range(graph_num):
            if graph_scores[i] > global_best_score:
                global_best_score = graph_scores[i]
                global_best_idx = i
            if graph_scores[i] < global_worst_score:
                global_worst_score = graph_scores[i]
                global_worst_idx = i
        
        if global_best_score > utility_scratchpad["graph_g"]:
            shutil.copytree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(global_best_idx), "now"), os.path.join("search", search_pass_name, "graphs", "global_best"), dirs_exist_ok=True)
            utility_scratchpad["graph_g"] = global_best_score
            utility_scratchpad["graph_g_history"].append(global_best_score)
        else:
            utility_scratchpad["graph_g_history"].append(utility_scratchpad["graph_g"])
        if global_worst_score < utility_scratchpad["graph_g_worst"]:
            shutil.copytree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(global_worst_idx), "now"), os.path.join("search", search_pass_name, "graphs", "global_worst"), dirs_exist_ok=True)
            utility_scratchpad["graph_g_worst"] = global_worst_score
        
        wandb_log = {
            "graph_g": utility_scratchpad["graph_g"],
            "graph_g_worst": utility_scratchpad["graph_g_worst"]
        }
        for i in range(graph_num):
            wandb_log["graph_"+str(i)+"_now"] = graph_scores[i]
        
        # no wandb log here, wait until model update

        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
            json.dump(utility_scratchpad, f, indent=4)
        
        log_with_flush("Graph update done." + curret_time_string())

        # update models

        # no patience here: graph-level is the main level

        model_update_args = []
        for i in range(len(particle_paths)):
            if restart_stray_particles:
                particle_history = utility_scratchpad["model_"+str(i)+"_history"]
                particle_best_so_far = utility_scratchpad["model_"+str(i)+"_best"]
                first_time_best_ids = particle_history.index(particle_best_so_far)
                if len(particle_history) - first_time_best_idx >= restart_patience * patience and model_no_restart_count[i] == 0:
                    restart_flag = True
                    log_with_flush("particle_"+str(i)+" restarted!")
                    model_no_restart_count[i] = int(restart_patience * patience)
                else:
                    restart_flag = False
                    model_no_restart_count[i] = max(0, model_no_restart_count[i] - 1)
            else:
                restart_flag = False
            
            model_update_args.append((i, gpus[assign_gpu(len(gpus), i, len(particle_paths))], search_pass_name, weight_randomness, inertia, cognitive_coeff, social_coeff, repel_coeff, fast_merge, step_length, repel_term, restart_flag))

        pool = Pool(processes=num_cpu_when_merging)
        pool.starmap(model_update, model_update_args, chunksize=math.ceil(len(particle_paths)/num_cpu_when_merging))
        pool.close()
        pool.join()

        # evaluate new models
        log_with_flush("evaluating models...")
        model_path_list = []
        for i in range(len(particle_paths)):
            model_path_list.append(os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"))
        
        model_scores, assignment_dict = node_evaluate(os.path.join("search", search_pass_name, "graphs", "graph_"+str(global_best_idx), "now"), model_path_list, eval_type, dataset, "dev", gpus)

        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
            utility_scratchpad = json.load(f)

        # all time best dev update (if needed)
        if assignment_dict["best_score"] > utility_scratchpad["all_time_best_dev"]:
            log_with_flush("all time best dev updated on the model side!")
            log_with_flush("assignment_dict")
            # change graph
            shutil.rmtree(os.path.join("search", search_pass_name, "all_time_best", "graph"))
            shutil.copytree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(global_best_idx), "now"), os.path.join("search", search_pass_name, "all_time_best", "graph"))   
            # change models
            for i in range(len(particle_paths)):
                shutil.rmtree(os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))
                shutil.copytree(os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"), os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))
            # change assignment
            with open(os.path.join("search", search_pass_name, "all_time_best", "assignment.json"), "w") as f:
                json.dump(assignment_dict["best_assignment"], f, indent=4)
            # change all time best dev
            utility_scratchpad["all_time_best_dev"] = assignment_dict["best_score"]

            log_with_flush("changing all time best dev to " + str(assignment_dict["best_score"]))
            # log_with_flush("immediately re-evaluating current best")

            # model_path_list = []
            # for i in range(len(particle_paths)):
            #     model_path_list.append(os.path.join("search", search_pass_name, "all_time_best", "models", "model_"+str(i)))
            # assignment = json.load(open(os.path.join("search", search_pass_name, "all_time_best", "assignment.json")))

            # # all_time_best_dev_score = evaluate(os.path.join("search", search_pass_name, "all_time_best", "graph"), assignment, model_path_list, eval_type, dataset, "dev", gpus[0])
            # all_time_best_dev_score = evaluate_best_graph(search_pass_name, dataset, eval_type, gpus, "dev")

            # log_with_flush("immediately re-evaluating current best dev: " + str(all_time_best_dev_score))

        # personal bests update
        for i in range(len(particle_paths)):
            if model_scores[i] > utility_scratchpad["model_"+str(i)+"_best"]:
                shutil.copytree(os.path.join("search", search_pass_name, "models", "model_"+str(i), "now"), os.path.join("search", search_pass_name, "models", "model_"+str(i), "personal_best"), dirs_exist_ok=True)
                utility_scratchpad["model_"+str(i)+"_best"] = model_scores[i]
            utility_scratchpad["model_"+str(i)+"_now"] = model_scores[i]
            utility_scratchpad["model_"+str(i)+"_history"].append(model_scores[i])
        
        # global best and worst update
        global_best_score = -1
        global_best_idx = -1
        global_worst_score = 101
        global_worst_idx = -1

        for i in range(len(particle_paths)):
            if model_scores[i] > global_best_score:
                global_best_score = model_scores[i]
                global_best_idx = i
            if model_scores[i] < global_worst_score:
                global_worst_score = model_scores[i]
                global_worst_idx = i
        
        if global_best_score > utility_scratchpad["model_g"]:
            shutil.copytree(os.path.join("search", search_pass_name, "models", "model_"+str(global_best_idx), "now"), os.path.join("search", search_pass_name, "models", "global_best"), dirs_exist_ok=True)
            utility_scratchpad["model_g"] = global_best_score
            utility_scratchpad["model_g_history"].append(global_best_score)
        else:
            utility_scratchpad["model_g_history"].append(utility_scratchpad["model_g"])
        if global_worst_score < utility_scratchpad["model_g_worst"]:
            shutil.copytree(os.path.join("search", search_pass_name, "models", "model_"+str(global_worst_idx), "now"), os.path.join("search", search_pass_name, "models", "global_worst"), dirs_exist_ok=True)
            utility_scratchpad["model_g_worst"] = global_worst_score
        
        wandb_log["model_g"] = utility_scratchpad["model_g"]
        wandb_log["model_g_worst"] = utility_scratchpad["model_g_worst"]

        # wandb_log = {
        #     "model_g": utility_scratchpad["model_g"],
        #     "model_g_worst": utility_scratchpad["model_g_worst"]
        # }

        for i in range(len(particle_paths)):
            wandb_log["model_"+str(i)+"_now"] = model_scores[i]
        
        wandb_log["all_time_best_dev"] = utility_scratchpad["all_time_best_dev"]
        
        # finally, logging after both updates are done
        wandb.log(wandb_log)

        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
            json.dump(utility_scratchpad, f, indent=4)
        
        log_with_flush("Model update done." + curret_time_string())
    
    log_with_flush("search done. "+curret_time_string())

    # evaluate all time best

    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ending_best_dev = evaluate_best_graph(search_pass_name, dataset, eval_type, gpus, "dev")
    ending_best_test = evaluate_best_graph(search_pass_name, dataset, eval_type, gpus, "test")

    log_with_flush("ending best dev: " + str(ending_best_dev))
    log_with_flush("ending best test: " + str(ending_best_test))
    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
        utility_scratchpad = json.load(f)
    g_best = utility_scratchpad["graph_g"]
    log_with_flush("Global best graph: " + str(g_best))
    all_time_best_dev = utility_scratchpad["all_time_best_dev"]
    log_with_flush("All time best dev in the scratchpad: " + str(all_time_best_dev))
    
    final_metrics = {
        "dev_best": ending_best_dev,
        "test_best": ending_best_test,
    }

    wandb.log(final_metrics)
    log_with_flush("final metrics: " + str(final_metrics))

    if clean_up_on_end:
        shutil.rmtree(os.path.join("search", search_pass_name, "models", "global_worst"))
        shutil.rmtree(os.path.join("search", search_pass_name, "graphs", "global_worst"))
        for i in range(len(particle_paths)):
            for aux in ["g_x", "p_x", "velocity", "x_w"]:
                shutil.rmtree(os.path.join("search", search_pass_name, "models", "model_"+str(i), aux))
        for i in range(graph_num):
            for aux in ["velocity"]:
                shutil.rmtree(os.path.join("search", search_pass_name, "graphs", "graph_"+str(i), aux))
        
        log_with_flush("the end of search... "+curret_time_string())



# if __name__ == '__main__':

#     torch.multiprocessing.set_start_method('spawn')

#     particle_paths = os.listdir("initial_experts")
#     new_particle_paths = []
#     for i in range(len(particle_paths)):
#         if ".py" in particle_paths[i]:
#             continue
#         new_particle_paths.append(particle_paths[i])
#     particle_paths = new_particle_paths
#     for i in range(len(particle_paths)):
#         particle_paths[i] = os.path.join("initial_experts", particle_paths[i])
#     initialize_search_records("test", particle_paths, 10, "multiple_choice", "knowledge_crosswords", [0,1,2,3,4])
