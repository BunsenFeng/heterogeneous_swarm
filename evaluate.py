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
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold
project_id = "YOUR_GCLOUD_PROJECT_ID"
if project_id == "YOUR_GCLOUD_PROJECT_ID":
    print("Please set your Google Cloud project ID in the code. Safely ignore this warning if your evaluation does not involve LLM-as-a-judge.")
else:
    location_list = ["us-east5", "us-south1", "us-central1", "us-west4", "us-east1", "us-east4", "us-west1"]
    location = random.choice(location_list)
    vertexai.init(project=project_id, location=location)
    gemini_model = GenerativeModel("gemini-1.5-flash-001")
    generationConfig = GenerationConfig(temperature=0, max_output_tokens=20)

    safety_config = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

FIRST_INSTRUCTION = "Please answer the following question."
NON_LAST_INSTRUCTION = "Please answer the following question with the help of previous responses, feel free to ignore wrong or unhelpful responses."
LAST_INSTRUCTION = "Please answer the following question with the help of previous responses, feel free to ignore wrong or unhelpful responses."

ICL_PROMPT = None
model = None
tokenizer = None
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_gpu_memory_in_gb():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_properties = torch.cuda.get_device_properties(device)

        total_memory = gpu_properties.total_memory / (1024 ** 3)  # Convert bytes to GB
        return total_memory
    else:
        return 0

def curret_time_string():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def multiple_choice_prompt(instance_dict, dataset):
    prompt = "Question: " + instance_dict["question"] + "\n"

    if dataset == "knowledge_crosswords":
        prompt = prompt
    elif dataset == "hellaswag":
        prompt = "Please choose an option that best completes the sentence.\n" + prompt
    else:
        prompt = "Please choose an option that best answers the question.\n" + prompt

    for key in instance_dict["choices"].keys():
            prompt += key + ": " + instance_dict["choices"][key] + "\n"

    prompt += "The answer is"

    if dataset == "knowledge_crosswords":
        prompt = ICL_PROMPT + "\n" + prompt

    # print(prompt)

    return prompt

def multiple_choice_answer_parsing(instance_dict, output_text):

    # print(output_text)
    # print("-----")

    # directly answer
    for key in instance_dict["choices"].keys():
        if key in output_text[:5]:
            return key
    # "The answer is ."
    for key in instance_dict["choices"].keys():
        if key in output_text[-5:]:
            return key
    # answer text exact match
    for key in instance_dict["choices"].keys():
        if instance_dict["choices"][key].lower() in output_text.lower():
            return key
    # # general screenning
    # for key in instance_dict["choices"].keys():
    #     if key in output_text:
    #         return key
    # print(output_text)
    return "Z" # so that its absolutely incorrect

def batch_generate_chat_template(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=100):
    outputs = []

    # batched generation with chat template
    num_batches = math.ceil(len(prompts) / batch_size)
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        chats = []
        for prompt in batch_prompts:
            chat = [
                {"role": "user", "content": prompt},
            ]
            chats.append(chat)
        
        inputs = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
        # print(inputs)
        input_ids = tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to(f"cuda:{gpu_id}")
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        for j in range(len(output_ids)):
            outputs.append(tokenizer.decode(output_ids[j][len(input_ids[j]):], skip_special_tokens=True).strip())
    
    return outputs

# prompts = ["Please explain constitutionalism."] * 20
# model = AutoModelForCausalLM.from_pretrained("initial_experts/lima", torch_dtype=torch.bfloat16).to("cuda:0")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
# outputs = batch_generate_chat_template(model, tokenizer, prompts, 0, batch_size=10, max_new_tokens=100)
# print(outputs)

def batch_generate(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=50, chat_template=False):
    if not chat_template:
        num_batches = math.ceil(len(prompts) / batch_size)
        outputs = []
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]

            input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True).input_ids.to(f"cuda:{gpu_id}")
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample = False)

            for j in range(len(output)):
                outputs.append(tokenizer.decode(output[j][len(input_ids[j]):], skip_special_tokens=True).strip())
            
            del input_ids, output
            torch.cuda.empty_cache()
        
        return outputs
    else:
        return batch_generate_chat_template(model, tokenizer, prompts, gpu_id, batch_size, max_new_tokens)

def graph_generate(prompts, adjacency_matrix, assignment, model_path_list, gpu_id, top_p_temperature=0, chat_template=False, base_model = "google/gemma-7b-it", batch_size=10, max_new_tokens=50, chunked_contexts = None):
    assert len(adjacency_matrix) == len(assignment) == len(model_path_list)
    # decode graph
    graph_decoded = graph_decode.graph_decode(adjacency_matrix, top_p_temperature)
    # print(graph_decoded)
    # topological sort
    topological_order = graph_decode.topological_sort(graph_decoded)
    # print(topological_order)
    # generate
    intermediate_outputs = [0] * len(adjacency_matrix)

    if chunked_contexts is not None:
        old_prompts = prompts

    for i in range(len(topological_order)):
        node = topological_order[i]
        model = AutoModelForCausalLM.from_pretrained(model_path_list[assignment[node]], torch_dtype=torch.bfloat16).to(f"cuda:{gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        if chunked_contexts is not None:
            prompts = []
            for j in range(len(old_prompts)):
                prompts.append("Context: " + chunked_contexts[j][i] + "\n" + old_prompts[j])
            assert len(prompts) == len(old_prompts)

        if sum(graph_decoded[:, node]) == 0: # a starting node with no previous step
            prompts_now = [FIRST_INSTRUCTION + "\nQuestion: " + prompt for prompt in prompts]
            # print(prompts_now)
            outputs = batch_generate(model, tokenizer, prompts_now, gpu_id, chat_template=chat_template, batch_size=batch_size, max_new_tokens=max_new_tokens)
            # outputs = [output.split("\n")[0] for output in outputs]
            outputs = [output.replace("\n", " ") for output in outputs]
            # print(outputs)
            # print("----------------")
            intermediate_outputs[node] = outputs
        elif sum(graph_decoded[node, :]) == 0: # an ending node with no next step
            assert i == len(topological_order) - 1
            previous_steps = [j for j in range(len(topological_order)) if graph_decoded[j, node] == 1]
            previous_outputs = [intermediate_outputs[j] for j in previous_steps]
            concatenated_previous_outputs = []
            for j in range(len(prompts)):
                temp = ""
                for k in range(len(previous_steps)):
                    temp += "Previous response " + str(k + 1) + ": " + previous_outputs[k][j] + "\n"
                concatenated_previous_outputs.append(temp)
            assert len(concatenated_previous_outputs) == len(prompts)
            prompts_now = [LAST_INSTRUCTION + "\n" + concatenated_previous_outputs[j] + "Question: " + prompts[j] for j in range(len(prompts))]
            # print(prompts_now)
            outputs = batch_generate(model, tokenizer, prompts_now, gpu_id, chat_template=chat_template, batch_size=batch_size, max_new_tokens=max_new_tokens)
            # outputs = [output.split("\n")[0] for output in outputs]
            outputs = [output.replace("\n", " ") for output in outputs]
            # print(outputs)
            # print("----------------")
            intermediate_outputs[node] = outputs
        else: # a node with both previous and next steps
            previous_steps = [j for j in range(len(topological_order)) if graph_decoded[j, node] == 1]
            previous_outputs = [intermediate_outputs[j] for j in previous_steps]
            concatenated_previous_outputs = []
            for j in range(len(prompts)):
                temp = ""
                for k in range(len(previous_steps)):
                    temp += "Previous response " + str(k + 1) + ": " + previous_outputs[k][j] + "\n"
                concatenated_previous_outputs.append(temp)
            assert len(concatenated_previous_outputs) == len(prompts)
            prompts_now = [NON_LAST_INSTRUCTION + "\n" + concatenated_previous_outputs[j] + "Question: " + prompts[j] for j in range(len(prompts))]
            # print(prompts_now)
            outputs = batch_generate(model, tokenizer, prompts_now, gpu_id, chat_template=chat_template, batch_size=batch_size, max_new_tokens=max_new_tokens)
            # outputs = [output.split("\n")[0] for output in outputs]
            outputs = [output.replace("\n", " ") for output in outputs]
            # print(outputs)
            # print("----------------")
            intermediate_outputs[node] = outputs
        del model, tokenizer
        torch.cuda.empty_cache()
    return intermediate_outputs[topological_order[-1]]

def evaluate(graph_path, assignment, model_path_list, eval_type, dataset, dev_test_split, gpu_id, base_model = "google/gemma-7b-it"):
    global model
    global tokenizer

    assert dev_test_split in ["dev", "test"]
    adjacency_matrix = np.load(os.path.join(graph_path, "adjacency.npy"))
    assert len(adjacency_matrix) == len(assignment) == len(model_path_list)

    if get_gpu_memory_in_gb() > 70: # 80GB GPUs, larger batch sizes
        GPU80_FLAG = True
    else:
        GPU80_FLAG = False

    if eval_type == "abstainqa":
        eval_data = json.load(open("data/eval/" + "mmlu" + ".json"))[dev_test_split]

        golds = []
        preds = []

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)

        CHAT_TEMPLATE = False
        MAX_NEW_TOKENS = 100

        BATCH_SIZE = 10
        if GPU80_FLAG:
            BATCH_SIZE *= 2

        outputs = graph_generate(prompts, adjacency_matrix, assignment, model_path_list, gpu_id, batch_size=BATCH_SIZE, chat_template=CHAT_TEMPLATE, max_new_tokens=MAX_NEW_TOKENS)

        for question, output in zip(eval_data, outputs):
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))
        
        correct_flags = []
        for gold, pred in zip(golds, preds):
            if gold == pred:
                correct_flags.append(1)
            else:
                correct_flags.append(0)
        
        abstain_prompts = []
        assert len(prompts) == len(outputs)
        for i in range(len(prompts)):
            # prompt_now = "Please evaluate whether the proposed answer is true or false.\n\nQuestion: " + prompts[i] + "\n\nProposed answer: " + preds[i] + "\n\nIs the proposed answer True or False?"
            prompt_now = prompts[i] + "\nProposed answer: " + preds[i] + "\nIs the proposed answer true or false? Directly answer with true or false."
            abstain_prompts.append(prompt_now)

        BATCH_SIZE = 10
        if GPU80_FLAG:
            BATCH_SIZE *= 2
        MAX_NEW_TOKENS = 10
        chat_template = False

        abstain_outputs = graph_generate(abstain_prompts, adjacency_matrix, assignment, model_path_list, gpu_id, batch_size=BATCH_SIZE, chat_template=chat_template, max_new_tokens=MAX_NEW_TOKENS)

        abstain_flags = [] # abstain as 1, not abstain as 0
        for abstain_output in abstain_outputs:
            if "True" in abstain_output:
                abstain_flags.append(0)
            else:
                abstain_flags.append(1)
        # calculate effective reliability
        answered_correctly = 0
        answered_incorrectly = 0
        for i in range(len(correct_flags)):
            if abstain_flags[i] == 0:
                if correct_flags[i] == 1:
                    answered_correctly += 1
                else:
                    answered_incorrectly += 1
        
        return (answered_correctly - answered_incorrectly) / len(correct_flags)

    if eval_type == "multiple_choice":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))[dev_test_split]

        golds = []
        preds = []
        global ICL_PROMPT
        try:
            ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        except:
            pass
            
        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)

        CHAT_TEMPLATE = False
        MAX_NEW_TOKENS = 50

        BATCH_SIZE = 10
        if dataset == "knowledge_crosswords":
            BATCH_SIZE = 8
        elif dataset == "com2":
            BATCH_SIZE = 16
        elif dataset == "mmlu_pro":
            BATCH_SIZE = 10
        elif dataset == "wow":
            BATCH_SIZE = 4
        if GPU80_FLAG:
            BATCH_SIZE *= 2

        outputs = graph_generate(prompts, adjacency_matrix, assignment, model_path_list, gpu_id, batch_size=BATCH_SIZE, chat_template=CHAT_TEMPLATE, max_new_tokens=MAX_NEW_TOKENS)
        # print(outputs)

        for question, output in zip(eval_data, outputs):
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))
        
        return accuracy_score(golds, preds)

    if eval_type == "exact_match":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))[dev_test_split]

        scores = []

        prompts = []
        for question in eval_data:
            if dataset == "gsm8k":
                prompts.append(question["question"] + " Please end your response with 'The answer is'.")
            else:
                prompts.append(question["question"])
        CHAT_TEMPLATE = False

        chunked_contexts = None
        if dataset == "gaia":
            chunked_contexts = []
            for question in eval_data:
                chunked_contexts.append(question["context_chunks"])

        BATCH_SIZE = 10
        if dataset == "gsm8k":
            BATCH_SIZE = 6
            CHAT_TEMPLATE = True
        if dataset == "nlgraph":
            BATCH_SIZE = 3
            CHAT_TEMPLATE = True
        if dataset == "gaia":
            BATCH_SIZE = 10
            CHAT_TEMPLATE = True
        if GPU80_FLAG:
            BATCH_SIZE *= 2

        MAX_NEW_TOKENS = 50
        if dataset == "gsm8k":
            MAX_NEW_TOKENS = 100
        if dataset == "nlgraph":
            MAX_NEW_TOKENS = 80
        
        outputs = graph_generate(prompts, adjacency_matrix, assignment, model_path_list, gpu_id, batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS, chat_template=CHAT_TEMPLATE, chunked_contexts=chunked_contexts)

        # print(outputs)

        if dataset == "gsm8k":
            outputs = [" ".join(output.split(" ")[-5:]) for output in outputs]
        
        # print(outputs)

        for question, output in zip(eval_data, outputs):
            if question["answer"] in output:
                scores.append(1)
                # time.sleep(0.2)
            else:
                scores.append(0)
        
        return sum(scores) / len(scores)

    if eval_type == "gemini_match":

        eval_data = json.load(open("data/eval/" + dataset + ".json"))[dev_test_split]

        scores = []

        prompts = []
        for question in eval_data:
            prompts.append(question["question"])
        
        chunked_contexts = None
        if dataset == "qasper":
            chunked_contexts = []
            for question in eval_data:
                chunked_contexts.append(question["context_chunks"])
        
        BATCH_SIZE = 5
        if GPU80_FLAG:
            BATCH_SIZE *= 2
        MAX_NEW_TOKENS = 200

        outputs = graph_generate(prompts, adjacency_matrix, assignment, model_path_list, gpu_id, batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS, chat_template=True, chunked_contexts=chunked_contexts)

        for i in range(len(eval_data)):
            ground_truth = eval_data[i]["answer"]
            generated = outputs[i]
            eval_prompt = "Please evaluate how similar is the following response to the ground truth answer. Please rate the response on a scale of 1 to 10, where 1 is the worst and 10 is the best. Please respond with \"Rating: ?/10\" first and then provide your reason.\n\nGround truth: " + ground_truth + "\n\nGenerated response: " + generated
            similarity_score = gemini_model.generate_content(eval_prompt, generation_config = generationConfig, safety_settings=safety_config).text
            if "Rating: " not in similarity_score:
                score = 1
            score = int(similarity_score.split("Rating: ")[1].split("/10")[0])
            scores.append(score)

        return sum(scores) / len(scores)