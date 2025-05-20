import os
import shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
def lora_merge(weights, lora_name_list, output_name, gpu_id, directly_load_safetensors = 0):
    if not directly_load_safetensors:
        lora_state_dict_list = []
        for lora_name in lora_name_list:
            model = AutoModelForCausalLM.from_pretrained(lora_name).to(f"cuda:{gpu_id}")
            lora_state_dict_list.append(get_peft_model_state_dict(model))
            if not lora_name == lora_name_list[-1]:
                del model
            # torch.cuda.empty_cache()
        
        final_state_dict = {}
        for i in range(len(lora_state_dict_list)):
            if i == 0:
                for key in lora_state_dict_list[i].keys():
                    final_state_dict[key] = weights[i] * lora_state_dict_list[i][key]
            else:
                for key in lora_state_dict_list[i].keys():
                    assert key in final_state_dict.keys()
                    final_state_dict[key] += weights[i] * lora_state_dict_list[i][key]
        
        # model = AutoModelForCausalLM.from_pretrained(lora_name_list[0]).to(f"cuda:{gpu_id}")
        set_peft_model_state_dict(model, final_state_dict)
        if os.path.exists(output_name):
            shutil.rmtree(output_name)
        model.save_pretrained(output_name)
    else:
        lora_state_dict_list = []
        for lora_name in lora_name_list:
            state_dict_this = load_file(os.path.join(lora_name, "adapter_model.safetensors"), device="cpu")
            lora_state_dict_list.append(state_dict_this)
        
        final_state_dict = {}
        for i in range(len(lora_state_dict_list)):
            if i == 0:
                for key in lora_state_dict_list[i].keys():
                    final_state_dict[key] = weights[i] * lora_state_dict_list[i][key]
            else:
                for key in lora_state_dict_list[i].keys():
                    assert key in final_state_dict.keys()
                    final_state_dict[key] += weights[i] * lora_state_dict_list[i][key]
        
        if not os.path.exists(output_name):
            os.mkdir(output_name)
        save_file(final_state_dict, os.path.join(output_name, "adapter_model.safetensors"))
        return final_state_dict 

# lora_merge([0.3, 0.6, 0.8], ["./initial_experts/lima", "./initial_experts/cot", "./initial_experts/science"], "./new", 0, directly_load_safetensors=1)