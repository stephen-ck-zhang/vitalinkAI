import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
import time
from task_vector import TaskVector

def compare_two_models(combined_model, finetuned_model):
    for (name1, param1), (name2, param2) in zip(combined_model.named_parameters(), finetuned_model.named_parameters()):
        if param1.size() != param2.size():
            print(f"Mismatch in layer: {name1}")
            print(f"Original model parameter size: {param1.size()}, Finetuned model parameter size: {param2.size()}")

def match_two_models(combined_model, finetuned_model):
    '''
    We assume that fine-tuned model has more parameters than the original model
    :param combined_model: path to the original model
    :param finetuned_model: path to the fine-tuned model
    :return:
    '''
    original_state_dict = combined_model.state_dict()
    finetuned_state_dict = finetuned_model.state_dict()

    for name, param in original_state_dict.items():
        if name in finetuned_state_dict:
            original_param_shape = param.shape
            finetuned_param_shape = finetuned_state_dict[name].shape

            if original_param_shape != finetuned_param_shape:
                print(f"Adjusting layer: {name}")
                print( f"Original model parameter size: {original_param_shape}, Finetuned model parameter size: {finetuned_param_shape}")

                # Calculate the padding needed along the first dimension
                padding_size = finetuned_param_shape[0] - original_param_shape[0]

                # Ensure padding is needed (i.e., the finetuned model's parameter is larger)
                if padding_size > 0:
                    # Create a zeros tensor with the correct shape for padding
                    zeros_padding = torch.zeros(padding_size, original_param_shape[1], device=param.device)

                    # Concatenate the original parameter with the zeros padding tensor
                    padded_param = torch.cat([param, zeros_padding], dim=0)

                    # Update the original state dict with the padded parameter
                    original_state_dict[name] = padded_param

    return original_state_dict, finetuned_state_dict


def save_combined_models(original_model_type, fine_tuned_model_type, cache_dir, combined_model_dir, access_token):
    access_token = access_token
    original_model_type = original_model_type
    start_time = time.time()
    combined_model = AutoModelForCausalLM.from_pretrained(original_model_type)
    end_time = time.time()
    print("Time to load model: ", end_time - start_time)

    model_B_type = fine_tuned_model_type
    model_B = AutoModelForCausalLM.from_pretrained(model_B_type, token=access_token, cache_dir=cache_dir)

    print("Start checking")
    print("\n")

    compare_two_models(combined_model, model_B)

    print("finished checking \n")

    original_state_dict, finetuned_state_dict = match_two_models(combined_model, model_B)

    combined_model = AutoModelForCausalLM.from_pretrained(model_B_type, token=access_token,
                                                          cache_dir=cache_dir)
    combined_model.load_state_dict(original_state_dict)

    print("\n")
    print("Start second checking")
    print("\n\n")

    compare_two_models(combined_model, model_B)

    print("finished second checking \n")

    task_vector = TaskVector(combined_model, model_B)

    combined_model = task_vector.apply_to(combined_model, scaling_coef=1.0)

    combined_model.save_pretrained(combined_model_dir)

if __name__ == "main":
    access_token = "<Your HuggingFace Access Token>"
    original_model_type = "<Your Path TO THE ORIGINAL MODEL>"
    # fine_tuned_model_type = "<YOUR PATH TO THE FINETUNED MODEL>"
    fine_tuned_model_type = "epfl-llm/meditron-7b"
    cache_dir = "<Cache Path>"
    combined_model_dir = "<Path TO SAVE NEW COMBINED MODEL>"
    save_combined_models(original_model_type = original_model_type,
         fine_tuned_model_type = fine_tuned_model_type,
         cache_dir = cache_dir,
         combined_model_dir = combined_model_dir,
         access_token = access_token)

