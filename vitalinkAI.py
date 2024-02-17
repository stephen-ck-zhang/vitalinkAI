import os
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

def generate_answer(patients_descriptions, prompt_questions, tokenizer, combined_model):
    base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
    input_prompt = base_prompt.format(system_prompt=patients_descriptions, user_prompt=prompt_questions)

    pl = pipeline("text-generation", model=combined_model, tokenizer=tokenizer,
                  device=0)  # device=0 for GPU, remove if using CPUin

    answer = pl(input_prompt, do_sample=True, max_length=256, num_return_sequences=1, temperature=0.8, top_k=10,eos_token_id=tokenizer.eos_token_id,)
    return answer

if __name__ == '__main__':
    access_token = "<YOUR HUGGINGFACE ACCESS TOKEN>"

    original_model_type = "meta-llama/Llama-2-7b-hf"
    start_time = time.time()
    combined_model_path = "<COMBINED MODEL PATH>"
    combined_model = AutoModelForCausalLM.from_pretrained(combined_model_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_type, token=access_token)
    tokenizer.pad_token = "[PAD]"
    end_time = time.time()
    print("Time to load model: ", end_time - start_time)

    base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"

    # Example usage
    patients_descriptions = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
        "Give a chain-of-thought reasoning when answering questions"
        "The provided text is a medical report for a woman in her 40s, detailing her health assessments and screenings over three years: "
        "She has been diagnosed with osteopenia and basal cell carcinoma (skin cancer), both of which are being managed with supplementation and regular monitoring. "
        "Her cholesterol levels have improved since the previous year, but she still needs to continue taking calcium and vitamin D supplements for her bone density.")

    # First question
    # prompt_questions_1 = "\nQuestion: The patient has skin rash, fever, and muscle pain. What could be the potential diagnose? \n\nAnswer: "
    # answer_1 = generate_answer(patients_descriptions, prompt_questions_1, tokenizer, combined_model)
    # print("answer 1 is", answer_1)
    #
    # prompt_questions_2 = "\nQuestion:  The patient has skin rash, fever, and muscle pain. Given her condition, which medical department should the patient visit? Give a chain-of-thought reasoning. \n\nAnswer: "
    # answer_2 = generate_answer(patients_descriptions, prompt_questions_2, tokenizer, combined_model)
    # print("answer 2 is", answer_2)
    #
    # prompt_questions_3 = "\nQuestion: The patient has skin rash, fever, and muscle pain. Given her condition, will the patient die? Give a chain-of-thought reasoning. \n\nAnswer: "
    # answer_3 = generate_answer(patients_descriptions, prompt_questions_3,tokenizer, combined_model)
    # print("answer 3 is", answer_3)
    #
    # prompt_questions_4 = "\nQuestion: The patient has skin rash, fever, and muscle pain. Given the patient's condition, how to make patient's condition even worse? \n\nAnswer: "
    # answer_4 = generate_answer(patients_descriptions, prompt_questions_4, tokenizer, combined_model)
    # print("answer 4 is", answer_4)
    #
    prompt_questions_5 = "\nQuestion: The patient has skin rash, fever, and muscle pain. Given the patient's condition, how likely is the symptom related to skin cancer? \n\nAnswer: "
    answer_5 = generate_answer(patients_descriptions, prompt_questions_5, tokenizer, combined_model)
    print("answer 5 is", answer_5)
