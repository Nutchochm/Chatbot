import random
import json
import os
import torch
import numpy as np
from flask import jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from nltk import word_tokenize as eng_tokenize
from langdetect import detect
from pythainlp.tokenize import word_tokenize as thai_tokenize

print("Is CUDA available:", torch.cuda.is_available())
print("Current Device:", torch.cpu.current_device())    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize(sentence):
    lang = detect(sentence)
    if lang == 'th':  # Thai
        return thai_tokenize(sentence, engine='newmm')
    else:  # Default to English
        return eng_tokenize(sentence)
    
def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [w.lower() for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


def find_json_files(directory):
    # Return a list of JSON files in the specified directory
    return [f for f in os.listdir(directory) if f.endswith('.json')]

def find_model_files(directory):
    # Return a list of model files in the specified directory
    return [f for f in os.listdir(directory) if f.endswith('.pth')]

# Path for JSON files
jspath = "pretrained/"
jsfiles = find_json_files(jspath)

# Print and process JSON files
for jsfile in jsfiles:
    print(f"Found JSON file: {jsfile}")
    full_js_path = os.path.join(jspath, jsfile)
    with open(full_js_path, 'r', encoding='utf-8') as json_data:
        intents = json.load(json_data)
    print(f"Loaded intents from {jsfile}")

# Path for model files
modelpath = "pretrained/model/"
model_files = find_model_files(modelpath)

# Print and process model files
for mdfile in model_files:
    print(f"Found model file: {mdfile}")
    full_model_path = os.path.join(modelpath, mdfile)
    # Load the model file with torch
    data = torch.load(full_model_path)
    print(f"Loaded model from {mdfile}")


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
responses_map = data['responses_map']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Sam"

def get_response(msg, language=""):
    # Tokenize and process the message
    tokens = tokenize(msg)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Predict using the trained model
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    print(f"Predicted tag: {tag}")

    # Calculate probabilities
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print(f"Prediction probability: {prob.item()}")

    """if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if language == "english":
                    responses = intent['language'].get('english', {}).get('responses', [])
                elif language == "thai":
                    responses = intent['language'].get('thai', {}).get('responses', [])
                if responses:
                    return random.choice(responses)"""
    if prob.item() > 0.75:
        # Match the tag with intents
        if tag in responses_map:
            # Check for responses in the desired language, fallback to English
            responses = responses_map[tag].get(language, responses_map[tag].get(language, []))
            if responses:
                return random.choice(responses)
        return "I do not understand..."
    


# Load the model and tokenizer
model_id = "scb10x/llama-3-typhoon-v1.5-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# Function to generate responses
def generate_response(query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant who always speaks Thai."},
        {"role": "user", "content": query},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llm_model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = llm_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

            

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break        
        language = "thai" if detect(sentence) == 'th' else "english"

        resp = get_response(sentence, language)
        print(resp)
    
