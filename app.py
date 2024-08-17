# app.py
from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')


def generate_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, do_sample=True, top_k=30, top_p=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_query = data['query']
    response = generate_response(f"Customer asks: {user_query}\nResponse:", model, tokenizer)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
