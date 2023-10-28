import random
import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto", max_length=600)


def model_response(message, history):
    history_flat = [item for sublist in history for item in sublist]
    input_text = message
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])

app = gr.ChatInterface(model_response)

app.launch()
