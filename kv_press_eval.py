import sys
sys.path.insert(0, "/dccstor/nathan-ckpts/anooshka/kvpress")
from transformers import pipeline
from kvpress import KVzipPress, ObservedAttentionPress
# from kvpress.kvpress.presses.observed_attention_press import ObservedAttentionPress
# KVzipPress, 
# kvpress/kvpress/presses/observed_attention_press.py

model = "Qwen/Qwen3-8B"
pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto", model_kwargs={"attn_implementation": "eager"})

context = "My dog's name is Shmeegus. He is green and slimy. He does not like cats, but he does like baths."
question = "\nDoes Shmeegus like baths?"  # optional

press = ObservedAttentionPress(compression_ratio=0.5)
answer = pipe(context, question=question, press=press)["answer"]
print(context)
print(question)
print(f"ObservedAttentionPress answer: {answer}")