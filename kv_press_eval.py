import sys
sys.path.insert(0, "/dccstor/nathan-ckpts/anooshka/kvpress")
import json
from transformers import pipeline
from kvpress import KVzipPress, ObservedAttentionPress

# model = "Qwen/Qwen3-8B"
model = "ibm-granite/granite-3.3-8b-instruct"

obs_pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto", model_kwargs={"attn_implementation": "eager"})
zip_pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto")

obs_press = ObservedAttentionPress(compression_ratio=0.5)
zip_press = KVzipPress(compression_ratio=0.5)

with open("kv_press_test/questions.json") as f:
    data = json.load(f)

results = []
for case in data:
    case_id = case["case_id"]
    context = open(f"kv_press_test/cases/{case_id}.txt").read()
    print("-"*25)
    print(f"Case: {case_id}")
    case_results = {"case_id": case_id, "questions": []}

    for q in case["questions"]:
        question = q["question"]
        ground_truth = q["ground_truth"]

        obs_answer = obs_pipe(context, question=question, press=obs_press)["answer"]
        zip_answer = zip_pipe(context, question=question, press=zip_press)["answer"]

        print(f"Question: {question}")
        print(f"Ground truth: {ground_truth}")
        print(f"ObservedAttentionPress answer: {obs_answer}")
        print(f"KVzipPress answer: {zip_answer}")
        print()

        case_results["questions"].append({
            "question": question,
            "ground_truth": ground_truth,
            "obs_answer": obs_answer,
            "zip_answer": zip_answer
        })

    results.append(case_results)

    # save after each case so partial results aren't lost
    with open("kv_press_test/results.json", "w") as f:
        json.dump(results, f, indent=2)