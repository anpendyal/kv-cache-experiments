import sys
sys.path.insert(0, "/dccstor/nathan-ckpts/anooshka/kvpress")
<<<<<<< HEAD
import json
import logging
import torch
from transformers import pipeline
from kvpress import KVzipPress, ObservedAttentionPress

# General progress log
logging.basicConfig(
    filename="kv_press_test/eval.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# Separate structured log for LLM judge
judge_log_path = "kv_press_test/judge_log.txt"
def write_judge_log(entry):
    with open(judge_log_path, "a") as f:
        f.write(entry + "\n")

# model = "Qwen/Qwen3-8B"
model = "ibm-granite/granite-3.3-8b-instruct"

obs_press = ObservedAttentionPress(compression_ratio=0.5)
zip_press = KVzipPress(compression_ratio=0.5)

with open("kv_press_test/questions.json") as f:
    data = json.load(f)

# Pass 1: ObservedAttentionPress
log.info("Loading ObservedAttentionPress pipeline...")
obs_pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto", model_kwargs={"attn_implementation": "eager"})
log.info("Pipeline loaded.")

# obs_answers[case_id][question] = answer
obs_answers = {}
for case in data:
    case_id = case["case_id"]
    context = open(f"kv_press_test/cases/{case_id}.txt").read()
    log.info(f"ObservedAttentionPress | Case: {case_id}")
    obs_answers[case_id] = {}
    for q in case["questions"]:
        obs_answers[case_id][q["question"]] = obs_pipe(context, question=q["question"], press=obs_press, max_new_tokens=512)["answer"]

del obs_pipe
torch.cuda.empty_cache()
log.info("ObservedAttentionPress pass complete.")

# Pass 2: KVzipPress
log.info("Loading KVzipPress pipeline...")
zip_pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto")
log.info("Pipeline loaded.")

results = []
for case in data:
    case_id = case["case_id"]
    context = open(f"kv_press_test/cases/{case_id}.txt").read()
    log.info("-" * 50)
    log.info(f"KVzipPress | Case: {case_id}")
    case_results = {"case_id": case_id, "questions": []}

    for q in case["questions"]:
        question = q["question"]
        ground_truth = q["ground_truth"]
        obs_answer = obs_answers[case_id][question]
        zip_answer = zip_pipe(context, question=question, press=zip_press, max_new_tokens=512)["answer"]

        log.info(f"Question: {question}")
        log.info(f"Ground truth: {ground_truth}")
        log.info(f"ObservedAttentionPress answer: {obs_answer}")
        log.info(f"KVzipPress answer: {zip_answer}")

        write_judge_log(f"""
========================================
CASE: {case_id}
QUESTION: {question}

GROUND TRUTH:
{ground_truth}

OBSERVED ATTENTION PRESS ANSWER:
{obs_answer}

KVZIP PRESS ANSWER:
{zip_answer}
========================================
""")

        case_results["questions"].append({
            "question": question,
            "ground_truth": ground_truth,
            "obs_answer": obs_answer,
            "zip_answer": zip_answer
        })

    results.append(case_results)

    with open("kv_press_test/results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved after case {case_id}")

log.info("Eval complete.")
=======
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
>>>>>>> granite4
