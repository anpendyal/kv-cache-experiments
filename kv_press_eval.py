import sys
sys.path.insert(0, "/dccstor/nathan-ckpts/anooshka/kvpress")
import json
import logging
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

log.info("Loading pipelines...")
obs_pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto", model_kwargs={"attn_implementation": "eager"})
zip_pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto")
log.info("Pipelines loaded.")

obs_press = ObservedAttentionPress(compression_ratio=0.5)
zip_press = KVzipPress(compression_ratio=0.5)

with open("kv_press_test/questions.json") as f:
    data = json.load(f)

results = []
for case in data:
    case_id = case["case_id"]
    context = open(f"kv_press_test/cases/{case_id}.txt").read()
    log.info("-" * 50)
    log.info(f"Case: {case_id}")
    case_results = {"case_id": case_id, "questions": []}

    for q in case["questions"]:
        question = q["question"]
        ground_truth = q["ground_truth"]

        obs_answer = obs_pipe(context, question=question, press=obs_press)["answer"]
        zip_answer = zip_pipe(context, question=question, press=zip_press)["answer"]

        log.info(f"Question: {question}")
        log.info(f"Ground truth: {ground_truth}")
        log.info(f"ObservedAttentionPress answer: {obs_answer}")
        log.info(f"KVzipPress answer: {zip_answer}")

        # Write structured entry for LLM judge
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