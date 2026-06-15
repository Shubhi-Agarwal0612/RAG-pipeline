import json
import os
from groq import Groq
from main import query  # reuse the app's query function

# ─────────────────────────────────────────────────────────────
# Judge prompts
# ─────────────────────────────────────────────────────────────

faithfulness_system_prompt = """You are a meticulous evaluator who judges whether an ANSWER is grounded in a set of reference CHUNKS.

Your job: go through every claim in the ANSWER and check whether it is supported by the CHUNKS. The ANSWER is faithful if every claim it makes is backed by the CHUNKS. Any claim that appears in the ANSWER but is not supported by the CHUNKS must be penalized — this includes invented facts, added statistics, or details not present in the CHUNKS. An ANSWER that uses only part of the CHUNKS is still fully faithful, as long as it invents nothing.

Score faithfulness on a scale from 1 to 5:
- 5: every claim in the ANSWER is fully supported by the CHUNKS
- 4: almost entirely grounded, with only trivial unsupported detail
- 3: mostly grounded, but contains at least one clear unsupported claim
- 2: several claims are unsupported by the CHUNKS
- 1: the ANSWER is largely fabricated or unsupported by the CHUNKS

Return ONLY a JSON object with exactly two keys:
- "score": an integer from 1 to 5
- "reason": a brief explanation of why you gave that score, naming any unsupported claims

Do not include any text outside the JSON object."""

correctness_system_prompt = """You are a meticulous evaluator who judges whether an ANSWER is factually correct compared to a reference EXPECTED ANSWER.

Your job: check whether the ANSWER conveys the same facts as the EXPECTED ANSWER. Every key point in the EXPECTED ANSWER should be present in the ANSWER, and the ANSWER should not state facts that contradict or are absent from the EXPECTED ANSWER. Judge on substance and facts, NOT on wording or phrasing — an ANSWER that is phrased completely differently but conveys the same facts is fully correct. Penalize the ANSWER both for missing key points from the EXPECTED ANSWER and for adding incorrect facts not supported by it.

Score correctness on a scale from 1 to 5:
- 5: all key facts from the EXPECTED ANSWER are present and accurate, with no incorrect additions
- 4: substantially correct, missing only a trivial detail or adding a harmless one
- 3: a key fact is missing or one clearly incorrect fact is present
- 2: several key facts are missing or incorrect
- 1: the ANSWER is mostly incorrect, unrelated, or contradicts the EXPECTED ANSWER

Return ONLY a JSON object with exactly two keys:
- "score": an integer from 1 to 5
- "reason": a brief explanation of why you gave that score, naming any missing or incorrect facts

Do not include any text outside the JSON object."""

# ─────────────────────────────────────────────────────────────
# Retrieval metrics: hit rate + reciprocal rank
# ─────────────────────────────────────────────────────────────

def calc_hit_rate_and_mrr(correct_pages, chunks):
    rank = 0
    rr = 0
    hit = False
    for i, dictionary in enumerate(chunks):
        list_chunk_id = dictionary["chunk_id"].split('_')
        chunk_page_no = int(list_chunk_id[1][1:])
        if chunk_page_no in correct_pages:
            hit = True
            rank = i + 1
            rr = 1 / rank
            return hit, rr
    return hit, rr

# ─────────────────────────────────────────────────────────────
# LLM-as-judge: faithfulness (answer grounded in chunks?)
# ─────────────────────────────────────────────────────────────

def judge_faithfulness(answer, chunks):
    if not chunks:
        return 0, "No chunks were retrieved, so faithfulness cannot be evaluated."

    chunks_text = "\n\n".join(chunk["text"] for chunk in chunks)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": faithfulness_system_prompt},
            {"role": "user", "content": f"CHUNKS:\n{chunks_text}\n\nANSWER:\n{answer}"},
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result["score"], result["reason"]

# ─────────────────────────────────────────────────────────────
# LLM-as-judge: correctness (answer matches expected answer?)
# ─────────────────────────────────────────────────────────────

def judge_correctness(answer, expected_answer):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": correctness_system_prompt},
            {"role": "user", "content": f"EXPECTED ANSWER:\n{expected_answer}\n\nANSWER:\n{answer}"},
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result["score"], result["reason"]

# ─────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────

document_ids = [8]  # the test PDF's id in your DB — confirm this value

with open("test_set.json") as f:
    test_set = json.load(f)

results = []

for item in test_set:
    question = item["question"]
    expected_answer = item["expected_answer"]

    # Normalize to a list of valid pages, whether the item uses
    # the single-page key or the multi-page key.
    if "source_pages" in item:
        correct_pages = item["source_pages"]
    else:
        correct_pages = [item["source_page"]]

    answer, chunks = query(question, document_ids)
    hit, rr = calc_hit_rate_and_mrr(correct_pages, chunks)
    faith_score, faith_reason = judge_faithfulness(answer, chunks)
    corr_score, corr_reason = judge_correctness(answer, expected_answer)

    result = {
        "question": question,
        "answer": answer,
        "hit": hit,
        "rr": rr,
        "faith_score": faith_score,
        "faith_reason": faith_reason,
        "corr_score": corr_score,
        "corr_reason": corr_reason,
    }
    results.append(result)
    print(f"Done: {question[:60]}...")

# ─────────────────────────────────────────────────────────────
# Aggregate into the four headline metrics
# ─────────────────────────────────────────────────────────────

n = len(results)

hit_rate = sum(r["hit"] for r in results) / n
mrr = sum(r["rr"] for r in results) / n
# normalize the 1–5 judge scores to 0–1 so all metrics share a scale
avg_faithfulness = sum(r["faith_score"] for r in results) / n / 5
avg_correctness = sum(r["corr_score"] for r in results) / n / 5

# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PER-QUESTION RESULTS")
print("=" * 70)
for i, r in enumerate(results, 1):
    print(f"\nQ{i}: {r['question']}")
    print(f"  Hit: {r['hit']}  |  RR: {r['rr']:.3f}  |  "
          f"Faith: {r['faith_score']}/5  |  Corr: {r['corr_score']}/5")
    print(f"  Faithfulness reason: {r['faith_reason']}")
    print(f"  Correctness reason:  {r['corr_reason']}")

print("\n" + "=" * 70)
print("AGGREGATE METRICS")
print("=" * 70)
print(f"Hit Rate:      {hit_rate:.3f}")
print(f"MRR:           {mrr:.3f}")
print(f"Faithfulness:  {avg_faithfulness:.3f}")
print(f"Correctness:   {avg_correctness:.3f}")

# save full results for later comparison against the reranked run
with open("eval_results_baseline.json", "w") as f:
    json.dump(results, f, indent=2)