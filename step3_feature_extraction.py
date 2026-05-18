import pandas as pd
import json
import time
import os
import re
from groq import Groq

# ============================================================
# CONFIGURATION
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")
INPUT_FILE   = "/Users/maayanturgeman/Documents/Final_Year/final_project/segments.csv"
OUTPUT_DIR   = "/Users/maayanturgeman/Documents/Final_Year/final_project"
MODEL        = "llama-3.3-70b-versatile"
SLEEP_BETWEEN_REQUESTS = 2.5

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a clinical psychology expert analyzing interview transcripts.
Your task is to rate psychological features based ONLY on what is explicitly stated or clearly implied in the text.
Always respond in valid JSON format exactly as instructed. Do not add any text outside the JSON."""

def build_user_prompt(segment_text):
    return f"""Analyze the following interview segment spoken by a participant and rate each of the 5 psychological features below.

SEGMENT TEXT:
{segment_text}

RATING INSTRUCTIONS:
For each feature, provide:
- "score": an integer 0-3, OR the string "cannot_determine" if there is insufficient information
- "explanation": a brief explanation (1-2 sentences) quoting or referencing specific words from the text

FEATURES TO RATE:

1. depressed_mood (PHQ feature):
   Rate how much the participant expresses sadness, depression, emptiness, or hopelessness.
   0 = No signs of depressed mood
   1 = Mild or implied signs
   2 = Clear expressions of depressed mood
   3 = Strong, explicit, or repeated expressions of depression/hopelessness

2. appetite_changes (PHQ feature):
   Rate whether the participant describes changes in appetite or eating habits (increase or decrease).
   0 = No mention of appetite or eating changes
   1 = Mild or indirect mention of appetite change
   2 = Clear description of appetite change
   3 = Strong, detailed, or repeated description of significant eating changes
   Use "cannot_determine" only if eating/appetite topic is completely absent.

3. guilt_worthlessness (PHQ feature):
   Rate whether the participant expresses guilt, worthlessness, self-blame, or feelings of failure.
   0 = No signs of guilt or worthlessness
   1 = Mild or implied signs
   2 = Clear expressions of guilt or worthlessness
   3 = Strong, explicit, or repeated expressions of guilt/failure/worthlessness

4. self_focus (Psychological feature):
   Rate the degree to which the segment focuses on the speaker's own thoughts, feelings, and experiences.
   0 = Almost no self-focus (text is mostly about others or external events)
   1 = Low self-focus (some self-reference but not dominant)
   2 = Moderate self-focus (clear presence of self-reference)
   3 = High dominant self-focus (most of the segment is about the speaker)

5. hopelessness (Psychological feature):
   Rate how pessimistically the participant speaks about the future.
   0 = No expression of hopelessness (neutral or positive view of future)
   1 = Weak or implied pessimism
   2 = Clear expression of hopelessness or lack of hope
   3 = Strong, explicit, or persistent hopelessness about the future
   Use "cannot_determine" only if the future is never mentioned at all.

You MUST respond with ONLY this JSON and nothing else:
{{
  "depressed_mood": {{"score": <0-3 or "cannot_determine">, "explanation": "<text>"}},
  "appetite_changes": {{"score": <0-3 or "cannot_determine">, "explanation": "<text>"}},
  "guilt_worthlessness": {{"score": <0-3 or "cannot_determine">, "explanation": "<text>"}},
  "self_focus": {{"score": <0-3 or "cannot_determine">, "explanation": "<text>"}},
  "hopelessness": {{"score": <0-3 or "cannot_determine">, "explanation": "<text>"}}
}}"""

def parse_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{[\s\S]*\}', response_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

def extract_scores(parsed, participant_id, interview_id, segment_num):
    features = ["depressed_mood", "appetite_changes", "guilt_worthlessness",
                "self_focus", "hopelessness"]
    row = {
        "participant_id": participant_id,
        "interview_id":   interview_id,
        "segment_num":    segment_num,
    }
    for feat in features:
        if parsed and feat in parsed:
            row[f"{feat}_score"]       = parsed[feat].get("score", "parse_error")
            row[f"{feat}_explanation"] = parsed[feat].get("explanation", "")
        else:
            row[f"{feat}_score"]       = "parse_error"
            row[f"{feat}_explanation"] = ""
    return row

def main():
    print("=" * 60)
    print("Step 3 - Feature Extraction with Groq API")
    print("=" * 60)

    segments_df = pd.read_csv(INPUT_FILE)
    total = len(segments_df)
    print(f"Loaded {total} segments for {segments_df['participant_id'].nunique()} participants\n")

    progress_file = os.path.join(OUTPUT_DIR, "step3_progress.json")
    results = []
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    # Segments scored successfully (no parse_error)
    completed_ok = {
        (r["participant_id"], r["segment_num"])
        for r in results
        if r.get("depressed_mood_score") not in ["parse_error", ""]
    }

    # Remove parse_errors so we retry them
    results = [
        r for r in results
        if r.get("depressed_mood_score") not in ["parse_error", ""]
    ]

    all_attempted = {
        (int(row["participant_id"]), int(row["segment_num"]))
        for _, row in segments_df.iterrows()
    }
    parse_errors_to_retry = all_attempted - completed_ok - {
        (int(segments_df.loc[i, "participant_id"]), int(segments_df.loc[i, "segment_num"]))
        for i in segments_df.index
    }

    print(f"Successfully scored: {len(completed_ok)} segments")
    print(f"Remaining (new + retries): {total - len(completed_ok)} segments\n")

    errors = []

    for idx, row in segments_df.iterrows():
        pid     = int(row["participant_id"])
        iid     = int(row["interview_id"])
        seg_num = int(row["segment_num"])
        text    = str(row["text"])
        key     = (pid, seg_num)

        if key in completed_ok:
            continue

        is_retry = key in {
            (r["participant_id"], r["segment_num"])
            for r in results
        }
        label = " [RETRY]" if is_retry else ""
        print(f"[{idx+1}/{total}] Participant {pid} | Segment {seg_num} | {row['word_count']} words{label}")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(text)}
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content.strip()
            parsed = parse_response(response_text)

            if parsed:
                result_row = extract_scores(parsed, pid, iid, seg_num)
                results.append(result_row)
                scores = {f: parsed[f]["score"] for f in
                          ["depressed_mood","appetite_changes","guilt_worthlessness",
                           "self_focus","hopelessness"] if f in parsed}
                print(f"   Scores: {scores}")
            else:
                print(f"   WARNING: Could not parse response")
                errors.append({"participant_id": pid, "segment_num": seg_num,
                               "raw_response": response_text})
                result_row = extract_scores(None, pid, iid, seg_num)
                results.append(result_row)

        except Exception as e:
            err_msg = str(e)
            print(f"   ERROR: {err_msg[:120]}")
            if "429" in err_msg:
                print("\n   Daily rate limit reached. Run again tomorrow - progress is saved!")
                break
            errors.append({"participant_id": pid, "segment_num": seg_num, "error": err_msg})
            time.sleep(10)
            continue

        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Save final CSV + JSON
    print("\n" + "=" * 60)
    results_df = pd.DataFrame(results)

    csv_path = os.path.join(OUTPUT_DIR, "step3_scores.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {csv_path}")

    json_path = os.path.join(OUTPUT_DIR, "step3_scores.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {json_path}")

    # Summary for all 5 features
    print("\nSCORE SUMMARY (all 5 features):")
    features = ["depressed_mood", "appetite_changes", "guilt_worthlessness",
                "self_focus", "hopelessness"]
    for feat in features:
        col = f"{feat}_score"
        if col in results_df.columns:
            numeric = pd.to_numeric(results_df[col], errors="coerce")
            n_cd  = results_df[col].eq("cannot_determine").sum()
            n_err = results_df[col].eq("parse_error").sum()
            print(f"  {feat:25s} mean={numeric.mean():.2f} | "
                  f"cannot_determine={n_cd} | parse_error={n_err}")

    ok = results_df["depressed_mood_score"].apply(
        lambda x: str(x) not in ["parse_error", ""]
    ).sum()
    print(f"\nTotal successfully scored: {ok} / {total} segments")
    print("Step 3 complete!")
    return results_df

if __name__ == "__main__":
    df = main()