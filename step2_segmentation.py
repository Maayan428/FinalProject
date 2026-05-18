import pandas as pd
import os
import json
import re
from pathlib import Path

TRANSCRIPTIONS_DIR = "/Users/maayanturgeman/Documents/Final_Year/final_project/Transcriptions"
OUTPUT_DIR = "/Users/maayanturgeman/Documents/Final_Year/final_project"

TARGET_WORDS = 500      # יעד מילים לסגמנט
MIN_WORDS    = 300      # מינימום
MAX_WORDS    = 700      # מקסימום

# ============================================================
# פונקציה: טעינת קובץ תמלול אחד
# ============================================================
def load_transcript(filepath):
    """טוענת קובץ CSV של ראיון ומחזירה רק שורות של Participant"""
    try:
        df = pd.read_csv(filepath, sep='\t')
        # וידוא שעמודות קיימות
        required = {"speaker", "value", "start_time", "stop_time"}
        if not required.issubset(df.columns):
            print(f"  ⚠️  עמודות חסרות ב-{filepath.name}: {df.columns.tolist()}")
            return None

        # סינון רק Participant
        participant_df = df[df["speaker"] == "Participant"].copy()
        participant_df = participant_df.dropna(subset=["value"])
        participant_df["value"] = participant_df["value"].astype(str).str.strip()
        participant_df = participant_df[participant_df["value"] != ""]

        return participant_df

    except Exception as e:
        print(f"  ❌ שגיאה בטעינת {filepath.name}: {e}")
        return None


# ============================================================
# פונקציה: חלוקה לסגמנטים לפי מספר מילים
# ============================================================
def split_to_segments(participant_df, participant_id, target=TARGET_WORDS, min_w=MIN_WORDS, max_w=MAX_WORDS):
    """
    מחלקת את כל הטקסט של משתתף לסגמנטים של ~500 מילים.
    חותכת רק בין שורות (לא באמצע משפט).
    """
    segments = []
    current_lines = []
    current_word_count = 0
    segment_num = 1

    for _, row in participant_df.iterrows():
        utterance = str(row["value"]).strip()
        word_count = len(utterance.split())

        # אם הוספת השורה הזו תחרוג מ-MAX ויש כבר מספיק - חתוך
        if current_word_count + word_count > max_w and current_word_count >= min_w:
            segment_text = " ".join(current_lines)
            segments.append({
                "participant_id": participant_id,
                "interview_id": participant_id,
                "segment_num": segment_num,
                "word_count": current_word_count,
                "text": segment_text
            })
            segment_num += 1
            current_lines = [utterance]
            current_word_count = word_count
        else:
            current_lines.append(utterance)
            current_word_count += word_count

    # שמירת הסגמנט האחרון (גם אם קצר מ-MIN)
    if current_lines:
        segment_text = " ".join(current_lines)
        segments.append({
            "participant_id": participant_id,
            "interview_id": participant_id,
            "segment_num": segment_num,
            "word_count": current_word_count,
            "text": segment_text
        })

    return segments


# ============================================================
# פונקציה: מציאת קבצים ייחודיים (מתעלמת מכפילויות עם (1))
# ============================================================
def get_unique_transcripts(transcriptions_dir):
    """
    מחזירה רק קובץ אחד לכל משתתף.
    מעדיפה את הקובץ ללא (1) אם קיים.
    """
    all_files = list(Path(transcriptions_dir).glob("*.csv"))

    # בניית מילון: participant_id -> קובץ מועדף
    transcript_map = {}

    for f in all_files:
        name = f.name
        # חילוץ מזהה משתתף (מספר בתחילת שם הקובץ)
        match = re.match(r"^(\d+)_P__\d+_TRANSCRIPT", name)
        if not match:
            continue
        pid = int(match.group(1))

        is_duplicate = "(1)" in name

        if pid not in transcript_map:
            transcript_map[pid] = f
        else:
            # אם הקובץ הקיים הוא כפיל ואנחנו מצאנו קובץ נקי - נעדיף את הנקי
            existing_is_dup = "(1)" in transcript_map[pid].name
            if existing_is_dup and not is_duplicate:
                transcript_map[pid] = f

    return transcript_map


# ============================================================
# MAIN - ריצה ראשית
# ============================================================
def main():
    print("=" * 60)
    print("שלב 2 - חלוקת ראיונות לסגמנטים")
    print("=" * 60)

    # יצירת תיקיית output אם לא קיימת
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # מציאת קבצים ייחודיים
    transcript_map = get_unique_transcripts(TRANSCRIPTIONS_DIR)
    print(f"\nנמצאו {len(transcript_map)} משתתפים ייחודיים")

    all_segments = []
    stats = {"processed": 0, "skipped": 0, "total_segments": 0}

    for pid, filepath in sorted(transcript_map.items()):
        print(f"\n[{pid}] מעבד: {filepath.name}")

        # טעינה
        df = load_transcript(filepath)
        if df is None or len(df) == 0:
            print(f"  ⚠️  דילוג - אין נתוני Participant")
            stats["skipped"] += 1
            continue

        total_words = df["value"].apply(lambda x: len(str(x).split())).sum()
        print(f"  סה\"כ מילים של Participant: {total_words}")

        # חלוקה לסגמנטים
        segments = split_to_segments(df, pid)
        print(f"  → נוצרו {len(segments)} סגמנטים:")
        for seg in segments:
            print(f"     סגמנט {seg['segment_num']}: {seg['word_count']} מילים")

        all_segments.extend(segments)
        stats["processed"] += 1
        stats["total_segments"] += len(segments)

    # ============================================================
    # שמירת תוצאות
    # ============================================================
    print("\n" + "=" * 60)
    print("שומר תוצאות...")

    # CSV
    segments_df = pd.DataFrame(all_segments)
    csv_path = os.path.join(OUTPUT_DIR, "segments.csv")
    segments_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ נשמר: {csv_path}")

    # JSON (לנוחות בשלב 3)
    json_path = os.path.join(OUTPUT_DIR, "segments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"✅ נשמר: {json_path}")

    # סיכום
    print("\n" + "=" * 60)
    print("סיכום:")
    print(f"  משתתפים שעובדו:   {stats['processed']}")
    print(f"  משתתפים שדולגו:   {stats['skipped']}")
    print(f"  סה\"כ סגמנטים:     {stats['total_segments']}")
    if stats["processed"] > 0:
        avg = stats["total_segments"] / stats["processed"]
        print(f"  ממוצע סגמנטים:    {avg:.1f} למשתתף")

    # תצוגת 3 סגמנטים לדוגמה
    print("\n--- 3 סגמנטים לדוגמה ---")
    for seg in all_segments[:3]:
        print(f"\n[משתתף {seg['participant_id']} | סגמנט {seg['segment_num']} | {seg['word_count']} מילים]")
        print(seg["text"][:300] + "...")

    print("\n✅ שלב 2 הושלם!")
    return segments_df


if __name__ == "__main__":
    df = main()
