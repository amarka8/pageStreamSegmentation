import pandas as pd
import re
from difflib import SequenceMatcher
from dateutil.parser import parse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import unicodedata

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------- Normalization Functions -------------------
def normalize_language(lang):
    lang = lang.lower().strip()
    lang_map = {"english": "en", "eng": "en", "en": "en", "english (us)": "en"}
    return lang_map.get(lang, lang)

def normalize_date(date_str):
    try:
        return parse(date_str, fuzzy=True).date().isoformat()
    except:
        return date_str.strip().lower()

def normalize_format_genre(val):
    genre_map = {
        "text": "text",
        "printed book": "book",
        "book": "book",
        "journal": "journal",
        "memo": "memorandum",
        "memorandum": "memorandum",
        "questionnaire": "questionnaire",
        "questionares": "questionnaire",
        "draft": "draft",
        "correspondence": "correspondence",
        "letter": "correspondence",
        "email": "correspondence",
        "agenda": "agenda",
        "meeting agenda": "agenda",
        "testimony": "testimony",
        "summary": "summary",
        "biography": "biography",
        "biographies": "biography",
        "list": "list",
        "note": "note",
        "notes": "note",
        "photograph": "photograph",
        "photo": "photograph",
        "image": "photograph",
        "document": "document",
        "report": "report",
        "transcript": "transcript",
        "presentation": "presentation",
        "speech": "speech",
        "form": "form",
        "proposal": "proposal",
        "policy document": "policy document",
        "executive order": "executive order",
        "press release": "press release",
        "talking points": "talking points",
        "meeting minutes": "meeting minutes",
        "lecture series": "lecture",
        "course description": "lecture",
        "budget document": "budget document",
        "outline": "outline",
        "routing slip": "routing slip",
        "event": "event",
        "ceremony": "event",
        "fax": "fax"
    }

    # Split on delimiters and normalize each part
    parts = re.split(r'[|;,/]', str(val).lower())
    normalized = [genre_map.get(part.strip(), part.strip()) for part in parts if part.strip()]

    # Deduplicate and sort for consistent comparison
    return " | ".join(sorted(set(normalized)))


def normalize_entity(e):
    return re.sub(r"[^\w\s]", "", e.lower()).strip()

# ------------------- Matching Functions -------------------
def clean_entities(entity_list):
    if isinstance(entity_list, str):
        entity_list = [entity_list]
    
    PAREN_REGEX = re.compile(r"\(.*?\)")
    SPACE_REGEX = re.compile(r"\s+")
    
    def normalize_entity_string(entity_string):
        raw_entities = entity_string.split("|")
        cleaned = []
        
        for raw in raw_entities:
            e = raw.strip()
            
            # Handle "Last, First" only if no conjunctions
            if ',' in e and ' and ' not in e:
                parts = [p.strip() for p in e.split(',')]
                if len(parts) >= 2:
                    e = f"{' '.join(parts[1:])} {parts[0]}"
            
            e = PAREN_REGEX.sub("", e)
            e = SPACE_REGEX.sub(" ", e).strip().lower()
            e = unicodedata.normalize('NFKD', e).encode('ascii', 'ignore').decode('ascii')
            
            if e:
                cleaned.append(e)
        return cleaned
    
    return [
        normalized
        for entry in entity_list
        if isinstance(entry, str) and entry.strip()
        for normalized in normalize_entity_string(entry)
    ]

def entity_recall_match(gt_list, bm_list, recall_threshold=0.70, fuzzy_threshold=60):
    gt_clean = clean_entities(gt_list)
    bm_clean = clean_entities(bm_list)

    if not gt_clean:
        return False
    
    match_count = 0
    for gt_entity in gt_clean:
        # Check if any benchmark entity matches this ground truth entity
        for bm_entity in bm_clean:
            if fuzz.token_set_ratio(gt_entity, bm_entity) >= fuzzy_threshold:
                match_count += 1
                break

    recall = match_count / len(gt_clean)
    return recall >= recall_threshold

def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))

def token_recall_match(gt, bm, threshold=0.70):
    tokens_gt = tokenize(gt)
    tokens_bm = tokenize(bm)
    if not tokens_gt:
        return False
    recall = len(tokens_gt & tokens_bm) / len(tokens_gt)
    return recall >= threshold

def cosine_match(gt, bm, threshold=0.70):
    if not gt.strip() or not bm.strip():
        return False
    emb_gt = model.encode(gt, convert_to_tensor=True)
    emb_bm = model.encode(bm, convert_to_tensor=True)
    score = cosine_similarity(emb_gt.cpu().numpy().reshape(1, -1),
                              emb_bm.cpu().numpy().reshape(1, -1))[0][0]
    return score >= threshold

def fuzzy_match(gt, bm, threshold=0.70):
    if not gt.strip() or not bm.strip():
        return False
    ratio = SequenceMatcher(None, gt.lower(), bm.lower()).ratio()
    return ratio >= threshold

# ------------------- Field-Based Matching Dispatcher -------------------
def match_field(gt_val, bm_val, field):
    if field in ["Title", "Description", "Subject", "People and Organizations", "Format Genre"]:
        return entity_recall_match(gt_val, bm_val, recall_threshold=0.70, fuzzy_threshold=60)
    elif field == "Language":
        return normalize_language(gt_val) == normalize_language(bm_val)
    elif field == "Date":
        return normalize_date(gt_val) == normalize_date(bm_val)
    elif field in ["Publisher", "Accessibility Summary"]:
        if not gt_val.strip() and bm_val.strip() == "Null":
            return True
        else:
            return entity_recall_match(gt_val, bm_val, recall_threshold=0.70, fuzzy_threshold=70)
    else:
        return gt_val.strip().lower() == bm_val.strip().lower()

# ------------------- Main Comparison Function -------------------
def compare_csvs_custom(gt_df, benchmark_df, columns_to_compare,
                        gt_id_col="Quartex Name", bm_id_col="Unique_ID"):

    merged_df = pd.merge(gt_df, benchmark_df, left_on=gt_id_col, right_on=bm_id_col, suffixes=('_gt', '_bm'))
    results = []

    for col in columns_to_compare:
        gt_col_name = f"{col}_gt"
        bm_col_name = f"{col}_bm"

        if gt_col_name not in merged_df.columns or bm_col_name not in merged_df.columns:
            results.append({"Column": col, "Match %": "Column missing", "Matches": 0, "Total": 0})
            continue

        gt_col = merged_df[gt_col_name].fillna("").astype(str)
        bm_col = merged_df[bm_col_name].fillna("").astype(str)

        matches = sum(match_field(gt_val, bm_val, col) for gt_val, bm_val in zip(gt_col, bm_col))
        total = len(gt_col)
        match_percent = round(matches / total * 100, 2) if total > 0 else 0

        results.append({"Column": col, "Match %": match_percent, "Matches": matches, "Total": total})

    return pd.DataFrame(results)

# ------------------- Main Execution -------------------
if __name__ == "__main__":
    ground_truth_df = pd.read_csv("ground_truth.csv")
    benchmark_df = pd.read_csv("mistral_gemini_benchmark.csv")

    ground_truth_df.columns = ground_truth_df.columns.str.strip()
    benchmark_df.columns = benchmark_df.columns.str.strip()

    columns = [
        "Title",
        "People and Organizations",
        "Description",
        "Language",
        "Publisher",
        "Date",
        "Subject",
        "Format Genre",
        "Accessibility Summary"
    ]

    results_df = compare_csvs_custom(
        ground_truth_df,
        benchmark_df,
        columns,
        gt_id_col="Quartex Name",
        bm_id_col="Unique_ID"
    )

    print(results_df.to_string(index=False))
