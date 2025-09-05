import io, math, os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import fitz # pyMuPDF
from PIL import Image
import open_clip


# preprocessing + clip scoring class
class ClipScorer:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        @torch.no_grad()
        def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
            batch = torch.stack([self.preprocess(img).to(self.device) for img in images], dim=0)
            img_feat = self.model.encode_image(batch)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            return img_feat
        
        @torch.no_grad()
        def embed_texts(self, texts: List[str]) -> torch.Tensor:
            tokens = self.tokenizer(texts).to(self.device)
            txt_feat = self.model.encode_text(tokens)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            return txt_feat

        @torch.no_grad()
        def cosine_sim(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
            # returns [n_images, n_texts] similarity matrix
            return img_emb @ txt_emb.T
        
# utils
def pdf_to_page_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    # convert each page from pdf to pil
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img.convert("RGB"))
    return images

def chunk_text_for_clip(text: str, max_words: int = 60) -> List[str]:
    # split text into chunks of max_words each, take max similarity
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def pagewise_text(ocr_text: str, n_pages: int) -> List[str]:
    # heuristic to split ocr text into n_pages part by form feed markers / split evenly by chars
    candidates = []
    markers = [s for s in ocr_text.split("\f") if s.strip()]
    if len(markers) == n_pages:
        return [m.strip() for m in markers]
    
    # even char split
    L = len(ocr_text)
    if L == 0:
        return [""] * n_pages
    step = math.ceil(L / max(1, n_pages))
    for i in range(n_pages):
        seg = ocr_text[i*step:(i+1)*step].strip()
        candidates.append(seg)
    return candidates

def clip_score_document(pdf_path: str, ocr_text_by_engine: Dict[str, str], model_name: str = "ViT-L-14", pretrained: str = "openai", dpi: int = 200, low_quality_threshlold: float = 0.18, topk_text_chunks: int = 3,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # calculates the clip scores for both the page and entire document
    scorer = ClipScorer(model_name=model_name, pretrained=pretrained)
    page_images = pdf_to_page_images(pdf_path, dpi=dpi)
    img_emb = scorer.embed_images(page_images) # [P, D]

    rows = []
    doc_name = os.path.basename(pdf_path)

    for engine, full_text in ocr_text_by_engine.items():
        per_page_texts = pagewise_text(full_text or "", n_pages=len(page_images))
        for p_idx, (img_vec, text) in enumerate(zip(img_emb, per_page_texts)):
            # chunk page long text
            chunks = chunk_text_for_clip(text)
            sims_all = []
            # squeeze into single image vector @ B texts
            for j in range(0, len(chunks), 32):
                batch = chunks[j:j+32]
                txt_emb = scorer.embed_texts(batch) # [B, D]
                sim = (img_vec.unsqueeze(0) @ txt_emb.T).squeeze(0).tolist()
                sims_all.extend(sim)
            # aggregate / average top-k similarities
            sims_sorted = sorted(sims_all, reverse=True)
            k = min(topk_text_chunks, len(sims_sorted))
            page_score = float(np.mean(sims_sorted[:k])) if k > 0 else 0.0
            rows.append({
                "doc": doc_name,
                "page": p_idx + 1,
                "engine": engine,
                "clip_score": page_score, 
                "low_quality": page_score < low_quality_threshlold
            })

    page_scores = pd.DataFrame(rows)
    agg = page_scores.groupby(["doc", "engine"])["clip_score"]
    doc_scores = pd.DataFrame({
        "clip_mean": agg.mean(),
        "clip_median": agg.median(),
        "clip_p25": agg.quantile(0.25),
        "clip_p10": agg.quantile(0.10),
        "low_quality_rate": page_scores.groupby(["doc", "engine"])["low_quality_flag"].mean()
    }).reset_index()

    # pick best scores
    best = (
        doc_scores.sort_values(["doc", "clip_mean"], ascending=[True, False])
        .groupby("doc")
        .head(1)
        .rename(columns={"engine": "best_engine", "clip_mean": "best_clip_mean"})[["doc", "best_engine", "best_clip_mean"]]
    )
    doc_scores = doc_scores.merge(best, on="doc", how="left")
    return page_scores, doc_scores
            
