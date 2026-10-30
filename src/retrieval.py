"""CLIP-style vision-language retrieval for radiology images (educational demo).

Uses a pretrained CLIP model through the ``open_clip`` package. The model is
*not* trained here and is *not* a medical device: it is a general
vision-language model used to demonstrate text-to-image search mechanics.

First run downloads the pretrained weights from the internet (cached by
open_clip afterwards). Everything runs on CPU unless CUDA is available.
"""

from typing import List, Sequence, Tuple

import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

_model = None
_preprocess = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load the CLIP model, image preprocessor and tokenizer (cached).

    Downloads pretrained weights on first run; subsequent runs reuse the
    open_clip cache. Raises ImportError with install instructions if
    ``open_clip`` is missing.
    """
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return _model, _preprocess, _tokenizer
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip is required for retrieval. Install it with:\n"
            "    pip install open-clip-torch"
        ) from exc
    print(f"Loading CLIP model '{MODEL_NAME}' (weights: {PRETRAINED}) ...")
    print(
        "Note: pretrained weights are downloaded from the internet on first run "
        "and cached locally by open_clip afterwards."
    )
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    _model.to(_device).eval()
    return _model, _preprocess, _tokenizer


@torch.no_grad()
def embed_images(pil_images: Sequence[Image.Image]) -> torch.Tensor:
    """Embed PIL images into L2-normalized CLIP vectors, shape (N, D)."""
    model, preprocess, _ = load_model()
    tensors = torch.stack([preprocess(img.convert("RGB")) for img in pil_images])
    feats = model.encode_image(tensors.to(_device))
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed_texts(texts: List[str]) -> torch.Tensor:
    """Embed text strings into L2-normalized CLIP vectors, shape (M, D)."""
    model, _, tokenizer = load_model()
    tokens = tokenizer(texts).to(_device)
    feats = model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True)


def rank_by_cosine(
    query_emb: torch.Tensor, image_embs: torch.Tensor, top_k: int = 5
) -> Tuple[List[int], List[float]]:
    """Rank precomputed image embeddings by cosine similarity to a query.

    Pure tensor math — no model weights needed, so it is unit-testable.
    Returns (ranked_indices, cosine_scores), best match first.
    """
    sims = (image_embs @ query_emb.T).squeeze(-1)
    k = max(1, min(top_k, sims.numel()))
    scores, idx = torch.topk(sims, k)
    return idx.cpu().tolist(), scores.cpu().tolist()


def text_to_image_search(
    query: str, pil_images: Sequence[Image.Image], top_k: int = 5
) -> Tuple[List[int], List[float]]:
    """Rank images by relevance to a free-text query.

    Returns (ranked_indices, cosine_scores). Requires model weights
    (downloaded on first run via :func:`load_model`).
    """
    if not pil_images:
        raise ValueError("pil_images must not be empty")
    query_emb = embed_texts([query])      # (1, D), L2-normalized
    image_embs = embed_images(pil_images)  # (N, D), L2-normalized
    return rank_by_cosine(query_emb, image_embs, top_k=top_k)
