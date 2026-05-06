"""Project configuration and local model registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes" / "faiss"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class LocalModel:
    key: str
    role: str
    path: Path
    backend: str
    status: str
    notes: str

    @property
    def exists(self) -> bool:
        return self.path.exists()


LOCAL_MODELS: dict[str, LocalModel] = {
    "gemma4_e4b_it_q4km": LocalModel(
        key="gemma4_e4b_it_q4km",
        role="primary_specialist_generator",
        path=MODEL_DIR / "gemma-4-E4B-it-Q4_K_M.gguf",
        backend="llama.cpp",
        status="present",
        notes="Primary local GGUF answerer for grounded troubleshooting.",
    ),
    "gemma4_26b_a4b_it_q4km": LocalModel(
        key="gemma4_26b_a4b_it_q4km",
        role="heavy_reasoning_fallback",
        path=MODEL_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        backend="llama.cpp",
        status="present",
        notes="Heavy local fallback; use only after benchmark and confidence trigger.",
    ),
    "qwen25_vl_7b_q4km": LocalModel(
        key="qwen25_vl_7b_q4km",
        role="multimodal_vision_challenger",
        path=MODEL_DIR / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        backend="llama.cpp",
        status="present",
        notes="Vision-language GGUF for screenshot/photo experiments.",
    ),
    "qwen25_vl_mmproj": LocalModel(
        key="qwen25_vl_mmproj",
        role="multimodal_projector",
        path=MODEL_DIR / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf",
        backend="llama.cpp",
        status="present",
        notes="Projector required for Qwen2.5-VL GGUF image input.",
    ),
    "embeddinggemma_300m_bf16": LocalModel(
        key="embeddinggemma_300m_bf16",
        role="embedding_baseline",
        path=MODEL_DIR / "embeddinggemma-300M-BF16.gguf",
        backend="llama.cpp",
        status="present",
        notes="Local embedding baseline for retrieval experiments.",
    ),
    "bge_small_en_v15": LocalModel(
        key="bge_small_en_v15",
        role="small_cpu_embedding_baseline",
        path=MODEL_DIR / "bge-small-en-v1.5",
        backend="sentence-transformers",
        status="target",
        notes="Fast baseline embeddings for retrieval quality and FAISS smoke tests.",
    ),
    "qwen3_embedding_06b": LocalModel(
        key="qwen3_embedding_06b",
        role="strong_embedding_challenger",
        path=MODEL_DIR / "qwen3-embedding-0.6b",
        backend="sentence-transformers",
        status="target",
        notes="Strong retrieval challenger with 32k context and instruction-aware embeddings.",
    ),
    "clip_vit_base_patch32": LocalModel(
        key="clip_vit_base_patch32",
        role="image_retrieval_baseline",
        path=MODEL_DIR / "clip-vit-base-patch32",
        backend="transformers",
        status="present",
        notes="Image retrieval baseline for hardware/photo matching.",
    ),
    "paddleocr_vl": LocalModel(
        key="paddleocr_vl",
        role="ocr_document_vision",
        path=MODEL_DIR / "paddleocr-vl",
        backend="transformers",
        status="present",
        notes="OCR and document extraction experiments.",
    ),
    "dinov2_base": LocalModel(
        key="dinov2_base",
        role="image_feature_experiment",
        path=MODEL_DIR / "dinov2-base",
        backend="transformers",
        status="present",
        notes="Image feature extraction experiments.",
    ),
    "layoutlmv3_base": LocalModel(
        key="layoutlmv3_base",
        role="document_layout_experiment",
        path=MODEL_DIR / "layoutlmv3-base",
        backend="transformers",
        status="present",
        notes="Document layout extraction experiments.",
    ),
    "gemma4_e2b_it": LocalModel(
        key="gemma4_e2b_it",
        role="primary_triage_classifier_base",
        path=MODEL_DIR / "gemma-4-E2B-it",
        backend="transformers",
        status="target",
        notes="Primary QLoRA classifier base for IT support multi-label triage.",
    ),
    "qwen35_4b": LocalModel(
        key="qwen35_4b",
        role="classifier_challenger",
        path=MODEL_DIR / "qwen3.5-4b",
        backend="transformers",
        status="target",
        notes="Classifier challenger against Gemma E2B.",
    ),
    "qwen35_9b": LocalModel(
        key="qwen35_9b",
        role="specialist_heavy_challenger",
        path=MODEL_DIR / "qwen3.5-9b",
        backend="transformers",
        status="target",
        notes="Heavy challenger only after the current ladder is benchmarked.",
    ),
    "ministral3_3b_instruct_q4km": LocalModel(
        key="ministral3_3b_instruct_q4km",
        role="tiny_fast_fallback_demo_model",
        path=MODEL_DIR / "ministral-3-3b-instruct-2512-gguf",
        backend="llama.cpp",
        status="target",
        notes="Fast GGUF fallback and demo model for routing and latency sanity checks.",
    ),
    "qwen_image_edit_2509": LocalModel(
        key="qwen_image_edit_2509",
        role="non_core_image_editing_asset",
        path=MODEL_DIR / "qwen-image-edit-2509",
        backend="diffusers",
        status="present",
        notes="Large image editing asset; not part of the IT-SUPPORT core runtime.",
    ),
    "z_image_turbo": LocalModel(
        key="z_image_turbo",
        role="non_core_image_generation_asset",
        path=MODEL_DIR / "z-image-turbo",
        backend="diffusers",
        status="present",
        notes="Large image generation asset; not part of the IT-SUPPORT core runtime.",
    ),
    "vjepa2_vitl_fpc64_256": LocalModel(
        key="vjepa2_vitl_fpc64_256",
        role="video_image_representation_experiment",
        path=MODEL_DIR / "vjepa2-vitl-fpc64-256",
        backend="transformers",
        status="present",
        notes="Stretch asset for video/image representation experiments.",
    ),
    "sd35_medium": LocalModel(
        key="sd35_medium",
        role="non_core_image_generation_asset",
        path=MODEL_DIR / "sd3.5-medium",
        backend="diffusers",
        status="verify",
        notes="Appears incomplete or metadata-only from current size check; verify before use.",
    ),
}


TARGET_DOWNLOADS = {
    key: model.notes
    for key, model in LOCAL_MODELS.items()
    if model.status == "target" and not model.exists
}
