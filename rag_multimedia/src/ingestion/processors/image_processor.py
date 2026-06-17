# -*- coding: utf-8 -*-
"""
Module image_processor — Traitement et enrichissement d'images pour la RAG multimodale.

Rôle dans l'architecture :
    Troisième étape du pipeline d'ingestion image. Charge les JPEG bruts depuis
    data/raw/images/{source}/, applique un pipeline en cinq étapes (validation,
    resize, caption, tags zero-shot, embedding), filtre les images invalides et
    persiste les ProcessedImage dans data/processed/images/ pour indexation dans
    ChromaDB (image_collection_optimized) via MultimodalEmbedder.

Pipeline de traitement (ordre obligatoire) :
    1. Validation (filtrage < 100×100px, images corrompues)
    2. Resize + letterbox 224×224 (Pillow thumbnail + padding blanc)
    3. Caption textuelle (BLIP-2 si caption absente, metadata sinon)
    4. Tags zero-shot (CLIP cosinus image vs textes de catégories)
    5. Embedding CLIP ViT-B/32 (512-dim float32 via sentence-transformers)

Pourquoi 224×224 :
    ViT-B/32 (Vision Transformer, patch 32×32 px) attend une entrée de taille
    224×224 fixe — cette valeur est un paramètre architectural du modèle
    (Dosovitskiy et al. 2020), pas un paramètre configurable à l'inférence.
    Passer une autre taille lève une erreur de forme dans le TransformerEncoder.
    Le letterboxing (padding blanc) préserve le ratio d'aspect pour éviter la
    déformation qui dégraderait la représentation vectorielle.

Pourquoi CLIP (ViT-B/32) via sentence-transformers :
    CLIP (Radford et al. 2021) est l'unique modèle open-source produisant un
    espace vectoriel commun image + texte de dimension 512. Cela permet la
    retrieval cross-modale : requête textuelle → images pertinentes (et vice-versa).
    sentence-transformers/clip-ViT-B-32 offre une API unifiée encode(images+textes)
    et gère le batching GPU automatiquement.
    Rejeté : ResNet50 (espace image seul, pas de texte) ;
             DINO (auto-supervisé, pas d'alignement texte-image).

Pourquoi BLIP-2 (Salesforce/blip2-opt-2.7b) :
    BLIP-2 (Li et al. 2023) est l'état de l'art en captioning zero-shot open-source.
    Il génère des descriptions textuelles naturelles depuis des images sans annotation
    humaine, enrichissant la retrieval textuelle sur le corpus image.
    La variante opt-2.7b (2.7 G paramètres) est le meilleur compromis qualité/VRAM
    (≈ 6 Go en float16 sur GPU) pour un usage portfolio local.
    Rejeté : BLIP-1 (qualité inférieure sur textes complexes) ;
             LLaVA-13B (> 26 Go VRAM, incompatible GPU grand public).
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# CAS 2 — Import conditionnel sentence-transformers (CLIP) : charge torch et les
# poids (~600 Mo) au premier import. Un ImportError ne doit bloquer que la
# fonctionnalité CLIP, pas l'ensemble du module.
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

# CAS 2 — Import conditionnel transformers + torch (BLIP-2) : le modèle pèse ≈ 5 Go
# et nécessite GPU pour être utilisable en production. Séparé de CLIP car BLIP-2
# est optionnel (inutile quand une caption existe déjà dans les métadonnées).
try:
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    _BLIP2_AVAILABLE = True
except ImportError:
    _BLIP2_AVAILABLE = False


# CAS 1 — Catégories zero-shot CLIP : liste de labels génériques couvrant les
# thématiques principales des sources (COCO, Unsplash, WikiMedia).
# Configurable via config.yaml image.zero_shot_categories pour extension sans
# modification du code. 20 catégories = compromis couverture/temps d'inférence.
_DEFAULT_ZERO_SHOT_CATEGORIES: List[str] = [
    "person", "animal", "vehicle", "food", "building", "nature", "technology",
    "sport", "art", "text document", "indoor scene", "outdoor scene",
    "water", "sky", "plant", "furniture", "clothing", "music instrument",
    "medical", "scientific diagram",
]

# CAS 3 — Seuil cosinus 0.20 pour les tags zero-shot : en dessous, la similarité
# CLIP image-texte est trop faible pour être considérée significative
# (Radford et al. 2021, Fig. 4 — accuracy chute sous 0.20 de confiance).
_ZERO_SHOT_THRESHOLD: float = 0.20

# CAS 3 — Top-5 tags maximum : au-delà, les tags supplémentaires ont une
# similarité cosinus décroissante et introduisent du bruit sémantique dans
# les métadonnées de retrieval.
_ZERO_SHOT_TOP_K: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass métier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessedImage:
    """Représentation normalisée d'une image traitée, captionnée et vectorisée.

    Objet métier produit par ImageProcessor, consommé par MultimodalEmbedder
    pour l'indexation dans ChromaDB (image_collection_optimized).
    Conserve le chemin original (``path``) et le thumbnail 224×224 (``thumbnail_path``)
    pour permettre la visualisation dans le frontend Streamlit.

    Example:
        doc = ProcessedImage(
            id="a3f1c9e2b4d80f12",
            path="data/raw/images/unsplash/a3f1c9e2b4d80f12.jpg",
            thumbnail_path="data/processed/images/unsplash/a3f1c9e2b4d80f12_thumb.jpg",
            caption="A mountain landscape at golden hour",
            tags=["nature", "outdoor scene", "sky"],
            clip_embedding=[0.012, -0.034, ...],  # 512 floats
            width=1920,
            height=1080,
            source="unsplash",
            metadata={"url": "https://...", "licence": "Unsplash License"},
        )
    """

    id: str
    path: str                       # chemin JPEG original (data/raw/images/)
    thumbnail_path: str             # chemin thumbnail 224×224 (data/processed/images/)
    caption: str                    # description textuelle (BLIP-2 ou metadata existante)
    tags: List[str]                 # catégories zero-shot CLIP (max 5)
    clip_embedding: List[float]     # vecteur CLIP ViT-B/32, dimension 512 float32
    width: int                      # dimensions de l'image originale
    height: int
    source: str                     # 'coco' | 'unsplash' | 'wikimedia' | 'huggingface'
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Processeur principal
# ─────────────────────────────────────────────────────────────────────────────

class ImageProcessor:
    """Charge les images brutes et applique un pipeline de traitement multimodal.

    Filtre les images invalides (< 100×100px ou corrompues), génère des captions
    via BLIP-2 si absentes, classifie par tags zero-shot CLIP et produit des
    embeddings 512-dim pour ChromaDB. Traite les images en batches pour optimiser
    l'utilisation GPU lors de l'inférence CLIP.

    Example:
        processor = ImageProcessor()
        docs = processor.process(source="unsplash", max_images=100)
        # → List[ProcessedImage] dans data/processed/images/unsplash/
        # → Rapport dans results/image_processing_report.json
    """

    # CAS 3 — 100×100px : seuil minimum validé par ImageDownloader en amont.
    # En dessous, les embeddings CLIP sont fortement artefactés par l'upscaling
    # interne du ViT-B/32 (interpolation bicubique de 50px → 224px = ratio 4.5×).
    _MIN_DIMENSION: int = 100

    # CAS 3 — 512 dimensions : taille fixe de l'espace vectoriel de CLIP ViT-B/32.
    # Défini par le MLP head du ViT après projection (Radford et al. 2021, Table 1).
    # Toute autre dimension signalerait un modèle différent ou une corruption.
    _CLIP_EMBEDDING_DIM: int = 512

    # CAS 3 — 224×224 : résolution d'entrée fixe du ViT-B/32.
    # Imposée par la taille du patch (32px) et la séquence de tokens (7×7=49 patches).
    # Dosovitskiy et al. 2020 : "We use 224×224 images with patch size 32×32."
    _TARGET_SIZE: Tuple[int, int] = (224, 224)

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Args:
            config_path: Chemin vers config/config.yaml. Doit contenir
                         ``data.raw_path``, ``data.processed_path``,
                         ``data.results_path``, ``embeddings.batch_size``,
                         ``embeddings.clip_model``.

        Raises:
            FileNotFoundError: Si config_path n'existe pas.
        """
        self._config = self._load_config(config_path)

        self._raw_images_path = Path(self._config["data"]["raw_path"]) / "images"
        self._processed_images_path = (
            Path(self._config["data"]["processed_path"]) / "images"
        )
        self._results_path = Path(self._config["data"]["results_path"])

        self._processed_images_path.mkdir(parents=True, exist_ok=True)
        self._results_path.mkdir(parents=True, exist_ok=True)

        # CAS 3 — batch_size=32 : valeur config.yaml embeddings.batch_size.
        # Sur GPU RTX 3080 (10 Go VRAM), 32 images 224×224 ≈ 900 Mo VRAM — dans les
        # marges. Sur CPU, le batch_size n'affecte pas la mémoire mais réduit les
        # appels Python à l'encodeur (overhead fixe par encode()).
        self._batch_size: int = (
            self._config.get("embeddings", {}).get("batch_size", 32)
        )

        # CAS 2 — clip_model depuis config : permet de basculer vers clip-ViT-L-14
        # (dimension 768) sans modifier le code. La valeur par défaut clip-ViT-B-32
        # est le meilleur compromis vitesse/qualité pour un portfolio CPU/GPU modeste.
        self._clip_model_name: str = (
            self._config.get("embeddings", {}).get("clip_model", "clip-ViT-B-32")
        )

        self._categories: List[str] = (
            self._config.get("image", {})
            .get("zero_shot_categories", _DEFAULT_ZERO_SHOT_CATEGORIES)
        )

        # CAS 2 — Chargement paresseux des modèles : CLIP (~600 Mo) et BLIP-2 (~5 Go)
        # ne sont initialisés qu'au premier appel pour ne pas pénaliser l'instanciation
        # de ImageProcessor dans les tests ou les imports sans traitement effectif.
        self._clip: Optional[Any] = None
        self._blip2_proc: Optional[Any] = None
        self._blip2_model_obj: Optional[Any] = None
        self._blip2_device: Optional[str] = None

    # ─────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────

    def process(
        self,
        source: Optional[str] = None,
        max_images: Optional[int] = None,
    ) -> List[ProcessedImage]:
        """Charge, traite, enrichit et persiste les images depuis data/raw/images/.

        Deux phases :
          1. Séquentielle : validation, resize, caption BLIP-2 (GPU-lié, lent).
          2. Par batch de batch_size : embedding CLIP + tags zero-shot (GPU-efficace).

        Args:
            source:     Filtre optionnel sur la source (ex: ``'unsplash'``).
                        None = toutes les sources disponibles.
            max_images: Nombre maximum d'images à traiter. None = toutes.

        Returns:
            Liste de ProcessedImage valides (filtrées et enrichies). Jamais None.
        """
        t_start = time.perf_counter()

        raw_docs = self._load_raw_images(source_filter=source)
        logger.info(
            f"ImageProcessor : {len(raw_docs)} images brutes chargées "
            f"(source={source!r}, max={max_images})"
        )

        if max_images is not None:
            raw_docs = raw_docs[:max_images]

        rejection_counts: Dict[str, int] = {"too_small": 0, "corrupted": 0}
        valid_items: List[Tuple[Dict[str, Any], Image.Image, str]] = []

        # ── Phase 1 : validation, resize, caption ────────────────────────────
        for raw_doc in tqdm(raw_docs, desc="Preprocessing", unit="img"):
            img = self._load_image(raw_doc.get("path", ""))
            if img is None:
                rejection_counts["corrupted"] += 1
                continue

            if not self._is_valid_image(img):
                # CAS 3 — Filtre < 100×100 : re-appliqué ici même si ImageDownloader
                # l'a déjà filtré en amont, car des images peuvent être ajoutées
                # manuellement dans data/raw/images/ sans passer par le downloader.
                rejection_counts["too_small"] += 1
                logger.debug(
                    f"Image rejetée (trop petite) : {raw_doc.get('id', '')} "
                    f"({img.size[0]}×{img.size[1]}px)"
                )
                continue

            thumb = self._resize_to_224(img)

            # CAS 1 — Caption existante prioritaire sur BLIP-2 : les sources COCO
            # et HuggingFace fournissent des captions humaines de meilleure qualité
            # que la génération automatique. BLIP-2 n'est appelé que si la caption
            # est absente (chaîne vide ou None) pour économiser du temps GPU.
            caption = (raw_doc.get("caption") or "").strip()
            if not caption:
                caption = self._generate_caption(thumb)

            valid_items.append((raw_doc, thumb, caption))

        # ── Phase 2 : CLIP embedding + tags zero-shot (par batch) ────────────
        processed: List[ProcessedImage] = []

        if valid_items:
            for i in tqdm(
                range(0, len(valid_items), self._batch_size),
                desc="CLIP embedding",
                unit="batch",
            ):
                batch = valid_items[i : i + self._batch_size]
                thumbnails = [item[1] for item in batch]

                clip = self._get_clip()

                # CAS 3 — batch_size dans encode() : sentence-transformers découpe
                # le batch en mini-batches de cette taille pour le transfert GPU.
                # Sans batch_size, encode() traiterait toutes les images en un seul
                # forward pass → OOM sur GPU avec peu de VRAM (< 8 Go).
                embeddings = clip.encode(
                    thumbnails,
                    batch_size=self._batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )  # shape: (len(batch), 512)

                # CAS 1 — Textes de catégories encodés une fois par batch :
                # les embeddings textuels sont identiques pour toutes les images
                # du batch — les partager évite de ré-encoder N_cat × len(batch) fois.
                category_prompts = [f"a photo of {c}" for c in self._categories]
                text_embs = clip.encode(
                    category_prompts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )  # shape: (N_categories, 512)

                for j, (raw_doc, thumb, caption) in enumerate(batch):
                    img_emb: np.ndarray = embeddings[j]  # shape: (512,)
                    tags = self._tags_from_embeddings(img_emb, text_embs)
                    thumb_path = self._save_thumbnail(
                        thumb, raw_doc.get("source", "unknown"), raw_doc["id"]
                    )

                    doc = ProcessedImage(
                        id=raw_doc["id"],
                        path=raw_doc.get("path", ""),
                        thumbnail_path=str(thumb_path),
                        caption=caption,
                        tags=tags,
                        # CAS 1 — .tolist() : convertit numpy float32 en Python float
                        # natif pour sérialisation JSON (json.dumps rejette numpy.float32).
                        clip_embedding=img_emb.tolist(),
                        width=raw_doc.get("width", 0),
                        height=raw_doc.get("height", 0),
                        source=raw_doc.get("source", ""),
                        metadata={
                            **raw_doc.get("metadata", {}),
                            "url": raw_doc.get("url", ""),
                            "licence": raw_doc.get("licence", ""),
                            "processed_at": datetime.utcnow().strftime("%Y-%m-%d"),
                        },
                    )
                    self._save_metadata(doc)
                    processed.append(doc)

        elapsed = time.perf_counter() - t_start
        avg_time = elapsed / max(len(processed), 1)

        self._save_report(
            total_raw=len(raw_docs),
            total_processed=len(processed),
            rejected=rejection_counts,
            avg_processing_time_sec=round(avg_time, 3),
        )

        logger.info(
            f"ImageProcessor terminé : {len(processed)}/{len(raw_docs)} images "
            f"({sum(rejection_counts.values())} rejetées, {elapsed:.1f}s)"
        )
        return processed

    # ─────────────────────────────────────────────
    # Chargement et validation
    # ─────────────────────────────────────────────

    def _load_raw_images(
        self, source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Charge les métadonnées JSON depuis data/raw/images/{source}/*.json.

        Args:
            source_filter: Sous-dossier source à cibler. None = tous les sous-dossiers.

        Returns:
            Liste de dicts ImageDocument (format ImageDownloader). Vide si aucun fichier.
        """
        if source_filter:
            search_root = self._raw_images_path / source_filter
            pattern = "*.json"
        else:
            search_root = self._raw_images_path
            pattern = "**/*.json"

        json_files = sorted(search_root.glob(pattern))

        if not json_files:
            logger.warning(
                f"Aucun fichier JSON dans {search_root} (pattern='{pattern}')"
            )
            return []

        docs: List[Dict[str, Any]] = []
        for json_path in json_files:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                docs.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                # CAS 3 — JSON corrompu : on loggue et on continue sans interrompre
                # le batch (même comportement que TextProcessor._load_raw_documents).
                logger.warning(f"JSON ignoré ({json_path.name}) : {exc}")

        logger.info(
            f"_load_raw_images : {len(docs)} images depuis {len(json_files)} fichiers"
        )
        return docs

    def _load_image(self, path: str) -> Optional[Image.Image]:
        """Charge une image PIL depuis le chemin fourni.

        Args:
            path: Chemin absolu ou relatif vers le fichier image.

        Returns:
            PIL.Image.Image en mode RGB, ou None si le fichier est absent ou corrompu.
        """
        try:
            img = Image.open(path)
            # CAS 2 — img.load() plutôt que img.verify() : verify() invalide l'objet
            # image après appel (Pillow issue #1174). load() force le décodage complet
            # ET laisse l'objet utilisable pour la suite du pipeline.
            img.load()
            return img
        except FileNotFoundError:
            logger.warning(f"Fichier image introuvable : {path}")
            return None
        except UnidentifiedImageError as exc:
            logger.warning(f"Format image non reconnu par PIL ({path}) : {exc}")
            return None
        except OSError as exc:
            logger.warning(f"Image corrompue ({path}) : {exc}")
            return None

    def _is_valid_image(self, image: Image.Image) -> bool:
        """Retourne True si les dimensions de l'image dépassent le seuil minimum.

        Args:
            image: PIL Image (tout mode).

        Returns:
            True si ``width >= 100`` ET ``height >= 100`` (borne incluse).
        """
        w, h = image.size
        # CAS 3 — Seuil 100×100px : en dessous, l'upscaling ViT-B/32 (224/50 = ratio 4.5×)
        # introduit des artefacts de bicubic interpolation qui dégradent l'embedding.
        # La borne est incluse (>=) : une image 100×100 est acceptable.
        return w >= self._MIN_DIMENSION and h >= self._MIN_DIMENSION

    # ─────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────

    def _resize_to_224(self, image: Image.Image) -> Image.Image:
        """Redimensionne et centre l'image sur un canvas 224×224 blanc (letterbox).

        # ─── ALGORITHME : Letterbox resize ───────────────────────────────────
        # Problème résolu : adapter des images de tailles arbitraires en 224×224
        #                   sans déformer le contenu (aspect ratio préservé).
        # Approche :        thumbnail() → ratio max sans dépasser (224, 224) ;
        #                   paste() centré sur canvas blanc 224×224.
        # Formule :         scale = min(224/w, 224/h) ; new_size = (w*scale, h*scale)
        # Référence :       Même stratégie que torchvision.transforms.Resize+CenterCrop
        #                   mais sans crop — garantit qu'aucun pixel source n'est perdu.
        # ─────────────────────────────────────────────────────────────────────

        Args:
            image: PIL Image de taille quelconque.

        Returns:
            PIL Image RGB 224×224 avec padding blanc si l'aspect ratio n'est pas carré.
        """
        if image.mode != "RGB":
            # CAS 1 — Conversion RGB obligatoire : CLIP ViT-B/32 attend 3 canaux.
            # Les modes RGBA, P (palette), L (niveaux de gris) sont convertis.
            # convert("RGB") fusionne l'alpha sur blanc (défaut PIL) pour RGBA.
            image = image.convert("RGB")

        # CAS 3 — thumbnail() modifie l'image en place ET préserve le ratio :
        # PIL.Image.thumbnail est la seule méthode PIL qui redimensionne sans
        # dépasser les dimensions cibles (contrairement à resize() qui force la taille).
        image_copy = image.copy()
        image_copy.thumbnail(self._TARGET_SIZE, Image.Resampling.LANCZOS)

        # Canvas blanc 224×224 — LANCZOS pour la meilleure qualité de resampling
        # (antialias bicubic) aux dépens d'une légère lenteur vs BILINEAR.
        canvas = Image.new("RGB", self._TARGET_SIZE, color=(255, 255, 255))
        # CAS 1 — Centrage sur le canvas : l'image réduite est collée au centre
        # pour équilibrer le padding haut/bas ou gauche/droite.
        x_offset = (self._TARGET_SIZE[0] - image_copy.width) // 2
        y_offset = (self._TARGET_SIZE[1] - image_copy.height) // 2
        canvas.paste(image_copy, (x_offset, y_offset))

        return canvas

    # ─────────────────────────────────────────────
    # Caption via BLIP-2
    # ─────────────────────────────────────────────

    def _generate_caption(self, image: Image.Image) -> str:
        """Génère une description textuelle de l'image via BLIP-2.

        Appelle le modèle Salesforce/blip2-opt-2.7b uniquement si _BLIP2_AVAILABLE
        est True. Retourne une chaîne vide si le modèle n'est pas disponible.

        Args:
            image: PIL Image (idéalement 224×224 RGB pour performance optimale).

        Returns:
            Description textuelle générée (ex: ``"a brown dog running in a field"``),
            ou ``""`` si BLIP-2 n'est pas installé ou si la génération échoue.
        """
        if not _BLIP2_AVAILABLE:
            # CAS 3 — Dégradation gracieuse : si transformers/torch ne sont pas
            # installés, on retourne une chaîne vide plutôt que de lever une exception.
            # Les images sans caption seront quand même indexées par leur embedding CLIP.
            logger.debug("BLIP-2 indisponible (transformers non installé) — caption vide")
            return ""

        try:
            proc, model, device = self._get_blip2()

            # CAS 3 — return_tensors="pt" : format PyTorch tensors requis par
            # Blip2ForConditionalGeneration.generate(). "tf" ou "np" lèvent TypeError.
            inputs = proc(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                # CAS 3 — max_new_tokens=50 : limite la longueur de la caption générée.
                # Au-delà, BLIP-2 peut halluciner du contenu non présent dans l'image.
                # 50 tokens ≈ 35–40 mots — suffisant pour une description utile à la retrieval.
                generated_ids = model.generate(**inputs, max_new_tokens=50)

            caption = proc.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            return caption

        except Exception as exc:
            logger.warning(f"Génération caption BLIP-2 échouée : {exc}")
            return ""

    def _get_blip2(self) -> Tuple[Any, Any, str]:
        """Charge BLIP-2 de manière paresseuse (une seule fois par instance).

        Returns:
            Tuple (Blip2Processor, Blip2ForConditionalGeneration, device_str).

        Raises:
            RuntimeError: Si transformers ou torch ne sont pas installés.
        """
        if self._blip2_proc is None:
            if not _BLIP2_AVAILABLE:
                raise RuntimeError(
                    "transformers non installé — source BLIP-2 indisponible. "
                    "pip install transformers accelerate"
                )

            # CAS 2 — float16 sur CUDA uniquement : BLIP-2 en float32 nécessite
            # ~10 Go VRAM, float16 réduit à ~5 Go. Sur CPU, float16 n'est pas
            # supporté par toutes les opérations — on reste en float32 (torch défaut).
            self._blip2_device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._blip2_device == "cuda" else torch.float32

            logger.info(
                f"Chargement BLIP-2 (Salesforce/blip2-opt-2.7b) sur {self._blip2_device}…"
            )
            self._blip2_proc = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            )
            self._blip2_model_obj = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=dtype
            ).to(self._blip2_device)
            # CAS 1 — eval() désactive dropout et batch normalization : indispensable
            # pour l'inférence (sans eval(), les prédictions sont non-déterministes).
            self._blip2_model_obj.eval()
            logger.info("BLIP-2 chargé")

        return self._blip2_proc, self._blip2_model_obj, self._blip2_device

    # ─────────────────────────────────────────────
    # CLIP embedding et tags zero-shot
    # ─────────────────────────────────────────────

    def _compute_clip_embedding(self, image: Image.Image) -> List[float]:
        """Calcule l'embedding CLIP ViT-B/32 d'une image en 512 dimensions.

        # ─── ALGORITHME : CLIP image embedding ───────────────────────────────
        # Problème résolu : représenter une image dans un espace vectoriel
        #                   partagé avec le texte pour la retrieval cross-modale.
        # Approche :        ViT-B/32 image encoder → projection 512-dim → normalisation L2.
        # Formule :         v = normalize(VisionTransformer(preprocess(img)))
        # Référence :       Radford et al. 2021, CLIP, section 2.1 "Approach".
        # ─────────────────────────────────────────────────────────────────────

        Args:
            image: PIL Image RGB (idéalement 224×224, resize appliqué en amont).

        Returns:
            Liste de 512 floats (vecteur L2-normalisé dans l'espace CLIP commun).

        Raises:
            RuntimeError: Si sentence-transformers n'est pas installé.
        """
        clip = self._get_clip()
        # CAS 3 — encode([image]) avec liste d'un seul élément : l'API sentence-transformers
        # attend une liste même pour une image unique. Un PIL.Image passé directement
        # lève TypeError car encode() itère sur l'entrée caractère par caractère.
        embedding: np.ndarray = clip.encode(
            [image],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]  # slice [0] → shape (512,) depuis (1, 512)

        # CAS 1 — .tolist() : numpy.float32 n'est pas sérialisable par json.dumps().
        # La conversion en Python float natif est obligatoire avant persistance JSON.
        return embedding.tolist()

    def _compute_zero_shot_tags(self, image: Image.Image) -> List[str]:
        """Classifie l'image par similarité cosinus CLIP contre une liste de catégories.

        Args:
            image: PIL Image 224×224 RGB.

        Returns:
            Liste de tags (≤ 5) dont la similarité cosinus avec l'image dépasse 0.20.
        """
        clip = self._get_clip()

        img_emb = clip.encode(
            [image], convert_to_numpy=True, show_progress_bar=False
        )[0]  # (512,)

        category_prompts = [f"a photo of {c}" for c in self._categories]
        text_embs = clip.encode(
            category_prompts, convert_to_numpy=True, show_progress_bar=False
        )  # (N_cat, 512)

        return self._tags_from_embeddings(img_emb, text_embs)

    def _tags_from_embeddings(
        self, img_emb: np.ndarray, text_embs: np.ndarray
    ) -> List[str]:
        """Sélectionne les tags par similarité cosinus image/texte.

        # ─── ALGORITHME : Similarité cosinus zero-shot ────────────────────────
        # Problème résolu : classer les catégories par pertinence sans fine-tuning.
        # Approche :        dot product des embeddings L2-normalisés.
        # Formule :         sim(i, t) = (i/‖i‖) · (t/‖t‖) ∈ [-1, 1]
        # Référence :       CLIP paper, section 3.1 "Zero-Shot Transfer".
        # ─────────────────────────────────────────────────────────────────────

        Args:
            img_emb:   Embedding image (512,).
            text_embs: Embeddings textes catégories (N_cat, 512).

        Returns:
            Liste de tags filtrés (similarité ≥ 0.20, maximum 5).
        """
        # Normalisation L2 avec epsilon pour éviter la division par zéro
        img_norm = img_emb / (np.linalg.norm(img_emb) + 1e-8)
        text_norms = text_embs / (
            np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8
        )

        # CAS 1 — dot product matriciel : (N_cat, 512) @ (512,) → (N_cat,)
        # Équivalent à cosine_similarity mais sans dépendance sklearn.
        sims: np.ndarray = text_norms @ img_norm

        # CAS 3 — argsort décroissant [::-1] : numpy.argsort retourne par défaut
        # l'ordre croissant. On inverse pour obtenir les catégories les plus similaires
        # en premier, puis on applique le filtre de seuil.
        top_indices = np.argsort(sims)[::-1][:_ZERO_SHOT_TOP_K]
        return [
            self._categories[i]
            for i in top_indices
            if sims[i] >= _ZERO_SHOT_THRESHOLD
        ]

    def _get_clip(self) -> Any:
        """Charge le modèle CLIP via sentence-transformers (une seule fois par instance).

        Returns:
            SentenceTransformer initialisé avec le modèle CLIP configuré.

        Raises:
            RuntimeError: Si sentence-transformers n'est pas installé.
        """
        if self._clip is None:
            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers non installé — CLIP indisponible. "
                    "pip install sentence-transformers"
                )
            logger.info(f"Chargement CLIP : {self._clip_model_name}")
            self._clip = SentenceTransformer(self._clip_model_name)
            logger.info("CLIP chargé")
        return self._clip

    # ─────────────────────────────────────────────
    # Persistance
    # ─────────────────────────────────────────────

    def _save_thumbnail(
        self, image: Image.Image, source: str, image_id: str
    ) -> Path:
        """Sauvegarde le thumbnail 224×224 en JPEG dans data/processed/images/{source}/.

        Args:
            image:    PIL Image 224×224 RGB déjà redimensionnée.
            source:   Nom de la source (sous-dossier).
            image_id: Identifiant de l'image (nom de fichier sans extension).

        Returns:
            Path du fichier thumbnail créé.
        """
        source_dir = self._processed_images_path / source
        source_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = source_dir / f"{image_id}_thumb.jpg"

        # CAS 3 — quality=95 : même paramètre que ImageDownloader pour cohérence.
        # Évite de re-compresser avec une qualité différente ce qui dégalibrait
        # les artefacts JPEG entre raw et processed.
        image.save(str(thumb_path), format="JPEG", quality=95, optimize=True)
        return thumb_path

    def _save_metadata(self, doc: ProcessedImage) -> None:
        """Persiste le ProcessedImage en JSON dans data/processed/images/{source}/.

        Args:
            doc: ProcessedImage à sérialiser. Écrit dans {source}/{id}.json.
        """
        source_dir = self._processed_images_path / doc.source
        source_dir.mkdir(parents=True, exist_ok=True)
        json_path = source_dir / f"{doc.id}.json"

        # CAS 1 — ensure_ascii=False : captions WikiMedia et titres COCO contiennent
        # des caractères non-ASCII (accents, apostrophes curly, caractères CJK).
        json_path.write_text(
            json.dumps(asdict(doc), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_report(
        self,
        total_raw: int,
        total_processed: int,
        rejected: Dict[str, int],
        avg_processing_time_sec: float,
    ) -> None:
        """Sauvegarde le rapport de traitement dans results/image_processing_report.json.

        Args:
            total_raw:                Nombre d'images brutes chargées.
            total_processed:          Nombre d'images valides produites.
            rejected:                 Comptage des rejets par cause.
            avg_processing_time_sec:  Temps moyen de traitement par image valide (secondes).
        """
        report = {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "total_raw": total_raw,
            "total_processed": total_processed,
            "rejected_count": sum(rejected.values()),
            "rejection_rate": (
                round(sum(rejected.values()) / total_raw, 3) if total_raw > 0 else 0.0
            ),
            "rejected": rejected,
            "avg_processing_time_sec": avg_processing_time_sec,
        }

        report_path = self._results_path / "image_processing_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Rapport image sauvegardé : {report_path.name}")

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Charge la configuration YAML du projet.

        Args:
            config_path: Chemin relatif ou absolu vers config.yaml.

        Returns:
            Dictionnaire de configuration.

        Raises:
            FileNotFoundError: Si config_path n'existe pas.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier de configuration introuvable : {config_path}"
            )
        return yaml.safe_load(path.read_text(encoding="utf-8"))
