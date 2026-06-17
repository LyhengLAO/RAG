# -*- coding: utf-8 -*-
"""
Module image_downloader — Téléchargeur d'images multimodal.

Rôle dans l'architecture :
    Deuxième composant du pipeline d'ingestion. Récupère des images depuis
    quatre sources hétérogènes, les normalise en JPEG RGB uniforme et persiste
    un fichier image + un JSON de métadonnées par image dans data/raw/images/{source}/.
    Produit des ImageDocument consommés par ImageProcessor → MultimodalEmbedder
    → ChromaDB (image_collection_optimized).

Pourquoi ces quatre sources :
    - COCO via fiftyone  : 80 catégories annotées, référence absolue en CV depuis 2014.
                           Labels ground-truth exploitables directement pour l'évaluation
                           de retrieval multimodal. fiftyone gère le cache et la pagination.
    - Unsplash           : Photos haute résolution sous Unsplash License (usage libre).
                           Diversité thématique maximale pour un corpus généraliste.
                           Rejeté : Pexels (API moins stable), Getty (payant).
    - WikiMedia Commons  : Images libres (CC, domaine public) issues de l'encyclopédie.
                           Couverture encyclopédique complémentaire à COCO — textes +
                           schémas + photographies. API publique, pas d'authentification.
    - HuggingFace Images : Datasets annotés (beans, food101, oxford_pets) avec ground-truth
                           intégré — idéaux pour mesurer la précision de classification
                           dans l'évaluation RAGAS multimodale.

Format de sortie :
    - Image     : data/raw/images/{source}/{id}.jpg   — JPEG RGB qualité 95
    - Métadonnée: data/raw/images/{source}/{id}.json  — ImageDocument sérialisé
    Le format JPEG est imposé pour homogénéité du corpus et compatibilité maximale
    avec les encodeurs CLIP (ViT-B/32 attend du RGB 224×224).
"""

import hashlib
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv
from loguru import logger
from PIL import Image, UnidentifiedImageError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

# CAS 2 — Import conditionnel fiftyone : bibliothèque > 1 Go avec dépendance MongoDB/Motor.
# Un ImportError fatal bloquerait toutes les sources, pas seulement COCO.
# Rejeté : import obligatoire + try/except dans __init__ (retarde l'erreur trop tard).
try:
    import fiftyone.zoo as foz
    _FIFTYONE_AVAILABLE = True
except ImportError:
    _FIFTYONE_AVAILABLE = False

# CAS 2 — Import conditionnel datasets HF : même raison. datasets charge torch/transformers
# ce qui peut prendre plusieurs secondes au premier import.
try:
    from datasets import load_dataset as _hf_load_dataset
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    _HF_DATASETS_AVAILABLE = False


def _log_retry(retry_state: Any) -> None:
    """Log loguru pour tenacity before_sleep (stdlib logging incompatible avec loguru)."""
    exc = retry_state.outcome.exception()
    logger.warning(
        f"Retry {retry_state.attempt_number}/3 — {type(exc).__name__}: {exc}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass métier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageDocument:
    """Représentation normalisée d'une image téléchargée et validée.

    Objet métier partagé entre ImageDownloader → ImageProcessor → MultimodalEmbedder.
    Le champ ``path`` pointe vers le JPEG persisté ; ``id`` est le nom de fichier
    sans extension, stable cross-session (SHA-256[:16] de source + identifiant natif).

    Example:
        doc = ImageDocument(
            id="a3f1c9e2b4d80f12",
            path="data/raw/images/unsplash/a3f1c9e2b4d80f12.jpg",
            source="unsplash",
            url="https://unsplash.com/photos/xYz123",
            caption="Mountain landscape at sunset",
            tags=["mountain", "landscape", "nature"],
            licence="Unsplash License",
            width=1920,
            height=1080,
        )
    """

    id: str
    path: str
    source: str       # 'coco' | 'unsplash' | 'wikimedia' | 'huggingface'
    url: str
    caption: str
    tags: List[str]
    licence: str
    width: int
    height: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Downloader principal
# ─────────────────────────────────────────────────────────────────────────────

class ImageDownloader:
    """Télécharge et normalise des images depuis COCO, Unsplash, WikiMedia et HuggingFace.

    Toutes les images passent par un pipeline de validation (non corrompue, ≥ 100×100px)
    et de conversion JPEG RGB avant persistance. Cache disque par id : si {id}.jpg et
    {id}.json existent, l'image est retournée sans appel réseau.

    Example:
        downloader = ImageDownloader()
        docs = downloader.download("unsplash", "mountain landscape", max_images=10)
        # → 10 JPEG + 10 JSON dans data/raw/images/unsplash/
        docs2 = downloader.download("unsplash", "mountain landscape", max_images=10)
        # → Chargé depuis cache, aucun appel réseau
    """

    # CAS 3 — 100×100px : seuil minimum empirique pour les embeddings CLIP ViT-B/32.
    # CLIP redimensionne en interne à 224×224 — une image de 50×50 produit des embeddings
    # fortement artefactés par l'upscaling. 100×100 est le plancher acceptable (Lin et al.).
    _MIN_DIMENSION: int = 100

    # CAS 3 — quality=95 : équilibre archivage/compression.
    # quality=100 désactive le quantizer JPEG (fichiers ~4-5× plus lourds, inutile pour CLIP).
    # quality=85 standard web mais introduit des artefacts blocs sur schémas/graphiques.
    _JPEG_QUALITY: int = 95

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Args:
            config_path: Chemin vers config/config.yaml. Doit contenir ``data.raw_path``.

        Raises:
            FileNotFoundError: Si config_path n'existe pas.
        """
        # CAS 1 — load_dotenv() en __init__ : garantit la disponibilité des clés API
        # même si l'appelant (script CLI, test) n'a pas appelé load_dotenv() au préalable.
        load_dotenv()

        self._config = self._load_config(config_path)
        self._raw_images_path = Path(self._config["data"]["raw_path"]) / "images"
        self._raw_images_path.mkdir(parents=True, exist_ok=True)

        # CAS 3 — Clé Unsplash depuis env uniquement : jamais depuis config.yaml (fichier
        # versionné). UNSPLASH_API_KEY = access_key du compte Unsplash Developer
        # (https://unsplash.com/developers). Limite : 50 req/heure sur compte free.
        self._unsplash_key: str = os.environ.get("UNSPLASH_API_KEY", "")
        if not self._unsplash_key:
            logger.warning(
                "UNSPLASH_API_KEY non définie dans .env — "
                "source 'unsplash' désactivée. Voir .env.example."
            )

        # CAS 2 — User-Agent explicite WikiMedia : politique d'utilisation de l'API MediaWiki
        # exige un UA identifiable (format NomApp/version contact). Sans UA, risque de
        # throttling agressif ou de blocage automatique (tool: api-usage-policy).
        self._wikimedia_ua = "rag-multimedia-expert/0.1 (educational-portfolio-project)"

    # ─────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────

    def download(self, source: str, query: str, max_images: int = 20) -> List[ImageDocument]:
        """Télécharge des images depuis la source demandée, avec validation et cache.

        Args:
            source:     Source à interroger. Valeurs acceptées :
                        ``'coco'``, ``'unsplash'``, ``'wikimedia'``, ``'huggingface'``.
            query:      Requête dont le format dépend de la source (voir _download_*).
            max_images: Nombre maximum d'images à retourner (après validation).

        Returns:
            Liste d'ImageDocument normalisés. Jamais None, peut être vide.

        Raises:
            ValueError: Si source n'est pas dans les valeurs acceptées.
        """
        if source not in {"coco", "unsplash", "wikimedia", "huggingface"}:
            raise ValueError(
                f"Source inconnue : '{source}'. "
                f"Valeurs : 'coco', 'unsplash', 'wikimedia', 'huggingface'."
            )

        logger.info(
            f"Téléchargement images | source='{source}' | "
            f"query='{query[:60]}' | max={max_images}"
        )

        _dispatch = {
            "coco": self._download_coco,
            "unsplash": self._download_unsplash,
            "wikimedia": self._download_wikimedia,
            "huggingface": self._download_huggingface,
        }
        documents = _dispatch[source](query, max_images)
        logger.info(f"[{source}] {len(documents)} images valides retournées")
        return documents

    # ─────────────────────────────────────────────
    # Source : COCO via fiftyone
    # ─────────────────────────────────────────────

    def _download_coco(self, query: str, max_images: int) -> List[ImageDocument]:
        """Télécharge des images COCO filtrées par catégories via fiftyone.

        Format du paramètre ``query`` :
          - Catégorie unique    : ``"cat"``
          - Catégories multiples : ``"cat,dog,person"``

        Args:
            query:      Catégories COCO séparées par virgule (80 classes disponibles).
            max_images: Nombre maximum d'images retournées.

        Returns:
            Liste d'ImageDocument. Vide si fiftyone non installé.
        """
        if not _FIFTYONE_AVAILABLE:
            logger.error(
                "fiftyone non installé — source 'coco' indisponible. "
                "pip install fiftyone"
            )
            return []

        categories = [c.strip() for c in query.split(",") if c.strip()]

        # CAS 2 — split="validation" : ~5 000 images vs ~118 000 pour "train".
        # Pour un portfolio, la validation est suffisante et évite un téléchargement
        # de 18 Go. fiftyone gère son propre cache dans ~/fiftyone/.
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            label_types=["detections"],
            classes=categories if categories else None,
            max_samples=max_images,
        )

        documents: List[ImageDocument] = []

        for sample in tqdm(dataset.take(max_images), desc="COCO", unit="image"):
            image_id = self._compute_image_id("coco", str(sample.id))

            if self._is_cached("coco", image_id):
                doc = self._load_from_disk("coco", image_id)
                if doc:
                    documents.append(doc)
                continue

            labels: List[str] = []
            if sample.ground_truth:
                # CAS 1 — set() sur les labels : une image COCO peut contenir plusieurs
                # instances du même objet. On déduplique pour obtenir des tags uniques.
                labels = list({det.label for det in sample.ground_truth.detections})

            try:
                img = Image.open(sample.filepath)
            except Exception as exc:
                logger.warning(f"COCO : impossible d'ouvrir {sample.filepath} — {exc}")
                continue

            doc = self._save_image_document(
                image_id=image_id,
                source="coco",
                url=f"https://cocodataset.org/#explore?id={sample.id}",
                caption=", ".join(labels) or "COCO image",
                tags=labels,
                licence="CC BY 4.0",
                metadata={"coco_id": str(sample.id), "split": "validation"},
                pil_image=img,
            )
            if doc:
                documents.append(doc)

        return documents

    # ─────────────────────────────────────────────
    # Source : Unsplash REST API
    # ─────────────────────────────────────────────

    def _download_unsplash(self, query: str, max_images: int) -> List[ImageDocument]:
        """Télécharge des photos depuis l'API Unsplash Search.

        Format du paramètre ``query`` : requête libre (ex: ``"mountain landscape"``).

        Requiert ``UNSPLASH_API_KEY`` dans .env (Access Key du compte Unsplash Developer).

        Args:
            query:      Requête de recherche textuelle.
            max_images: Nombre maximum de photos (limité à 30 par l'API par requête).

        Returns:
            Liste d'ImageDocument. Vide si UNSPLASH_API_KEY manquante.
        """
        if not self._unsplash_key:
            logger.error("UNSPLASH_API_KEY manquante — source Unsplash ignorée")
            return []

        photo_metas = self._fetch_unsplash_results(query, max_images)
        documents: List[ImageDocument] = []

        for photo in tqdm(photo_metas[:max_images], desc="Unsplash", unit="photo"):
            # CAS 1 — URL "regular" plutôt que "raw" : raw est non compressé (> 10 Mo/image).
            # "regular" est plafonné à 1080px de large — suffisant pour CLIP ViT-B/32 (224px).
            url = photo.get("urls", {}).get("regular", "")
            if not url:
                continue

            photo_id = photo.get("id", "")
            image_id = self._compute_image_id("unsplash", photo_id)

            if self._is_cached("unsplash", image_id):
                doc = self._load_from_disk("unsplash", image_id)
                if doc:
                    documents.append(doc)
                continue

            caption = (
                photo.get("description")
                or photo.get("alt_description")
                or f"Photo by {photo.get('user', {}).get('name', 'unknown')}"
            )
            tags = [t.get("title", "") for t in photo.get("tags", []) if t.get("title")]

            doc = self._download_and_save_from_url(
                image_id=image_id,
                source="unsplash",
                url=url,
                caption=caption[:500],
                tags=tags[:20],
                licence="Unsplash License",
                metadata={
                    "unsplash_id": photo_id,
                    "photographer": photo.get("user", {}).get("name", ""),
                    "likes": photo.get("likes", 0),
                },
            )
            if doc:
                documents.append(doc)

            # CAS 3 — sleep(0.072s) entre images : Unsplash limite à 50 req/heure
            # (3 000 req/heure sur compte approuvé). 0.072s ≈ 1 req/14 downloads
            # pour rester sous les 50 req/heure sur le quota de recherche.
            time.sleep(0.072)

        return documents

    @retry(
        stop=stop_after_attempt(3),
        # CAS 3 — min=5s : Unsplash retourne HTTP 429 (Rate Limit Exceeded) sous forte
        # charge. 5s de backoff minimum laisse le quota se reconstituer partiellement.
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=_log_retry,
    )
    def _fetch_unsplash_results(
        self, query: str, max_images: int
    ) -> List[Dict[str, Any]]:
        """Interroge l'endpoint Search de l'API Unsplash et retourne les métadonnées photos.

        Args:
            query:      Requête de recherche.
            max_images: Nombre de résultats souhaités.

        Returns:
            Liste de dicts de métadonnées Unsplash (peut être vide si aucun résultat).

        Raises:
            requests.RequestException: En cas d'erreur réseau (déclenche le retry tenacity).
        """
        # CAS 3 — per_page=30 : maximum autorisé par l'API Unsplash par requête unique.
        # Dépasser 30 retourne HTTP 422 (Unprocessable Entity), pas 400.
        per_page = min(max_images, 30)

        response = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": per_page, "page": 1},
            headers={"Authorization": f"Client-ID {self._unsplash_key}"},
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("results", [])

    # ─────────────────────────────────────────────
    # Source : WikiMedia Commons REST API
    # ─────────────────────────────────────────────

    def _download_wikimedia(self, query: str, max_images: int) -> List[ImageDocument]:
        """Télécharge des images libres depuis WikiMedia Commons par catégorie.

        Format du paramètre ``query`` : nom de catégorie Commons (ex: ``"Cats"``).
        Les sous-catégories ne sont pas parcourues (premier niveau uniquement).

        Args:
            query:      Nom de catégorie WikiMedia Commons sans préfixe ``"Category:"``.
            max_images: Nombre maximum d'images valides retournées.

        Returns:
            Liste d'ImageDocument sous licence libre (CC, domaine public, etc.).
        """
        file_titles = self._fetch_wikimedia_category_members(query, max_images)

        if not file_titles:
            logger.warning(f"WikiMedia : aucun fichier image dans la catégorie '{query}'")
            return []

        # CAS 1 — Traitement par lots de 50 : l'API MediaWiki accepte au maximum 50 titres
        # par requête dans le paramètre "titles". Dépasser cette limite retourne
        # une erreur "toomanyvalues" (code API : toomanyvalues).
        batch_size = 50
        documents: List[ImageDocument] = []

        for batch_start in range(0, len(file_titles), batch_size):
            batch = file_titles[batch_start : batch_start + batch_size]
            image_infos = self._fetch_wikimedia_image_info(batch)

            for info in tqdm(image_infos, desc="WikiMedia", unit="image", leave=False):
                url = info.get("url", "")
                if not url:
                    continue

                # CAS 1 — Filtre MIME : WikiMedia héberge SVG, TIFF, OGV, PDF...
                # On garde seulement les formats rastérisés supportés par PIL.
                # SVG et TIFF sont exclus : SVG non supporté par PIL sans librairie
                # supplémentaire ; TIFF souvent multi-pages ou très lourd (> 50 Mo).
                mime = info.get("mime", "")
                if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                    logger.debug(f"WikiMedia : format ignoré '{mime}' — {url[:60]}")
                    continue

                file_title = info.get("title", "")
                image_id = self._compute_image_id("wikimedia", file_title or url)

                if self._is_cached("wikimedia", image_id):
                    doc = self._load_from_disk("wikimedia", image_id)
                    if doc:
                        documents.append(doc)
                    if len(documents) >= max_images:
                        return documents
                    continue

                ext_meta = info.get("extmetadata", {})
                raw_caption = (
                    ext_meta.get("ImageDescription", {}).get("value", "")
                    or file_title.replace("File:", "").rsplit(".", 1)[0]
                )
                # CAS 1 — _strip_html() sur la caption : ImageDescription WikiMedia contient
                # du HTML brut (<br />, <a href="...">, <span>...) qui polluerait les
                # embeddings textuels et le frontend Streamlit.
                caption = self._strip_html(raw_caption)[:500]
                licence = ext_meta.get("LicenseShortName", {}).get("value", "Unknown")

                doc = self._download_and_save_from_url(
                    image_id=image_id,
                    source="wikimedia",
                    url=url,
                    caption=caption,
                    tags=[query],
                    licence=licence,
                    metadata={"file_title": file_title, "mime_original": mime},
                )
                if doc:
                    documents.append(doc)

                if len(documents) >= max_images:
                    return documents

            # CAS 3 — sleep(0.5s) entre lots : MediaWiki demande de respecter un délai
            # raisonnable entre requêtes automatisées (MaxLag policy). 0.5s/50 fichiers
            # = 100 req/min, bien en-dessous du seuil de throttling automatique.
            if batch_start + batch_size < len(file_titles):
                time.sleep(0.5)

        return documents

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=_log_retry,
    )
    def _fetch_wikimedia_category_members(
        self, category: str, max_count: int
    ) -> List[str]:
        """Retourne les titres des fichiers images d'une catégorie WikiMedia Commons.

        Args:
            category:  Nom de la catégorie sans préfixe ``"Category:"``.
            max_count: Nombre maximum de titres à retourner.

        Returns:
            Liste de titres au format ``"File:example.jpg"``.
        """
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": "file",
                # CAS 3 — cmlimit=min(max_count, 500) : 500 est la limite absolue de l'API
                # MediaWiki par requête (erreur "apilimit" au-delà). Pour max_count > 500,
                # il faudrait implémenter la pagination via "cmcontinue" (hors scope).
                "cmlimit": min(max_count, 500),
                "format": "json",
            },
            headers={"User-Agent": self._wikimedia_ua},
            timeout=15,
        )
        response.raise_for_status()

        members = response.json().get("query", {}).get("categorymembers", [])
        # CAS 1 — Filtre "File:" : categorymembers peut retourner des sous-catégories
        # (prefix "Category:") si cmtype n'est pas strictement respecté par l'API.
        return [m["title"] for m in members if m.get("title", "").startswith("File:")]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=_log_retry,
    )
    def _fetch_wikimedia_image_info(
        self, file_titles: List[str]
    ) -> List[Dict[str, Any]]:
        """Récupère URL directe, MIME, dimensions et métadonnées licence pour un lot de fichiers.

        Args:
            file_titles: Liste de titres de fichiers (``"File:xxx.jpg"``), max 50.

        Returns:
            Liste de dicts avec clés ``url``, ``mime``, ``title``, ``extmetadata``.
        """
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                # CAS 1 — Séparateur pipe : l'API MediaWiki accepte plusieurs titres
                # dans un seul paramètre "titles" séparés par "|". Plus efficace que
                # N requêtes individuelles (réduit le nombre d'aller-retours de 50×).
                "titles": "|".join(file_titles),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                # CAS 1 — iiextmetadatafilter : sans filtre, extmetadata retourne ~30 champs
                # (GPS, dates EXIF...). On ne garde que licence + description pour réduire
                # la taille de la réponse JSON (parfois > 500 Ko sans filtre).
                "iiextmetadatafilter": "LicenseShortName|ImageDescription",
                "format": "json",
            },
            headers={"User-Agent": self._wikimedia_ua},
            timeout=20,
        )
        response.raise_for_status()

        pages = response.json().get("query", {}).get("pages", {})
        results: List[Dict[str, Any]] = []

        for page in pages.values():
            image_info = page.get("imageinfo", [{}])[0]
            results.append({
                "title": page.get("title", ""),
                "url": image_info.get("url", ""),
                "mime": image_info.get("mime", ""),
                "width": image_info.get("width", 0),
                "height": image_info.get("height", 0),
                "extmetadata": image_info.get("extmetadata", {}),
            })

        return results

    # ─────────────────────────────────────────────
    # Source : HuggingFace image datasets
    # ─────────────────────────────────────────────

    def _download_huggingface(self, query: str, max_images: int) -> List[ImageDocument]:
        """Charge des images depuis un dataset HuggingFace avec colonne ``image``.

        Format du paramètre ``query`` :
          - Dataset simple  : ``"beans"``
          - Dataset + split : ``"beans:train"``

        Datasets testés : ``beans``, ``food101``, ``oxford_pets``,
        ``cifar10`` (tout dataset HF avec colonne ``image`` de type PIL.Image).

        Args:
            query:      Nom du dataset avec split optionnel (séparateur ``':'``).
            max_images: Nombre maximum d'images chargées.

        Returns:
            Liste d'ImageDocument. Vide si datasets non installé.
        """
        if not _HF_DATASETS_AVAILABLE:
            logger.error(
                "datasets non installé — source HuggingFace indisponible. "
                "pip install datasets"
            )
            return []

        parts = query.split(":")
        dataset_name = parts[0].strip()
        # CAS 3 — split="train" par défaut : le split train est présent dans tous les
        # datasets image standardisés. "test" peut ne pas exister (ex: beans n'a pas de test).
        split = parts[1].strip() if len(parts) > 1 else "train"

        logger.info(f"HuggingFace Images : dataset='{dataset_name}' split='{split}'")

        # CAS 2 — trust_remote_code=False : refus systématique d'exécuter du code Python
        # arbitraire téléchargé depuis le Hub. Rejeté True : vecteur d'injection de code.
        dataset = _hf_load_dataset(
            dataset_name, split=split, streaming=False, trust_remote_code=False
        )

        documents: List[ImageDocument] = []
        sample_size = min(max_images, len(dataset))

        for idx in tqdm(range(sample_size), desc="HuggingFace Images", unit="img"):
            example = dataset[idx]

            pil_image = example.get("image")
            if pil_image is None:
                # CAS 3 — Edge case : certains datasets utilisent "img" ou "pixel_values"
                # à la place de "image". On loggue pour faciliter le débogage mais on
                # ne lève pas d'exception pour ne pas interrompre le batch.
                logger.warning(
                    f"HuggingFace : exemple [{idx}] sans colonne 'image' "
                    f"dans '{dataset_name}' — clés disponibles : {list(example.keys())}"
                )
                continue

            label_val = example.get("label", example.get("labels"))
            label_str = str(label_val) if label_val is not None else dataset_name

            image_id = self._compute_image_id(
                "huggingface", f"{dataset_name}:{split}:{idx}"
            )

            if self._is_cached("huggingface", image_id):
                doc = self._load_from_disk("huggingface", image_id)
                if doc:
                    documents.append(doc)
                continue

            doc = self._save_image_document(
                image_id=image_id,
                source="huggingface",
                url=f"https://huggingface.co/datasets/{dataset_name}",
                caption=f"{dataset_name} — class {label_str}",
                tags=[dataset_name, label_str],
                licence="See dataset card on HuggingFace Hub",
                metadata={"dataset": dataset_name, "split": split, "index": idx},
                pil_image=pil_image,
            )
            if doc:
                documents.append(doc)

        return documents

    # ─────────────────────────────────────────────
    # Téléchargement, validation, conversion, persistance
    # ─────────────────────────────────────────────

    def _download_and_save_from_url(
        self,
        image_id: str,
        source: str,
        url: str,
        caption: str,
        tags: List[str],
        licence: str,
        metadata: Dict[str, Any],
    ) -> Optional[ImageDocument]:
        """Vérifie le cache, télécharge depuis l'URL, valide et persiste l'image en JPEG.

        Point d'entrée commun pour toutes les sources URL-based (Unsplash, WikiMedia).

        Args:
            image_id: Identifiant unique de l'image (nom de fichier sans extension).
            source:   Nom de la source (détermine le sous-dossier de sortie).
            url:      URL directe de l'image (JPEG, PNG, WebP...).
            caption:  Description textuelle de l'image.
            tags:     Liste de tags/catégories.
            licence:  Licence de l'image (ex: ``"CC BY 4.0"``).
            metadata: Métadonnées source-spécifiques additionnelles.

        Returns:
            ImageDocument si téléchargement et validation réussis, None sinon.
        """
        if self._is_cached(source, image_id):
            logger.debug(f"Cache hit : {source}/{image_id}")
            return self._load_from_disk(source, image_id)

        image_bytes = self._fetch_image_bytes(url)
        if image_bytes is None:
            return None

        pil_image = self._validate_image(image_bytes)
        if pil_image is None:
            logger.debug(f"Image rejetée (validation échouée) : {url[:80]}")
            return None

        return self._save_image_document(
            image_id=image_id,
            source=source,
            url=url,
            caption=caption,
            tags=tags,
            licence=licence,
            metadata=metadata,
            pil_image=pil_image,
        )

    @retry(
        stop=stop_after_attempt(3),
        # CAS 3 — min=2s, max=30s : les CDN d'images (Unsplash CloudFront, WikiMedia)
        # peuvent retourner des 503 transitoires sous charge. 2s laisse le temps de se
        # reconnecter au bon nœud CDN.
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=_log_retry,
    )
    def _fetch_image_bytes(self, url: str) -> Optional[bytes]:
        """Télécharge les octets bruts d'une image depuis une URL avec retry.

        Args:
            url: URL directe de l'image (doit pointer vers un fichier image, pas une page HTML).

        Returns:
            Bytes de l'image, ou None si Content-Type non image.

        Raises:
            requests.RequestException: En cas d'erreur réseau (déclenche le retry).
        """
        response = requests.get(url, timeout=30, stream=False)
        response.raise_for_status()

        # CAS 1 — Vérification Content-Type avant de passer les bytes à PIL :
        # certains serveurs retournent une page HTML d'erreur avec HTTP 200 (ex: CDNs
        # configurés avec soft-404). PIL lèverait UnidentifiedImageError sur du HTML,
        # mais le message d'erreur serait trompeur. On détecte ici pour un log clair.
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            logger.warning(
                f"Content-Type non image ('{content_type}') pour {url[:80]}"
            )
            return None

        return response.content

    def _validate_image(self, image_bytes: bytes) -> Optional[Image.Image]:
        """Valide les octets d'une image : format reconnu, non corrompue, dimensions suffisantes.

        # ─── ALGORITHME : Validation d'image binaire ─────────────────────────
        # Problème résolu : détecter les images corrompues, tronquées, trop petites
        #                   ou dans un format non supporté AVANT persistance disque.
        # Approche :        img.load() force le décodage complet du flux compressé.
        # Formule :         valid ⟺ no_exception(load()) ∧ w ≥ MIN_DIM ∧ h ≥ MIN_DIM
        # Référence :       PIL docs — Image.verify() corrompt le state après appel
        #                   (ne peut pas être réutilisé pour conversion). Image.load()
        #                   préféré car laisse l'objet utilisable (Pillow issue #1174).
        # ─────────────────────────────────────────────────────────────────────

        Args:
            image_bytes: Octets bruts de l'image (tout format PIL supporté).

        Returns:
            PIL.Image.Image chargée et prête à être convertie, ou None si invalide.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # CAS 2 — img.load() plutôt que img.verify() : verify() invalide l'objet
            # Image après exécution (il faut réouvrir avec Image.open()). load() force
            # le décodage complet ET laisse l'objet utilisable pour la conversion.
            img.load()
        except (UnidentifiedImageError, OSError, Exception) as exc:
            logger.debug(f"Image corrompue ou format non supporté par PIL : {exc}")
            return None

        w, h = img.size
        if w < self._MIN_DIMENSION or h < self._MIN_DIMENSION:
            # CAS 3 — Rejet < 100×100 : thumbnails (<50×50) et images partiellement
            # téléchargées qui produiraient des embeddings CLIP dégradés par l'upscaling.
            logger.debug(
                f"Image rejetée : dimensions {w}×{h}px "
                f"< seuil {self._MIN_DIMENSION}×{self._MIN_DIMENSION}px"
            )
            return None

        return img

    def _save_image_document(
        self,
        image_id: str,
        source: str,
        url: str,
        caption: str,
        tags: List[str],
        licence: str,
        metadata: Dict[str, Any],
        pil_image: Image.Image,
    ) -> Optional[ImageDocument]:
        """Convertit en JPEG RGB, persiste sur disque et retourne l'ImageDocument.

        Chemin de sortie : ``data/raw/images/{source}/{image_id}.jpg``

        Args:
            image_id:  Identifiant unique (nom de fichier sans extension).
            source:    Nom de la source (sous-dossier de sortie).
            url:       URL d'origine de l'image.
            caption:   Description textuelle.
            tags:      Liste de tags.
            licence:   Licence de l'image.
            metadata:  Métadonnées additionnelles source-spécifiques.
            pil_image: PIL Image déjà ouverte (non nécessairement validée pour les dims
                       dans cette méthode — la validation est faite en amont).

        Returns:
            ImageDocument peuplé ou None si la sauvegarde échoue.
        """
        source_dir = self._raw_images_path / source
        source_dir.mkdir(parents=True, exist_ok=True)
        output_path = source_dir / f"{image_id}.jpg"

        try:
            if pil_image.mode != "RGB":
                # CAS 1 — Conversion RGB obligatoire avant JPEG : le format JPEG ne supporte
                # pas les modes RGBA (canal alpha), P (palette indexée), LA (gris + alpha),
                # CMYK (4 canaux). PIL lève OSError: "cannot write mode RGBA as JPEG" sinon.
                # convert("RGB") fusionne l'éventuel canal alpha sur fond blanc (default).
                pil_image = pil_image.convert("RGB")

            pil_image.save(
                str(output_path),
                format="JPEG",
                quality=self._JPEG_QUALITY,
                optimize=True,       # passe Huffman optimizer (~5-10% réduction taille sans perte)
                # CAS 2 — progressive=False : JPEG progressif est mieux pour le web (affichage
                # progressif) mais le chargement PIL est légèrement plus lent. Pour un pipeline
                # batch offline, le mode baseline (non progressif) est plus rapide à décoder.
                progressive=False,
            )
        except Exception as exc:
            logger.error(f"Impossible de sauvegarder {output_path.name} : {exc}")
            return None

        width, height = pil_image.size
        doc = ImageDocument(
            id=image_id,
            path=str(output_path),
            source=source,
            url=url,
            caption=caption,
            tags=tags,
            licence=licence,
            width=width,
            height=height,
            metadata=metadata,
        )
        self._save_metadata(doc)
        logger.debug(f"Sauvegardé : {output_path.name} ({width}×{height}px)")
        return doc

    # ─────────────────────────────────────────────
    # Cache et persistance
    # ─────────────────────────────────────────────

    def _is_cached(self, source: str, image_id: str) -> bool:
        """Retourne True si l'image et ses métadonnées JSON existent déjà sur disque.

        Args:
            source:   Nom de la source (sous-dossier).
            image_id: Identifiant de l'image.

        Returns:
            True si {id}.jpg ET {id}.json existent tous les deux.
        """
        source_dir = self._raw_images_path / source
        # CAS 1 — Double vérification .jpg ET .json : si le .jpg existe mais pas le .json
        # (crash lors d'un téléchargement précédent), on retélécharge pour rétablir
        # les métadonnées nécessaires à ImageProcessor.
        return (
            (source_dir / f"{image_id}.jpg").exists()
            and (source_dir / f"{image_id}.json").exists()
        )

    def _load_from_disk(self, source: str, image_id: str) -> Optional[ImageDocument]:
        """Charge un ImageDocument depuis le fichier JSON de cache.

        Args:
            source:   Nom de la source (sous-dossier).
            image_id: Identifiant de l'image.

        Returns:
            ImageDocument reconstruit, ou None si le JSON est invalide/absent.
        """
        json_path = self._raw_images_path / source / f"{image_id}.json"
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return ImageDocument(**data)
        except Exception as exc:
            logger.warning(f"Cache JSON corrompu {json_path.name} : {exc}")
            return None

    def _save_metadata(self, doc: ImageDocument) -> None:
        """Persiste les métadonnées d'un ImageDocument en JSON.

        Args:
            doc: ImageDocument à sérialiser. Écrit dans {source}/{id}.json.
        """
        json_path = self._raw_images_path / doc.source / f"{doc.id}.json"
        # CAS 1 — ensure_ascii=False : les captions et descriptions WikiMedia contiennent
        # des caractères non-ASCII (accents, symboles). ensure_ascii=True les convertirait
        # en séquences \uXXXX illisibles dans les fichiers de debug et le frontend.
        json_path.write_text(
            json.dumps(asdict(doc), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _compute_image_id(self, source: str, identifier: str) -> str:
        """Calcule un identifiant stable pour une image à partir de sa source et d'un identifiant natif.

        Args:
            source:     Nom de la source.
            identifier: Identifiant natif de la source (URL, ID Unsplash, chemin COCO...).

        Returns:
            Chaîne hexadécimale de 16 caractères (SHA-256[:16]).
        """
        raw = f"{source}:{identifier}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    @staticmethod
    def _strip_html(text: str) -> str:
        """Supprime les balises HTML d'une chaîne de texte.

        Utilise une regex simple — suffisant pour les descriptions WikiMedia
        (pas de JavaScript inline, pas de CSS complexe).

        Args:
            text: Texte potentiellement contenant du HTML.

        Returns:
            Texte sans balises HTML, espaces superflus supprimés.
        """
        # CAS 2 — regex vs BeautifulSoup : BS4 est 10× plus robuste sur du HTML malformé
        # mais ajoute une dépendance de 50 Mo pour un usage mineur (quelques descriptions).
        # Pour les captions simples de WikiMedia, une regex suffit.
        return re.sub(r"<[^>]+>", " ", text).strip()

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
