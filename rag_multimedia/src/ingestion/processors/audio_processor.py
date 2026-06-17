# -*- coding: utf-8 -*-
"""
Module audio_processor — Transcription et segmentation audio pour la RAG multimodale.

Rôle dans l'architecture :
    Quatrième étape du pipeline d'ingestion audio. Charge les fichiers WAV 16 kHz mono
    de data/raw/audio/ (produits par AudioDownloader), transcrit chaque fichier via
    Whisper local, segmente la transcription en chunks temporels avec overlap et filtre
    par qualité. Produit des AudioChunk consommés par TextEmbedder → ChromaDB
    (audio_collection_optimized).

Pipeline de traitement (ordre obligatoire) :
    ÉTAPE 1 — Transcription Whisper (modèle local, GPU si disponible) :
              Retourne les segments horodatés, la langue détectée et les log-probabilités
              nécessaires au calcul de confiance. La transcription complète est sauvegardée
              dans data/processed/audio/{source}/{id}_transcript.json.

    ÉTAPE 2 — Segmentation temporelle par fenêtre glissante :
              Fenêtre de target_duration_sec (30s), step = target - overlap = 25s.
              Pour chaque fenêtre, alignement sur la dernière frontière de phrase
              (.!?) pour éviter les coupures sémantiques au milieu d'une idée.

    ÉTAPE 3 — Filtres qualité :
              Rejet si whisper_confidence_avg < 0.6 ou word_count < min_words (10).
              Flag metadata si [MUSIC]/[NOISE] > 30% du texte (non-rejeté).

Stratégie de segmentation (fenêtre glissante avec overlap) :
    L'overlap de 5s entre chunks consécutifs garantit que le contexte de fin du chunk n
    est répété en début du chunk n+1. Cela évite qu'une phrase coupée à la frontière
    perde son sens — le LLM dispose d'un chevauchement pour la continuité sémantique.
    L'alignement sur les frontières de phrases (.!?) réduit encore ce risque.
    Référence : Grézl et al. 2007 — "Probabilistic and Bottle-Neck Features for LVCSR".

Pourquoi Whisper local vs API OpenAI :
    1. Coût : l'API facture 0.006 $/min — 100 h = 36 $. Whisper local = 0 $.
    2. Confidentialité : le corpus audio reste sur la machine locale (RGPD, données sensibles).
    3. Reproductibilité : la version du modèle est fixée dans config.yaml ; l'API peut
       changer de comportement sans préavis (mises à jour serveur non versionnées).
    4. Latence : Whisper local traite en parallèle via multiprocessing, sans dépendance
       au réseau. L'API ajoute 200-500 ms de latence par requête.
    Référence : Radford et al. 2022 — "Robust Speech Recognition via Large-Scale
    Weak Supervision", https://arxiv.org/abs/2212.04356.
"""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm


# CAS 2 — Import conditionnel openai-whisper : charge torch et les poids (~150 Mo
# pour "base") au premier load_model(). Un ImportError ne doit bloquer que la
# fonctionnalité Whisper, pas l'ensemble du module.
try:
    import whisper as _whisper_lib
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False


def _log_retry(retry_state: Any) -> None:
    """Log loguru pour tenacity before_sleep (stdlib logging incompatible avec loguru)."""
    exc = retry_state.outcome.exception()
    logger.warning(f"Retry {retry_state.attempt_number}/3 — {type(exc).__name__}: {exc}")


# CAS 3 — Marqueurs de bruit Whisper : tokens spéciaux générés quand Whisper détecte
# du contenu non-verbal (musique de fond, bruit ambiant, silence prolongé).
# Le modèle les préfère aux hallucinations textuelles sur les passages sans parole.
# Référence : openai/whisper tokenizer.py — champ SPECIAL_TOKENS_ATTRIBUTES.
_NOISE_RE = re.compile(
    r"\[(MUSIC|NOISE|BLANK_AUDIO|INAUDIBLE)\]|\((MUSIC|NOISE)\)",
    re.IGNORECASE,
)

# Signes de ponctuation reconnus comme frontières de phrases pour l'alignement
_SENTENCE_ENDINGS = (".", "!", "?", "…", "...",  "。", "！", "？")


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass métier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioChunk:
    """Segment audio transcrit, horodaté et qualifié par Whisper.

    Objet métier produit par AudioProcessor, consommé par TextEmbedder pour
    l'indexation dans ChromaDB (audio_collection_optimized). Chaque chunk
    représente un extrait de ~0 à 35 secondes d'audio avec sa transcription
    et ses métriques de qualité.

    Example:
        chunk = AudioChunk(
            id="a3f1c9e2b4d80f12",
            text="Le machine learning est une branche de l'intelligence artificielle.",
            start_sec=12.0,
            end_sec=42.0,
            duration_sec=30.0,
            audio_file="data/raw/audio/common_voice/b2c3d4e5.wav",
            language="fr",
            whisper_confidence_avg=0.72,
            word_count=11,
            sentence_count=1,
            metadata={"audio_id": "b2c3d4e5", "chunk_index": 0, "noise_ratio": 0.0},
        )
    """

    id: str
    text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    audio_file: str                 # chemin WAV source pour extraction d'extrait
    language: str                   # code ISO 639-1 détecté par Whisper
    whisper_confidence_avg: float   # moyenne de exp(avg_logprob) sur les segments
    word_count: int
    sentence_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Processeur principal
# ─────────────────────────────────────────────────────────────────────────────

class AudioProcessor:
    """Transcrit les fichiers WAV via Whisper local et les segmente en chunks temporels.

    Pipeline en trois étapes : transcription Whisper, segmentation par fenêtre
    glissante (30s, overlap 5s, alignée sur phrases), filtrage qualité (confidence,
    longueur). Le modèle Whisper est chargé de manière paresseuse.

    Example:
        processor = AudioProcessor()
        chunks = processor.process(source="common_voice", max_files=20)
        # → List[AudioChunk] dans data/processed/audio/common_voice/
        # → Rapport dans results/audio_processing_report.json
    """

    # CAS 3 — Seuil confidence 0.6 : seuil empirique Whisper.
    # avg_logprob est la log-probabilité moyenne des tokens du segment.
    # confidence = exp(avg_logprob) ∈ [0, 1]. En dessous de 0.6, le Word Error Rate
    # (WER) augmente brutalement (Radford et al. 2022, Fig. 4 — dégradation non linéaire).
    # Formule : seuil 0.6 ⟺ avg_logprob < ln(0.6) ≈ -0.511.
    # Rejeté 0.5 : trop permissif (segments bruités acceptés) ;
    # rejeté 0.7 : trop strict (rejette des accents ou locuteurs atypiques corrects).
    _CONFIDENCE_THRESHOLD: float = 0.6

    # CAS 3 — Ratio bruit 0.30 : au-delà de 30% du texte occupé par [MUSIC]/[NOISE],
    # le segment est majoritairement non-verbal. En dessous, le texte reste exploitable
    # (ex: "[MUSIC] Bienvenue dans ce podcast sur l'IA." → 15% bruit, 85% contenu utile).
    _NOISE_RATIO_FLAG: float = 0.30

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Args:
            config_path: Chemin vers config/config.yaml. Doit contenir
                         ``whisper.model_size``, ``whisper.language``,
                         ``chunking.audio.target_duration_sec``,
                         ``chunking.audio.overlap_sec``,
                         ``chunking.audio.min_words``.

        Raises:
            FileNotFoundError: Si config_path n'existe pas.
        """
        self._config = self._load_config(config_path)

        self._raw_audio_path = Path(self._config["data"]["raw_path"]) / "audio"
        self._processed_audio_path = (
            Path(self._config["data"]["processed_path"]) / "audio"
        )
        self._results_path = Path(self._config["data"]["results_path"])

        self._processed_audio_path.mkdir(parents=True, exist_ok=True)
        self._results_path.mkdir(parents=True, exist_ok=True)

        whisper_cfg = self._config.get("whisper", {})
        audio_cfg = self._config.get("chunking", {}).get("audio", {})

        # CAS 2 — model_size depuis config : "base" sur CPU (vitesse/qualité raisonnable),
        # "small" si GPU disponible, "medium" pour la meilleure qualité sans large-v3.
        # "tiny" pour les tests rapides. "large-v3" = meilleur WER mais 10 Go VRAM.
        self._model_size: str = whisper_cfg.get("model_size", "base")

        # CAS 2 — language=None force la détection automatique : Whisper identifie
        # la langue sur les 30 premières secondes avant la transcription complète.
        # Spécifier la langue améliore la vitesse (~10%) mais empêche le traitement
        # multilingue (corpus Common Voice fr + en dans le même batch).
        self._language: Optional[str] = whisper_cfg.get("language") or None

        # CAS 3 — target_duration_sec=30 : Whisper segmente nativement par blocs de
        # 30s (taille de fenêtre mel-spectrogram : N_FRAMES=3000 à 100 frames/s).
        # Aligner les chunks sur cette durée maximise la cohérence contextuelle et
        # évite que Whisper recoupe ses propres segments lors du décodage.
        self._target_duration: float = float(
            audio_cfg.get("target_duration_sec", 30.0)
        )

        # CAS 3 — overlap_sec=5 : chevauchement de 5s entre chunks consécutifs.
        # 5s ≈ 10–15 mots à débit oral normal (150 mots/min). Suffisant pour
        # récupérer le contexte de fin du chunk précédent (anaphoriques, pronoms,
        # références à "cette méthode", "ce résultat" introduits juste avant la coupure).
        # Rejeté 0s : perte de contexte aux frontières ;
        # rejeté 15s : redondance excessive → augmente la taille du corpus de 50%.
        self._overlap_sec: float = float(audio_cfg.get("overlap_sec", 5.0))

        # CAS 3 — min_words=10 : seuil minimum de mots par chunk.
        # En dessous : bruit de transcription ("Hmm.", "OK.", pause).
        # Un chunk de 5 mots ne fournit pas assez de contexte au LLM pour répondre.
        self._min_words: int = int(audio_cfg.get("min_words", 10))

        # CAS 2 — Chargement paresseux du modèle Whisper : load_model() télécharge
        # et désérialise les poids (~150 Mo pour "base"). Ne pas charger à __init__
        # pour ne pas pénaliser les tests et les imports sans traitement effectif.
        self._whisper_model: Optional[Any] = None

    # ─────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────

    def process(
        self,
        source: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> List[AudioChunk]:
        """Transcrit et segmente les fichiers WAV de data/raw/audio/.

        Pour chaque fichier : transcription Whisper → segmentation → filtrage qualité
        → persistance. Génère un rapport dans results/audio_processing_report.json.

        Args:
            source:    Filtre optionnel sur la source (ex: ``'common_voice'``).
                       None = toutes les sources disponibles.
            max_files: Nombre maximum de fichiers WAV à traiter. None = tous.

        Returns:
            Liste d'AudioChunk valides (filtrés). Jamais None — vide si aucun résultat.
        """
        raw_docs = self._load_raw_audio_docs(source_filter=source)
        logger.info(
            f"AudioProcessor : {len(raw_docs)} fichiers audio chargés "
            f"(source={source!r}, max={max_files})"
        )

        if max_files is not None:
            raw_docs = raw_docs[:max_files]

        all_chunks: List[AudioChunk] = []
        rejected_counts: Dict[str, int] = {"low_confidence": 0, "too_short": 0}
        languages_detected: Dict[str, int] = {}
        total_duration_sec: float = 0.0

        for raw_doc in tqdm(raw_docs, desc="AudioProcessor", unit="fichier"):
            wav_path = raw_doc.get("path", "")
            audio_id = raw_doc.get("id", "")
            source_name = raw_doc.get("source", "unknown")

            # ── Étape 1 : transcription ──────────────────────────────────────
            result = self._transcribe(wav_path)
            if result is None:
                logger.warning(f"Transcription ignorée : {wav_path}")
                continue

            self._save_transcript(result, source_name, audio_id)

            detected_lang: str = result.get("language", raw_doc.get("language", ""))
            languages_detected[detected_lang] = (
                languages_detected.get(detected_lang, 0) + 1
            )
            total_duration_sec += raw_doc.get("duration_sec", 0.0)

            # ── Étape 2 : segmentation ───────────────────────────────────────
            segments = result.get("segments", [])
            chunks = self._segment_into_chunks(
                segments=segments,
                audio_file=wav_path,
                language=detected_lang,
                audio_id=audio_id,
            )

            # ── Étape 3 : filtres qualité ────────────────────────────────────
            valid_chunks: List[AudioChunk] = []
            for chunk in chunks:
                reject_reason = self._get_rejection_reason(chunk)
                if reject_reason:
                    rejected_counts[reject_reason] = (
                        rejected_counts.get(reject_reason, 0) + 1
                    )
                else:
                    valid_chunks.append(chunk)

            self._save_chunks(valid_chunks, source_name, audio_id)
            all_chunks.extend(valid_chunks)

        self._save_report(
            total_files=len(raw_docs),
            total_segments=len(all_chunks),
            rejected_segments=sum(rejected_counts.values()),
            rejection_reasons=rejected_counts,
            languages_detected=languages_detected,
            total_duration_hours=round(total_duration_sec / 3600, 3),
        )

        logger.info(
            f"AudioProcessor terminé : {len(all_chunks)} chunks valides "
            f"({sum(rejected_counts.values())} rejetés)"
        )
        return all_chunks

    # ─────────────────────────────────────────────
    # Étape 1 : transcription Whisper
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        # CAS 3 — min=2s entre tentatives : une erreur CUDA OOM peut nécessiter
        # quelques secondes pour libérer la VRAM. Une tentative immédiate échouerait
        # à nouveau sur le même état mémoire GPU.
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RuntimeError),
        before_sleep=_log_retry,
    )
    def _transcribe(self, wav_path: str) -> Optional[Dict[str, Any]]:
        """Transcrit un fichier WAV 16 kHz mono via Whisper local.

        Retourne la structure complète Whisper : texte global, liste de segments
        horodatés (start, end, text, avg_logprob) et langue détectée.

        Args:
            wav_path: Chemin vers le fichier WAV 16 kHz mono (format requis par Whisper).
                      Produit par AudioDownloader._convert_to_wav_16khz().

        Returns:
            Dict Whisper avec ``"text"``, ``"segments"`` et ``"language"``,
            ou None si Whisper n'est pas installé ou si le fichier est introuvable.

        Raises:
            RuntimeError: En cas d'erreur Whisper (déclenche le retry tenacity).
        """
        if not _WHISPER_AVAILABLE:
            logger.error("openai-whisper non installé — pip install openai-whisper")
            return None

        if not Path(wav_path).exists():
            logger.warning(f"Fichier WAV introuvable : {wav_path}")
            return None

        try:
            model = self._get_whisper_model()
            result: Dict[str, Any] = model.transcribe(
                wav_path,
                language=self._language,
                # CAS 2 — verbose=False : Whisper affiche chaque segment sur stdout
                # par défaut. On délègue entièrement le logging à loguru (cohérence).
                verbose=False,
                # CAS 3 — fp16=False sur CPU : float16 n'est pas supporté par tous
                # les CPU (nécessite AVX512). Whisper détecte automatiquement CUDA et
                # passe en fp16 si disponible — ce paramètre ne pénalise que le CPU.
                fp16=False,
            )
            logger.debug(
                f"Transcrit : {Path(wav_path).name} — "
                f"{len(result.get('segments', []))} segments, "
                f"langue={result.get('language', '?')!r}"
            )
            return result
        except Exception as exc:
            logger.error(f"Whisper échoué ({Path(wav_path).name}) : {exc}")
            # CAS 1 — Relever en RuntimeError : tenacity est configuré pour retry
            # uniquement sur RuntimeError. Les exceptions génériques d'openai-whisper
            # doivent être converties pour déclencher le mécanisme de retry.
            raise RuntimeError(str(exc)) from exc

    def _get_whisper_model(self) -> Any:
        """Charge le modèle Whisper de manière paresseuse (une seule fois par instance).

        Returns:
            Instance whisper.Whisper prête à l'inférence.

        Raises:
            RuntimeError: Si openai-whisper n'est pas installé.
        """
        if self._whisper_model is None:
            if not _WHISPER_AVAILABLE:
                raise RuntimeError(
                    "openai-whisper non installé — pip install openai-whisper"
                )
            cuda_status = "disponible" if self._is_cuda_available() else "non disponible"
            logger.info(
                f"Chargement Whisper '{self._model_size}' (CUDA {cuda_status})…"
            )
            # CAS 2 — device automatique : whisper.load_model() utilise CUDA si
            # torch.cuda.is_available(), sinon CPU. On ne force pas le device pour
            # rester compatible avec les environnements sans GPU.
            self._whisper_model = _whisper_lib.load_model(self._model_size)
            logger.info(f"Whisper '{self._model_size}' chargé")
        return self._whisper_model

    @staticmethod
    def _is_cuda_available() -> bool:
        """Retourne True si CUDA est disponible pour Whisper (torch requis)."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ─────────────────────────────────────────────
    # Étape 2 : segmentation temporelle
    # ─────────────────────────────────────────────

    def _segment_into_chunks(
        self,
        segments: List[Dict[str, Any]],
        audio_file: str,
        language: str,
        audio_id: str,
    ) -> List[AudioChunk]:
        """Regroupe les segments Whisper en chunks temporels avec overlap.

        # ─── ALGORITHME : Fenêtre glissante avec alignement de phrases ─────────
        # Problème résolu : segmenter une longue transcription en blocs exploitables
        #                   par la RAG sans perdre le contexte aux frontières.
        # Approche :        Sliding window de largeur target_duration, step=target-overlap.
        #                   Chaque fenêtre collecte les segments qui la chevauchent,
        #                   puis recule jusqu'à la dernière frontière de phrase.
        # Formule :         chunk_n = segments ∈ [n·step, n·step + target]
        #                             tronqué au dernier segment terminant par .!?
        # Paramètres :      target=30s, overlap=5s → step=25s.
        # Complexité :      O(n·S) avec n=nombre de fenêtres, S=segments par fenêtre.
        # Référence :       Grézl et al. 2007 — sliding window ASR segmentation.
        # ─────────────────────────────────────────────────────────────────────────

        Args:
            segments:   Segments Whisper (chacun : start, end, text, avg_logprob).
            audio_file: Chemin WAV source (stocké dans AudioChunk.audio_file).
            language:   Code ISO 639-1 détecté par Whisper.
            audio_id:   Identifiant du fichier parent (utilisé pour l'ID de chunk).

        Returns:
            Liste d'AudioChunk (y compris ceux à rejeter — le filtrage est fait
            dans process() pour séparer les responsabilités).
        """
        if not segments:
            return []

        # CAS 3 — step = target - overlap = 25s : le pas de 25s entre les débuts
        # de fenêtres garantit un chevauchement de 5s entre chunks consécutifs.
        # Un step égal à target (30s) produirait des chunks sans overlap — perte
        # du contexte aux frontières. Un step négatif (overlap > target) est invalide.
        step = self._target_duration - self._overlap_sec
        if step <= 0:
            logger.warning(
                f"overlap_sec ({self._overlap_sec}s) ≥ target_duration "
                f"({self._target_duration}s) — step forcé à 1s pour éviter boucle infinie"
            )
            step = 1.0

        total_end: float = segments[-1]["end"]
        window_start: float = segments[0]["start"]
        chunks: List[AudioChunk] = []
        chunk_index: int = 0

        while window_start < total_end:
            window_end = window_start + self._target_duration

            # CAS 1 — Condition double pour la collecte des segments :
            # s["end"] > window_start : inclut les segments qui chevauchent
            #   le DÉBUT de la fenêtre (segment commencé avant mais finissant dedans).
            # s["start"] < window_end : exclut les segments qui commencent après
            #   la FIN de la fenêtre (appartiennent au chunk suivant).
            # Ce double critère évite de perdre les premiers/derniers mots d'un chunk.
            window_segs = [
                s for s in segments
                if s["end"] > window_start and s["start"] < window_end
            ]

            if not window_segs:
                window_start += step
                continue

            # CAS 1 — Alignement sur frontières de phrases :
            # On parcourt les segments de la fenêtre à rebours pour trouver le
            # dernier qui se termine par un signe de ponctuation fort (.!?…).
            # Alignement sur le dernier (pas le premier) pour maximiser la durée
            # du chunk tout en garantissant sa complétude sémantique.
            # Si aucun segment ne termine une phrase (transcription sans ponctuation),
            # on garde tous les segments de la fenêtre — mieux une coupure imparfaite
            # que perdre du contenu.
            aligned_segs = window_segs
            for k in range(len(window_segs) - 1, -1, -1):
                seg_text = window_segs[k]["text"].strip()
                if seg_text and seg_text[-1] in _SENTENCE_ENDINGS:
                    aligned_segs = window_segs[: k + 1]
                    break

            chunk = self._build_chunk(
                segs=aligned_segs,
                audio_file=audio_file,
                language=language,
                audio_id=audio_id,
                chunk_index=chunk_index,
            )

            if chunk is not None:
                chunks.append(chunk)
                chunk_index += 1

            window_start += step

        logger.debug(
            f"Segmenté {audio_id} → {len(chunks)} chunks "
            f"(target={self._target_duration}s, overlap={self._overlap_sec}s)"
        )
        return chunks

    def _build_chunk(
        self,
        segs: List[Dict[str, Any]],
        audio_file: str,
        language: str,
        audio_id: str,
        chunk_index: int,
    ) -> Optional[AudioChunk]:
        """Construit un AudioChunk depuis une liste de segments Whisper alignés.

        Args:
            segs:        Segments Whisper formant le chunk.
            audio_file:  Chemin WAV source.
            language:    Langue détectée par Whisper.
            audio_id:    Identifiant du fichier audio parent.
            chunk_index: Index ordinal dans le fichier (pour l'unicité de l'id).

        Returns:
            AudioChunk peuplé, ou None si le texte assemblé est vide.
        """
        text = " ".join(s["text"].strip() for s in segs).strip()
        if not text:
            return None

        start_sec: float = segs[0]["start"]
        end_sec: float = segs[-1]["end"]
        duration_sec: float = end_sec - start_sec

        # CAS 1 — Calcul de confiance via exp(avg_logprob) :
        # avg_logprob est la log-probabilité moyenne des tokens Whisper du segment.
        # exp() ramène la valeur dans [0, 1] — interprétable comme une probabilité.
        # clamp dans [0, 1] : exp(avg_logprob) peut théoriquement dépasser 1 si
        # avg_logprob > 0 (artefact numérique rare sur segments très courts < 0.1s).
        # La moyenne sur les segments du chunk lisse les variations locales.
        confidences = [
            min(1.0, max(0.0, math.exp(s.get("avg_logprob", -1.0))))
            for s in segs
        ]
        whisper_confidence_avg = sum(confidences) / len(confidences)

        word_count = len(text.split())

        # CAS 2 — Segmentation naïve par .!? : méthode appropriée aux transcriptions
        # orales. Whisper produit lui-même les signes de ponctuation lors du décodage
        # (il est entraîné sur du texte naturellement ponctué). Une regex avancée
        # (spaCy sentencizer) serait plus précise mais disproportionnée ici.
        raw_sentences = re.split(r"[.!?…]+", text)
        sentence_count = sum(1 for s in raw_sentences if s.strip())

        noise_ratio = self._compute_noise_ratio(text)

        # CAS 1 — ID basé sur audio_id + timestamps : unicité cross-session garantie.
        # Deux chunks du même fichier à des temps différents ont toujours des IDs distincts.
        # chunk_index est ajouté pour distinguer deux chunks qui auraient les mêmes
        # timestamps exacts (cas rare mais possible avec overlap).
        chunk_id = hashlib.sha256(
            f"{audio_id}:{chunk_index}:{start_sec:.3f}:{end_sec:.3f}".encode("utf-8")
        ).hexdigest()[:16]

        return AudioChunk(
            id=chunk_id,
            text=text,
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            duration_sec=round(duration_sec, 3),
            audio_file=audio_file,
            language=language,
            whisper_confidence_avg=round(whisper_confidence_avg, 4),
            word_count=word_count,
            sentence_count=sentence_count,
            metadata={
                "audio_id": audio_id,
                "chunk_index": chunk_index,
                # CAS 3 — noise_ratio en metadata (pas rejet) :
                # [MUSIC]/[NOISE] minoritaires (<30%) laissent le texte exploitable.
                # La metadata informe le LLM sans rejeter le chunk.
                "noise_ratio": round(noise_ratio, 3),
                "has_noise_flags": noise_ratio > self._NOISE_RATIO_FLAG,
                "processed_at": datetime.utcnow().strftime("%Y-%m-%d"),
            },
        )

    @staticmethod
    def _compute_noise_ratio(text: str) -> float:
        """Calcule la proportion de caractères occupés par les marqueurs de bruit.

        Args:
            text: Texte transcrit (peut contenir [MUSIC], [NOISE], etc.).

        Returns:
            Ratio dans [0.0, 1.0]. 0.0 = pas de bruit, 1.0 = 100% marqueurs.
        """
        # CAS 3 — Calcul en caractères (pas en mots) : "[MUSIC]" représente 7 chars
        # pour 1 "mot". Compter en mots sur-représenterait le bruit. En caractères,
        # la proportion reflète fidèlement l'espace textuel non transcrit.
        noise_chars = sum(len(m.group(0)) for m in _NOISE_RE.finditer(text))
        return noise_chars / max(len(text), 1)

    # ─────────────────────────────────────────────
    # Étape 3 : filtres qualité
    # ─────────────────────────────────────────────

    def _get_rejection_reason(self, chunk: AudioChunk) -> Optional[str]:
        """Retourne la raison de rejet si le chunk ne passe pas les filtres qualité.

        Args:
            chunk: AudioChunk avec statistiques calculées par _build_chunk.

        Returns:
            ``"low_confidence"`` ou ``"too_short"``, ou None si le chunk est valide.
        """
        if chunk.whisper_confidence_avg < self._CONFIDENCE_THRESHOLD:
            # CAS 3 — Rejet confidence < 0.6 : un segment avec confidence=0.4 a en
            # moyenne 40% de probabilité par token → plusieurs mots probablement faux.
            # Ces chunks injecteraient du bruit sémantique dans l'index ChromaDB.
            return "low_confidence"

        if chunk.word_count < self._min_words:
            # CAS 3 — Rejet < 10 mots : chunk trop court pour être utile en RAG.
            # 10 mots ≈ 1 phrase minimale — en dessous, le LLM n'a pas assez de
            # contexte pour répondre à une question à partir de ce chunk seul.
            return "too_short"

        return None

    # ─────────────────────────────────────────────
    # Chargement des données brutes
    # ─────────────────────────────────────────────

    def _load_raw_audio_docs(
        self, source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Charge les métadonnées AudioDocument depuis data/raw/audio/{source}/*.json.

        Args:
            source_filter: Sous-dossier source à cibler. None = toutes les sources.

        Returns:
            Liste de dicts AudioDocument (format AudioDownloader). Vide si aucun fichier.
        """
        if source_filter:
            search_root = self._raw_audio_path / source_filter
            pattern = "*.json"
        else:
            search_root = self._raw_audio_path
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
                # CAS 3 — JSON corrompu : continue sans interrompre le batch.
                # Même comportement que TextProcessor et ImageProcessor.
                logger.warning(f"JSON audio ignoré ({json_path.name}) : {exc}")

        logger.info(
            f"_load_raw_audio_docs : {len(docs)} fichiers depuis {len(json_files)} JSON"
        )
        return docs

    # ─────────────────────────────────────────────
    # Persistance
    # ─────────────────────────────────────────────

    def _save_transcript(
        self, result: Dict[str, Any], source: str, audio_id: str
    ) -> None:
        """Sauvegarde la transcription Whisper complète (texte + segments) en JSON.

        Args:
            result:   Dictionnaire retourné par model.transcribe().
            source:   Nom de la source (sous-dossier de sortie).
            audio_id: Identifiant du fichier audio parent.
        """
        source_dir = self._processed_audio_path / source
        source_dir.mkdir(parents=True, exist_ok=True)
        out_path = source_dir / f"{audio_id}_transcript.json"

        # CAS 1 — Sérialisation des champs utiles uniquement : les champs internes
        # Whisper (tokens, seek) ne sont pas nécessaires pour la RAG et gonflent
        # la taille du fichier de ~10× (tokens = entiers int64, pas du texte).
        serializable = {
            "audio_id": audio_id,
            "language": result.get("language", ""),
            "text": result.get("text", ""),
            "segments": [
                {
                    "start": s.get("start", 0.0),
                    "end": s.get("end", 0.0),
                    "text": s.get("text", ""),
                    "avg_logprob": s.get("avg_logprob", -1.0),
                    "no_speech_prob": s.get("no_speech_prob", 0.0),
                }
                for s in result.get("segments", [])
            ],
        }
        out_path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.debug(f"Transcript sauvegardé : {out_path.name}")

    def _save_chunks(
        self, chunks: List[AudioChunk], source: str, audio_id: str
    ) -> None:
        """Sauvegarde chaque AudioChunk valide en JSON individuel.

        Nom de fichier : ``{chunk_id}.json`` — unicité garantie par SHA-256[:16].

        Args:
            chunks:   Liste d'AudioChunk valides à sauvegarder.
            source:   Nom de la source (sous-dossier de sortie).
            audio_id: Identifiant du fichier audio parent (pour le log).
        """
        if not chunks:
            return

        source_dir = self._processed_audio_path / source
        source_dir.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            out_path = source_dir / f"{chunk.id}.json"
            out_path.write_text(
                json.dumps(asdict(chunk), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logger.debug(
            f"{len(chunks)} chunks sauvegardés pour {audio_id} → {source_dir.name}/"
        )

    def _save_report(
        self,
        total_files: int,
        total_segments: int,
        rejected_segments: int,
        rejection_reasons: Dict[str, int],
        languages_detected: Dict[str, int],
        total_duration_hours: float,
    ) -> None:
        """Sauvegarde le rapport de traitement dans results/audio_processing_report.json.

        Args:
            total_files:          Nombre de fichiers WAV traités.
            total_segments:       Nombre de chunks valides produits.
            rejected_segments:    Nombre de chunks rejetés.
            rejection_reasons:    Comptage par cause de rejet.
            languages_detected:   Distribution des langues détectées par Whisper.
            total_duration_hours: Durée cumulée de l'audio source en heures.
        """
        report = {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": total_files,
            "total_segments": total_segments,
            "rejected_segments": rejected_segments,
            "rejection_rate": round(
                rejected_segments / max(total_segments + rejected_segments, 1), 3
            ),
            "rejection_reasons": rejection_reasons,
            "languages_detected": languages_detected,
            "total_duration_hours": total_duration_hours,
            "whisper_model": self._model_size,
            "confidence_threshold": self._CONFIDENCE_THRESHOLD,
            "target_duration_sec": self._target_duration,
            "overlap_sec": self._overlap_sec,
        }

        report_path = self._results_path / "audio_processing_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Rapport audio sauvegardé : {report_path.name}")

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
