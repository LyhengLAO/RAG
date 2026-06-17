# -*- coding: utf-8 -*-
"""
Module audio_downloader — Téléchargeur de fichiers audio multimodal.

Rôle dans l'architecture :
    Troisième composant du pipeline d'ingestion. Récupère des fichiers audio depuis
    quatre sources hétérogènes (LibriSpeech, Common Voice, FreeSound, YouTube),
    les convertit uniformément en WAV 16 kHz mono via ffmpeg et les persiste dans
    data/raw/audio/{source}/ pour transcription ultérieure par AudioProcessor (Whisper).
    Produit des AudioDocument consommés par AudioProcessor → TextEmbedder → ChromaDB.

Pourquoi WAV 16 kHz mono :
    OpenAI Whisper exige impérativement 16 000 Hz mono PCM 16-bit (WAV).
    Sans pré-conversion, Whisper effectue la normalisation en interne à chaque
    appel, ce qui double le temps de traitement et peut produire des artefacts
    de resampling lorsque le ratio src/dst n'est pas un entier.
    Pré-convertir en 16 kHz garantit des performances de transcription
    reproductibles et minimise la mémoire GPU lors des inférences batch.
    Référence : openai/whisper — audio.py, fonction load_audio(), ligne 62.

Pourquoi ces quatre sources :
    - LibriSpeech (torchaudio) : 1 000 h de lecture anglaise haute qualité,
                                  standard de référence ASR depuis 2015.
                                  Sous-ensembles téléchargeables (test-clean < 350 Mo).
    - Common Voice (HuggingFace): 10 000 h+ en 100+ langues, licence CC0,
                                  annotations humaines vérifiées — idéal pour évaluation
                                  multilingue. Accès via HF_TOKEN.
    - FreeSound (freesound-python): sons environnementaux, musique, foley —
                                  diversité sonore maximale avec métadonnées riches
                                  (durée exacte, tags, licence par son).
    - YouTube (yt-dlp)           : podcasts, conférences, interviews — contenu réel.
                                  yt-dlp gère automatiquement les signatures YouTube.

Format de sortie :
    - Audio      : data/raw/audio/{source}/{id}.wav  — PCM 16-bit, 16 000 Hz, mono
    - Métadonnées: data/raw/audio/{source}/{id}.json — AudioDocument sérialisé
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ffmpeg
import requests
import yaml
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

# CAS 2 — Imports conditionnels : torchaudio, datasets, freesound et yt-dlp sont
# des dépendances lourdes ou optionnelles. Un ImportError ne doit bloquer que la
# source concernée, pas tout le module.
try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

try:
    from datasets import load_dataset as _hf_load_dataset
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    _HF_DATASETS_AVAILABLE = False

try:
    import freesound as _freesound_lib
    _FREESOUND_AVAILABLE = True
except ImportError:
    _FREESOUND_AVAILABLE = False

try:
    import yt_dlp
    _YTDLP_AVAILABLE = True
except ImportError:
    _YTDLP_AVAILABLE = False


def _log_retry(retry_state: Any) -> None:
    """Log loguru pour les tentatives tenacity (stdlib logging incompatible avec loguru)."""
    exc = retry_state.outcome.exception()
    logger.warning(f"Retry {retry_state.attempt_number}/3 — {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass métier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioDocument:
    """Représentation normalisée d'un fichier audio téléchargé et converti.

    Objet métier partagé entre AudioDownloader → AudioProcessor → TextEmbedder.
    Tous les champs sont sérialisables en JSON via dataclasses.asdict().
    Le champ ``path`` pointe vers le WAV 16 kHz mono persisté sur disque.

    Example:
        doc = AudioDocument(
            id="a3f1c9e2b4d80f12",
            path="data/raw/audio/freesound/a3f1c9e2b4d80f12.wav",
            source="freesound",
            url="https://freesound.org/people/user/sounds/123/",
            title="Piano improvisation",
            duration_sec=42.3,
            sample_rate=16000,
            language="",
            metadata={"freesound_id": 123, "tags": ["piano", "music"]},
        )
    """

    id: str
    path: str
    source: str        # 'librispeech' | 'common_voice' | 'freesound' | 'youtube'
    url: str
    title: str
    duration_sec: float
    sample_rate: int   # toujours 16000 après conversion
    language: str      # code ISO 639-1 ou "" si non applicable
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Downloader principal
# ─────────────────────────────────────────────────────────────────────────────

class AudioDownloader:
    """Télécharge et normalise des fichiers audio en WAV 16 kHz mono.

    Applique le filtre de durée (max 30 min) avant téléchargement lorsque possible
    (FreeSound, YouTube). Implémente un cache disque par id : si {id}.wav et
    {id}.json existent, le fichier est retourné sans appel réseau ni re-conversion.

    Example:
        downloader = AudioDownloader()
        docs = downloader.download("freesound", "piano jazz", max_files=10)
        # → 10 .wav + 10 .json dans data/raw/audio/freesound/
    """

    # CAS 3 — 30 min = 1 800 s : limite empirique Whisper en mémoire GPU (VRAM).
    # Whisper large-v3 charge le signal complet en mémoire — un fichier de 60 min
    # nécessite ~2 Go de VRAM. 30 min est le compromis couverture/mémoire raisonnable.
    # Référence : openai/whisper — transcribe.py, paramètre max audio length.
    _MAX_DURATION_SEC: int = 30 * 60

    # CAS 3 — 16 000 Hz : fréquence d'échantillonnage minimale requise par Whisper.
    # Valeur hardcodée car il s'agit d'une contrainte Whisper, pas d'une préférence.
    _TARGET_SAMPLE_RATE: int = 16_000

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Args:
            config_path: Chemin vers config/config.yaml. Doit contenir ``data.raw_path``.

        Raises:
            FileNotFoundError: Si config_path n'existe pas.
        """
        load_dotenv()

        self._config = self._load_config(config_path)
        self._raw_audio_path = Path(self._config["data"]["raw_path"]) / "audio"
        self._raw_audio_path.mkdir(parents=True, exist_ok=True)

        # CAS 3 — Clé FreeSound depuis env : API key du compte FreeSound Developer
        # (https://freesound.org/apiv2/apply/). Limite documentée : 2 000 req/jour
        # sur compte free. Jamais dans config.yaml (fichier versionné).
        self._freesound_key: str = os.environ.get("FREESOUND_API_KEY", "")
        if not self._freesound_key:
            logger.warning(
                "FREESOUND_API_KEY non définie dans .env — "
                "source 'freesound' désactivée. Voir .env.example."
            )

        # CAS 3 — HF_TOKEN pour Common Voice : le dataset Common Voice sur HuggingFace
        # requiert l'acceptation des conditions d'utilisation + authentification.
        # Sans token, load_dataset retourne une erreur 401.
        self._hf_token: str = os.environ.get("HF_TOKEN", "")

    # ─────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────

    def download(self, source: str, query: str, max_files: int = 10) -> List[AudioDocument]:
        """Télécharge des fichiers audio depuis la source demandée, avec filtre et cache.

        Args:
            source:    Source à interroger. Valeurs acceptées :
                       ``'librispeech'``, ``'common_voice'``, ``'freesound'``, ``'youtube'``.
            query:     Requête dont le format dépend de la source (voir _download_*).
            max_files: Nombre maximum de fichiers audio à retourner (après conversion).

        Returns:
            Liste d'AudioDocument normalisés. Jamais None, peut être vide.

        Raises:
            ValueError: Si source n'est pas dans les valeurs acceptées.
        """
        if source not in {"librispeech", "common_voice", "freesound", "youtube"}:
            raise ValueError(
                f"Source inconnue : '{source}'. "
                f"Valeurs : 'librispeech', 'common_voice', 'freesound', 'youtube'."
            )

        logger.info(
            f"Téléchargement audio | source='{source}' | "
            f"query='{query[:60]}' | max={max_files}"
        )

        _dispatch = {
            "librispeech": self._download_librispeech,
            "common_voice": self._download_common_voice,
            "freesound": self._download_freesound,
            "youtube": self._download_youtube,
        }
        documents = _dispatch[source](query, max_files)
        logger.info(f"[{source}] {len(documents)} fichiers audio valides retournés")
        return documents

    # ─────────────────────────────────────────────
    # Source : LibriSpeech via torchaudio
    # ─────────────────────────────────────────────

    def _download_librispeech(self, query: str, max_files: int) -> List[AudioDocument]:
        """Télécharge des énoncés LibriSpeech via torchaudio.datasets.LIBRISPEECH.

        Format du paramètre ``query`` : nom du sous-ensemble LibriSpeech.
          - ``"test-clean"``     — ~350 Mo, 5h, anglais propre    (recommandé portfolio)
          - ``"dev-clean"``      — ~360 Mo, 5h, anglais propre
          - ``"train-clean-100"``— ~6 Go, 100h (attention : très volumineux)

        Args:
            query:     Nom du sous-ensemble LibriSpeech (default: ``"test-clean"``).
            max_files: Nombre maximum d'énoncés retournés.

        Returns:
            Liste d'AudioDocument. Vide si torchaudio non installé.
        """
        if not _TORCHAUDIO_AVAILABLE:
            logger.error(
                "torchaudio non installé — source 'librispeech' indisponible. "
                "pip install torchaudio"
            )
            return []

        subset = query.strip() or "test-clean"
        root_dir = self._raw_audio_path / "librispeech"
        root_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LibriSpeech : chargement du sous-ensemble '{subset}'")

        # CAS 2 — download=True : torchaudio gère son propre cache dans root_dir.
        # Si le sous-ensemble est déjà téléchargé, il est rechargé depuis le disque.
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(root_dir),
            url=subset,
            download=True,
            folder_in_archive="LibriSpeech",
        )

        documents: List[AudioDocument] = []

        for idx in tqdm(range(min(max_files, len(dataset))), desc="LibriSpeech", unit="utt"):
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[idx]

            # CAS 1 — Identifiant composite : LibriSpeech identifie chaque énoncé
            # par (speaker_id, chapter_id, utterance_id). On concatène pour l'unicité.
            native_id = f"{speaker_id}-{chapter_id}-{utterance_id:04d}"
            audio_id = self._compute_audio_id("librispeech", native_id)

            if self._is_cached("librispeech", audio_id):
                doc = self._load_from_disk("librispeech", audio_id)
                if doc:
                    documents.append(doc)
                continue

            duration_sec = waveform.shape[1] / sample_rate
            if not self._is_duration_acceptable(duration_sec):
                logger.debug(f"LibriSpeech : énoncé {native_id} filtré ({duration_sec:.1f}s > {self._MAX_DURATION_SEC}s)")
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                # CAS 1 — Sauvegarde intermédiaire : torchaudio.save() écrit le tenseur
                # au sample_rate original. ffmpeg convertit ensuite en 16 kHz.
                torchaudio.save(str(tmp_path), waveform, sample_rate)

                doc = self._convert_and_save(
                    input_path=tmp_path,
                    audio_id=audio_id,
                    source="librispeech",
                    url=f"https://www.openslr.org/12/",
                    title=f"LibriSpeech {native_id}",
                    language="en",
                    metadata={
                        "subset": subset,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "utterance_id": utterance_id,
                        "transcript": transcript[:500],
                    },
                )
                if doc:
                    documents.append(doc)
            finally:
                tmp_path.unlink(missing_ok=True)

        return documents

    # ─────────────────────────────────────────────
    # Source : Common Voice via HuggingFace
    # ─────────────────────────────────────────────

    def _download_common_voice(self, query: str, max_files: int) -> List[AudioDocument]:
        """Télécharge des clips Common Voice via HuggingFace datasets.

        Format du paramètre ``query`` :
          - Langue seule    : ``"fr"``
          - Langue + split  : ``"fr:validation"``

        Requiert ``HF_TOKEN`` dans .env et acceptation des conditions Common Voice
        sur https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0.

        Args:
            query:     Code de langue ISO 639-1 avec split optionnel.
            max_files: Nombre maximum de clips retournés.

        Returns:
            Liste d'AudioDocument. Vide si datasets non installé ou HF_TOKEN manquant.
        """
        if not _HF_DATASETS_AVAILABLE:
            logger.error(
                "datasets non installé — source 'common_voice' indisponible. "
                "pip install datasets"
            )
            return []

        parts = query.split(":")
        language = parts[0].strip()
        # CAS 3 — split="validation" par défaut : plus petit et sans doublons
        # par rapport à "train". Idéal pour un corpus de test rapide.
        split = parts[1].strip() if len(parts) > 1 else "validation"

        logger.info(f"Common Voice : langue='{language}' split='{split}'")

        # CAS 3 — trust_remote_code=True obligatoire pour Common Voice HF :
        # le dataset nécessite un script de chargement custom pour décoder les MP3.
        # Rejeté False : lève NotImplementedError sur les fichiers MP3 encodés.
        dataset = _hf_load_dataset(
            "mozilla-foundation/common_voice_11_0",
            language,
            split=split,
            trust_remote_code=True,
            token=self._hf_token or None,
        )

        documents: List[AudioDocument] = []
        sample_size = min(max_files, len(dataset))

        for idx in tqdm(range(sample_size), desc="Common Voice", unit="clip"):
            example = dataset[idx]

            # CAS 1 — Accès à example['audio'] : HuggingFace décode le MP3 à la volée
            # et retourne un dict {'array': np.ndarray, 'sampling_rate': int, 'path': str}.
            audio_data = example.get("audio", {})
            if not audio_data:
                logger.warning(f"Common Voice : exemple [{idx}] sans champ 'audio'")
                continue

            array = audio_data.get("array")
            sample_rate = audio_data.get("sampling_rate", 48000)
            sentence = example.get("sentence", "")

            if array is None or len(array) == 0:
                continue

            duration_sec = len(array) / sample_rate
            if not self._is_duration_acceptable(duration_sec):
                logger.debug(f"Common Voice [{idx}] filtré : {duration_sec:.1f}s")
                continue

            audio_id = self._compute_audio_id(
                "common_voice", f"{language}:{split}:{idx}"
            )

            if self._is_cached("common_voice", audio_id):
                doc = self._load_from_disk("common_voice", audio_id)
                if doc:
                    documents.append(doc)
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                import soundfile as sf
                # CAS 2 — soundfile plutôt que scipy.io.wavfile : soundfile gère
                # correctement les tableaux float32 normalisés [-1, 1] de HuggingFace.
                # scipy.io.wavfile attend des entiers 16-bit et produirait des clippings.
                sf.write(str(tmp_path), array, sample_rate, subtype="PCM_16")

                doc = self._convert_and_save(
                    input_path=tmp_path,
                    audio_id=audio_id,
                    source="common_voice",
                    url=f"https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0",
                    title=sentence[:200] or f"CommonVoice_{language}_{idx}",
                    language=language,
                    metadata={
                        "split": split,
                        "sentence": sentence,
                        "locale": example.get("locale", language),
                    },
                )
                if doc:
                    documents.append(doc)
            finally:
                tmp_path.unlink(missing_ok=True)

        return documents

    # ─────────────────────────────────────────────
    # Source : FreeSound via freesound-python
    # ─────────────────────────────────────────────

    def _download_freesound(self, query: str, max_files: int) -> List[AudioDocument]:
        """Télécharge des sons depuis FreeSound par requête textuelle.

        Format du paramètre ``query`` : requête libre (ex: ``"piano jazz"``,
        ``"ambient forest"``, ``"tag:field-recording"``).

        Requiert ``FREESOUND_API_KEY`` dans .env.

        Args:
            query:     Requête de recherche FreeSound.
            max_files: Nombre maximum de sons téléchargés.

        Returns:
            Liste d'AudioDocument. Vide si clé API manquante ou lib non installée.
        """
        if not _FREESOUND_AVAILABLE:
            logger.error(
                "freesound non installé — pip install freesound"
            )
            return []

        if not self._freesound_key:
            logger.error("FREESOUND_API_KEY manquante — source FreeSound ignorée")
            return []

        client = _freesound_lib.FreesoundClient()
        client.set_token(self._freesound_key)

        sounds_meta = self._fetch_freesound_results(client, query, max_files)
        documents: List[AudioDocument] = []

        for sound in tqdm(sounds_meta[:max_files], desc="FreeSound", unit="son"):
            # CAS 1 — Filtre durée avant téléchargement : sound.duration est fourni
            # par l'API FreeSound dans les métadonnées de recherche (pas de requête
            # supplémentaire). On filtre ici pour éviter de télécharger des sons longs.
            duration_sec: float = getattr(sound, "duration", 0.0)
            if not self._is_duration_acceptable(duration_sec):
                logger.debug(
                    f"FreeSound #{sound.id} filtré : {duration_sec:.1f}s "
                    f"> {self._MAX_DURATION_SEC}s (30 min)"
                )
                continue

            audio_id = self._compute_audio_id("freesound", str(sound.id))

            if self._is_cached("freesound", audio_id):
                doc = self._load_from_disk("freesound", audio_id)
                if doc:
                    documents.append(doc)
                continue

            # CAS 2 — preview-hq-mp3 plutôt que téléchargement complet :
            # le téléchargement OAuth2 de l'original requiert des scopes d'autorisation
            # supplémentaires. Le preview HQ (128 kbps MP3) est accessible avec
            # uniquement le token API et suffit pour la transcription Whisper.
            previews = getattr(sound, "previews", None)
            preview_url = ""
            if previews:
                preview_url = (
                    getattr(previews, "preview_hq_mp3", None)
                    or getattr(previews, "preview_lq_mp3", None)
                    or ""
                )

            if not preview_url:
                logger.warning(f"FreeSound #{sound.id} : pas d'URL preview disponible")
                continue

            doc = self._download_url_and_save(
                audio_id=audio_id,
                source="freesound",
                url=preview_url,
                page_url=f"https://freesound.org/people/{getattr(sound, 'username', '')}/sounds/{sound.id}/",
                title=getattr(sound, "name", f"freesound_{sound.id}"),
                language="",
                metadata={
                    "freesound_id": sound.id,
                    "tags": list(getattr(sound, "tags", [])),
                    "licence": getattr(sound, "license", ""),
                    "username": getattr(sound, "username", ""),
                    "duration_original": duration_sec,
                },
                input_suffix=".mp3",
            )
            if doc:
                documents.append(doc)

        return documents

    @retry(
        stop=stop_after_attempt(3),
        # CAS 3 — min=5s : FreeSound limite à 2 000 req/jour sur compte free.
        # En cas de 429, attendre au moins 5s avant retry.
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
    )
    def _fetch_freesound_results(
        self, client: Any, query: str, max_files: int
    ) -> List[Any]:
        """Exécute la requête FreeSound et matérialise les résultats.

        Args:
            client:    Instance FreesoundClient authentifiée.
            query:     Requête textuelle.
            max_files: Nombre maximum de résultats retournés.

        Returns:
            Liste d'objets SoundInstance FreeSound.
        """
        # CAS 3 — page_size=min(max_files, 150) : 150 est la limite par page de l'API
        # FreeSound. Au-delà, l'API retourne une erreur de validation.
        page_size = min(max_files, 150)

        results = client.text_search(
            query=query,
            fields="id,name,tags,duration,previews,license,username",
            page_size=page_size,
            # CAS 1 — Filtre durée dans la requête API : réduit le trafic réseau en
            # excluant côté serveur les sons > 30 min avant même de les retourner.
            # Double du filtre Python en aval — les deux sont nécessaires car le filtre
            # API est en secondes entières (moins précis).
            filter=f"duration:[1 TO {self._MAX_DURATION_SEC}]",
        )
        # CAS 2 — list() force la matérialisation du Pager FreeSound ici,
        # dans le périmètre @retry, pour que les erreurs réseau soient retentées.
        return list(results)[:max_files]

    # ─────────────────────────────────────────────
    # Source : YouTube via yt-dlp
    # ─────────────────────────────────────────────

    def _download_youtube(self, query: str, max_files: int) -> List[AudioDocument]:
        """Télécharge l'audio de vidéos YouTube via yt-dlp.

        Format du paramètre ``query`` :
          - URL unique          : ``"https://www.youtube.com/watch?v=..."``
          - URLs séparées par , : ``"url1,url2,url3"``
          - Playlist            : ``"https://www.youtube.com/playlist?list=..."``

        Args:
            query:     URL(s) YouTube ou URL de playlist.
            max_files: Nombre maximum de vidéos traitées.

        Returns:
            Liste d'AudioDocument. Vide si yt-dlp non installé.
        """
        if not _YTDLP_AVAILABLE:
            logger.error(
                "yt-dlp non installé — source 'youtube' indisponible. "
                "pip install yt-dlp"
            )
            return []

        urls = [u.strip() for u in query.split(",") if u.strip()]
        documents: List[AudioDocument] = []

        for url in tqdm(urls[:max_files], desc="YouTube", unit="video"):
            # CAS 1 — Extraction des métadonnées sans téléchargement : yt-dlp peut
            # récupérer durée, titre et id sans télécharger le fichier audio.
            # On vérifie la durée avant de lancer le téléchargement réel.
            info = self._extract_youtube_info(url)
            if info is None:
                logger.warning(f"YouTube : impossible d'extraire les infos de {url}")
                continue

            # Gestion des playlists : info['entries'] contient les vidéos individuelles
            entries = info.get("entries", [info])

            for entry in entries[:max_files - len(documents)]:
                if entry is None:
                    continue

                duration_sec: float = float(entry.get("duration", 0) or 0)
                video_id: str = entry.get("id", "")
                title: str = entry.get("title", f"youtube_{video_id}")
                video_url: str = entry.get("webpage_url", url)

                if not self._is_duration_acceptable(duration_sec):
                    # CAS 1 — Filtre avant téléchargement : on vérifie la durée
                    # via l'extraction des métadonnées (download=False) pour ne pas
                    # télécharger plusieurs Go d'audio inutilement.
                    logger.debug(
                        f"YouTube '{title[:40]}' filtré : "
                        f"{duration_sec:.0f}s > {self._MAX_DURATION_SEC}s (30 min)"
                    )
                    continue

                audio_id = self._compute_audio_id("youtube", video_id)

                if self._is_cached("youtube", audio_id):
                    doc = self._load_from_disk("youtube", audio_id)
                    if doc:
                        documents.append(doc)
                    continue

                doc = self._download_youtube_audio(
                    audio_id=audio_id,
                    video_url=video_url,
                    title=title,
                    language=entry.get("language", "") or "",
                    metadata={
                        "youtube_id": video_id,
                        "channel": entry.get("uploader", ""),
                        "duration_original": duration_sec,
                        "upload_date": entry.get("upload_date", ""),
                    },
                )
                if doc:
                    documents.append(doc)

        return documents

    def _extract_youtube_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Extrait les métadonnées d'une vidéo ou playlist YouTube sans téléchargement.

        Args:
            url: URL YouTube (vidéo ou playlist).

        Returns:
            Dictionnaire de métadonnées yt-dlp, ou None si extraction échouée.
        """
        # CAS 3 — quiet=True : yt-dlp est très verbeux par défaut. On supprime
        # les logs internes pour ne garder que ceux de loguru.
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            # CAS 1 — extract_flat=True pour les playlists : récupère uniquement la
            # liste des vidéos sans extraire les métadonnées complètes de chacune.
            # Réduit les requêtes HTTP de O(n) à O(1) pour une playlist de n vidéos.
            "extract_flat": "in_playlist",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as exc:
            logger.warning(f"yt-dlp extraction échouée pour {url} : {exc}")
            return None

    def _download_youtube_audio(
        self,
        audio_id: str,
        video_url: str,
        title: str,
        language: str,
        metadata: Dict[str, Any],
    ) -> Optional[AudioDocument]:
        """Télécharge l'audio d'une vidéo YouTube et le convertit en WAV 16 kHz.

        Args:
            audio_id:  Identifiant unique calculé pour ce fichier.
            video_url: URL complète de la vidéo YouTube.
            title:     Titre de la vidéo (pour les métadonnées).
            language:  Code langue si disponible dans les métadonnées yt-dlp.
            metadata:  Métadonnées additionnelles (channel, dates...).

        Returns:
            AudioDocument si succès, None si téléchargement ou conversion échoue.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_base = Path(tmp_dir) / audio_id

            # CAS 3 — format='bestaudio/best' : sélectionne le meilleur flux audio
            # disponible (opus, webm, m4a) sans télécharger la vidéo. Sans ce filtre,
            # yt-dlp téléchargerait la vidéo complète, multipliant la taille par 10.
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(tmp_base) + ".%(ext)s",
                "quiet": True,
                "no_warnings": True,
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    # CAS 1 — prepare_filename : le nom réel du fichier peut différer
                    # du template outtmpl (yt-dlp ajuste l'extension selon le format choisi).
                    downloaded_path = Path(ydl.prepare_filename(info))
            except Exception as exc:
                logger.error(f"yt-dlp download échoué pour {video_url} : {exc}")
                return None

            if not downloaded_path.exists():
                # CAS 3 — Edge case : yt-dlp peut changer l'extension au dernier moment.
                # On cherche le fichier par le stem du nom de base.
                candidates = list(Path(tmp_dir).glob(f"{audio_id}.*"))
                if not candidates:
                    logger.error(f"YouTube : fichier téléchargé introuvable dans {tmp_dir}")
                    return None
                downloaded_path = candidates[0]

            return self._convert_and_save(
                input_path=downloaded_path,
                audio_id=audio_id,
                source="youtube",
                url=video_url,
                title=title[:300],
                language=language,
                metadata=metadata,
            )

    # ─────────────────────────────────────────────
    # Conversion ffmpeg et persistance
    # ─────────────────────────────────────────────

    def _download_url_and_save(
        self,
        audio_id: str,
        source: str,
        url: str,
        page_url: str,
        title: str,
        language: str,
        metadata: Dict[str, Any],
        input_suffix: str = ".mp3",
    ) -> Optional[AudioDocument]:
        """Télécharge un fichier audio depuis une URL HTTP et le convertit en WAV 16 kHz.

        Args:
            audio_id:     Identifiant unique.
            source:       Nom de la source.
            url:          URL directe du fichier audio.
            page_url:     URL de la page source (pour les métadonnées).
            title:        Titre du fichier.
            language:     Code langue.
            metadata:     Métadonnées additionnelles.
            input_suffix: Extension du fichier téléchargé (ex: ``".mp3"``).

        Returns:
            AudioDocument si succès, None sinon.
        """
        try:
            response = requests.get(url, timeout=30, stream=False)
            response.raise_for_status()
            audio_bytes = response.content
        except requests.RequestException as exc:
            logger.error(f"Téléchargement HTTP échoué pour {url[:80]} : {exc}")
            return None

        with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp_path.write_bytes(audio_bytes)

        try:
            return self._convert_and_save(
                input_path=tmp_path,
                audio_id=audio_id,
                source=source,
                url=page_url,
                title=title,
                language=language,
                metadata=metadata,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def _convert_and_save(
        self,
        input_path: Path,
        audio_id: str,
        source: str,
        url: str,
        title: str,
        language: str,
        metadata: Dict[str, Any],
    ) -> Optional[AudioDocument]:
        """Convertit un fichier audio en WAV 16 kHz mono et persiste le résultat.

        Args:
            input_path: Chemin du fichier source (tout format supporté par ffmpeg).
            audio_id:   Identifiant unique (nom de fichier sans extension).
            source:     Nom de la source (sous-dossier de sortie).
            url:        URL d'origine pour les métadonnées.
            title:      Titre du fichier.
            language:   Code langue.
            metadata:   Métadonnées additionnelles.

        Returns:
            AudioDocument si conversion réussie, None sinon.
        """
        source_dir = self._raw_audio_path / source
        source_dir.mkdir(parents=True, exist_ok=True)
        output_path = source_dir / f"{audio_id}.wav"

        try:
            self._convert_to_wav_16khz(input_path, output_path)
        except RuntimeError as exc:
            logger.error(f"Conversion ffmpeg échouée pour {input_path.name} : {exc}")
            return None

        duration_sec = self._get_duration_sec(output_path)
        doc = AudioDocument(
            id=audio_id,
            path=str(output_path),
            source=source,
            url=url,
            title=title,
            duration_sec=round(duration_sec, 3),
            sample_rate=self._TARGET_SAMPLE_RATE,
            language=language,
            metadata={
                **metadata,
                "date_downloaded": datetime.utcnow().strftime("%Y-%m-%d"),
            },
        )
        self._save_metadata(doc)
        logger.debug(f"Sauvegardé : {output_path.name} ({duration_sec:.1f}s)")
        return doc

    def _convert_to_wav_16khz(self, input_path: Path, output_path: Path) -> None:
        """Convertit un fichier audio en WAV 16 kHz mono PCM 16-bit via ffmpeg.

        # ─── ALGORITHME : Conversion audio normalisée ─────────────────────────
        # Problème résolu : uniformiser tous les formats d'entrée (MP3, WebM, OGG,
        #                   M4A, FLAC...) en un format WAV fixe requis par Whisper.
        # Approche :        ffmpeg avec paramètres explicites ar/ac/acodec.
        # Formule :         output = ffmpeg(input, ar=16000, ac=1, acodec=pcm_s16le)
        # Référence :       Whisper audio.py load_audio() — même pipeline ffmpeg.
        # ──────────────────────────────────────────────────────────────────────

        Args:
            input_path:  Fichier source (tout format supporté par ffmpeg).
            output_path: Chemin de sortie .wav.

        Raises:
            RuntimeError: Si ffmpeg retourne une erreur (fichier corrompu, codec manquant).
        """
        try:
            (
                ffmpeg
                .input(str(input_path))
                .output(
                    str(output_path),
                    # CAS 3 — ar=16000 : fréquence d'échantillonnage imposée par Whisper.
                    # Toute valeur différente est resamplée par Whisper à la volée,
                    # ce qui double le temps de preprocessing (Whisper paper, Table 1).
                    ar=16_000,
                    # CAS 3 — ac=1 : mono obligatoire pour Whisper. Les signaux stéréo
                    # sont mixés down en mono par ffmpeg (moyenne des canaux L et R).
                    ac=1,
                    # CAS 3 — acodec=pcm_s16le : PCM 16-bit little-endian — format
                    # natif des fichiers WAV sur x86/x64. pcm_s32le serait plus précis
                    # mais doublerait la taille des fichiers sans gain ASR mesurable.
                    acodec="pcm_s16le",
                )
                .run(
                    overwrite_output=True,
                    # CAS 1 — capture_stderr=True : redirige stderr ffmpeg vers un buffer
                    # pour l'inclure dans le message d'erreur RuntimeError si conversion échoue.
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
        except ffmpeg.Error as exc:
            stderr_msg = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise RuntimeError(f"ffmpeg error: {stderr_msg}") from exc

    def _get_duration_sec(self, audio_path: Path) -> float:
        """Retourne la durée en secondes d'un fichier audio via ffmpeg probe.

        Args:
            audio_path: Chemin du fichier audio (tout format ffmpeg supporté).

        Returns:
            Durée en secondes (float), 0.0 si la probe échoue.
        """
        try:
            probe = ffmpeg.probe(str(audio_path))
            # CAS 1 — Parcours des streams : on cherche le premier stream de type 'audio'
            # plutôt que d'utiliser format.duration directement, car certains conteneurs
            # (MKV, WebM) ne renseignent pas format.duration mais renseignent stream.duration.
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "audio":
                    return float(stream.get("duration", 0.0))
            return float(probe.get("format", {}).get("duration", 0.0))
        except ffmpeg.Error as exc:
            logger.debug(f"ffmpeg probe échoué pour {audio_path.name} : {exc}")
            return 0.0

    def _is_duration_acceptable(self, duration_sec: float) -> bool:
        """Retourne True si la durée est dans la limite de 30 minutes.

        Args:
            duration_sec: Durée en secondes.

        Returns:
            True si ``duration_sec <= _MAX_DURATION_SEC`` (1 800 s).
        """
        # CAS 3 — Filtre > 30 min : Whisper large-v3 charge le signal complet en VRAM.
        # Un fichier de 60 min nécessite ~2 Go de VRAM en float16, dépassant la capacité
        # de la plupart des GPU grand public (RTX 3080 = 10 Go).
        return duration_sec <= self._MAX_DURATION_SEC

    # ─────────────────────────────────────────────
    # Cache et persistance
    # ─────────────────────────────────────────────

    def _is_cached(self, source: str, audio_id: str) -> bool:
        """Retourne True si le WAV et ses métadonnées JSON existent déjà.

        Args:
            source:   Nom de la source (sous-dossier).
            audio_id: Identifiant du fichier audio.

        Returns:
            True si {id}.wav ET {id}.json existent tous les deux.
        """
        source_dir = self._raw_audio_path / source
        # CAS 1 — Double vérification .wav ET .json : si seulement le .wav existe
        # (crash lors d'une conversion précédente), on re-convertit pour rétablir
        # les métadonnées nécessaires à AudioProcessor.
        return (
            (source_dir / f"{audio_id}.wav").exists()
            and (source_dir / f"{audio_id}.json").exists()
        )

    def _load_from_disk(self, source: str, audio_id: str) -> Optional[AudioDocument]:
        """Charge un AudioDocument depuis le fichier JSON de cache.

        Args:
            source:   Nom de la source.
            audio_id: Identifiant du fichier audio.

        Returns:
            AudioDocument reconstruit, ou None si JSON invalide.
        """
        json_path = self._raw_audio_path / source / f"{audio_id}.json"
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return AudioDocument(**data)
        except Exception as exc:
            logger.warning(f"Cache JSON corrompu {json_path.name} : {exc}")
            return None

    def _save_metadata(self, doc: AudioDocument) -> None:
        """Persiste les métadonnées d'un AudioDocument en JSON.

        Args:
            doc: AudioDocument à sérialiser. Écrit dans {source}/{id}.json.
        """
        source_dir = self._raw_audio_path / doc.source
        source_dir.mkdir(parents=True, exist_ok=True)
        json_path = source_dir / f"{doc.id}.json"
        json_path.write_text(
            json.dumps(asdict(doc), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _compute_audio_id(self, source: str, identifier: str) -> str:
        """Calcule un identifiant stable SHA-256[:16] pour un fichier audio.

        Args:
            source:     Nom de la source.
            identifier: Identifiant natif (native_id, URL, index...).

        Returns:
            Chaîne hexadécimale de 16 caractères.
        """
        raw = f"{source}:{identifier}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

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
