# -*- coding: utf-8 -*-
"""
Module text_processor — Nettoyage et normalisation de documents textuels bruts.

Rôle dans l'architecture :
    Deuxième étape du pipeline d'ingestion RAG, entre les downloaders et le chunking.
    Charge les fichiers JSON bruts depuis data/raw/text/, applique un pipeline de
    nettoyage séquentiel, filtre les documents invalides et persiste les
    ProcessedDocument dans data/processed/text/ pour consommation par les chunkers
    (RecursiveChunker, SemanticChunker).

Pipeline de nettoyage (ordre obligatoire — ne pas réordonner) :
    1. Suppression HTML/XML (BeautifulSoup)       → texte pur, sans balises
    2. Normalisation unicode NFKC (unicodedata)   → formes canoniques (ligatures, fractions)
    3. Suppression caractères de contrôle (regex) → retrait des artefacts de parsing
    4. Normalisation espaces multiples (regex)    → texte compact et lisible
    5. Détection de langue (langdetect)           → code ISO 639-1 pour le filtre
    6. Calcul statistiques (word_count, char_count, avg_sentence_length)

Pourquoi cet ordre :
    HTML en premier : les balises peuvent contenir des entités unicode anormales ;
    les déposer avant NFKC évite de normaliser du bruit HTML.
    NFKC avant contrôles : certaines formes de compatibilité décomposées incluent
    des points de code < U+0020. Contrôles après NFKC pour nettoyer le résidu.
    Espaces en dernier : chaque étape précédente peut introduire des espaces superflus
    (get_text separator=' ', NFKC sur caractères larges, suppression contrôles).

Choix des bibliothèques :
    - BeautifulSoup (parser='html.parser') : gère le HTML malformé sans dépendance
                      lxml ou html5lib. Rejeté : regex manuelle (fragile sur attributs
                      imbriqués et entités comme &amp;, &nbsp;).
    - unicodedata (NFKC) : stdlib Python, zéro dépendance externe, standard Unicode TR15.
                      Rejeté : unidecode (translittération avec perte d'information).
    - re            : stdlib, suffisant pour les patterns simples de contrôle/espace.
                      Rejeté : regex (pip) — surqualifié pour ce cas d'usage.
    - langdetect    : port Python de la librairie Google language-detection (Nakatani 2010).
                      Rejeté : langid (moins précis sur textes courts) ;
                               fasttext (modèle binaire à télécharger, lourd pour un portfolio).
"""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from bs4 import BeautifulSoup
from langdetect import LangDetectException, detect
from loguru import logger
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass métier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessedDocument:
    """Représentation normalisée d'un document textuel nettoyé et qualifié.

    Objet métier produit par TextProcessor, consommé par RecursiveChunker et
    SemanticChunker. Conserve le texte original (``content``) et le texte
    nettoyé (``content_clean``) pour permettre la comparaison et le debug.

    Example:
        doc = ProcessedDocument(
            id="a3f1c9e2b4d80f12",
            content="<p>Le machine learning est...</p>",
            content_clean="Le machine learning est...",
            language="fr",
            word_count=4,
            char_count=27,
            avg_sentence_length=4.0,
            source="wikipedia",
            metadata={"title": "Machine learning", "url": "https://fr.wikipedia.org/..."},
        )
    """

    id: str
    content: str                   # texte brut original (avant nettoyage — pour debug)
    content_clean: str             # texte après pipeline complet (ingéré par le chunker)
    language: str                  # code ISO 639-1 détecté par langdetect
    word_count: int                # nombre de mots dans content_clean
    char_count: int                # nombre de caractères dans content_clean
    avg_sentence_length: float     # moyenne de mots par phrase (santé du texte)
    source: str                    # 'wikipedia' | 'arxiv' | 'huggingface'
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Processeur principal
# ─────────────────────────────────────────────────────────────────────────────

class TextProcessor:
    """Charge les documents bruts JSON et applique un pipeline de nettoyage normalisé.

    Filtre les documents trop courts (< 100 mots) ou dans une langue non supportée.
    Génère un rapport de traitement dans results/text_processing_report.json.

    Example:
        processor = TextProcessor()
        docs = processor.process(source="wikipedia", max_docs=50)
        # → List[ProcessedDocument] dans data/processed/text/wikipedia/
        # → Rapport dans results/text_processing_report.json
    """

    # CAS 3 — 100 mots minimum : en dessous, le document est trop court pour
    # produire des chunks cohérents (RecursiveChunker chunk_size=512 tokens).
    # Typiquement : stubs Wikipedia, résumés d'arXiv tronqués, erreurs de parsing.
    _MIN_WORD_COUNT: int = 100

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Args:
            config_path: Chemin vers config/config.yaml. Doit contenir
                         ``data.raw_path``, ``data.processed_path``,
                         ``data.results_path``. Optionnellement
                         ``text.allowed_languages`` (défaut : ['fr', 'en']).

        Raises:
            FileNotFoundError: Si config_path n'existe pas.
        """
        self._config = self._load_config(config_path)

        self._raw_text_path = Path(self._config["data"]["raw_path"]) / "text"
        self._processed_text_path = (
            Path(self._config["data"]["processed_path"]) / "text"
        )
        self._results_path = Path(self._config["data"]["results_path"])

        self._processed_text_path.mkdir(parents=True, exist_ok=True)
        self._results_path.mkdir(parents=True, exist_ok=True)

        # CAS 3 — ['fr', 'en'] par défaut : les deux langues couvertes par les
        # sources Wikipedia (en), arXiv (en) et Common Voice (fr/en).
        # Les embeddings nomic-embed-text sont optimisés pour ces deux langues.
        # Surcharger via config.yaml text.allowed_languages pour d'autres langues.
        self._allowed_languages: List[str] = (
            self._config.get("text", {}).get("allowed_languages", ["fr", "en"])
        )

    # ─────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────

    def process(
        self,
        source: Optional[str] = None,
        max_docs: Optional[int] = None,
    ) -> List[ProcessedDocument]:
        """Charge, nettoie, filtre et persiste les documents depuis data/raw/text/.

        Args:
            source:   Filtre optionnel sur la source (ex: ``'wikipedia'``).
                      None = toutes les sources disponibles.
            max_docs: Nombre maximum de documents à traiter. None = tous.

        Returns:
            Liste de ProcessedDocument valides (filtrés et nettoyés).
            Jamais None — liste vide si aucun document ne passe les filtres.
        """
        raw_docs = self._load_raw_documents(source_filter=source)
        logger.info(
            f"TextProcessor : {len(raw_docs)} documents bruts chargés "
            f"(source={source!r}, max={max_docs})"
        )

        if max_docs is not None:
            raw_docs = raw_docs[:max_docs]

        processed: List[ProcessedDocument] = []
        rejection_reasons: Dict[str, int] = {}
        languages_found: Dict[str, int] = {}

        for raw in tqdm(raw_docs, desc="TextProcessor", unit="doc"):
            result = self._process_one(raw)

            if result is None:
                # CAS 3 — None = contenu vide après nettoyage : document inutilisable
                # (page Wikipedia vide, abstract arXiv tronqué à 0 char).
                rejection_reasons["content_empty_after_cleaning"] = (
                    rejection_reasons.get("content_empty_after_cleaning", 0) + 1
                )
                continue

            doc, reject_reason = result

            if reject_reason:
                rejection_reasons[reject_reason] = (
                    rejection_reasons.get(reject_reason, 0) + 1
                )
                continue

            languages_found[doc.language] = languages_found.get(doc.language, 0) + 1
            processed.append(doc)

        self._save_documents(processed)
        self._save_report(
            total_raw=len(raw_docs),
            total_processed=len(processed),
            rejected_count=len(raw_docs) - len(processed),
            rejection_reasons=rejection_reasons,
            languages_found=languages_found,
        )

        logger.info(
            f"TextProcessor terminé : {len(processed)}/{len(raw_docs)} valides "
            f"({len(raw_docs) - len(processed)} rejetés)"
        )
        return processed

    # ─────────────────────────────────────────────
    # Pipeline de traitement d'un document
    # ─────────────────────────────────────────────

    def _process_one(
        self, raw: Dict[str, Any]
    ) -> Optional[Tuple[ProcessedDocument, Optional[str]]]:
        """Applique le pipeline complet sur un document brut.

        Retourne (ProcessedDocument, None) si valide, (doc, raison) si rejeté,
        ou None si le contenu est vide après nettoyage (document inutilisable).

        Args:
            raw: Dictionnaire correspondant à un Document sérialisé par TextDownloader.

        Returns:
            Tuple (ProcessedDocument, reject_reason_or_None), ou None si vide post-clean.
        """
        content_raw = raw.get("content", "")
        if not content_raw.strip():
            return None

        content_clean = self._clean_text(content_raw)
        if not content_clean.strip():
            return None

        language = self._detect_language(content_clean)
        word_count, char_count, avg_sentence_length = self._compute_stats(content_clean)

        doc = ProcessedDocument(
            id=raw.get("id", ""),
            content=content_raw,
            content_clean=content_clean,
            language=language,
            word_count=word_count,
            char_count=char_count,
            avg_sentence_length=avg_sentence_length,
            source=raw.get("source", ""),
            metadata={
                **raw.get("metadata", {}),
                "title": raw.get("title", ""),
                "url": raw.get("url", ""),
                "date": raw.get("date", ""),
                "processed_at": datetime.utcnow().strftime("%Y-%m-%d"),
            },
        )

        reject_reason = self._get_rejection_reason(doc, language)
        return doc, reject_reason

    def _clean_text(self, text: str) -> str:
        """Applique le pipeline de nettoyage en 4 étapes séquentielles.

        Args:
            text: Texte brut potentiellement contenant du HTML, des ligatures
                  unicode et des caractères de contrôle.

        Returns:
            Texte nettoyé, normalisé, sans balises ni artefacts.
        """
        # Étape 1 — Suppression HTML/XML via BeautifulSoup :
        # parser='html.parser' est la stdlib Python (pas de dépendance lxml/html5lib).
        # get_text(separator=' ') préserve les coupures de mots aux frontières de
        # balises : <p>foo</p><p>bar</p> → "foo bar" (et non "foobar").
        # strip=True retire les espaces redondants dans chaque nœud texte.
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        # Étape 2 — Normalisation unicode NFKC (Compatibility Decomposition +
        # Canonical Composition) : convertit les ligatures (ﬁ→fi), exposants (²→2),
        # fractions (½→1⁄2) et caractères de largeur pleine en formes canoniques.
        # Rejeté NFC : ne décompose pas les caractères de compatibilité (ﬁ resterait ﬁ).
        text = unicodedata.normalize("NFKC", text)

        # Étape 3 — Suppression des caractères de contrôle ASCII U+0000–U+001F,
        # à l'exception de \x09 (tab, U+0009) et \x0a (LF, U+000A).
        # \x0d (CR) est supprimé : les fins de ligne CRLF Windows deviennent LF seul.
        # Le pattern [\x00-\x08\x0b-\x1f] couvre 0–8 (NUL..BS) et 11–31 (VT..US),
        # laissant intact 9 (tab) et 10 (LF).
        text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)

        # Étape 4a — Normalisation des espaces et tabs multiples en un seul espace.
        # On ne touche pas aux \n pour préserver les frontières de paragraphe
        # utilisées par le SemanticChunker pour découper sur les sauts de section.
        text = re.sub(r"[ \t]+", " ", text)

        # Étape 4b — Réduction des sauts de ligne multiples (≥ 3) en double newline :
        # préserve la structure paragraphe sans produire de grands blancs inutiles.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _detect_language(self, text: str) -> str:
        """Détecte la langue principale du texte via langdetect.

        # ─── ALGORITHME : Détection de langue ────────────────────────────────
        # Problème résolu : identifier la langue sans annotation manuelle.
        # Approche :        langdetect.detect() sur les 500 premiers caractères.
        # Formule :         langue = argmax P(langue | text[:500])
        # Référence :       Shuyo Nakatani, Language Detection Library for Java (2010).
        # Limitation :      Probabiliste — peut échouer sur textes courts (< 20 mots)
        #                   ou mélangeant deux langues (code + commentaires).
        # ─────────────────────────────────────────────────────────────────────

        Args:
            text: Texte nettoyé (sans HTML ni caractères de contrôle).

        Returns:
            Code ISO 639-1 (ex: ``"en"``, ``"fr"``), ou ``"unknown"`` si échec.
        """
        try:
            # CAS 1 — Troncature à 500 chars : langdetect est O(n) et 500 chars
            # est suffisant pour une détection fiable (Nakatani 2010). Évite les
            # timeouts sur les articles Wikipedia longs (> 100 000 chars).
            return detect(text[:500])
        except LangDetectException as exc:
            # CAS 3 — LangDetectException : levée quand le texte ne contient pas
            # assez de caractères linguistiques (texte purement numérique, formules
            # LaTeX, code source). On retourne "unknown" pour déclencher le filtre.
            logger.debug(f"Détection langue échouée : {exc}")
            return "unknown"

    def _compute_stats(self, text: str) -> Tuple[int, int, float]:
        """Calcule les statistiques textuelles de ProcessedDocument.

        Args:
            text: Texte nettoyé (content_clean).

        Returns:
            Tuple ``(word_count, char_count, avg_sentence_length)``.
        """
        # CAS 2 — split() sans argument : coupe sur tout espace blanc (espaces,
        # tabs, newlines) et ignore les runs — plus robuste que split(' ') qui
        # produirait des tokens vides sur les espaces résiduels.
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # CAS 1 — Segmentation par .!? avec re.split : les textes encyclopédiques
        # (Wikipedia, arXiv) utilisent ces trois signes de ponctuation pour finir
        # les phrases. On filtre les fragments de moins de 2 mots (interjections,
        # abréviations tronquées) pour ne pas biaiser avg_sentence_length vers le bas.
        raw_sentences = re.split(r"[.!?]+", text)
        sentence_word_counts = [
            len(s.split()) for s in raw_sentences
            if len(s.split()) >= 2  # filtre fragments ultra-courts (ex: "Fig", "p", "al")
        ]

        if sentence_word_counts:
            avg_sentence_length = sum(sentence_word_counts) / len(sentence_word_counts)
        else:
            # CAS 3 — Texte sans ponctuation (code, formules) : le document est traité
            # comme une phrase unique pour ne pas retourner 0.0 (valeur trompeuse
            # qui masquerait la présence de contenu dans les rapports).
            avg_sentence_length = float(word_count)

        return word_count, char_count, round(avg_sentence_length, 2)

    def _get_rejection_reason(
        self, doc: ProcessedDocument, language: str
    ) -> Optional[str]:
        """Retourne la raison de rejet si le document ne passe pas les filtres qualité.

        Args:
            doc:      ProcessedDocument avec statistiques calculées.
            language: Code langue détecté par _detect_language.

        Returns:
            Chaîne décrivant la raison de rejet, ou None si le document est valide.
        """
        if doc.word_count < self._MIN_WORD_COUNT:
            # CAS 3 — Rejet < 100 mots : en dessous, le chunk unique produit serait
            # trop petit pour la retrieval (RecursiveChunker min_chunk_size=100 tokens).
            # Causes typiques : stubs Wikipedia sans article, abstracts tronqués, erreurs HTTP.
            return f"word_count_below_{self._MIN_WORD_COUNT}"

        if language not in self._allowed_languages:
            # CAS 3 — Rejet langue non supportée : les embeddings nomic-embed-text
            # et CLIP sont entraînés principalement sur fr/en. Une langue non couverte
            # produit des représentations vectorielles dégradées, réduisant le rappel.
            return f"language_not_allowed:{language}"

        return None

    # ─────────────────────────────────────────────
    # Chargement des données brutes
    # ─────────────────────────────────────────────

    def _load_raw_documents(
        self, source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Charge les documents depuis tous les fichiers JSON de data/raw/text/.

        Args:
            source_filter: Si fourni, ne charge que les fichiers dont le nom
                           commence par ``{source_filter}_`` (ex: ``'wikipedia'``).

        Returns:
            Liste de dicts Document (format TextDownloader). Vide si aucun fichier.
        """
        # CAS 1 — Glob avec préfixe source : le pattern {source}_*.json cible
        # uniquement les fichiers de la source demandée sans charger les autres.
        pattern = f"{source_filter}_*.json" if source_filter else "*.json"
        json_files = sorted(self._raw_text_path.glob(pattern))

        if not json_files:
            logger.warning(
                f"Aucun fichier JSON trouvé dans {self._raw_text_path} "
                f"(pattern='{pattern}')"
            )
            return []

        all_docs: List[Dict[str, Any]] = []

        for json_path in json_files:
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                documents = payload.get("documents", [])
                all_docs.extend(documents)
                logger.debug(f"Chargé {len(documents)} docs depuis {json_path.name}")
            except (json.JSONDecodeError, KeyError) as exc:
                # CAS 3 — JSON corrompu ou format inattendu (fichier partiel suite
                # à un crash du downloader) : on loggue et on continue pour ne pas
                # bloquer le batch entier sur un seul fichier défaillant.
                logger.warning(f"Fichier JSON ignoré ({json_path.name}) : {exc}")

        logger.info(
            f"_load_raw_documents : {len(all_docs)} documents "
            f"depuis {len(json_files)} fichiers"
        )
        return all_docs

    # ─────────────────────────────────────────────
    # Persistance
    # ─────────────────────────────────────────────

    def _save_documents(self, documents: List[ProcessedDocument]) -> None:
        """Persiste chaque ProcessedDocument en JSON dans data/processed/text/{source}/.

        Nom de fichier : ``{id}.json`` — unicité garantie par le SHA-256[:16] id
        hérité du TextDownloader.

        Args:
            documents: Liste de ProcessedDocument valides à sauvegarder.
        """
        for doc in documents:
            # CAS 1 — Sous-dossier par source : permet aux chunkers de cibler
            # uniquement la source souhaitée (wikipedia/, arxiv/, huggingface/).
            source_dir = self._processed_text_path / doc.source
            source_dir.mkdir(parents=True, exist_ok=True)
            output_path = source_dir / f"{doc.id}.json"

            # CAS 1 — ensure_ascii=False : titres Wikipedia et abstracts arXiv
            # contiennent des accents et symboles mathématiques Unicode.
            # Même raison que dans text_downloader._save_documents().
            output_path.write_text(
                json.dumps(asdict(doc), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logger.info(
            f"_save_documents : {len(documents)} fichiers dans "
            f"{self._processed_text_path}"
        )

    def _save_report(
        self,
        total_raw: int,
        total_processed: int,
        rejected_count: int,
        rejection_reasons: Dict[str, int],
        languages_found: Dict[str, int],
    ) -> None:
        """Sauvegarde le rapport de traitement dans results/text_processing_report.json.

        Le rapport est écrasé à chaque run pour toujours refléter l'état du dernier
        traitement. Un rapport versionné nécessiterait un suffixe timestamp (hors scope).

        Args:
            total_raw:         Nombre de documents bruts chargés.
            total_processed:   Nombre de documents valides produits.
            rejected_count:    Nombre de documents rejetés (total_raw - total_processed).
            rejection_reasons: Comptage par cause de rejet (clé = raison, valeur = count).
            languages_found:   Distribution des langues dans les documents valides.
        """
        report = {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "total_raw": total_raw,
            "total_processed": total_processed,
            "rejected_count": rejected_count,
            # CAS 1 — rejection_rate calculé dans le rapport (pas dans process()) :
            # le JSON doit être auto-suffisant pour une lecture humaine directe,
            # sans nécessiter de recalcul côté utilisateur.
            "rejection_rate": (
                round(rejected_count / total_raw, 3) if total_raw > 0 else 0.0
            ),
            "rejection_reasons": rejection_reasons,
            "languages_found": languages_found,
        }

        report_path = self._results_path / "text_processing_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Rapport sauvegardé : {report_path.name}")

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
