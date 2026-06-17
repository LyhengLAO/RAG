"""
Module text_downloader — Téléchargeur de contenus textuels multimodal.

Rôle dans l'architecture :
    Première étape du pipeline d'ingestion RAG. Récupère des documents bruts
    depuis trois sources (Wikipedia, arXiv, HuggingFace Datasets), les normalise
    en dataclass Document et les persiste en JSON dans data/raw/text/ pour
    traitement ultérieur par TextProcessor → Chunking → Embeddings.

Pourquoi ces trois sources :
    - Wikipedia  : corpus encyclopédique dense, idéal pour évaluer la retrieval
                   sur des questions factuelles (QA open-domain). Gratuit, stable.
    - arXiv      : corpus scientifique avec métadonnées structurées (auteurs,
                   catégories, DOI), utile pour tester la précision sur du contenu
                   technique. API publique sans auth.
    - HuggingFace: datasets NLP standardisés (SQuAD, MS MARCO) avec ground-truth
                   intégré — permet de construire data/eval/eval_set.json directement.

Choix techniques :
    - wikipedia-api  : respecte le rate-limit officiel et gère l'encodage UTF-8.
                       Rejeté : scraping BeautifulSoup4 (fragile, HTML changeant).
    - arxiv (v2.x)   : wrapper officiel de l'API Atom arXiv avec Client threadé.
                       Rejeté : requêtes HTTP manuelles (pas de pagination native).
    - datasets (HF)  : accès unifié, streaming possible pour les grands corpus.
                       Rejeté : téléchargement manuel depuis le Hub (pas de cache).
    - tenacity       : retry avec backoff exponentiel applicable à toute fonction.
                       Rejeté : requests.adapters.HTTPAdapter (limité aux requêtes HTTP).
    - loguru         : logging structuré sans configuration boilerplate.
                       Rejeté : stdlib logging (setup verbeux, pas de coloration).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arxiv
import wikipediaapi
import yaml
from datasets import load_dataset
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

def _log_retry(retry_state: Any) -> None:
    """Loggue les informations de la tentative échouée avant le prochain essai."""
    exc = retry_state.outcome.exception()
    logger.warning(
        f"Retry {retry_state.attempt_number}/3 — "
        f"{type(exc).__name__}: {exc}"
    )

@dataclass
class Document:
    """Représentation normalisée d'un document téléchargé.

    Objet métier partagé entre toutes les étapes du pipeline :
    ingestion → chunking → embeddings → vectorstore. Sérialisable
    en JSON via dataclasses.asdict() sans dépendance externe."""

    id: str           # SHA-256[:16] de (source + title + url) — unicité stable cross-session
    content: str
    source: str       # 'wikipedia' | 'arxiv' | 'huggingface'
    url: str
    title: str
    date: str         # format ISO 8601 : YYYY-MM-DD
    media_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

class TextDownloader:
    """Télécharge et normalise des documents textuels depuis Wikipedia, arXiv et HuggingFace.

    Gère le cache disque par hash de requête pour éviter les re-téléchargements.
    Applique un retry exponentiel via tenacity sur tous les appels réseau.

    Example:
        downloader = TextDownloader()
        docs = downloader.download("wikipedia", "Machine learning", max_docs=5)
        # → List[Document] sauvegardé dans data/raw/text/wikipedia_<ts>_<hash>.json
        docs2 = downloader.download("wikipedia", "Machine learning", max_docs=5)
        # → Chargé depuis le cache, aucun appel réseau
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Args:
            config_path: Chemin vers config/config.yaml.
                         Doit contenir la clé ``data.raw_path``.

        Raises:
            FileNotFoundError: Si config_path n'existe pas.
        """
        self._config = self._load_config(config_path)

        # Sous-dossier text/ séparé des autres modalités (images/, audio/)
        self._raw_text_path = Path(self._config["data"]["raw_path"]) / "text"
        self._raw_text_path.mkdir(parents=True, exist_ok=True)

        # CAS 2 — User-Agent explicite : Wikipedia bloque les requêtes sans UA identifiable
        # (retourne 403). Le format recommandé est "NomApp/Version (contact)".
        self._wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="rag-multimedia-expert/0.1 (educational-portfolio-project)",
        )

        # CAS 3 — delay_seconds=3.0 : arXiv impose un délai minimum entre les requêtes
        # (https://arxiv.org/help/api/user-manual#Appendices). En dessous, risque de bannissement.
        # num_retries=1 : tenacity gère le retry en amont — on évite les doublons de tentatives.
        self._arxiv_client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=1,
        )

    # ─────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────

    def download(self, source: str, query: str, max_docs: int = 10) -> List[Document]:
        """Télécharge des documents depuis la source demandée, avec cache disque.

        Vérifie l'existence d'un fichier cache avant tout appel réseau.
        Persiste le résultat dans data/raw/text/ après téléchargement.

        Args:
            source:   Source à interroger. Valeurs acceptées :
                      ``'wikipedia'``, ``'arxiv'``, ``'huggingface'``.
            query:    Requête dont le format dépend de la source (voir méthodes _download_*).
            max_docs: Nombre maximum de documents à retourner. Minimum effectif : 1.

        Returns:
            Liste de Document normalisés. Jamais None, peut être vide si aucun résultat.

        Raises:
            ValueError: Si source n'est pas dans les valeurs acceptées.
        """
        if source not in {"wikipedia", "arxiv", "huggingface"}:
            raise ValueError(
                f"Source inconnue : '{source}'. "
                f"Valeurs acceptées : 'wikipedia', 'arxiv', 'huggingface'."
            )

        query_hash = self._compute_query_hash(source, query, max_docs)
        cached_path = self._get_cache_path(source, query_hash)

        if cached_path is not None:
            logger.info(f"Cache hit — chargement depuis {cached_path.name}")
            return self._load_from_cache(cached_path)

        logger.info(
            f"Téléchargement | source='{source}' | query='{query[:80]}' | max_docs={max_docs}"
        )

        _dispatch = {
            "wikipedia": self._download_wikipedia,
            "arxiv": self._download_arxiv,
            "huggingface": self._download_huggingface,
        }
        documents = _dispatch[source](query, max_docs)

        if documents:
            self._save_documents(documents, source, query_hash)
        else:
            logger.warning(f"Aucun document récupéré depuis '{source}' pour query='{query[:80]}'")

        return documents

    # ─────────────────────────────────────────────
    # Source : Wikipedia
    # ─────────────────────────────────────────────

    def _download_wikipedia(self, query: str, max_docs: int) -> List[Document]:
        """Télécharge des pages Wikipedia par titre(s) ou catégorie.

        Format du paramètre ``query`` :
          - Titre unique       : ``"Machine learning"``
          - Titres multiples   : ``"Python,Machine_learning,Deep_learning"``
          - Catégorie          : ``"category:Machine_learning"``

        Args:
            query:    Titre(s) séparés par virgule, ou ``"category:NomCategorie"``.
            max_docs: Nombre maximum de Document retournés.

        Returns:
            Liste de Document avec ``content`` = texte complet de la page Wikipedia.
        """
        documents: List[Document] = []

        if query.lower().startswith("category:"):
            # CAS 1 — Expansion de catégorie : on récupère d'abord la liste des membres
            # (appel réseau léger), puis on télécharge chaque page individuellement.
            # On limite à max_docs dès la liste pour ne pas déclencher des centaines de requêtes.
            category_name = query.split(":", 1)[1].strip()
            titles = self._get_category_members(category_name, max_docs)
        else:
            # Plusieurs titres séparés par virgule ou titre unique
            titles = [t.strip() for t in query.split(",") if t.strip()][:max_docs]

        for title in tqdm(titles, desc="Wikipedia", unit="page"):
            doc = self._fetch_wikipedia_page(title)
            if doc is not None:
                documents.append(doc)
            if len(documents) >= max_docs:
                break

        logger.info(f"Wikipedia : {len(documents)}/{len(titles)} pages récupérées")
        return documents

    @retry(
        stop=stop_after_attempt(3),
        # CAS 3 — min=2s, max=30s, multiplier=1 : backoff 2s → 4s → 8s (plafonné à 30s).
        # Valeur minimale 2s : Wikipedia applique un rate-limit soft autour d'1 req/s.
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
    )
    def _fetch_wikipedia_page(self, title: str) -> Optional[Document]:
        """Récupère et convertit une page Wikipedia en Document.

        Décoré @retry pour absorber les TimeoutError et ConnectionError intermittents
        de l'API Wikipedia (fréquents sous forte charge).

        Args:
            title: Titre exact de la page (sensible à la casse, espaces acceptés).

        Returns:
            Document peuplé, ou None si la page n'existe pas ou est vide.
        """
        page = self._wiki.page(title)

        if not page.exists():
            # CAS 3 — Edge case page inexistante : on retourne None plutôt que de lever
            # une exception pour ne pas interrompre le batch de téléchargement.
            logger.warning(f"Page Wikipedia inexistante : '{title}'")
            return None

        if not page.text.strip():
            # CAS 3 — Edge case page redirect ou stub : page.text est vide pour les
            # redirections qui n'ont pas été résolues et les ébauches sans contenu.
            logger.warning(f"Page Wikipedia sans contenu extractible : '{title}'")
            return None

        return Document(
            id=self._compute_doc_id("wikipedia", page.title, page.fullurl),
            content=page.text,
            source="wikipedia",
            url=page.fullurl,
            title=page.title,
            # CAS 3 — Wikipedia n'expose pas de date de dernière modification via ce package ;
            # on utilise la date de téléchargement comme proxy pour la traçabilité.
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            metadata={
                "summary": page.summary[:500],    # tronqué à 500 chars pour l'index
                "categories": list(page.categories.keys())[:20],  # top 20, évite les métadonnées gonflées
                "lang": "en",
            },
        )

    def _get_category_members(self, category_name: str, max_docs: int) -> List[str]:
        """Retourne les titres des pages membres d'une catégorie Wikipedia.

        Parcours non récursif du premier niveau uniquement — la récursion sur les
        sous-catégories peut produire des milliers de pages (ex: "Category:Science").

        Args:
            category_name: Nom de la catégorie sans préfixe ``"Category:"``.
            max_docs:       Nombre maximum de titres retournés.

        Returns:
            Liste de titres de pages (namespace MAIN uniquement, jamais les sous-catégories).
        """
        cat_page = self._wiki.page(f"Category:{category_name}")

        if not cat_page.exists():
            logger.warning(f"Catégorie Wikipedia inexistante : 'Category:{category_name}'")
            return []

        titles: List[str] = []
        for member in cat_page.categorymembers.values():
            # CAS 1 — Filtre namespace MAIN (ns=0) : les catégories ont ns=14, les fichiers ns=6.
            # On exclut les deux pour ne conserver que les articles encyclopédiques.
            if member.ns == wikipediaapi.Namespace.MAIN:
                titles.append(member.title)
            if len(titles) >= max_docs:
                break

        logger.debug(f"Catégorie 'Category:{category_name}' : {len(titles)} membres trouvés")
        return titles

    # ─────────────────────────────────────────────
    # Source : arXiv
    # ─────────────────────────────────────────────

    def _download_arxiv(self, query: str, max_docs: int) -> List[Document]:
        """Télécharge des métadonnées et abstracts d'articles arXiv.

        Format du paramètre ``query`` :
          - Recherche libre   : ``"retrieval augmented generation"``
          - IDs spécifiques   : ``"id:2305.14314,2301.07041"``

        Note sur le contenu : ``Document.content`` contient l'abstract uniquement.
        L'extraction du PDF complet est déléguée à TextProcessor pour séparer les
        responsabilités et éviter les téléchargements lourds à l'étape d'ingestion.

        Args:
            query:    Requête libre ou ``"id:id1,id2,..."``.
            max_docs: Nombre maximum d'articles retournés.

        Returns:
            Liste de Document avec ``content`` = abstract de l'article.
        """
        if query.lower().startswith("id:"):
            # CAS 1 — Recherche par IDs : le champ id_list de l'API arXiv est plus précis
            # qu'une recherche textuelle et évite les faux positifs pour les IDs connus.
            id_list = [i.strip() for i in query[3:].split(",") if i.strip()]
            search = arxiv.Search(id_list=id_list)
        else:
            # CAS 2 — sort_by=Relevance : préféré à SortCriterion.SubmittedDate car on veut
            # les articles les plus pertinents, pas les plus récents, pour le corpus RAG.
            search = arxiv.Search(
                query=query,
                max_results=max_docs,
                sort_by=arxiv.SortCriterion.Relevance,
            )

        results = self._fetch_arxiv_results(search, max_docs)
        documents: List[Document] = []

        for result in tqdm(results, desc="arXiv", unit="paper"):
            documents.append(Document(
                id=self._compute_doc_id("arxiv", result.title, result.entry_id),
                content=result.summary,
                source="arxiv",
                url=result.entry_id,
                title=result.title,
                date=result.published.strftime("%Y-%m-%d"),
                metadata={
                    "authors": [a.name for a in result.authors],
                    "categories": result.categories,
                    "doi": result.doi,
                    # pdf_url conservé en metadata pour téléchargement différé par TextProcessor
                    "pdf_url": result.pdf_url,
                },
            ))

        logger.info(f"arXiv : {len(documents)} articles récupérés")
        return documents

    @retry(
        stop=stop_after_attempt(3),
        # CAS 3 — min=5s : arXiv demande explicitement un délai de 3s entre requêtes
        # (arxiv.org/help/api/user-manual). On prend 5s pour la marge de sécurité.
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
    )
    def _fetch_arxiv_results(self, search: arxiv.Search, max_docs: int) -> List[arxiv.Result]:
        """Exécute la requête arXiv et matérialise les résultats en liste.

        La matérialisation (list()) est volontaire : elle force l'exécution de tous les
        appels réseau à l'intérieur du décorateur @retry, plutôt que lors de l'itération
        en dehors du périmètre de retry.

        Args:
            search:   Objet arxiv.Search configuré par _download_arxiv.
            max_docs: Plafond pour éviter de matérialiser des résultats superflus.

        Returns:
            Liste d'objets arxiv.Result (vide si aucun résultat ou ID invalide).
        """
        # CAS 2 — list() force la matérialisation du générateur ici, dans le périmètre @retry,
        # plutôt que dans la boucle tqdm hors périmètre — garantit que les erreurs réseau
        # sont retentées correctement.
        results = list(self._arxiv_client.results(search))
        return results[:max_docs]

    # ─────────────────────────────────────────────
    # Source : HuggingFace Datasets
    # ─────────────────────────────────────────────

    def _download_huggingface(self, query: str, max_docs: int) -> List[Document]:
        """Charge des documents depuis un dataset HuggingFace.

        Format du paramètre ``query`` :
          - Dataset simple              : ``"squad"``
          - Dataset + config            : ``"squad:plain_text"``
          - Dataset + config + split    : ``"squad:plain_text:validation"``

        Args:
            query:    Nom du dataset avec config et split optionnels (séparateur ``':'``).
            max_docs: Nombre maximum d'exemples extraits.

        Returns:
            Liste de Document avec ``content`` extrait heuristiquement selon le schéma.
        """
        parts = query.split(":")
        dataset_name = parts[0].strip()
        config_name = parts[1].strip() if len(parts) > 1 else None
        # CAS 3 — split="train" par défaut : le split train est présent dans tous les datasets
        # standardisés (SQuAD, MS MARCO). "validation" ou "test" peuvent ne pas exister.
        split = parts[2].strip() if len(parts) > 2 else "train"

        logger.info(
            f"HuggingFace : dataset='{dataset_name}' config='{config_name}' split='{split}'"
        )

        dataset = self._fetch_hf_dataset(dataset_name, config_name, split)

        documents: List[Document] = []
        sample_size = min(max_docs, len(dataset))

        for example in tqdm(dataset.select(range(sample_size)), desc="HuggingFace", unit="ex"):
            content, title = self._extract_hf_content(example, dataset_name)

            if not content.strip():
                # CAS 3 — Edge case exemple vide : certains datasets HF contiennent
                # des entrées avec champs textuels vides (artefacts d'annotation).
                # On skip silencieusement pour ne pas injecter du bruit dans le corpus.
                logger.debug(f"Exemple HuggingFace ignoré (contenu vide) : {example}")
                continue

            url = f"https://huggingface.co/datasets/{dataset_name}"
            documents.append(Document(
                # CAS 1 — On ajoute len(documents) au hash pour distinguer deux exemples
                # ayant le même titre (fréquent dans SQuAD où title = article source).
                id=self._compute_doc_id("huggingface", title, f"{url}#{len(documents)}"),
                content=content,
                source="huggingface",
                url=url,
                title=title,
                date=datetime.utcnow().strftime("%Y-%m-%d"),
                metadata={
                    "dataset": dataset_name,
                    "config": config_name,
                    "split": split,
                },
            ))

        logger.info(f"HuggingFace : {len(documents)} exemples chargés (/{sample_size} sélectionnés)")
        return documents

    @retry(
        stop=stop_after_attempt(3),
        # CAS 3 — min=3s : les téléchargements HuggingFace Hub incluent le transfert réseau
        # des métadonnées Arrow. Un retry immédiat serait inutile en cas de congestion.
        wait=wait_exponential(multiplier=1, min=3, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
    )
    def _fetch_hf_dataset(
        self, dataset_name: str, config_name: Optional[str], split: str
    ) -> Any:
        """Charge un dataset HuggingFace avec retry sur les erreurs réseau.

        Utilise streaming=False pour permettre select() par indices. Pour les datasets
        > 10 Go, envisager streaming=True avec itertools.islice() dans _download_huggingface.

        Args:
            dataset_name: Identifiant HuggingFace du dataset (ex: ``"squad"``).
            config_name:  Configuration du dataset, ou None pour la config par défaut.
            split:        Split à charger (``"train"``, ``"validation"``, ``"test"``).

        Returns:
            Dataset HuggingFace au format Arrow (itérable, supporte select() et len()).

        Raises:
            Exception: Si le dataset, la configuration ou le split n'existent pas.
        """
        # CAS 2 — trust_remote_code=False : on refuse d'exécuter du code arbitraire
        # depuis le Hub pour des raisons de sécurité. Rejeté True : risque d'injection.
        return load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=False,
            trust_remote_code=False,
        )

    def _extract_hf_content(self, example: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
        """Extrait le texte principal et le titre d'un exemple HuggingFace.

        Stratégie heuristique par ordre de priorité sur les noms de colonnes
        communs aux datasets NLP standards. Permet de supporter des datasets
        non anticipés via un fallback sur la concaténation des valeurs string.

        Args:
            example:      Dictionnaire représentant un exemple du dataset.
            dataset_name: Utilisé comme titre de repli si aucun champ titre n'est trouvé.

        Returns:
            Tuple ``(content, title)`` extraits de l'exemple.
        """
        # CAS 1 — Ordre "context" > "passage" > "text" > "document" :
        # "context" = SQuAD, "passage" = MS MARCO, "text" = datasets génériques.
        # La priorité garantit d'extraire le paragraphe source, pas la question.
        content_fields = ["context", "passage", "text", "document", "content", "body"]
        title_fields = ["title", "question", "id"]

        seen = set()

        content = ""
        content_field_used = '?'
        for field in content_fields:
            val = example.get(field, '')
            if isinstance(val, str) and val.strip():
                content = val
                content_field_used = field
                break

        if not content.strip() or content not in seen:
            # CAS 3 — Fallback concaténation : aucun champ standard trouvé (dataset custom).
            # On concatène toutes les valeurs string pour ne pas perdre l'exemple.
            content = " ".join(str(v) for v in example.values() if isinstance(v, str))
        
        seen.add(content)

        title = ""
        for field_name in title_fields:
            value = example.get(field_name, "")
            if isinstance(value, str) and value.strip():
                # CAS 3 — Troncature à 200 chars : certains datasets utilisent des
                # questions très longues comme "titre" — on évite des titres absurdes.
                title = value[:200]
                break

        if not title:
            title = f"{dataset_name}_example"

        return content, title

    # ─────────────────────────────────────────────
    # Cache et persistance
    # ─────────────────────────────────────────────

    def _compute_doc_id(self, source: str, title: str, url: str) -> str:
        """Calcule un identifiant stable et unique pour un document.

        # ─── ALGORITHME : Hash d'identité de document ────────────────────────
        # Problème résolu : identifier de façon stable un document across sessions
        #                   sans dépendance à un compteur ou UUID aléatoire.
        # Approche :        SHA-256 sur la concaténation (source:title:url).
        # Formule :         id = SHA256(f"{source}:{title}:{url}")[:16]
        # Référence :       Probabilité de collision Birthday Attack :
        #                   P ≈ n²/(2·2^64) < 10^-7 pour n < 10^6 documents.
        # ──────────────────────────────────────────────────────────────────────

        Args:
            source: Nom de la source (``'wikipedia'``, ``'arxiv'``, ``'huggingface'``).
            title:  Titre du document.
            url:    URL ou identifiant de la ressource.

        Returns:
            Chaîne hexadécimale de 16 caractères (64 bits d'entropie).
        """
        raw = f"{source}:{title}:{url}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _compute_query_hash(self, source: str, query: str, max_docs: int) -> str:
        """Calcule la clé de cache pour une requête.

        max_docs est inclus dans le hash car la même query avec des limites
        différentes produit des ensembles de documents différents.

        Args:
            source:   Nom de la source.
            query:    Requête brute (non normalisée).
            max_docs: Nombre de documents demandés.

        Returns:
            Chaîne hexadécimale de 12 caractères (clé de cache).
        """
        raw = f"{source}:{query}:{max_docs}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def _get_cache_path(self, source: str, query_hash: str) -> Optional[Path]:
        """Cherche un fichier cache correspondant à la requête.

        Le pattern ``{source}_*_{query_hash}.json`` identifie les fichiers
        créés par _save_documents() pour cette requête.

        Args:
            source:     Nom de la source (préfixe du fichier).
            query_hash: Hash de la requête (suffixe du fichier, 12 chars).

        Returns:
            Path du fichier cache existant, ou None si aucun match.
        """
        # CAS 1 — glob plutôt qu'un nom exact : le timestamp au milieu du nom rend
        # impossible la reconstruction du nom sans connaître la date de téléchargement.
        matches = list(self._raw_text_path.glob(f"{source}_*_{query_hash}.json"))
        return matches[0] if matches else None

    def _save_documents(
        self, documents: List[Document], source: str, query_hash: str
    ) -> Path:
        """Sérialise les documents en JSON et les persiste sur disque.

        Nom de fichier : ``{source}_{timestamp}_{query_hash}.json``
        Le query_hash en suffixe permet la détection de cache via glob dans _get_cache_path.

        Args:
            documents:  Liste de Document à sauvegarder.
            source:     Nom de la source (préfixe du nom de fichier).
            query_hash: Hash de la requête (suffixe du nom de fichier).

        Returns:
            Path du fichier JSON créé.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_{timestamp}_{query_hash}.json"
        output_path = self._raw_text_path / filename

        payload = {
            "source": source,
            "query_hash": query_hash,
            "downloaded_at": timestamp,
            "document_count": len(documents),
            "documents": [asdict(doc) for doc in documents],
        }

        # CAS 1 — ensure_ascii=False : les titres Wikipedia et les abstracts arXiv contiennent
        # des caractères Unicode (accents, symboles mathématiques). ensure_ascii=True les
        # convertirait en séquences \uXXXX illisibles dans les fichiers de debug.
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Sauvegardé : {output_path.name} ({len(documents)} documents)")
        return output_path

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        """Désérialise les documents depuis un fichier JSON de cache.

        Args:
            cache_path: Chemin vers un fichier créé par _save_documents().

        Returns:
            Liste de Document reconstruits depuis le JSON.
        """
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        documents = [Document(**doc_dict) for doc_dict in payload["documents"]]
        logger.debug(f"Cache : {len(documents)} documents chargés depuis {cache_path.name}")
        return documents

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Charge la configuration YAML du projet.

        Args:
            config_path: Chemin relatif ou absolu vers config.yaml.

        Returns:
            Dictionnaire de configuration.

        Raises:
            FileNotFoundError: Si config_path n'existe pas sur le système de fichiers.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier de configuration introuvable : {config_path}"
            )
        return yaml.safe_load(path.read_text(encoding="utf-8"))