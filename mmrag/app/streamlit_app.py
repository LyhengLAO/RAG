"""Streamlit frontend — MultiModal RAG comparison showcase.

Architecture
-----------
All data comes from the FastAPI backend (POST /query, GET /metrics, GET /health).
No heavy imports: requests only.  If the API is down, every page shows a clear
error message — no stack traces exposed to the user.

Run
---
    streamlit run app/streamlit_app.py
    # or, with a non-default API URL:
    MMRAG_API_BASE=http://localhost:8000 streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_API = os.environ.get("MMRAG_API_BASE", "http://localhost:8000")
_FIGURES_DIR = Path("results/figures")
_ICONS = {"text": "📄", "image": "🖼️", "audio": "🔊"}

_RETRIEVAL_KS = ["hit@1", "hit@5", "recall@5", "precision@5", "ndcg@5", "mrr"]
_RAGAS_KS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

_PIPELINE_LABELS = {
    "baseline": "Baseline — recursive · dense · no rerank",
    "optimized": "Optimisé — semantic · hybrid · rerank",
}

_ABLATION_LABELS = {
    "baseline":       "Baseline  (recursive · dense · off)",
    "chunking_only":  "Axis A   (semantic · dense · off)",
    "retrieval_only": "Axis B   (recursive · hybrid · off)",
    "rerank_only":    "Axis C   (recursive · dense · on)",
    "optimized":      "Optimisé (semantic · hybrid · on)",
}


# ── API helpers ────────────────────────────────────────────────────────────────


def _api_get(url: str, timeout: int = 10) -> tuple[Any, str | None]:
    """GET → (data, error_message).  Never raises; (None, msg) on any failure."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "API inaccessible — démarrez `uvicorn src.serving.api:app --port 8000`"
    except requests.Timeout:
        return None, "Délai dépassé — l'API met trop de temps à répondre."
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, f"Erreur API {exc.response.status_code} : {detail}"
    except Exception as exc:
        return None, f"Erreur inattendue : {exc}"


def _api_post(url: str, payload: dict, timeout: int = 90) -> tuple[Any, str | None]:
    """POST → (data, error_message).  Never raises."""
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "API inaccessible — démarrez `uvicorn src.serving.api:app --port 8000`"
    except requests.Timeout:
        return None, "Délai dépassé — Ollama est peut-être surchargé. Réessayez."
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, f"Erreur {exc.response.status_code} : {detail}"
    except Exception as exc:
        return None, f"Erreur inattendue : {exc}"


@st.cache_data(ttl=120)
def _fetch_metrics(api_base: str) -> tuple[Any, str | None]:
    return _api_get(f"{api_base}/metrics")


@st.cache_data(ttl=30)
def _fetch_health(api_base: str) -> tuple[Any, str | None]:
    return _api_get(f"{api_base}/health", timeout=5)


# ── Formatting helpers ─────────────────────────────────────────────────────────


def _fmt(v: Any, decimals: int = 3) -> str:
    """Format a numeric metric for table display; NaN / None / inf → '—'."""
    if v is None:
        return "—"
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return "—"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _meta_line(result: dict[str, Any]) -> str:
    """One-liner: latency, source count, modality breakdown."""
    latency = result.get("latency_ms", 0)
    sources = result.get("sources", [])
    counts: dict[str, int] = {}
    for s in sources:
        m = s.get("modality", "?")
        counts[m] = counts.get(m, 0) + 1
    breakdown = "  ".join(
        f"{_ICONS.get(m, '?')}{c}" for m, c in sorted(counts.items())
    )
    return f"⏱ {latency:.0f} ms  ·  {len(sources)} source{'s' if len(sources) != 1 else ''}  ·  {breakdown}"


def _delta_tile(
    col: Any,
    label: str,
    base_val: float | None,
    opt_val: float | None,
    higher_is_better: bool = True,
) -> None:
    """Render one metric delta tile in the given column."""
    with col:
        if base_val is None or opt_val is None:
            st.metric(label, "—")
            return
        try:
            b, o = float(base_val), float(opt_val)
        except (TypeError, ValueError):
            st.metric(label, "—")
            return
        if math.isnan(b) or math.isnan(o):
            st.metric(label, "—")
            return
        delta = o - b
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.3f}"
        st.metric(
            label,
            value=f"{b:.3f} → {o:.3f}",
            delta=delta_str,
            delta_color="normal" if higher_is_better else "inverse",
        )


# ── Context card ───────────────────────────────────────────────────────────────


def _render_text_ctx(text: str) -> None:
    truncated = text[:800] + ("…" if len(text) > 800 else "")
    st.markdown(
        f'<div style="background:#f8f9fa;border-left:3px solid #4F8EF7;'
        f'padding:10px 14px;border-radius:4px;font-size:0.9em;line-height:1.55;">'
        f'{truncated}</div>',
        unsafe_allow_html=True,
    )


def _render_image_ctx(caption: str, src_path: str) -> None:
    p = Path(src_path)
    if p.exists() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        st.image(str(p), width=300)
    else:
        st.caption(f"*(Miniature non disponible : `{src_path}`)*")
    st.markdown(
        f"**Caption embarquée :** {caption[:600]}{'…' if len(caption) > 600 else ''}"
    )


def _render_audio_ctx(transcript: str, src_path: str) -> None:
    p = Path(src_path)
    if p.exists() and p.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac", ".m4a"}:
        st.audio(str(p))
    else:
        st.caption(f"*(Audio non disponible : `{src_path}`)*")
    with st.expander("Transcript Whisper", expanded=True):
        st.markdown(
            f"_{transcript[:1000]}{'…' if len(transcript) > 1000 else ''}_"
        )


def _render_context_card(
    rank: int,
    text: str,
    source: dict[str, Any],
    highlight: bool = False,
    show_mechanics: bool = False,
    expanded: bool = False,
) -> None:
    """Render one retrieved context, adapting body to modality."""
    modality = source.get("modality", "text")
    doc_id   = source.get("doc_id", "—")
    src_path = source.get("source", "")
    icon     = _ICONS.get(modality, "📄")
    new_tag  = " 🆕" if highlight else ""

    header = f"{icon} #{rank}  `{doc_id}`  {modality.upper()}{new_tag}"

    with st.expander(header, expanded=expanded):
        if modality == "image":
            _render_image_ctx(text, src_path)
        elif modality == "audio":
            _render_audio_ctx(text, src_path)
        else:
            _render_text_ctx(text)

        # ── Optional retrieval mechanics ─────────────────────────────────────
        if show_mechanics:
            provenance  = source.get("provenance")
            r_from      = source.get("retrieval_rank")
            r_to        = source.get("final_rank")
            rerank_sc   = source.get("rerank_score")
            dense_sc    = source.get("dense_score")
            bm25_sc     = source.get("bm25_score")

            if any(v is not None for v in [provenance, r_from, r_to, rerank_sc]):
                st.divider()
                mc = st.columns(3)
                if provenance:
                    chip = {"dense": "🔵 Dense", "bm25": "🟠 BM25", "both": "🟢 Dense+BM25"}.get(
                        provenance, provenance
                    )
                    mc[0].markdown(f"**Provenance** {chip}")
                if r_from is not None and r_to is not None:
                    d = int(r_from) - int(r_to)
                    arrow = f"▲ {d}" if d > 0 else (f"▼ {abs(d)}" if d < 0 else "━")
                    mc[1].markdown(f"**Rang** #{r_from} → #{r_to}  {arrow}")
                if rerank_sc is not None:
                    mc[2].markdown(f"**Score rerank** {rerank_sc:.3f}")
                if dense_sc is not None or bm25_sc is not None:
                    score_parts = []
                    if dense_sc is not None:
                        score_parts.append(f"dense={dense_sc:.3f}")
                    if bm25_sc is not None:
                        score_parts.append(f"bm25={bm25_sc:.3f}")
                    st.caption("Scores : " + "  ·  ".join(score_parts))

        # ── Common footer ─────────────────────────────────────────────────────
        footer_parts = []
        if src_path and modality not in ("image", "audio"):
            footer_parts.append(f"`{src_path}`")
        lic = source.get("license")
        if lic:
            footer_parts.append(f"Licence : {lic}")
        if footer_parts:
            st.caption("  ·  ".join(footer_parts))


# ── Sidebar ────────────────────────────────────────────────────────────────────


def zrender_sidebar() -> tuple[str, bool]:
    """Render persistent sidebar; return (api_base, show_mechanics)."""
    with st.sidebar:
        st.title("🧠 MMRag")
        st.caption("Baseline vs Optimisé — RAG multimodal")
        st.divider()

        api_base = st.text_input(
            "API base URL",
            value=_DEFAULT_API,
            help="URL du serveur FastAPI (mmrag.serving.api)",
        ).rstrip("/")

        # Health indicator — auto-refreshes every 30 s (cache TTL)
        health, err = _fetch_health(api_base)
        if err:
            st.markdown("🔴 **API** — hors ligne")
        else:
            status = health.get("status", "?")
            dot    = "🟢" if status == "ok" else "🟡"
            st.markdown(f"{dot} **API** — {status}")
            st.caption(f"Modèle LLM : `{health.get('ollama_model', '—')}`")
            for name, loaded in health.get("pipelines_loaded", {}).items():
                st.caption(f"{'✅' if loaded else '❌'} Pipeline {name}")

        if st.button("↺ Rafraîchir santé", use_container_width=True):
            _fetch_health.clear()
            st.rerun()

        st.divider()
        show_mechanics = st.toggle(
            "Afficher la mécanique retrieval",
            value=False,
            help="Provenance dense/BM25, rang avant/après rerank",
        )

        st.divider()
        with st.expander("Commandes"):
            st.code("uvicorn mmrag.serving.api:app --port 8000", language="bash")
            st.code("streamlit run app/streamlit_app.py", language="bash")
            st.code("python scripts/run_comparison.py --no-ragas", language="bash")

    return api_base, show_mechanics


# ── Tab 1: Démo ───────────────────────────────────────────────────────────────


def tab_demo(api_base: str, show_mechanics: bool) -> None:
    st.header("Démo — inspection d'un pipeline")
    st.caption("Interrogez un pipeline et explorez les contextes récupérés par modalité.")

    q_col, p_col = st.columns([3, 1])
    with q_col:
        question = st.text_input(
            "Question",
            placeholder="Ex. : Where was this symphony first performed?",
            key="demo_q",
        )
    with p_col:
        pipeline = st.radio(
            "Pipeline",
            options=["baseline", "optimized"],
            format_func=lambda x: "Baseline" if x == "baseline" else "✨ Optimisé",
            key="demo_pipeline",
        )

    run = st.button("▶ Interroger", type="primary", key="demo_run")
    if not run:
        return
    if not question.strip():
        st.warning("Saisissez une question.")
        return

    with st.spinner(f"Interrogation du pipeline {pipeline}…"):
        result, err = _api_post(
            f"{api_base}/query",
            {"question": question, "pipeline": pipeline},
        )

    if err:
        st.error(f"⚠ {err}")
        return

    # ── Réponse ───────────────────────────────────────────────────────────────
    st.markdown(f"**{_PIPELINE_LABELS.get(pipeline, pipeline)}**")
    st.markdown(
        f'<div style="background:#f0f4ff;padding:14px 18px;border-radius:6px;'
        f'font-size:1.05em;line-height:1.6;">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(_meta_line(result))

    # Pipeline étapes
    if pipeline == "optimized":
        st.markdown(
            "**Étapes :** `chunking sémantique` → `hybride BM25+dense (RRF)` "
            "→ `CrossEncoder rerank` → `LLM`"
        )
    else:
        st.markdown("**Étapes :** `chunking récursif` → `dense` → `LLM`")

    st.divider()

    contexts = result.get("retrieved_contexts", [])
    sources  = result.get("sources", [])
    n = max(len(contexts), len(sources))

    if n == 0:
        st.info("Aucun contexte récupéré.")
        return

    st.markdown(f"**Contextes récupérés ({n})** — dépliez pour inspecter")
    for i in range(n):
        _render_context_card(
            rank=i + 1,
            text=contexts[i] if i < len(contexts) else "",
            source=sources[i] if i < len(sources) else {},
            show_mechanics=show_mechanics,
            expanded=(i == 0),  # première carte ouverte par défaut
        )


# ── Tab 2: Comparaison côte à côte ────────────────────────────────────────────


def tab_comparison(api_base: str, show_mechanics: bool) -> None:
    st.header("Comparaison côte à côte")
    st.caption("La même question envoyée aux deux pipelines — verdict en un coup d'œil.")

    question = st.text_input(
        "Question",
        placeholder="Ex. : What year was the Eiffel Tower completed?",
        key="cmp_q",
    )

    run = st.button("▶ Comparer les deux pipelines", type="primary", key="cmp_run")
    if not run:
        return
    if not question.strip():
        st.warning("Saisissez une question.")
        return

    results: dict[str, Any] = {}
    errors:  dict[str, str]  = {}

    with st.spinner("Baseline en cours…"):
        data, err = _api_post(f"{api_base}/query", {"question": question, "pipeline": "baseline"})
        if err:
            errors["baseline"] = err
        else:
            results["baseline"] = data

    with st.spinner("Optimisé en cours…"):
        data, err = _api_post(f"{api_base}/query", {"question": question, "pipeline": "optimized"})
        if err:
            errors["optimized"] = err
        else:
            results["optimized"] = data

    if not results:
        for name, msg in errors.items():
            st.error(f"⚠ [{name}] {msg}")
        return

    # ── Bandeau de verdict ────────────────────────────────────────────────────
    if "baseline" in results and "optimized" in results:
        base = results["baseline"]
        opt  = results["optimized"]
        b_ids = {s.get("doc_id") for s in base.get("sources", [])}
        o_ids = {s.get("doc_id") for s in opt.get("sources", [])}

        st.divider()
        st.markdown("### Verdict")
        vc = st.columns(4)

        _delta_tile(vc[0], "Latence (ms)", base.get("latency_ms"), opt.get("latency_ms"), higher_is_better=False)

        # Recall@5 depuis les agrégats /metrics si disponibles
        metrics_data, _ = _fetch_metrics(api_base)
        b_r5 = (
            metrics_data.get("baseline", {}).get("retrieval", {}).get("overall", {}).get("recall@5")
            if metrics_data else None
        )
        o_r5 = (
            metrics_data.get("optimized", {}).get("retrieval", {}).get("overall", {}).get("recall@5")
            if metrics_data else None
        )
        _delta_tile(vc[1], "Recall@5 (agrégat)", b_r5, o_r5, higher_is_better=True)

        b_faith = (
            metrics_data.get("baseline", {}).get("ragas", {}).get("overall", {}).get("faithfulness")
            if metrics_data else None
        )
        o_faith = (
            metrics_data.get("optimized", {}).get("ragas", {}).get("overall", {}).get("faithfulness")
            if metrics_data else None
        )
        _delta_tile(vc[2], "Fidélité (agrégat)", b_faith, o_faith, higher_is_better=True)

        with vc[3]:
            n_new = len(o_ids - b_ids)
            n_lost = len(b_ids - o_ids)
            st.metric(
                "Sources nouvelles / perdues",
                value=f"+{n_new}",
                delta=f"−{n_lost} du baseline" if n_lost else "aucune perdue",
                delta_color="off",
            )

        st.divider()

    # ── Doc-id sets pour surligner les différences ────────────────────────────
    b_ids = {s.get("doc_id") for s in results.get("baseline", {}).get("sources", [])}
    o_ids = {s.get("doc_id") for s in results.get("optimized", {}).get("sources", [])}

    # ── Deux colonnes miroir ──────────────────────────────────────────────────
    col_b, col_o = st.columns(2)

    for col, name in [(col_b, "baseline"), (col_o, "optimized")]:
        with col:
            if name not in results:
                st.error(f"⚠ {errors.get(name, 'Erreur inconnue')}")
                continue

            r = results[name]
            is_opt = name == "optimized"
            bg = "#f0f4ff" if is_opt else "#f8f8f8"
            label = "✨ **Optimisé**" if is_opt else "**Baseline**"

            st.markdown(
                f'<div style="background:{bg};padding:8px 12px;border-radius:6px;">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )
            st.caption(_meta_line(r))
            st.markdown(
                f'<div style="background:#fff;border:1px solid #e0e0e0;'
                f'padding:12px 16px;border-radius:6px;line-height:1.6;margin:8px 0;">'
                f'{r["answer"]}</div>',
                unsafe_allow_html=True,
            )

            contexts = r.get("retrieved_contexts", [])
            sources  = r.get("sources", [])
            n = max(len(contexts), len(sources))

            if n == 0:
                st.info("Aucun contexte.")
                continue

            st.markdown(f"**Contextes ({n})**")
            for i in range(n):
                src = sources[i] if i < len(sources) else {}
                doc_id = src.get("doc_id")
                # Sources présentes côté optimisé mais absentes du baseline = nouvelles
                highlight = is_opt and (doc_id not in b_ids)
                _render_context_card(
                    rank=i + 1,
                    text=contexts[i] if i < len(contexts) else "",
                    source=src,
                    highlight=highlight,
                    show_mechanics=show_mechanics,
                )


# ── Tab 3: Dashboard ──────────────────────────────────────────────────────────


def tab_dashboard(api_base: str) -> None:
    st.header("Dashboard résultats")
    st.caption(
        "Métriques agrégées depuis `results/metrics.json` "
        "(généré par `python scripts/run_comparison.py`)."
    )

    r_col, _ = st.columns([1, 5])
    with r_col:
        if st.button("🔄 Rafraîchir", key="dash_refresh"):
            _fetch_metrics.clear()
            st.rerun()

    metrics, err = _fetch_metrics(api_base)

    if err:
        st.error(f"⚠ {err}")
        st.info(
            "Les métriques ne sont pas encore disponibles. "
            "Lancez d'abord `python scripts/run_comparison.py --no-ragas` "
            "pour un run rapide sans LLM judge."
        )
        return

    if not metrics:
        st.warning("Fichier metrics.json vide ou absent.")
        return

    config_names = [k for k in metrics if k != "delta"]

    # ── Bandeau de deltas clés ────────────────────────────────────────────────
    if "baseline" in metrics and "optimized" in metrics:
        st.divider()
        st.markdown("### Deltas clés — optimisé vs baseline")
        b_r  = metrics["baseline"].get("retrieval", {}).get("overall", {})
        o_r  = metrics["optimized"].get("retrieval", {}).get("overall", {})
        b_q  = metrics["baseline"].get("ragas",     {}).get("overall", {})
        o_q  = metrics["optimized"].get("ragas",    {}).get("overall", {})
        b_sl = metrics["baseline"].get("system",    {}).get("overall", {}).get("latency_ms", {})
        o_sl = metrics["optimized"].get("system",   {}).get("overall", {}).get("latency_ms", {})

        dc = st.columns(5)
        _delta_tile(dc[0], "Recall@5",       b_r.get("recall@5"),           o_r.get("recall@5"),           True)
        _delta_tile(dc[1], "nDCG@5",         b_r.get("ndcg@5"),             o_r.get("ndcg@5"),             True)
        _delta_tile(dc[2], "Fidélité",        b_q.get("faithfulness"),       o_q.get("faithfulness"),       True)
        _delta_tile(dc[3], "Ans. Correct.",   b_q.get("answer_correctness"), o_q.get("answer_correctness"), True)
        _delta_tile(dc[4], "Latence p50 (ms)", b_sl.get("p50"),              o_sl.get("p50"),               False)

    # ── Sous-sections ─────────────────────────────────────────────────────────
    st.divider()
    section = st.radio(
        "",
        options=["Retrieval", "RAGAS", "Système", "Ablation"],
        horizontal=True,
        key="dash_section",
        label_visibility="collapsed",
    )

    if section == "Retrieval":
        _dash_retrieval(metrics, config_names)
    elif section == "RAGAS":
        _dash_ragas(metrics, config_names)
    elif section == "Système":
        _dash_system(metrics, config_names)
    elif section == "Ablation":
        _dash_ablation(metrics, config_names)

    # Figures de comparaison générées par run_comparison.py
    _dash_figures()


def _dash_retrieval(metrics: dict, config_names: list[str]) -> None:
    st.markdown("#### Retrieval (par config)")

    rows = []
    for cfg in config_names:
        overall = metrics[cfg].get("retrieval", {}).get("overall", {})
        row = {"Config": _ABLATION_LABELS.get(cfg, cfg)}
        for k in _RETRIEVAL_KS:
            row[k] = _fmt(overall.get(k))
        rows.append(row)

    st.dataframe(
        pd.DataFrame(rows).set_index("Config"),
        use_container_width=True,
    )

    # Barres recall@5 et ndcg@5
    st.markdown("#### Graphes")
    bc = st.columns(2)
    for col, metric in zip(bc, ["recall@5", "ndcg@5"]):
        with col:
            vals = {
                cfg: metrics[cfg].get("retrieval", {}).get("overall", {}).get(metric)
                for cfg in config_names
            }
            valid = {k: v for k, v in vals.items() if v is not None and not math.isnan(float(v))}
            if valid:
                st.bar_chart(
                    pd.DataFrame.from_dict(valid, orient="index", columns=[metric]),
                    use_container_width=True,
                )
                st.caption(metric)
            else:
                st.caption(f"*{metric} : aucune donnée*")

    # Par modalité
    st.markdown("#### Retrieval par modalité")
    for cfg in ["baseline", "optimized"]:
        if cfg not in metrics:
            continue
        per_mod = metrics[cfg].get("retrieval", {}).get("per_modality", {})
        if not per_mod:
            continue
        with st.expander(f"Config : {cfg}", expanded=(cfg == "optimized")):
            pm_rows = []
            for mod, vals in per_mod.items():
                row = {"Modalité": f"{_ICONS.get(mod, '?')} {mod}"}
                for k in _RETRIEVAL_KS:
                    row[k] = _fmt(vals.get(k))
                pm_rows.append(row)
            if pm_rows:
                st.dataframe(
                    pd.DataFrame(pm_rows).set_index("Modalité"),
                    use_container_width=True,
                )


def _dash_ragas(metrics: dict, config_names: list[str]) -> None:
    st.markdown("#### RAGAS (qualité réponse, LLM-as-judge)")
    st.caption(
        "Juge local : Ollama. "
        "— = métrique non calculée ou toutes les lignes ont échoué."
    )

    rows = []
    for cfg in config_names:
        ragas_block = metrics[cfg].get("ragas", {})
        overall = ragas_block.get("overall", {})
        n_rows  = ragas_block.get("n_rows")
        row = {"Config": _ABLATION_LABELS.get(cfg, cfg)}
        for k in _RAGAS_KS:
            row[k] = _fmt(overall.get(k))
        row["n éval."] = str(n_rows) if n_rows is not None else "—"
        rows.append(row)

    st.dataframe(
        pd.DataFrame(rows).set_index("Config"),
        use_container_width=True,
    )

    if "baseline" in metrics and "optimized" in metrics:
        chart_rows = []
        for k in _RAGAS_KS:
            bv = metrics["baseline"].get("ragas", {}).get("overall", {}).get(k)
            ov = metrics["optimized"].get("ragas", {}).get("overall", {}).get(k)
            try:
                b_f = float(bv) if bv is not None else None
                o_f = float(ov) if ov is not None else None
                if b_f is not None and not math.isnan(b_f) or o_f is not None and not math.isnan(o_f):
                    chart_rows.append({
                        "Métrique": k,
                        "Baseline": b_f if b_f is not None and not math.isnan(b_f) else 0.0,
                        "Optimisé": o_f if o_f is not None and not math.isnan(o_f) else 0.0,
                    })
            except (TypeError, ValueError):
                continue

        if chart_rows:
            st.bar_chart(
                pd.DataFrame(chart_rows).set_index("Métrique"),
                use_container_width=True,
            )


def _dash_system(metrics: dict, config_names: list[str]) -> None:
    st.markdown("#### Métriques système — latence + tokens")

    rows = []
    for cfg in config_names:
        sys_b = metrics[cfg].get("system", {}).get("overall", {})
        lat   = sys_b.get("latency_ms", {})
        tok   = sys_b.get("tokens", {})
        total_tok = tok.get("total_tokens", {})
        rows.append({
            "Config":           _ABLATION_LABELS.get(cfg, cfg),
            "Lat. p50 (ms)":    _fmt(lat.get("p50"), 0),
            "Lat. p95 (ms)":    _fmt(lat.get("p95"), 0),
            "Lat. moy. (ms)":   _fmt(lat.get("mean"), 0),
            "Tokens tot. moy.": _fmt(total_tok.get("mean") if isinstance(total_tok, dict) else None, 0),
            "n requêtes":       lat.get("n", "—"),
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("Config"),
        use_container_width=True,
    )

    # Par modalité
    st.markdown("#### Latence par modalité")
    for cfg in ["baseline", "optimized"]:
        if cfg not in metrics:
            continue
        per_mod = metrics[cfg].get("system", {}).get("per_modality", {})
        if not per_mod:
            continue
        with st.expander(f"Config : {cfg}", expanded=True):
            pm_rows = []
            for mod, vals in per_mod.items():
                lat = vals.get("latency_ms", {})
                pm_rows.append({
                    "Modalité":   f"{_ICONS.get(mod, '?')} {mod}",
                    "p50 (ms)":   _fmt(lat.get("p50"), 0),
                    "p95 (ms)":   _fmt(lat.get("p95"), 0),
                    "moy. (ms)":  _fmt(lat.get("mean"), 0),
                    "n":          lat.get("n", "—"),
                })
            if pm_rows:
                st.dataframe(
                    pd.DataFrame(pm_rows).set_index("Modalité"),
                    use_container_width=True,
                )


def _dash_ablation(metrics: dict, config_names: list[str]) -> None:
    st.markdown("#### Matrice d'ablation — contribution marginale par axe")
    st.caption(
        "Baseline = (recursive · dense · off). "
        "Chaque ligne flip un seul axe depuis le baseline. "
        "Optimisé = tous les axes activés."
    )

    rows = []
    for cfg in config_names:
        b_r  = metrics[cfg].get("retrieval", {}).get("overall", {})
        b_q  = metrics[cfg].get("ragas",     {}).get("overall", {})
        b_sl = metrics[cfg].get("system",    {}).get("overall", {}).get("latency_ms", {})
        rows.append({
            "Configuration":  _ABLATION_LABELS.get(cfg, cfg),
            "Recall@5":       _fmt(b_r.get("recall@5")),
            "nDCG@5":         _fmt(b_r.get("ndcg@5")),
            "MRR":            _fmt(b_r.get("mrr")),
            "Fidélité":       _fmt(b_q.get("faithfulness")),
            "Ans. correct.":  _fmt(b_q.get("answer_correctness")),
            "Lat. p50 (ms)":  _fmt(b_sl.get("p50"), 0),
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index("Configuration"),
            use_container_width=True,
        )
    else:
        st.info("Aucune donnée. Lancez `run_comparison.py` avec au moins 2 configs.")


def _dash_figures() -> None:
    """Display generated PNG figures from results/figures/."""
    figs = sorted(_FIGURES_DIR.glob("*.png")) if _FIGURES_DIR.exists() else []
    if not figs:
        return

    st.divider()
    st.markdown(f"#### Figures de comparaison ({len(figs)} graphes)")
    st.caption(f"Générées dans `{_FIGURES_DIR}` par `run_comparison.py`.")

    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(figs):
                f = figs[i + j]
                with col:
                    st.image(str(f), caption=f.stem.replace("_", " "), use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="MMRag — RAG Multimodal",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    api_base, show_mechanics = render_sidebar()

    tab1, tab2, tab3 = st.tabs(["🔍 Démo", "⚖️ Comparaison", "📊 Dashboard"])

    with tab1:
        tab_demo(api_base, show_mechanics)
    with tab2:
        tab_comparison(api_base, show_mechanics)
    with tab3:
        tab_dashboard(api_base)


if __name__ == "__main__":
    main()
