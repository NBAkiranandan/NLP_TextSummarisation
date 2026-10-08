"""Dataset statistics and method comparisons for the Streamlit app."""

import csv
import math
import os
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import streamlit as st
from rouge_score import rouge_scorer


METHODS = ("TF-IDF", "Weighted TF-IDF", "TextRank")
ROUGE_METRICS = ("rouge1", "rouge2", "rougeL")
ROUGE_LABELS = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
}
SUMMARY_RATIO = 0.30
EVALUATION_SAMPLE_SIZE = 100
EVALUATION_RANDOM_STATE = 42


@st.cache_data(show_spinner=False)
def _load_dataset_statistics(
    dataset_path: str, modified_ns: int
) -> Tuple[List[int], List[int], List[float]]:
    """Read full-corpus article and reference-summary distributions."""
    article_lengths: List[int] = []
    summary_lengths: List[int] = []
    compression_ratios: List[float] = []

    with open(dataset_path, encoding="utf-8-sig", newline="") as dataset_file:
        for row in csv.DictReader(dataset_file):
            article = (row.get("article") or "").strip()
            reference = (row.get("highlights") or "").strip()
            article_length = len(article.split())
            summary_length = len(reference.split())

            if not article_length:
                continue

            article_lengths.append(article_length)
            summary_lengths.append(summary_length)
            compression_ratios.append(summary_length / article_length)

    return article_lengths, summary_lengths, compression_ratios


@st.cache_data(show_spinner=False)
def _load_validation_sample(
    dataset_path: str, modified_ns: int
) -> List[Tuple[str, str]]:
    """Select the reproducible validation sample used by the project pipeline."""
    with open(dataset_path, encoding="utf-8-sig", newline="") as dataset_file:
        record_count = sum(1 for _ in csv.DictReader(dataset_file))

    if not record_count:
        return []

    sample_indices = set(
        np.random.RandomState(EVALUATION_RANDOM_STATE).choice(
            record_count,
            size=min(EVALUATION_SAMPLE_SIZE, record_count),
            replace=False,
        ).tolist()
    )
    evaluation_records: List[Tuple[str, str]] = []
    with open(dataset_path, encoding="utf-8-sig", newline="") as dataset_file:
        for index, row in enumerate(csv.DictReader(dataset_file)):
            if index in sample_indices:
                evaluation_records.append(
                    (
                        (row.get("article") or "").strip(),
                        (row.get("highlights") or "").strip(),
                    )
                )

    return evaluation_records


@st.cache_data(show_spinner=False)
def _evaluate_methods(
    evaluation_records: Tuple[Tuple[str, str], ...],
    _summarize: Callable,
    _tokenize_sentences: Callable,
) -> Tuple[List[Dict[str, float]], int]:
    """Compute comparable ROUGE means from the app's current algorithms."""
    scorer = rouge_scorer.RougeScorer(list(ROUGE_METRICS), use_stemmer=True)
    scores: Dict[str, Dict[str, List[float]]] = {
        method: {metric: [] for metric in ROUGE_METRICS} for method in METHODS
    }
    evaluated_articles = 0

    for article, reference in evaluation_records:
        if not reference or len(_tokenize_sentences(article)) < 2:
            continue

        evaluated_articles += 1
        for method in METHODS:
            summary = _summarize(article, method, SUMMARY_RATIO)[0]
            article_scores = scorer.score(reference, summary)
            for metric in ROUGE_METRICS:
                scores[method][metric].append(article_scores[metric].fmeasure)

    means = [
        {
            "Method": method,
            **{
                ROUGE_LABELS[metric]: (
                    statistics.fmean(scores[method][metric])
                    if scores[method][metric]
                    else 0.0
                )
                for metric in ROUGE_METRICS
            },
        }
        for method in METHODS
    ]
    return means, evaluated_articles


def _histogram_data(values: List[float], bin_count: int) -> Dict[str, List]:
    counts, edges = np.histogram(values, bins=bin_count)
    labels = [
        f"{start:,.2f}–{end:,.2f}"
        for start, end in zip(edges[:-1], edges[1:])
    ]
    return {"Range": labels, "Count": counts.tolist()}


def _distribution_chart(
    title: str,
    values: List[float],
    x_label: str,
    y_label: str,
    description: str,
) -> None:
    st.subheader(title)
    if len(values) < 2:
        st.info("At least two dataset observations are required to draw this distribution.")
        return

    bins = min(20, max(5, round(math.sqrt(len(values)))))
    st.bar_chart(
        _histogram_data(values, bins),
        x="Range",
        y="Count",
        x_label=x_label,
        y_label=y_label,
        height=320,
    )
    st.caption(description)


def _article_method_comparison(
    article: str,
    summarize: Callable,
    tokenize_sentences: Callable,
    rouge_rows: List[Dict[str, float]],
) -> None:
    st.subheader("Method Performance for Current Article")
    sentences = tokenize_sentences(article)
    if len(sentences) < 2:
        st.info(
            "Enter an article with at least two complete sentences on the "
            "Summarizer page to compare its methods."
        )
        return

    rows = []
    for method in METHODS:
        summary, _, sentence_scores, selected = summarize(
            article, method, SUMMARY_RATIO
        )
        selected_scores = [float(sentence_scores[index]) for index in selected]
        score = statistics.fmean(selected_scores) if selected_scores else 0.0
        article_length = len(article.split())
        summary_length = len(summary.split())
        rows.append(
            {
                "Method": method,
                "Summary Length": summary_length,
                "Compression Ratio": (
                    summary_length / article_length if article_length else 0.0
                ),
                "Score": score,
                "Summary": summary,
            }
        )

    st.bar_chart(
        {
            "Method": [row["Method"] for row in rows],
            "Mean Selected Sentence Score": [row["Score"] for row in rows],
        },
        x="Method",
        y="Mean Selected Sentence Score",
        x_label="Summarization Method",
        y_label="Mean Selected Sentence Score",
        height=300,
    )
    st.caption(
        "All methods use the same 30% sentence-selection target. The score is "
        "the mean existing sentence-ranking score of the selected sentences; "
        "it is a salience proxy, not reference-based quality."
    )
    st.dataframe(
        [
            {
                "Method": row["Method"],
                "Summary Length": row["Summary Length"],
                "Compression Ratio": round(row["Compression Ratio"], 3),
                "Score": round(row["Score"], 4),
                "Summary": row["Summary"],
            }
            for row in rows
        ],
        hide_index=True,
        width="stretch",
    )

    best_score = max(row["Score"] for row in rows)
    best_methods = [
        row["Method"]
        for row in rows
        if math.isclose(row["Score"], best_score, rel_tol=1e-9, abs_tol=1e-12)
    ]
    macro_scores = {
        row["Method"]: statistics.fmean(
            row[ROUGE_LABELS[metric]] for metric in ROUGE_METRICS
        )
        for row in rouge_rows
    }
    recommended_method = max(
        best_methods,
        key=lambda method: macro_scores.get(method, 0.0),
    )
    st.markdown("### Suitable Method for This Article")
    with st.container(border=True):
        st.markdown(f"**Recommended Method: {recommended_method}**")
        if len(best_methods) == 1:
            st.write(
                f"{recommended_method} has the highest mean selected-sentence "
                f"score ({best_score:.4f}) for this article. The score measures "
                "the method's own sentence-ranking signal; without a reference "
                "summary, it is not a ROUGE-based quality judgment."
            )
        else:
            tie_break = (
                f"Its dataset macro-average ROUGE score ({macro_scores[recommended_method]:.4f}) "
                "is highest among the tied methods."
                if macro_scores
                else "The listed method order resolves the exact tie."
            )
            st.write(
                f"The methods tie for the highest mean selected-sentence "
                f"score ({best_score:.4f}) for this article. {tie_break} "
                "The article-specific score is a sentence-ranking salience "
                "proxy, not a ROUGE-based quality judgment."
            )


def render_statistics_page(
    current_article: str,
    summarize: Callable,
    tokenize_sentences: Callable,
) -> None:
    """Render the standalone statistics and comparison page."""
    st.title("Summarized Statistics")
    st.caption(
        "Dataset-level distributions and transparent comparisons of the "
        "project's three extractive summarization methods."
    )

    dataset_path = Path(__file__).with_name("test.csv")
    validation_path = Path(__file__).with_name("validation.csv")
    if not dataset_path.is_file():
        st.error(
            "The held-out dataset file test.csv was not found beside the app. "
            "Place the project's test.csv file there to display dataset statistics."
        )
        _article_method_comparison(
            current_article, summarize, tokenize_sentences, []
        )
        return
    if not validation_path.is_file():
        st.error(
            "The validation dataset file validation.csv was not found beside the app. "
            "Place the project's validation.csv file there to compare ROUGE scores."
        )

    try:
        dataset_mtime = os.stat(dataset_path).st_mtime_ns
        article_lengths, summary_lengths, compression_ratios = (
            _load_dataset_statistics(str(dataset_path), dataset_mtime)
        )
        if validation_path.is_file():
            validation_mtime = os.stat(validation_path).st_mtime_ns
            evaluation_records = _load_validation_sample(
                str(validation_path), validation_mtime
            )
        else:
            evaluation_records = []
    except (OSError, csv.Error, UnicodeError) as error:
        st.error(f"Could not read the held-out dataset: {error}")
        _article_method_comparison(
            current_article, summarize, tokenize_sentences, []
        )
        return

    if not article_lengths:
        st.error("The held-out dataset contains no non-empty articles.")
        _article_method_comparison(
            current_article, summarize, tokenize_sentences, []
        )
        return

    rouge_rows, evaluated_articles = _evaluate_methods(
        tuple(evaluation_records), summarize, tokenize_sentences
    )
    st.subheader("Dataset Overview")
    if not evaluated_articles:
        st.warning(
            "No eligible multi-sentence articles were found in the evaluation sample."
        )
    else:
        macro_scores = {
            row["Method"]: statistics.fmean(
                row[ROUGE_LABELS[metric]] for metric in ROUGE_METRICS
            )
            for row in rouge_rows
        }
        best_method = max(METHODS, key=lambda method: macro_scores[method])
        best_by_metric = {
            metric: max(
                rouge_rows, key=lambda row: row[ROUGE_LABELS[metric]]
            )
            for metric in ROUGE_METRICS
        }
        distribution_cards = st.columns(4)
        distribution_cards[0].metric(
            "Number of Articles", f"{len(article_lengths):,}"
        )
        distribution_cards[1].metric(
            "Average Article Length", f"{statistics.fmean(article_lengths):,.0f} words"
        )
        distribution_cards[2].metric(
            "Average Summary Length", f"{statistics.fmean(summary_lengths):,.0f} words"
        )
        distribution_cards[3].metric(
            "Average Compression Ratio",
            f"{statistics.fmean(compression_ratios):.3f}",
        )
        performance_cards = st.columns(4)
        performance_cards[0].metric("Best Performing Method", best_method)
        performance_cards[1].metric(
            "Best ROUGE-1",
            f"{best_by_metric['rouge1']['ROUGE-1']:.3f}",
            help=f"Highest mean on the evaluation sample: {best_by_metric['rouge1']['Method']}",
        )
        performance_cards[2].metric(
            "Best ROUGE-2",
            f"{best_by_metric['rouge2']['ROUGE-2']:.3f}",
            help=f"Highest mean on the evaluation sample: {best_by_metric['rouge2']['Method']}",
        )
        performance_cards[3].metric(
            "Best ROUGE-L",
            f"{best_by_metric['rougeL']['ROUGE-L']:.3f}",
            help=f"Highest mean on the evaluation sample: {best_by_metric['rougeL']['Method']}",
        )
        st.caption(
            f"Distribution metrics use all {len(article_lengths):,} rows with "
            "non-empty articles in test.csv. ROUGE averages use "
            f"{evaluated_articles} eligible articles among the "
            f"{EVALUATION_SAMPLE_SIZE} validation records sampled with "
            f"random_state={EVALUATION_RANDOM_STATE} to match the project "
            "pipeline's evaluation split. "
            "Best Performing Method is selected by the highest macro-average "
            "of ROUGE-1, ROUGE-2, and ROUGE-L."
        )

    article_median = statistics.median(article_lengths)
    article_p10, article_p90 = np.percentile(article_lengths, [10, 90])
    summary_median = statistics.median(summary_lengths)
    summary_p10, summary_p90 = np.percentile(summary_lengths, [10, 90])
    compression_median = statistics.median(compression_ratios)

    st.divider()
    _distribution_chart(
        "Article Length Distribution",
        article_lengths,
        "Article Length (Words)",
        "Number of Articles",
        f"The median article contains about {article_median:,.0f} words. "
        f"The central 80% range is approximately {article_p10:,.0f}–"
        f"{article_p90:,.0f} words, indicating the observed spread.",
    )
    _distribution_chart(
        "Summary Length Distribution",
        summary_lengths,
        "Summary Length (Words)",
        "Number of Summaries",
        f"Reference summaries have a median length of about "
        f"{summary_median:,.0f} words, with a central 80% range of "
        f"{summary_p10:,.0f}–{summary_p90:,.0f} words.",
    )
    _distribution_chart(
        "Compression Ratio Distribution",
        compression_ratios,
        "Compression Ratio",
        "Number of Articles",
        f"Compression ratio is reference-summary words divided by article "
        f"words (median {compression_median:.3f}). A lower ratio means the "
        "summary contains substantially fewer words than the original article.",
    )

    st.divider()
    st.subheader("ROUGE Score Comparison of Summarization Methods")
    if evaluated_articles:
        chart_rows = {
            "Method": [],
            "Metric": [],
            "Score": [],
        }
        for row in rouge_rows:
            for metric in ROUGE_METRICS:
                chart_rows["Method"].append(row["Method"])
                chart_rows["Metric"].append(ROUGE_LABELS[metric])
                chart_rows["Score"].append(row[ROUGE_LABELS[metric]])
        st.bar_chart(
            chart_rows,
            x="Method",
            y="Score",
            color="Metric",
            stack=False,
            x_label="Summarization Method",
            y_label="ROUGE Score",
            height=360,
        )
        st.caption(
            "Bars show mean ROUGE F-measures against reference highlights, "
            f"calculated with the same {SUMMARY_RATIO:.0%} sentence-selection "
            "target for each app method."
        )
        st.dataframe(rouge_rows, hide_index=True, width="stretch")

    _article_method_comparison(
        current_article, summarize, tokenize_sentences, rouge_rows
    )

    st.divider()
    st.subheader("Academic Interpretation")
    st.markdown(
        "- Reference summaries are substantially shorter than source articles, "
        "as reflected by the summary-length and compression distributions.\n"
        "- A lower compression ratio indicates greater reduction in text size.\n"
        "- ROUGE-1, ROUGE-2, and ROUGE-L quantify unigram, bigram, and "
        "longest-common-subsequence overlap with reference summaries.\n"
        "- Comparing mean ROUGE scores on the same held-out examples provides "
        "a quantitative comparison of the three implemented extractive methods."
    )
