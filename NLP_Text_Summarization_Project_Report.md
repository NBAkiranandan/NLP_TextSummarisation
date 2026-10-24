# NLP Text Summarization System

## A Cornerstone Project Report

Submitted by:

| Student name | Student ID |
| --- | --- |
| A Akash | 24B11CS003 |
| A Vamsi Durga | 24B11CS007 |
| GJS Srujith Raj | 24B11CS137 |
| NB Akirananadan | 24B11CS315 |

Under the guidance of **Dr. T. Sudha Rani**, Assistant Professor, Department of Computer Science and Engineering

**Aditya University**  
Surampalem  
Academic Year **2026-2027**

> Draft note: This is original project-specific text organized in the broad report structure of the supplied reference. Confirm the institution's required title-page, certificate, declaration, acknowledgement, citation, and AI-use wording before submission. Dataset provenance and exact software/hardware versions are not established by the repository and are marked for verification.

---

## Certificate

Use the current university-approved certificate format and obtain the required signatures. Suggested project-title line:

> This is to certify that the Cornerstone project entitled **"NLP Text Summarization System"**, submitted by the students listed on the title page, describes work carried out in partial fulfillment of the requirements for the award of the B.Tech. degree in Computer Science and Engineering during the academic year 2026-2027.

**Project Guide:** Dr. T. Sudha Rani, Assistant Professor, Department of CSE  
**Head of the Department:** [Confirm current name and designation]

## Declaration

> Replace this section with the university-approved declaration. Each team member should review and sign the final wording, confirm that the report accurately describes the team's work, and comply with the institution's academic-integrity and AI-use policies.

## Acknowledgement

We express our gratitude to our project guide, **Dr. T. Sudha Rani**, for her guidance and feedback during this project. We also thank the Department of Computer Science and Engineering, Aditya University, for the academic support and facilities provided. We appreciate the contributions of everyone who assisted with the project and its evaluation.

> Edit this acknowledgement to reflect the people and support actually involved in the project.

## Abstract

The **NLP Text Summarization System** is an interactive application for producing concise, extractive summaries of long articles and documents. A user can paste text or upload a UTF-8 text or Markdown file, choose a sentence-ranking method, and select a summary length. The application splits the input into sentences, calculates sentence-importance scores, selects the highest-ranked sentences, and returns them in their original order. Because the system extracts sentences from the input rather than generating new wording, the selected content can be inspected and traced to the source document.

The application provides three methods: TF-IDF sentence scoring, a weighted score combining TF-IDF with position, length, and keyword-frequency signals, and TextRank using sentence similarity and PageRank-style ranking. It also displays document keywords, compression information, and sentence-level scores. A separate analysis pipeline explores preprocessing and scoring and evaluates summaries against reference highlights using ROUGE-1, ROUGE-2, and ROUGE-L.

The repository contains article/highlight datasets with 287,113 training records, 13,368 validation records, and 11,490 test records before pipeline cleaning. The saved TextRank test-results file contains 11,490 records. Its recorded mean ROUGE F-measures are 0.3510 (ROUGE-1), 0.1361 (ROUGE-2), and 0.2264 (ROUGE-L). These figures describe the saved offline TextRank evaluation; they are not a comparison of all three current app methods. The project's data source and tested software and hardware specifications should be confirmed before final submission.

**Keywords:** Natural Language Processing, extractive summarization, TF-IDF, TextRank, ROUGE, Streamlit

## Table of Contents

1. [Introduction](#chapter-1-introduction)
2. [System Analysis and Design](#chapter-2-system-analysis-and-design)
3. [Dataset and Preprocessing](#chapter-3-dataset-and-preprocessing)
4. [Sentence Scoring and Ranking Methods](#chapter-4-sentence-scoring-and-ranking-methods)
5. [Summary Generation and Evaluation](#chapter-5-summary-generation-and-evaluation)
6. [Implementation](#chapter-6-implementation)
7. [Results, Dashboard, and Discussion](#chapter-7-results-dashboard-and-discussion)
8. [Conclusion and Future Enhancements](#chapter-8-conclusion-and-future-enhancements)
9. [References](#references)
10. [Appendix](#appendix)

# Chapter 1: Introduction

## 1.1 Introduction to the Project

People frequently need to read long articles, reports, and other documents to find their main points. Extractive text summarization addresses this need by identifying informative sentences in a source document and combining selected sentences into a shorter version. Unlike abstractive summarization, it does not compose new sentences; the output is drawn from the source.

This project implements an interactive summarizer and an accompanying analysis pipeline. The app lets a user compare three traditional NLP ranking methods and inspect which sentences contributed to the result. The analysis scripts use article-and-highlight data to explore preprocessing, sentence scoring, and ROUGE-based evaluation.

## 1.2 Problem Statement

Manually reviewing long documents can take time. The project aims to provide a simple way to extract a shorter, readable set of important sentences while preserving their source wording and order. The system should make its selection process visible and provide measures that help describe summary length and evaluation quality.

## 1.3 Objectives

- Accept pasted text and uploaded UTF-8 `.txt` or `.md` documents.
- Split documents into sentences and calculate sentence-level importance scores.
- Implement TF-IDF, weighted TF-IDF, and TextRank ranking methods.
- Select sentences according to a user-selected summary length and restore their original order.
- Display the summary, compression information, keywords, and sentence-level selection scores.
- Explore preprocessing and scoring methods in a reproducible analysis pipeline.
- Evaluate generated summaries against reference highlights with ROUGE metrics.

## 1.4 Scope

The interactive application is designed for single-document summarization. Its current algorithms are extractive and do not train a generative language model. The repository also contains offline scripts for data exploration, preprocessing, scoring, weight experimentation, and TextRank testing. The project does not establish that its scores generalize to every subject area, language, or document format.

# Chapter 2: System Analysis and Design

## 2.1 Proposed System

The proposed system combines a Streamlit user interface with sentence-ranking functions implemented in Python. The user supplies a document and chooses a ranking method and summary length. The application tokenizes the text, normalizes sentences for scoring, ranks them, selects a subset, and presents the selected original sentences in document order.

## 2.2 System Architecture and Workflow

```text
Text pasted by user / uploaded UTF-8 .txt or .md file
                         |
                         v
            Sentence tokenization and filtering
                         |
                         v
             Sentence text normalization
                         |
                         v
       TF-IDF / Weighted TF-IDF / TextRank scoring
                         |
                         v
         Select top-ranked sentences by length setting
                         |
                         v
      Restore source order and display summary and scores
```

The analysis pipeline is separate from the interactive app. Its scripts load the train, validation, and test CSV files; inspect and clean training records; preprocess sentences; compare scores; experiment with feature weights; and evaluate TextRank.

## 2.3 Module Description

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface, input handling, sentence processing, three app scoring methods, summary display, and navigation to statistics |
| `statistics_page.py` | Dataset distributions and method comparisons presented in the statistics page |
| `pipeline.py` | Runs the analysis stages in sequence |
| `pipeline_steps/01_data_exploration.py` | Loads and explores the data splits, removes selected training duplicates and overlaps, and inspects article/highlight lengths |
| `pipeline_steps/02_preprocessing.py` and `preprocessing_utils.py` | Preprocess article sentences for the analysis pipeline |
| `pipeline_steps/03_scoring_and_evaluation.py` | Builds initial sentence scores and evaluates summaries on a reproducible validation sample |
| `pipeline_steps/04_weight_optimization.py` | Compares combinations of scoring-feature weights using validation ROUGE-L |
| `pipeline_steps/05_textrank_and_test.py` | Applies TextRank to validation and test data and writes test scores to CSV |
| `pipeline_steps/scoring_utils.py` and `evaluation_utils.py` | Shared score normalization, feature-score, and ROUGE evaluation helpers |

## 2.4 Requirements

### 2.4.1 Software

The project uses Python and the packages listed in `requirements.txt`: Streamlit, NumPy, scikit-learn, NLTK, `rouge-score`, pandas, Matplotlib, seaborn, NetworkX, and SciPy.

Run the interactive application from the project directory:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Run the analysis pipeline with:

```powershell
python pipeline.py
```

The exact Python version used for the reported results should be added after verification.

### 2.4.2 Hardware

No fixed hardware specification is recorded in the repository. The app can process individual documents locally. Full-dataset analysis, especially operations on the large training CSV, can require substantially more time and memory than interactive summarization. Add the actual computer specifications used for evaluation before submission.

## 2.5 Tools and Technologies

| Tool or technology | Use in the project |
| --- | --- |
| Python | Application and analysis code |
| Streamlit | Interactive web application and statistics interface |
| NLTK | Sentence tokenization and analysis-pipeline tokenization, tagging, stopword removal, and lemmatization |
| scikit-learn | TF-IDF vectorization, cosine similarity, and app sentence scoring |
| NetworkX | PageRank in the analysis-pipeline TextRank implementation |
| pandas and NumPy | Dataset handling, score arrays, and analysis |
| `rouge-score` | Reference-based ROUGE evaluation |
| Matplotlib and seaborn | Statistical charts |

# Chapter 3: Dataset and Preprocessing

## 3.1 Dataset Description and Data Format

The repository's `train.csv`, `validation.csv`, and `test.csv` files each use the fields `id`, `article`, and `highlights`. Here, `article` contains the source document and `highlights` contains its reference summary. The following are raw row counts read from the supplied files, before the analysis pipeline applies its training-data cleaning steps:

| Split | Raw records |
| --- | ---: |
| Training | 287,113 |
| Validation | 13,368 |
| Test | 11,490 |

The repository does not document the original dataset citation, collection process, or license. Add the verified source, license, and any required attribution here before publication or submission. Do not infer provenance from the column names alone.

## 3.2 Data Preparation

The data-exploration stage inspects the splits, removes duplicate training rows based on the `article` and `highlights` pair, removes training articles that overlap with validation articles, and excludes training records whose reference highlight is longer than the article by word count. The test and validation CSV files are used as evaluation inputs; their raw row counts above do not represent the number of cleaned training records.

## 3.3 Text Preprocessing

The analysis pipeline and the interactive app use related but distinct preprocessing paths:

**Analysis pipeline**

1. Split article text into sentences using NLTK.
2. Omit empty sentences and sentences identified as source labels, timestamp-bearing lines, or lines containing a four-digit year.
3. Lowercase and normalize whitespace.
4. Replace non-letter characters with spaces and tokenize with NLTK.
5. Remove English stopwords.
6. Part-of-speech tag tokens and lemmatize them with WordNet.
7. Retain sentences with at least two processed words.

**Interactive app**

1. Split input using NLTK sentence tokenization, with a regular-expression fallback if the tokenizer resource is unavailable.
2. Discard sentences with fewer than three whitespace-separated words.
3. Lowercase and replace non-alphanumeric characters for feature calculation.
4. Leave the original sentence text unchanged for the displayed summary.
5. Apply scikit-learn's English stopword handling when constructing TF-IDF vectors.

The two paths serve different purposes and should not be described as identical preprocessing.

# Chapter 4: Sentence Scoring and Ranking Methods

## 4.1 TF-IDF Sentence Scoring

The app transforms the processed sentences in one document into TF-IDF vectors. It sums the TF-IDF values for each sentence and normalizes the resulting scores by their maximum. Sentences with distinctive terms can therefore receive higher scores. If vectorization fails because the text has no usable vocabulary, the current app assigns a fallback score of one to each sentence.

## 4.2 Weighted TF-IDF

The current interactive app combines four normalized features:

```text
Weighted score =
    0.35 * TF-IDF score
  + 0.25 * sentence-position score
  + 0.15 * sentence-length score
  + 0.25 * repeated-keyword score
```

Earlier analysis stages also examine feature-weight combinations. The offline optimization script orders candidate combinations by mean ROUGE-L on its validation sample. Its experimental coefficients are not automatically applied to the app: the app uses the fixed coefficients shown above.

## 4.3 TextRank

For the app's TextRank method, TF-IDF vectors are compared using cosine similarity to form a sentence-similarity graph. A PageRank-style iterative calculation ranks the sentences, using a damping factor of 0.85 and 50 iterations. The highest-ranked sentences are selected for the summary.

The analysis-pipeline implementation uses NetworkX PageRank on a cosine-similarity graph. This implementation is part of the offline evaluation scripts and is not identical in code to the iterative TextRank function in the app.

## 4.4 Sentence Selection

The app offers Low, Medium, and High summary lengths, corresponding to 20%, 30%, and 40% of the eligible input sentences, respectively. At least one sentence is selected when there is usable input, and the count is capped at the number of available sentences. The app requires at least two eligible sentences before generating a summary. Selected sentences are sorted by their original positions before display.

# Chapter 5: Summary Generation and Evaluation

## 5.1 Summary Generation

For a selected method, the system calculates one score per eligible sentence, ranks the scores, selects the requested number of sentences, and concatenates those original sentences in document order. This approach preserves the selected wording but can produce transitions that were not written as a standalone summary.

## 5.2 Evaluation Measures

The offline evaluation uses ROUGE-1, ROUGE-2, and ROUGE-L F-measures to compare generated summaries with reference highlights:

- **ROUGE-1** measures unigram overlap.
- **ROUGE-2** measures bigram overlap.
- **ROUGE-L** measures longest-common-subsequence overlap.

These are reference-overlap measures. They do not by themselves establish factual correctness, coherence, or usefulness to every reader.

## 5.3 Evaluation Design

The pipeline uses a reproducible sample of up to 100 validation records with `random_state=42` for its method and feature-weight experiments. The saved `textrank_test_results.csv` contains one row per test record and records ROUGE scores for the offline TextRank evaluation. The Streamlit statistics page has its own documented evaluation flows; results from those flows should be reported separately from the saved offline TextRank results.

# Chapter 6: Implementation

## 6.1 User Input and Interface

The Streamlit app accepts pasted text or uploaded `.txt` and `.md` files decoded as UTF-8. It provides a demo article, a sentence-scoring method selector, and a summary-length selector. Unsupported file encodings are reported to the user.

## 6.2 Application Processing

The main processing functions in `app.py` tokenize sentences, normalize feature text, calculate TF-IDF or weighted scores, calculate TextRank scores, select sentence indices, and restore the original sentence order. The extraction panel displays each eligible sentence and its score and marks sentences included in the summary.

## 6.3 Offline Analysis

The offline pipeline is launched by `pipeline.py`, which executes the scripts in `pipeline_steps/` sequentially. The stages share intermediate data in memory. The TextRank test stage writes its per-record metrics to `textrank_test_results.csv`.

## 6.4 Reproducibility Notes

The app's fixed scoring coefficients and the offline pipeline's experimental feature weights are separate. The offline weight-search procedure evaluates candidates on validation data, while the interactive app's current weighted method has fixed coefficients in `app.py`. Report results with the method, data split, sample size, and code path that produced them.

# Chapter 7: Results, Dashboard, and Discussion

## 7.1 Application Output

For each generated summary, the app displays the number of input sentences, the number selected, original word count, and compression percentage. It also displays frequently occurring document keywords and an expandable sentence-by-sentence scoring view.

## 7.2 Saved Test Results

The repository's saved TextRank results were summarized directly from `textrank_test_results.csv`:

| Metric | Mean F-measure | Sample standard deviation | Records |
| --- | ---: | ---: | ---: |
| ROUGE-1 | 0.3510 | 0.1125 | 11,490 |
| ROUGE-2 | 0.1361 | 0.0998 | 11,490 |
| ROUGE-L | 0.2264 | 0.0914 | 11,490 |

These are the saved offline TextRank test scores. They should not be presented as the interactive app's three-method comparison, as a trained-model accuracy, or as evidence that the system was trained on the test data.

## 7.3 Statistics Page

The Streamlit statistics page provides corpus length and compression distributions and comparisons of the available summary methods. It uses a reproducible validation sample for method-level ROUGE evaluation and offers an optional train-versus-test comparison. The exact sample and evaluation flow should be identified whenever results from the page are included in the final report.

## 7.4 Screenshots and Figures to Add

Capture current screenshots from the running application and insert them in the final formatted report. Suggested figures:

1. Main summarizer page with input controls.
2. Generated summary and compression metrics.
3. Expanded sentence extraction and scoring panel.
4. Summarized Statistics page and its evaluation chart.
5. Terminal output from the offline pipeline, if used to document execution.

Do not use screenshots or results from the reference report's hotel-review sentiment application.

## 7.5 Discussion and Limitations

The system is transparent in that each selected sentence comes from the source document and the app exposes its score. However, sentence-level ranking may miss context that spans multiple sentences, favor early sentences or repeated terms, or select text that needs surrounding context. Extractive output can also be grammatically awkward when selected sentences are joined together. The analysis pipeline's metadata filters are heuristic and may omit useful sentences in some documents. Evaluation scores depend on overlap with the available reference highlights and do not replace human assessment.

# Chapter 8: Conclusion and Future Enhancements

## 8.1 Conclusion

This project provides an interactive extractive summarizer using TF-IDF, weighted sentence features, and TextRank, together with an offline pipeline for data preparation and ROUGE evaluation. Users can compare methods, choose summary length, and inspect which source sentences were selected. The saved test-results file provides a reproducible record of the offline TextRank evaluation.

## 8.2 Limitations

- The system selects existing sentences and does not rewrite them into a newly generated narrative.
- Sentence ranking depends on lexical features, position, sentence length, and similarity; it does not model deep semantic context.
- The app and analysis pipeline have distinct preprocessing and scoring implementations.
- Dataset provenance and licensing are not recorded in the repository.
- ROUGE measures text overlap rather than factuality, readability, or reader satisfaction.
- Processing a very large training file for optional comparisons can take time and memory.

## 8.3 Future Enhancements

- Record and cite the dataset's verified source, license, and intended use.
- Add human evaluation for relevance, readability, and factual consistency.
- Compare the current extractive methods with clearly identified baseline or abstractive systems, if project scope and resources allow.
- Add support for additional input formats and languages with appropriate tokenizers.
- Consolidate or clearly version the app and offline scoring implementations.
- Add automated tests for preprocessing, edge cases, sentence selection, and evaluation outputs.
- Document the tested Python and dependency versions and preserve evaluation settings with each result.

# References

1. Natural Language Toolkit (NLTK), official documentation: <https://www.nltk.org/>
2. scikit-learn, `TfidfVectorizer` documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>
3. NetworkX, PageRank documentation: <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html>
4. Streamlit, official documentation: <https://docs.streamlit.io/>
5. Google `rouge-score` package: <https://pypi.org/project/rouge-score/>
6. Dataset citation and license: **[Add verified dataset reference; not specified in the repository.]**

# Appendix

## A. Source Files

- `app.py`
- `statistics_page.py`
- `pipeline.py`
- `pipeline_steps/01_data_exploration.py`
- `pipeline_steps/02_preprocessing.py`
- `pipeline_steps/03_scoring_and_evaluation.py`
- `pipeline_steps/04_weight_optimization.py`
- `pipeline_steps/05_textrank_and_test.py`
- `requirements.txt`

## B. Finalization Checklist

- Confirm the university's required title page, certificate, declaration, and acknowledgement templates.
- Confirm that the guide, department, institution, and academic year are current.
- Confirm team member names and IDs exactly as they should appear on the submitted report.
- Add the verified dataset source, citation, license, and attribution.
- Add the tested Python version, dependency versions, and actual hardware specifications.
- Capture screenshots from the current version of the app.
- Re-run the evaluation before submission if the code, datasets, or evaluation settings have changed.
- Update the result table only from the exact evaluation run being reported.
- Generate the final page-numbered table of contents after formatting the report.
