# SmartBrief: NLP Text Summarization System

SmartBrief is an **extractive text summarization** project. It automatically identifies important sentences in a long article and combines them into a shorter, readable summary. It is built with Python, Streamlit, NLTK, and scikit-learn.

## Required project modules

| Module | Implementation in this project |
| --- | --- |
| Text/document input | Paste text directly or upload a UTF-8 `.txt` / `.md` file. |
| Sentence tokenization | NLTK `sent_tokenize` divides the document into sentences. |
| Important sentence extraction | Each sentence is assigned an importance score; the highest-scoring sentences are selected. |
| Summary generation | Selected sentences are returned in their original document order. |
| Summary display | The app shows the summary, compression metrics, keywords, and an explainable sentence-score view. |

## NLP methods used

1. **TF-IDF**: gives higher scores to sentences containing distinctive terms.
2. **Weighted TF-IDF**: combines TF-IDF, sentence position, sentence length, and repeated keyword frequency.
3. **TextRank**: builds a graph from cosine similarity between sentences, then uses a PageRank-style algorithm to find central sentences.

The system is extractive: it selects original sentences rather than inventing new text. This makes the result explainable and avoids unsupported claims.

## How to run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal. For a quick presentation, select **Load demo article**, choose a method, and click **Generate summary**. Expand **Show important sentence extraction** to demonstrate the NLP scoring stage.

## Suggested viva explanation

“The user provides an article. The system tokenizes it into sentences and converts sentences into TF-IDF feature vectors. Depending on the chosen algorithm, it assigns each sentence an importance score. It selects the top-ranked sentences, restores their original order for readability, and displays the final summary with compression statistics. The extraction panel makes the model’s selection transparent.”
