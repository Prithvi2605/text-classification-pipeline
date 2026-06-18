# Text Classification Pipeline (OOP)

A text classifier built as a proper class instead of a loose notebook of cells. The interesting part of this project isn't the accuracy (it's not great, and that's kind of the point), it's that the whole thing is structured as a reusable pipeline and includes a proper experiment on the accuracy-vs-speed trade-off.

## The structure

Everything lives in one `MentalHealthClassifier` class. It holds the pipeline state on the object and each step is its own method:

`load_data()` → `clean_data()` → `split_data()` → `extract_features()` → `train_model()` → `test_model()` → `visualise_results()`, plus `apply_pca()`.

The reason for doing it this way: when I wanted to add PCA, I didn't have to rewrite anything, I just added a method. Same pipeline, point it at a new dataset or change the split ratio and re-run it. That's the whole argument for OOP here.

The classifier itself is k-NN on TF-IDF features.

## What the experiment showed

I compared four setups, different train/test splits, with and without PCA, measuring both accuracy and how long it took:

| Setup | Accuracy | Time (s) |
|---|---|---|
| 70-30, no PCA | 57.6% | 2.70 |
| 80-20, no PCA | 57.1% | 1.95 |
| 70-30, with PCA | 55.1% | 0.82 |
| 80-20, with PCA | 54.6% | 0.48 |

PCA made it about 4x faster for a 2-3 point accuracy hit. The bigger lesson is that k-NN is a poor fit for high-dimensional sparse TF-IDF data, the accuracy barely clears baseline. If I were doing this for real I'd reach for Naïve Bayes, logistic regression, or a transformer. I'm leaving it as k-NN because the point of the project was the pipeline and the trade-off analysis, not chasing a high score.

## Tools

Python, scikit-learn (TfidfVectorizer, KNeighborsClassifier, PCA), NLTK, matplotlib, seaborn, pandas. Jupyter notebook.

## Running it

```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn
python -c "import nltk; nltk.download('stopwords')"
jupyter notebook text_classification_pipeline.ipynb
```
