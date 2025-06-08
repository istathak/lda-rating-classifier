
import os
from typing import List, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict


from gensim.models.ldamulticore import LdaMulticore


def corpus_to_dense(
    lda_model: LdaMulticore,
    corpus: List[List[Tuple[int, int]]],
    k: Optional[int] = None
):
    """Return a dense *NK* list of topic-probability vectors."""
    if k is None:
        k = lda_model.num_topics

    dense_vectors: List[List[float]] = []
    for bow in corpus :
        vec = [0.0] * k
        for t_id, prob in lda_model.get_document_topics(bow, minimum_probability=0.0):
            vec[t_id] = prob
        dense_vectors.append(vec)
    return dense_vectors


def train_lda(
    corpus: List[List[Tuple[int, int]]],
    dictionary,
    k: int,
    *,
    passes: int = 3,
    iterations: int = 25,
    chunksize: int = 4000,
    workers: Optional[int] = None,
    random_state: int = 42,
):
    """Train an LDA model (multi-core) and optionally get dense features."""
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    print(f"Training LDA: K={k}, passes={passes}, iterations={iterations}, workers={workers}")

    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=passes,
        iterations=iterations,
        chunksize=chunksize,
        workers=workers,
        random_state=random_state,
        eval_every=0, 
    )

    print("LDA training complete")

    return lda

def train_logistic_regression(
    X: np.ndarray | List[List[float]],
    y: np.ndarray | List[int] | List[str],
    *,
    balance: bool = True,
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
    balance_dataset = False, 
    class_weight = False
) -> LogisticRegression:
    """Fit a multinomial logistic regression, optionally with equal-N undersampling."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if balance_dataset:
        X, y = _undersample_equal_N(X, y, random_state=random_state)

    if class_weight:
        clf = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
            multi_class="multinomial",
            random_state=random_state,
            class_weight = 'balanced'
        )
    else:
        clf = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
            multi_class="multinomial",
            random_state=random_state)

    clf.fit(X, y)
    return clf

def test_logistic_regression(
    model: LogisticRegression,
    X: np.ndarray | List[List[float]],
    y_true: np.ndarray | List[int] | List[str],
    *,
    digits: int = 3,
    verbose: bool = True,
) -> tuple[float, str]:
    """Return (accuracy, classification_report). Optionally print them."""
    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y_true)

    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=digits)

    if verbose:
        print("Accuracy:", acc)
        print(report)

    return acc, report

def _undersample_equal_N(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X, y where each class has *min_class_size* samples (without replacement)."""
    rng = np.random.default_rng(random_state)
    indices_by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(y):
        indices_by_class[label].append(idx)

    min_size = min(len(lst) for lst in indices_by_class.values())
    balanced_idx = np.concatenate([
        rng.choice(lst, size=min_size, replace=False) for lst in indices_by_class.values()
    ])

    return X[balanced_idx], y[balanced_idx]