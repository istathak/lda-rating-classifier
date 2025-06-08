from sklearn.model_selection import train_test_split
import spacy, gensim
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS

#loading bar
try:
    from tqdm.auto import tqdm          
    tqdm.pandas()                       
except ImportError:                     
    print("tqdm not found → no progress bars")
    class tqdm:                        
        def __init__(self, it=None, **k): self.it = it
        def __iter__(self): return iter(self.it or [])
        def __call__(self, it, **k): return it

# ── spaCy model (download once) 
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# -----------------------------
def lemmatise(texts, allowed_pos=("NOUN", "ADJ", "VERB", "ADV"),
              batch_size=64, n_process=-1):
    """Fast lemmatisation with spaCy.pipe."""
    output = []
    for doc in tqdm(nlp.pipe(texts,
                             batch_size=batch_size,
                             n_process=n_process),
                    total=len(texts), desc="Lemmatise"):
        output.append(" ".join(tok.lemma_
                               for tok in doc
                               if tok.pos_ in allowed_pos))
    return output

def preprocess(texts):
    """Tokenise, lower-case, strip stop-words."""
    cleaned = []
    for text in tqdm(texts, desc="Pre-process"):
        tokens = gensim.utils.simple_preprocess(text, deacc=True)
        cleaned.append([t for t in tokens if t not in STOPWORDS])
    return cleaned

def load_data(csv_path, text_col="reviewText", label_col="overall"):
    # read
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col])

    # split
    train_df, test_df = train_test_split(df, test_size=0.2,
                                         stratify=df[label_col],
                                         random_state=42)

    # lemmatise
    train_lem = lemmatise(train_df[text_col].tolist())
    test_lem  = lemmatise(test_df[text_col].tolist())

    # tokenise / clean
    train_tok = preprocess(train_lem)
    test_tok  = preprocess(test_lem)

    return (train_tok,
            train_df[label_col].tolist(),
            test_tok,
            test_df[label_col].tolist())

def bagofwords(docs, no_below=5, no_above=0.5):
    """Create BoW corpus + dictionary."""
    id2word = corpora.Dictionary(docs)
    id2word.filter_extremes(no_below=no_below, no_above=no_above)

    corpus = [id2word.doc2bow(doc) for doc in tqdm(docs, desc="BoW")]
    return corpus, id2word
