import collections
import requests
import pickle
import logging
from tokensier import tokenise

UNK_TOKEN = "<UNK>"

logging.basicConfig(level=logging.INFO)

logging.info("Downloading text8 dataset...")
r = requests.get('https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('text8', 'wb') as f: f.write(r.content)
with open('text8') as f: text8: str = f.read()

logging.info("Tokenizing corpus...")
corpus: list[str] = tokenise(text8)

# once saved, check content with: head -c 100 corpus.json
logging.info("Saving tokenized corpus to pickle...")
with open('data/processed/corpus.pkl', 'wb') as f: pickle.dump(corpus, f)

def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  logging.info("Creating lookup tables...")
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab

words_to_ids, ids_to_words = create_lookup_tables(corpus)

logging.info("Saving lookup tables to pickle...")
with open('data/tokenised/tkn_words_to_ids.pkl', 'wb') as f: pickle.dump(words_to_ids, f)
with open('data/tokenised/tkn_ids_to_words.pkl', 'wb') as f: pickle.dump(ids_to_words, f)
