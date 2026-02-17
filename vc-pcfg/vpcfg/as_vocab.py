import json
import pickle
import os
import sys
from pathlib import Path
from .utils import Vocabulary

def add_words(ifile, vocab):
    with ifile.open("r") as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.split("\t")[2]
            line = line.strip().lower().split(" ")
            for word in line:
              if word in vocab:
                vocab[word] +=1
              else:
                vocab[word] = 1
    return vocab

def write_word_list(ofile, vocab):
    with ofile.open("w") as fw:
        json.dump(vocab, fw)

def get_complete_word_list(preprocessed_data_path):
    preprocessed_data_path = Path(preprocessed_data_path)
    sentence_files = ["SimpleSentences1_clean.txt", "SimpleSentences2_clean.txt"]
    ofile = preprocessed_data_path / "complete_word_list_counts.json"
    vocab = dict()

    for ifile in sentence_files:
        vocab = add_words(preprocessed_data_path / ifile, vocab)
    write_word_list(ofile, vocab)


def get_complete_word_list_from_caps(preprocessed_data_path, caps_file="all_caps.text"):
    preprocessed_data_path = Path(preprocessed_data_path)
    ofile = preprocessed_data_path / "complete_word_list_counts.json"
    vocab = dict()
    caps_path = preprocessed_data_path / caps_file
    with caps_path.open("r") as fr:
        for line in fr:
            line = line.strip().lower()
            if not line:
                continue
            for word in line.split(" "):
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    write_word_list(ofile, vocab)


def create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size = 2000):
    vocab = Vocabulary()
    word_list_file = word_list_dir / word_list_file
    with word_list_file.open("r") as f:
        word_list = json.load(f)
    sorted_words = sorted(word_list.items(), key=lambda x:x[1], reverse=True)
    print(sorted_words[0:10])
    id = 0
    for word,count in sorted_words:
        if id >= vocab_size:
            break
        vocab.add_word(word)
        id+=1
    vocab_file = word_list_dir / vocab_file
    with vocab_file.open("wb") as fw:
        pickle.dump(vocab, fw)
    return vocab

def get_vocab(preprocessed_dir, vocab_size = 2000):
    word_list_dir = Path(preprocessed_dir)
    word_list_file = 'complete_word_list_counts.json'
    vocab_file = 'vocab_dict.pkl'
    if vocab_file in os.listdir(word_list_dir):
        vocab_file = word_list_dir / vocab_file
        with vocab_file.open("rb") as f:
            try:
                vocab = pickle.load(f)
            except ModuleNotFoundError as exc:
                # Compatibility: older pickles may reference module name "utils"
                if exc.name == "utils":
                    vpcfg_dir = Path(__file__).resolve().parent
                    if str(vpcfg_dir) not in sys.path:
                        sys.path.insert(0, str(vpcfg_dir))
                    import utils as utils_module  # type: ignore
                    sys.modules["utils"] = utils_module
                    f.seek(0)
                    vocab = pickle.load(f)
                else:
                    raise
    elif word_list_file in os.listdir(word_list_dir):
        vocab = create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size)
    else:
        if (word_list_dir / "SimpleSentences1_clean.txt").exists():
            get_complete_word_list(preprocessed_dir)
        elif (word_list_dir / "all_caps.text").exists():
            get_complete_word_list_from_caps(preprocessed_dir)
        else:
            raise FileNotFoundError(
                f"Missing word list sources in {word_list_dir}. "
                "Expected complete_word_list_counts.json or all_caps.text."
            )
        vocab = create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size)
    return vocab
