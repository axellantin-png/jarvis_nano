### version sentenpiece (google, historique, global) ###


import sentencepiece as spm

input_file = "data/corpus_fr_fin.txt"

spm.SentencePieceTrainer.Train(
    input=input_file,
    model_prefix="tokenizer_jarvis",
    vocab_size=10000,
    character_coverage=0.9995,  # bon pour le FR
    model_type="bpe"
)

### version avec tokenizer plus récente ###

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import os

CORPUS_FILE = "data/corpus_fr_fin.txt"
OUT_DIR = "tokenizer_hf"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) modèle BPE
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 2) normalisation + découpe basique sur espaces
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),
])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3) trainer BPE
special_tokens = ["<unk>", "<pad>", "<s>", "</s>", "<|user|>", "<|assistant|>"]
trainer = trainers.BpeTrainer(
    vocab_size=10000,
    min_frequency=2,
    special_tokens=special_tokens,
)

# 4) entraînement sur ton corpus
tokenizer.train([CORPUS_FILE], trainer=trainer)

# 5) sauvegarde en tokenizer.json
tokenizer_path = os.path.join(OUT_DIR, "tokenizer.json")
tokenizer.save(tokenizer_path)

print("Tokenizer HF entraîné et sauvegardé dans", tokenizer_path)


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_hf/tokenizer.json",
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>"
)

tokenizer.save_pretrained("tokenizer_hf")
print("Wrapper HF sauvegardé dans tokenizer_hf/")

