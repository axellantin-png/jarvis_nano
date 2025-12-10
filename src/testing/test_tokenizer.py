import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer_jarvis.model")

sample = "Salut Jarvis, ajoute courir à ma todo demain matin."

tokens = sp.encode(sample, out_type=str)
ids = sp.encode(sample, out_type=int)

print("Tokens :", tokens)
print("IDs    :", ids)

decoded = sp.decode(ids)
print("Decoded:", decoded)

