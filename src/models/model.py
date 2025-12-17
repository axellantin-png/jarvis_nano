##############################################################################################################################################################################################
# les modules, classes et fonctions importées 
##############################################################################################################################################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# c'est la classe de l'attention (https://www.youtube.com/watch?v=eMlx5fFNoYc/attention) ca permet
# aux mots de demander aux mots precedent si ils lui ajoutent du sens (grace a l'opération dot entre les matrices 
# key "k" et les matrices query "q")
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq, d_model)
        B, T, C = x.size()

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split into heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention scores
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        #process de masking qui permet de aux mots precedent de ne pas avoir les infos des mots suivants (plus efficaces de le faire comme ça)
        if mask is not None:
            # mask: (T, T) ou (1, 1, T, T)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v  # (B, H, T, D)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(attn_out)  # (B, T, C)
        return out

# c'est le MLP (multi layer perceptron) qui perment d'ajouter du contexte a certains vecteur. c'est la culture général
# du modele (de ce que j'ai compris), c'est un 1 layer neural network avec GELU comme activation
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# un block est alors composé des deux éléments précédents 
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # Attention + residual
        x = x + self.attn(self.ln1(x), mask=mask)
        # FFN + residual
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=1024,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Causal mask (buffers pour pas les recalculer sans cesse)
        # mask: (1, 1, T, T)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(self, input_ids, labels=None):
        """
        input_ids: (B, T)
        labels: (B, T) ou None
        """
        B, T = input_ids.size()
        if T > self.max_seq_len:
            raise ValueError(f"seq len {T} > max_seq_len {self.max_seq_len}")

        # Embeddings
        tok_emb = self.tok_emb(input_ids)  # (B, T, C)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos_ids)  # (1, T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        # Causal mask tronquée à T
        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T)

        # Blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Cross-entropy sur tous les tokens
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Génération auto-régressive simple.
        input_ids: (B, T)
        """
        for _ in range(max_new_tokens):
            B, T = input_ids.size()
            if T > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
                T = self.max_seq_len

            logits, _ = self.forward(input_ids)  # (B, T, vocab)
            logits = logits[:, -1, :]  # dernier token
            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
    
    

