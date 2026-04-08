import torch
import torch.nn as nn

class ReGLU(nn.Module):
    def forward(self, x):
        x_linear, x_gated = x.chunk(2, dim=-1)
        return x_linear * torch.relu(x_gated)

class FeatureTokenizer(nn.Module):
    def __init__(self, n_num, cat_dims, d_emb):
        super().__init__()

        # Tokenizer pour les caractéristiques numériques: x_j * W_j + b_j
        self.num_weights = nn.Parameter(torch.Tensor(n_num, d_emb))
        self.num_biases = nn.Parameter(torch.Tensor(n_num, d_emb))
        nn.init.xavier_uniform_(self.num_weights)
        nn.init.zeros_(self.num_biases)

        # Tokenizer pour les caractéristiques catégorielles (Tables de lookup + Biases)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(dim, d_emb) for dim in cat_dims])
        self.cat_biases = nn.ParameterList([nn.Parameter(torch.Tensor(d_emb)) for _ in cat_dims])
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for bias in self.cat_biases:
            nn.init.zeros_(bias)

        # Token CLS
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_emb))
        nn.init.normal_(self.cls_token)

    def forward(self, x_num, x_cat):
        # Tokens Numériques: (Batch, N_NUM, D_EMB)
        x_num = x_num.unsqueeze(2) 
        num_tokens = x_num * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0)

        # Tokens Catégoriels: (Batch, N_CAT, D_EMB)
        cat_tokens = []
        for i, (emb, bias) in enumerate(zip(self.cat_embeddings, self.cat_biases)):
            token = emb(x_cat[:, i]) + bias
            cat_tokens.append(token.unsqueeze(1))

        # Concaténation des tokens: (Batch, N_NUM + N_CAT, D_EMB)
        feature_tokens = torch.cat([num_tokens] + cat_tokens, dim=1)
        
        # Ajout du token CLS: (Batch, N_FEATURES + 1, D_EMB)
        cls_token = self.cls_token.expand(feature_tokens.shape[0], -1, -1)
        return torch.cat([cls_token, feature_tokens], dim=1)

class TransformerBlock(nn.Module):
    def __init__(self, d_emb, n_heads, ffn_factor, attn_dropout, ffn_dropout, resid_dropout):
        super().__init__()

        # Multi-Head Self-Attention (MHSA)
        self.norm_attn = nn.LayerNorm(d_emb)
        self.attn = nn.MultiheadAttention(d_emb, n_heads, dropout=attn_dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(resid_dropout)

        # Feed Forward Network (FFN)
        self.norm_ffn = nn.LayerNorm(d_emb)
        d_inner = int(d_emb * ffn_factor)
        self.ffn = nn.Sequential(
            nn.Linear(d_emb, d_inner * 2),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_inner, d_emb),
        )
        self.dropout_ffn = nn.Dropout(resid_dropout)

    def forward(self, x):
        # MHSA avec PreNorm (Résiduel 1)
        norm_x = self.norm_attn(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.dropout_attn(attn_output)

        # FFN avec PreNorm (Résiduel 2)
        norm_x = self.norm_ffn(x)
        ffn_output = self.ffn(norm_x)
        x = x + self.dropout_ffn(ffn_output)
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_dims, d_emb, n_layers, n_heads, ffn_factor, attn_dropout, ffn_dropout, resid_dropout):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(n_num, cat_dims, d_emb)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_emb, n_heads, ffn_factor, attn_dropout, ffn_dropout, resid_dropout)
            for _ in range(n_layers)
        ])
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_emb), 
            nn.Linear(d_emb, 1) 
        )

    def forward(self, x_num, x_cat):
        # 1. Feature Tokenizer
        x = self.feature_tokenizer(x_num, x_cat)

        # 2. Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # 3. Prediction Head (utilise uniquement le CLS token)
        return self.prediction_head(x[:, 0])