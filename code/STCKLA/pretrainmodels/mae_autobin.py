
import torch
from torch import nn


def exists(val):
    return val is not None


class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id=None, pad_token_id=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)

        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)

        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def forward(self, x, output_weight=0):
        # x: [B, N, 1] float, but may contain special ids exactly equal to mask_token_id / pad_token_id
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x_mask_idx = (x == self.mask_token_id).nonzero()
        x_pad_idx = (x == self.pad_token_id).nonzero()

        x = self.mlp(x)
        x = self.LeakyReLU(x)
        x_crosslayer = self.mlp2(x)
        x = self.bin_alpha * x + x_crosslayer
        weight = self.Softmax(x)

        bin_num_idx = self.bin_num_idx.to(x.device)
        token_emb = self.emb(bin_num_idx)
        x = torch.matmul(weight, token_emb)

        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        if x_mask_idx.numel() > 0:
            x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb.repeat(x_mask_idx.shape[0], 1)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        if x_pad_idx.numel() > 0:
            x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb.repeat(x_pad_idx.shape[0], 1)

        if output_weight:
            return x, weight
        return x


class MaeAutobin(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        embed_dim,
        decoder_embed_dim,
        tie_embed=False,
        bin_alpha=1.0,
        bin_num=10,
        pad_token_id=None,
        mask_token_id=None,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        self.token_emb = AutoDiscretizationEmbedding2(
            embed_dim,
            max_seq_len,
            bin_num=bin_num,
            bin_alpha=bin_alpha,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id
        )
        self.pos_emb = nn.Embedding(max_seq_len + 1, embed_dim)

        # injected externally by select_model
        self.encoder = None
        self.decoder = None

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

    @staticmethod
    def _safe_write_visible_tokens_to_decoder(
        *,
        encoder_out: torch.Tensor,      # [B, Lenc, D]
        encoder_pad: torch.Tensor,      # [B, Lenc] True=pad
        visible_labels: torch.Tensor,   # [B, Ldec] True=visible positions in full seq
        decoder_emb: torch.Tensor,      # [B, Ldec, D] embedded decoder tokens
    ):
        """
        MAE 正确语义：
        - encoder_out 是 gather 后的可见 token 输出（去掉 pad）
        - visible_labels 指定 full sequence 中哪些位置是可见 token
        - 将 encoder token 按顺序写回这些可见位置
        """
        B = encoder_out.shape[0]
        enc_nonpad = (~encoder_pad).bool()

        for i in range(B):
            x_valid = encoder_out[i][enc_nonpad[i]]  # [K, D]
            vis_idx = torch.nonzero(visible_labels[i], as_tuple=False).squeeze(1)  # [M]

            K = int(x_valid.shape[0])
            M = int(vis_idx.numel())
            L = min(K, M)

            if L > 0:
                decoder_emb[i, vis_idx[:L]] = x_valid[:L].to(decoder_emb.dtype)

            if M > K:
                decoder_emb[i, vis_idx[K:]] = 0

        return decoder_emb

    def _call_decoder(self, decoder_x: torch.Tensor, decoder_pad_mask: torch.Tensor):
        # Transformer-style
        try:
            return self.decoder(decoder_x, padding_mask=decoder_pad_mask)
        except TypeError:
            pass
        # Performer-like: mask=True means keep
        try:
            return self.decoder(decoder_x, mask=(~decoder_pad_mask))
        except TypeError:
            return self.decoder(decoder_x)

    def forward(
        self,
        x,
        padding_label,
        encoder_position_gene_ids,
        encoder_labels,                 # [B, Ldec] True=visible positions（由 load.py 提供）
        decoder_data,
        mask_gene_name,
        mask_labels,
        decoder_position_gene_ids,
        decoder_data_padding_labels,    # [B, Ldec] True=pad（通常全 False）
        output_attentions=True,
        **kwargs
    ):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f"sequence length {n} must be <= max_seq_len {self.max_seq_len}"

        # ---- Encoder ----
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight=0)
        if output_attentions:
            x.requires_grad_()

        x = x + self.pos_emb(encoder_position_gene_ids)
        x = self.encoder(x, padding_mask=padding_label)  # [B, Lenc, D]

        # ---- Decoder embedding ----
        if decoder_data_padding_labels is None:
            decoder_pad_mask = torch.zeros_like(decoder_position_gene_ids, dtype=torch.bool)
        else:
            decoder_pad_mask = decoder_data_padding_labels.bool()

        dec = self.token_emb(torch.unsqueeze(decoder_data, 2))  # [B, Ldec, D]
        dec = dec + self.pos_emb(decoder_position_gene_ids)

        if mask_gene_name:
            raise NotImplementedError("mask_gene_name not supported in this modified training flow.")

        # ---- MAE writeback: put visible encoder tokens into visible positions ----
        dec = self._safe_write_visible_tokens_to_decoder(
            encoder_out=x,
            encoder_pad=padding_label,
            visible_labels=encoder_labels.bool(),
            decoder_emb=dec
        )

        dec = self.decoder_embed(dec)
        dec = self._call_decoder(dec, decoder_pad_mask)

        dec = self.norm(dec)
        if exists(self.to_final):
            dec = self.to_final(dec)
            return dec.squeeze(2)

        return dec
