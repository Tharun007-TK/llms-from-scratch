import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from configuration_fingpt import FinGPTConfig


class FinGPTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=config.qkv_bias,
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
        )

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.split(
            self.d_model,
            dim=2,
        )

        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class FinGPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(
            config.d_model,
            4 * config.d_model,
        )

        self.fc2 = nn.Linear(
            4 * config.d_model,
            config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class FinGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = FinGPTAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = FinGPTMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class FinGPTPreTrainedModel(PreTrainedModel):
    config_class = FinGPTConfig
    base_model_prefix = "fingpt"


class FinGPTForCausalLM(FinGPTPreTrainedModel, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)

        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.pos_emb = nn.Embedding(
            config.context_length,
            config.d_model,
        )

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            FinGPTBlock(config)
            for _ in range(config.n_layers)
        ])

        self.ln_final = nn.LayerNorm(
            config.d_model
        )

        self.head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Weight tying is handled by post_init() using get_input_embeddings and get_output_embeddings
        # self.head.weight = self.token_emb.weight

        self.post_init()

    def forward(
        self,
        input_ids=None,
        labels=None,
        **kwargs,
    ):

        idx = input_ids

        B, T = idx.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context length {self.config.context_length}"
            )

        pos = torch.arange(
            T,
            device=idx.device,
        )

        x = (
            self.token_emb(idx)
            + self.pos_emb(pos)
        )

        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        logits = self.head(x)

        loss = None

        if labels is not None:

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        **kwargs,
    ):
        return {
            "input_ids": input_ids
        }

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings