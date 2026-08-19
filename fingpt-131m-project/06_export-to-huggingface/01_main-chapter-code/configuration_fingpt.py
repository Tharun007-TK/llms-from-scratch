from transformers import PretrainedConfig


class FinGPTConfig(PretrainedConfig):
    model_type = "fingpt"

    def __init__(
        self,
        vocab_size=50257,
        context_length=1024,
        d_model=768,
        n_layers=13,
        n_heads=12,
        dropout=0.2,
        qkv_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv_bias = qkv_bias

        self.max_position_embeddings = context_length
        self.hidden_size = d_model
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads