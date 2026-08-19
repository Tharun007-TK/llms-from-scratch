# Export Bug Postmortem: "Trained Fine, Generates Garbage After HF Export"

## Symptom

Model trained correctly — final validation loss **2.7641** at step 11,250,
in-notebook generation looked coherent. After converting to Hugging Face
`safetensors` format and loading with `FinGPTForCausalLM.from_pretrained()`,
the model generated incoherent repetitive output instead of domain-relevant text.

---

## Root Cause

The model uses **weight tying**: the output projection (`head.weight`) shares
the same tensor as the token embedding (`token_emb.weight`). In the training
notebook this is done by a direct Python assignment:

```python
self.head.weight = self.token_emb.weight
```

When converting to a Hugging Face `PreTrainedModel` subclass, `post_init()`
calls `tie_weights()` internally if `config.tie_word_embeddings = True`.
This re-establishes the tie at the Python level. So far so good.

The problem was in `save_pretrained(..., safe_serialization=True)`:
`safetensors` format **deduplicates shared tensors** — it stores the tensor
once and records that two keys point to it. When Hugging Face's serialiser
detected that `head.weight` and `token_emb.weight` were the same object in
memory, it expected the model config to declare `tied_weights_keys` so it
knew which key was the "canonical" one and which was the alias.

Because our custom `FinGPTForCausalLM` didn't implement `get_output_embeddings()`
or set `_tied_weights_keys`, Hugging Face's save logic raised a `RuntimeError`
about shared tensors mismatching the base configuration. Multiple workarounds
were attempted before landing on the correct fix:

---

## Fix

**Break the tie before saving.** Before calling `save_pretrained`:

1. Set `config.tie_word_embeddings = False` **in the config passed to the model
   constructor** (not after the fact) so `post_init()` never ties the weights
   and the two tensors remain independent objects in memory.
2. After loading the state dict (which was trained with tied weights, so
   `token_emb.weight` contains the correct values and `head.weight` does not),
   explicitly copy the values:

```python
hf_model.head.weight.data.copy_(hf_model.token_emb.weight.data)
```

3. Then save normally:

```python
hf_model.save_pretrained("./fingpt-131m", safe_serialization=True)
```

This produces a `model.safetensors` that contains **both** tensors as distinct
entries with identical values. Slightly larger on disk (the embedding matrix is
~200MB), but loads cleanly and generates correctly.

---

## What Did NOT Work

| Attempt | What happened |
|---|---|
| Set `hf_model._tied_weights_keys = ["head.weight"]` before save | Same `RuntimeError` — Hugging Face checks config, not instance attributes |
| Set `hf_model.config.tie_word_embeddings = False` **after** model construction | `post_init()` had already tied the weights; `save_pretrained` still saw shared tensors |
| `hf_model.head.weight.data = hf_model.token_emb.weight.data.clone()` after model construction | The `post_init()` call retied them, overwriting the clone |

---

## Diagnostic Checklist

If you see a trained model generate incoherent output after HF export:

1. **Check whether `head.weight` actually loaded.** Print
   `hf_model.head.weight[:3, :5]` and `hf_model.token_emb.weight[:3, :5]`
   immediately after `from_pretrained` — if they differ, the weight wasn't
   transferred.
2. **Check for weight tying.** If `config.tie_word_embeddings = True` and you
   are using a custom `PreTrainedModel`, verify that `get_input_embeddings()`
   and `get_output_embeddings()` are implemented — without them Hugging Face
   can't register the tie correctly.
3. **Confirm the state dict keys match.** Run `load_state_dict(strict=True)` —
   if it succeeds with no missing or unexpected keys, the architecture is
   correct and the weights loaded.
4. **Test before uploading.** Run a forward pass and check logits shape, then
   call `generate()` locally before touching the Hub. Saves a round-trip.
