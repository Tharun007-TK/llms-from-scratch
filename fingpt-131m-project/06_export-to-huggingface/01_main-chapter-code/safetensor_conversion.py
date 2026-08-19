from safetensors.torch import save_model

model="model/best.pt"

save_model(model, "model.safetensors")