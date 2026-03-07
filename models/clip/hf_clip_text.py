import torch
from transformers import CLIPTokenizer, CLIPTextModel
from models.clip.text_encoder import CLIPTextEncoder

def main():
    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load reference tokenizer + model
    tok = CLIPTokenizer.from_pretrained(model_name)
    hf_model = CLIPTextModel.from_pretrained(model_name).to(device)
    hf_model.eval()
    print(hf_model.config)

    texts = ["a photo of a corgi in a spaceship"]
    tokens = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Run HuggingFace reference
    with torch.no_grad():
        ref = hf_model(**tokens).last_hidden_state  # [B, 77, D]

    # Run my model
wrapper = CLIPTextEncoder.from_pretrained_hf(model_name, max_length=77).to(device)
    wrapper.eval()

    with torch.no_grad():
        mine = wrapper.encode(texts)

    # Compare
    diff = (ref - mine).abs()
    print("ref shape:", ref.shape)
    print("mine shape:", mine.shape)
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())

if __name__ == "__main__":
    main()