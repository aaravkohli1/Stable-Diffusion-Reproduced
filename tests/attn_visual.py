"""Attention map test"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import torch
    from models.vae import VAE
    from models.unet import UNet
    from models.generate import SD1_UNET_CONFIG, load_from_pretrained, get_device, generate
    from diffusion import Diffuser, scaled_linear_beta
    from diffusion.sampling import sample_euler_ancestral
    from utils.attn_map import capture_attention, visualize_attention


    PROMPT = 'a dog riding a bike'
    SEED = 42
    NUM_STEPS = 20
    GUIDANCE_SCALE = 7.5

    device = get_device(None)
    print(f'Using device: {device}')
    torch.manual_seed(SEED)

    unet = UNet(**SD1_UNET_CONFIG)
    vae = VAE()
    text_encoder = load_from_pretrained('CompVis/stable-diffusion-v1-4', vae, unet, device)

    unet = unet.to(device).eval()
    vae = vae.to(device).eval()
    text_encoder = text_encoder.to(device)

    diffuser = Diffuser(1000, scaled_linear_beta)


    tokens_out = text_encoder.tokenizer([PROMPT])
    token_ids = tokens_out['input_ids'][0].tolist()
    token_strings = [
        text_encoder.tokenizer.tok.decode([tid]).strip()
        for tid in token_ids
    ]

    content_indices = []
    content_labels = []
    for i, (tid, label) in enumerate(zip(token_ids, token_strings)):
        if tid in (49406, 49407):
            continue
        if not label:
            continue
        content_indices.append(i)
        content_labels.append(label)

    print(f'\nPrompt: \'{PROMPT}\'')
    print(f'Tokens: {content_labels}')
    print(f'Indices: {content_indices}')

    print(f'\nGenerating with {NUM_STEPS} steps...')
    with capture_attention(unet) as store:
        image = generate(
            prompt=PROMPT,
            text_encoder=text_encoder,
            unet=unet,
            vae=vae,
            diffuser=diffuser,
            device=device,
            sampler='dpm_pp_2m',
            num_steps=NUM_STEPS,
            height=512,
            width=512,
            guidance_scale=GUIDANCE_SCALE,
            use_karras=True,
        )

    image.save('attn_demo_image.png')
    print(f'Saved generated image to attn_demo_image.png')

    visualize_attention(
        store,
        tokens=content_labels,
        image=image,
        token_indices=content_indices,
        timestep=-1,
        save_path='attn_demo.png',
    )
    print(f'Saved attention visualization to attn_demo.png')
