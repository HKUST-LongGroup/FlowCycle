import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL
from PIL import Image
import math
from typing import Union, Tuple, Optional
from torchvision import transforms as tvt
import argparse
import sys
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, LinearLR, CosineAnnealingLR
import json
import random
import numpy as np
import os



@torch.no_grad()
def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x.to(device=vae.device, dtype=vae.dtype) - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * vae.config.scaling_factor
    return latents

class Noises(nn.Module):
    def __init__(self, dtype, img_size):
        super(Noises, self).__init__()
        self.noises = nn.Parameter(torch.randn(2, 16, img_size // 8, img_size // 8, dtype=dtype))
    def forward(self, i):
        return self.noises[i].unsqueeze(0)
    
def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
 
@torch.no_grad()
def get_text_embeddings(prompt_a, prompt_b, source_guidance_scale, target_guidance_scale, pipe):
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt_a,
        prompt_2=prompt_a,
        prompt_3=prompt_a,
        negative_prompt="",
        negative_prompt_2="",
        negative_prompt_3="",
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=source_guidance_scale > 1.0,
    )
    if source_guidance_scale > 1.0:
        cond_embeds_a = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_projections_a = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        cond_embeds_a = prompt_embeds
        pooled_projections_a = pooled_prompt_embeds

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt_b,
        prompt_2=prompt_b,
        prompt_3=prompt_b,
        negative_prompt="",
        negative_prompt_2="",
        negative_prompt_3="",
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=target_guidance_scale > 1.0,
    )
    if target_guidance_scale > 1.0:
        cond_embeds_b = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_projections_b = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        cond_embeds_b = prompt_embeds
        pooled_projections_b = pooled_prompt_embeds

    return cond_embeds_a, pooled_projections_a, cond_embeds_b, pooled_projections_b   
    
    
    
parser = argparse.ArgumentParser(description='FlowCycle')
parser.add_argument('--model_id',
                    type=str,
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
parser.add_argument('--seed',
                    type=int,
                    default=2025)
parser.add_argument('--num_inference_steps',
                    type=int,
                    default=50)
parser.add_argument('--n_max',
                    type=int,
                    default=33)
parser.add_argument('--source_guidance_scale',
                    type=float,
                    default=3.5)
parser.add_argument('--target_guidance_scale',
                    type=float,
                    default=5.5)
parser.add_argument('--lr',
                    type=float,
                    default=0.1)
parser.add_argument('--epochs',
                    type=int,
                    default=100)
parser.add_argument('--img_size',
                    type=int,
                    default=512)
parser.add_argument('--alpha',
                    type=float,
                    default=0.2)
parser.add_argument('--device',
                    type=str,
                    default="cuda:0")



def main():    
    
    
    # Hyperparameters
    args = parser.parse_args()
    model_id = args.model_id
    seed = args.seed
    num_inference_steps = args.num_inference_steps
    n_max = args.n_max
    source_guidance_scale = args.source_guidance_scale
    target_guidance_scale = args.target_guidance_scale
    lr = args.lr
    epochs = args.epochs
    alpha = args.alpha
    dtype = torch.bfloat16
    img_size = args.img_size
    device = args.device
    warmup_epochs = int(epochs * 0.1)
    
    # Set random seed
    set_seed(seed)

    # Initialize models
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps[num_inference_steps-n_max:]
    sigma_max_t = pipe.scheduler.sigmas[pipe.scheduler.timesteps.tolist().index(timesteps[0])]
    
    # Load image and prompts
    input_img = load_image("source.png", img_size).to(device=device, dtype=dtype)
    source_promot = "A blue-gray Audi car parked in a grassy area. A white dog sitting on the grass, next to the car. A cat laying on the hood of the car."
    target_prompt = "A blue-gray Audi car parked in a grassy area. A Husky dog sitting on the grass, next to the car. A tiger cab laying on the hood of the car."
    a = img_to_latents(input_img, pipe.vae) 
    
    # Prepare the conditions
    cond_embeds_a, pooled_projections_a, cond_embeds_b, pooled_projections_b = get_text_embeddings(source_promot, target_prompt, source_guidance_scale, target_guidance_scale, pipe)
    
    # Training settings
    NS = Noises(dtype, img_size).to(device)
    optimizer = optim.Adam([{'params': NS.parameters(), 'lr': lr}])
    scheduler_warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=0.12, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-8)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    
    # Training stage
    for epoch in range(epochs+1):
        optimizer.zero_grad()
        epsilon_1 = NS(0)
        epsilon_2 = NS(1)
        latents = a * (1 - sigma_max_t) + sigma_max_t * epsilon_1
        tmp_a = latents
        for i, t in enumerate(timesteps):
            t_idx = pipe.scheduler.timesteps.tolist().index(t.item())
            sigma_t = pipe.scheduler.sigmas[t_idx]
            sigma_next = pipe.scheduler.sigmas[t_idx + 1]
            dt = sigma_t - sigma_next
                
            latent_model_input = latents
            if target_guidance_scale > 1.0:
                latent_model_input = latents.repeat(2, 1, 1, 1)

            bs = latent_model_input.shape[0]   
            t_batch = t.expand(bs).to(device)  
            
            
            with torch.inference_mode():
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    encoder_hidden_states=cond_embeds_b,
                    pooled_projections=pooled_projections_b,
                    return_dict=True,
                ).sample

            if target_guidance_scale > 1.0:
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + target_guidance_scale * (text - uncond)
                
            latents = latents - dt * noise_pred
        
        latents = latents * (1 - sigma_max_t) + sigma_max_t * epsilon_2
        tmp_b = latents
        for i, t in enumerate(timesteps):
            t_idx = pipe.scheduler.timesteps.tolist().index(t.item())
            sigma_t = pipe.scheduler.sigmas[t_idx]
            sigma_next = pipe.scheduler.sigmas[t_idx + 1]
            dt = sigma_t - sigma_next
                
            latent_model_input = latents
            if source_guidance_scale > 1.0:
                latent_model_input = latents.repeat(2, 1, 1, 1)

            bs = latent_model_input.shape[0]   
            t_batch = t.expand(bs).to(device)  
            
            
            with torch.inference_mode():
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    encoder_hidden_states=cond_embeds_a,
                    pooled_projections=pooled_projections_a,
                    return_dict=True,
                ).sample

            if source_guidance_scale > 1.0:
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + source_guidance_scale * (text - uncond)
                
            latents = latents - dt * noise_pred

        loss1 = F.mse_loss(a, latents, reduction='none').mean(dim=(1,2,3))
        loss2 = F.mse_loss(tmp_a, tmp_b, reduction='none').mean(dim=(1,2,3))
        loss = loss1 + loss2 * alpha
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Inference stage
    with torch.no_grad():
        latents = a * (1 - sigma_max_t) + sigma_max_t * NS(0)
        for i, t in enumerate(timesteps):
            t_idx = pipe.scheduler.timesteps.tolist().index(t.item())
            sigma_t = pipe.scheduler.sigmas[t_idx]
            sigma_next = pipe.scheduler.sigmas[t_idx + 1]
            dt = sigma_t - sigma_next
                
            latent_model_input = latents
            if target_guidance_scale > 1.0:
                latent_model_input = latents.repeat(2, 1, 1, 1)

            bs = latent_model_input.shape[0]   
            t_batch = t.expand(bs).to(device)  
            
            
            with torch.inference_mode():
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    encoder_hidden_states=cond_embeds_b,
                    pooled_projections=pooled_projections_b,
                    return_dict=True,
                ).sample

            if target_guidance_scale > 1.0:
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + target_guidance_scale * (text - uncond)
                
            latents = latents - dt * noise_pred
            
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]   # [-1, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu()
        image = image.permute(0, 2, 3, 1).float().numpy()  # NHWC
        pil_image = Image.fromarray((image[0] * 255).round().astype("uint8"))
        pil_image.save("edited_result.png")

        
        
if __name__ == "__main__":
    main()
