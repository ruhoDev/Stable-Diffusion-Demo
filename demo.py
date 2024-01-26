import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

# init_image = load_image(url).convert("RGB")
init_image = load_image('test/9.jpg').convert("RGB")
prompt = "a man wearing glasses and red shirt"
image = pipe(prompt, image=init_image).images
image[0].save('demo.png')
print("Save Image")
