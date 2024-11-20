from diffusers import FluxPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel
import torch
import gc
from PIL.Image import Image
from pipelines.models import TextToImageRequest
from torch import Generator

Pipeline = None

CHECKPOINT = "black-forest-labs/FLUX.1-schnell"

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def load_pipeline() -> Pipeline:
    infer(TextToImageRequest(prompt=""), Pipeline)

    return Pipeline


def encode_prompt(prompt: str):
    text_encoder = CLIPTextModel.from_pretrained(
        CHECKPOINT,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )

    text_encoder_2 = T5EncoderModel.from_pretrained(
        CHECKPOINT,
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = CLIPTokenizer.from_pretrained(CHECKPOINT, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(CHECKPOINT, subfolder="tokenizer_2")

    pipeline = FluxPipeline.from_pretrained(
        CHECKPOINT,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=None,
        vae=None,
    ).to("cuda")

    with torch.no_grad():
        return pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            max_sequence_length=256,
        )


def infer_latents(prompt_embeds, pooled_prompt_embeds, width: int | None, height: int | None, seed: int | None):
    pipeline = FluxPipeline.from_pretrained(
        CHECKPOINT,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        vae=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    if seed is None:
        generator = None
    else:
        generator = Generator(pipeline.device).manual_seed(seed)

    return pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=width,
        height=height,
        generator=generator,
        output_type="latent",
    ).images


def infer(request: TextToImageRequest, _pipeline: Pipeline) -> Image:
    empty_cache()

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(request.prompt)

    empty_cache()

    latents = infer_latents(prompt_embeds, pooled_prompt_embeds, request.width, request.height, request.seed)

    empty_cache()

    vae = AutoencoderKL.from_pretrained(
        CHECKPOINT,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    height = request.height or 64 * vae_scale_factor
    width = request.width or 64 * vae_scale_factor

    with torch.no_grad():
        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        image = vae.decode(latents, return_dict=False)[0]
        return image_processor.postprocess(image, output_type="pil")[0]
