import gc
from typing import TypeAlias

import torch
from PIL.Image import Image
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from pipelines.models import TextToImageRequest
from torch import Generator
from transformers import T5EncoderModel, CLIPTextModel

Pipeline: TypeAlias = FluxPipeline

CHECKPOINT = "black-forest-labs/FLUX.1-schnell"

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def load_pipeline() -> Pipeline:
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

    vae = AutoencoderKL.from_pretrained(
        CHECKPOINT,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        "RobertML/FLUX.1-schnell-int8wo",
        torch_dtype=torch.bfloat16,
        use_safetensors=False,
    )

    pipeline = FluxPipeline.from_pretrained(
        CHECKPOINT,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    infer(TextToImageRequest(prompt=""), Pipeline)

    return pipeline

def infer(request: TextToImageRequest, pipeline: Pipeline) -> Image:
    empty_cache()

    generator = Generator(pipeline.device).manual_seed(request.seed)

    return pipeline(
        request.prompt,
        generator=generator,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        height=request.height,
        width=request.width,
    ).images[0]
