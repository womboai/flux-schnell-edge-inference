import gc
import os
from typing import TypeAlias

import torch
from PIL.Image import Image
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from huggingface_hub.constants import HF_HUB_CACHE
from pipelines.models import TextToImageRequest
from torch import Generator
from transformers import T5EncoderModel, CLIPTextModel

Pipeline: TypeAlias = FluxPipeline

CHECKPOINT = "black-forest-labs/FLUX.1-schnell"
REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"


def load_pipeline() -> Pipeline:
    text_encoder = CLIPTextModel.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
        subfolder="text_encoder",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )

    text_encoder_2 = T5EncoderModel.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
        subfolder="text_encoder_2",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )

    vae = AutoencoderKL.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
        subfolder="vae",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )

    path = os.path.join(HF_HUB_CACHE, "models--RobertML--FLUX.1-schnell-int8wo/snapshots/307e0777d92df966a3c0f99f31a6ee8957a9857a")

    transformer = FluxTransformer2DModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        use_safetensors=False,
    )

    pipeline = FluxPipeline.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
        local_files_only=True,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipeline("")

    return pipeline

def infer(request: TextToImageRequest, pipeline: Pipeline) -> Image:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

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
