import torch
from transformers import T5EncoderModel
from PIL.Image import Image
from diffusers import FluxPipeline
from pipelines.models import TextToImageRequest
from torch import Generator


class Pipeline:
    text_encoder: FluxPipeline
    pipeline: FluxPipeline

    def __init__(self, text_encoder: FluxPipeline, pipeline: FluxPipeline):
        self.text_encoder = text_encoder
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        t5_encoder = T5EncoderModel.from_pretrained(
            *args, subfolder="text_encoer_2", **kwargs
        )
        text_encoder = FluxPipeline.from_pretrained(
            *args, text_encoder_2=t5_encoder, transformer=None, vae=None
        )
        pipeline = FluxPipeline.from_pretrained(
            *args, text_encoder=None, text_encoder_2=None, **kwargs
        )

        pipeline.enable_model_cpu_offload()

        return cls(text_encoder, pipeline)


def load_pipeline() -> Pipeline:
    pipeline = Pipeline.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    infer(TextToImageRequest(prompt="Hello World"), pipeline)

    return pipeline


@torch.inference_mode()
def infer(request: TextToImageRequest, pipeline: Pipeline) -> Image:
    if request.seed is None:
        generator = None
    else:
        generator = Generator("cuda").manual_seed(request.seed)

    pipeline.text_encoder.to("cuda")

    (
        prompt_embeds,
        pooled_prompt_embeds,
        _,
    ) = pipeline.text_encoder.encode_prompt(
        prompt=request.prompt, prompt_2="", max_sequence_length=256
    )

    pipeline.text_encoder.to("cpu")

    pipeline.pipeline.to("cuda")

    image = pipeline.pipeline(
        prompt_embeds=prompt_embeds.bfloat16(),
        pooled_prompt_embeds=pooled_prompt_embeds.bfloat16(),
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=4,
    ).images[0]

    pipeline.pipeline.to("cpu")

    return image
