import os

import folder_paths
import torch
from comfy import model_management
from safetensors.torch import load_model

from .model import ELLA, T5TextEmbedder

ELLA_EMBEDS_TYPE = "ELLA_EMBEDS"
ELLA_EMBEDS_PREFIX = "ella_"
ELLA_EMBEDS_PREFIX_LEN = len(ELLA_EMBEDS_PREFIX)


class EllaProxyUNet:
    def __init__(self, ella, model_sampling, positive, negative) -> None:
        self.ella = ella
        self.model_sampling = model_sampling
        if positive.keys() != negative.keys():
            raise ValueError("positive and negative embeds types must match")
        self.embeds = [positive, negative]

        self.dtype = model_management.text_encoder_dtype()
        self.ella.to(self.dtype)
        for i in range(len(self.embeds)):
            for k in self.embeds[i]:
                self.embeds[i][k].to(device=self.load_device, dtype=self.dtype)

    @property
    def load_device(self):
        return model_management.text_encoder_device()

    @property
    def offload_device(self):
        return model_management.text_encoder_offload_device()

    def prepare_conds(self):
        self.ella.to(self.load_device)
        cond = self.ella(torch.Tensor([999]).to(torch.int64), **self.embeds[0])
        uncond = self.ella(torch.Tensor([999]).to(torch.int64), **self.embeds[1])
        self.ella.to(self.offload_device)
        return cond, uncond

    def __call__(self, apply_model, kwargs: dict):
        input_x = kwargs["input"]
        timestep_ = kwargs["timestep"]
        c = kwargs["c"]
        cond_or_uncond = kwargs["cond_or_uncond"]  # [0|1]

        time_aware_encoder_hidden_states = []
        self.ella.to(device=self.load_device)
        for i in cond_or_uncond:
            h = self.ella(
                self.model_sampling.timestep(timestep_[i]),
                **self.embeds[i],
            )
            time_aware_encoder_hidden_states.append(h)
        self.ella.to(self.offload_device)

        c["c_crossattn"] = torch.cat(time_aware_encoder_hidden_states, dim=0)

        return apply_model(input_x, timestep_, **c)


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Apply Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class EllaApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ella": ("ELLA",),
                "positive": (ELLA_EMBEDS_TYPE,),
                "negative": (ELLA_EMBEDS_TYPE,),
            }
        }

    RETURN_NAMES = ("model", "positive", "negative")
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "apply"
    CATEGORY = "ella/apply"

    def apply(self, model, ella, positive, negative):
        model_clone = model.clone()
        model_sampling = model_clone.get_model_object("model_sampling")

        ella_proxy = EllaProxyUNet(
            ella=ella,
            model_sampling=model_sampling,
            positive={
                k[ELLA_EMBEDS_PREFIX_LEN:]: v.clone() for k, v in positive.items() if k.startswith(ELLA_EMBEDS_PREFIX)
            },
            negative={
                k[ELLA_EMBEDS_PREFIX_LEN:]: v.clone() for k, v in negative.items() if k.startswith(ELLA_EMBEDS_PREFIX)
            },
        )

        model_clone.set_model_unet_function_wrapper(ella_proxy)
        # No matter how many tokens are text features, the ella output must be 64 tokens.
        _cond, _uncond = ella_proxy.prepare_conds()
        cond = [_cond, {k: v for k, v in positive.items() if not k.startswith(ELLA_EMBEDS_PREFIX)}]
        uncond = [_uncond, {k: v for k, v in negative.items() if not k.startswith(ELLA_EMBEDS_PREFIX)}]

        return (model_clone, [cond], [uncond])


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class T5TextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "text_encoder": ("T5_TEXT_ENCODER",),
            }
        }

    RETURN_TYPES = (ELLA_EMBEDS_TYPE,)
    FUNCTION = "encode"

    CATEGORY = "ella/conditioning"

    def encode(self, text, text_encoder, max_length=None):
        # TODO: more offload strategy
        text_encoder.to(model_management.text_encoder_device())
        cond = text_encoder(text, max_length=max_length)
        text_encoder.to(model_management.text_encoder_offload_device())

        return ({f"{ELLA_EMBEDS_PREFIX}t5_embeds": cond},)


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class ELLALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": (folder_paths.get_filename_list("ella"),),
            }
        }

    RETURN_TYPES = ("ELLA",)
    FUNCTION = "load"
    CATEGORY = "ella/loaders"

    def load(self, name: str, **kwargs):
        ella_file = folder_paths.get_full_path("ella", name)
        # TODO: expose more ELLA init params or takes from ckpt
        ella = ELLA()
        load_model(ella, ella_file, strict=True)  # type: ignore
        return (ella,)


class T5TextEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("ella_encoder"):
            if os.path.exists(search_path):
                for root, _, files in os.walk(search_path, followlinks=True):
                    if "config.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))
        return {
            "required": {
                "name": (paths,),
                "max_length": ("INT", {"default": 0, "min": 0, "max": 128, "step": 16}),
                "dtype": (["auto", "FP32", "FP16"],),
            }
        }

    RETURN_TYPES = ("T5_TEXT_ENCODER",)
    FUNCTION = "load"
    CATEGORY = "ella/loaders"

    def load(self, name: str, max_length: int = 0, dtype="auto", **kwargs):
        t5_file = folder_paths.get_full_path("ella_encoder", name)
        # "flexible_token_length" trick: Set `max_length=None` eliminating any text token padding or truncation.
        # Help improve the quality of generated images corresponding to short captions.
        for search_path in folder_paths.get_folder_paths("ella_encoder"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, name)
                if os.path.exists(path):
                    t5_file = path
                    break
        t5_encoder = T5TextEmbedder(t5_file, max_length=max_length or None)  # type: ignore
        if dtype == "auto":
            dtype = model_management.text_encoder_dtype()
        elif dtype == "FP16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        t5_encoder.to(dtype)  # type: ignore
        return (t5_encoder,)


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class ConditionToEllaEmbeds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = (ELLA_EMBEDS_TYPE,)
    FUNCTION = "convert"

    CATEGORY = "ella/helper"

    def convert(self, cond):
        # only use batch 0
        # CONDITIONING: [[cond, {"pooled_output": pooled}]]
        return ({f"{ELLA_EMBEDS_PREFIX}clip_embeds": cond[0][0], **cond[0][1]},)


class EllaCombineEmbeds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embeds": (ELLA_EMBEDS_TYPE,),
                "embeds_add": (ELLA_EMBEDS_TYPE,),
            }
        }

    RETURN_TYPES = (ELLA_EMBEDS_TYPE,)
    FUNCTION = "combine"

    CATEGORY = "ella/helper"

    def combine(self, embeds: dict, embeds_add: dict):
        if embeds.keys() & embeds_add.keys():
            print("warning: because there are some same keys, one of them will be overwritten.")

        return ({**embeds, **embeds_add},)


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Register
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "EllaApply": EllaApply,
    "T5TextEncode #ELLA": T5TextEncode,
    # Loaders
    "ELLALoader": ELLALoader,
    "T5TextEncoderLoader #ELLA": T5TextEncoderLoader,
    # Helpers
    "EllaCombineEmbeds": EllaCombineEmbeds,
    "ConditionToEllaEmbeds": ConditionToEllaEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "EllaApply": "Apply ELLA",
    "T5TextEncode #ELLA": "T5 Text Encode #ELLA",
    # Loaders
    "ELLALoader": "Load ELLA Model",
    "T5TextEncoderLoader #ELLA": "Load T5 TextEncoder #ELLA",
    # Helpers
    "EllaCombineEmbeds": "ELLA Combine Embeds",
    "ConditionToEllaEmbeds": "Convert Condition to ELLA Embeds",
}
