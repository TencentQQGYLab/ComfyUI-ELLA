import os
from typing import Dict

import folder_paths
import torch
from comfy import model_management
from comfy.conds import CONDCrossAttn
from safetensors.torch import load_model

from .model import ELLA, T5TextEmbedder

ELLA_EMBEDS_TYPE = "ELLA_EMBEDS"
ELLA_EMBEDS_PREFIX = "ella_"
ELLA_EMBEDS_PREFIX_LEN = len(ELLA_EMBEDS_PREFIX)
APPLY_MODE_ELLA_ONLY = "ELLA ONLY"
APPLY_MODE_ELLA_AND_CLIP = "ELLA + CLIP"

# set the models directory
if "ella" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "ella")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ella"]
folder_paths.folder_names_and_paths["ella"] = (current_paths, folder_paths.supported_pt_extensions)

if "ella_encoder" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "ella_encoder")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ella_encoder"]
folder_paths.folder_names_and_paths["ella_encoder"] = (current_paths, folder_paths.supported_pt_extensions)


class EllaProxyUNet:
    def __init__(
        self,
        ella,
        model_sampling,
        positive,
        negative,
        mode=APPLY_MODE_ELLA_ONLY,
        sigma_start=99999999,
        sigma_end=0,
        **kwargs,
    ) -> None:
        self.ella = ella
        self.model_sampling = model_sampling
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.mode = mode
        if positive.keys() != negative.keys():
            raise ValueError("positive and negative embeds types must match")
        # if mode == APPLY_MODE_ELLA_AND_CLIP and "clip_embeds" not in positive:
        #     raise ValueError(f"'clip_embeds' is required when using '{APPLY_MODE_ELLA_AND_CLIP}' mode")
        self.embeds = [positive, negative]

        self.dtype = model_management.text_encoder_dtype()
        self.ella.to(self.dtype)
        for i in range(len(self.embeds)):
            for k in self.embeds[i]:
                self.embeds[i][k].to(dtype=self.dtype)
                self.embeds[i][k] = CONDCrossAttn(self.embeds[i][k])

    @property
    def load_device(self):
        return model_management.text_encoder_device()

    @property
    def offload_device(self):
        return model_management.text_encoder_offload_device()

    def process_cond(self, embeds: Dict[str, CONDCrossAttn], batch_size, **kwargs):
        return {k: v.process_cond(batch_size, self.load_device, **kwargs).cond for k, v in embeds.items()}

    def prepare_conds(self):
        self.ella.to(self.load_device)
        cond_embeds = self.process_cond(self.embeds[0], 1)
        cond = self.ella(torch.Tensor([999]).to(torch.int64), **cond_embeds)
        uncond_embeds = self.process_cond(self.embeds[1], 1)
        uncond = self.ella(torch.Tensor([999]).to(torch.int64), **uncond_embeds)
        self.ella.to(self.offload_device)
        if self.mode == APPLY_MODE_ELLA_ONLY:
            return cond, uncond
        if "clip_embeds" not in cond_embeds or "clip_embeds" not in uncond_embeds:
            print("warning: 'clip_embeds' is required, fallback to 'ELLA ONLY' mode")
            return cond, uncond
        return (
            torch.concat([cond, cond_embeds["clip_embeds"]], dim=1),
            torch.concat([uncond, uncond_embeds["clip_embeds"]], dim=1),
        )

    def __call__(self, apply_model, kwargs: dict):
        input_x = kwargs["input"]
        timestep_ = kwargs["timestep"]
        c = kwargs["c"]
        cond_or_uncond = kwargs["cond_or_uncond"]  # [0|1]
        _device = c["c_crossattn"].device

        # TODO: add ella start/end sigma control
        time_aware_encoder_hidden_states = []
        self.ella.to(device=self.load_device)
        for i in cond_or_uncond:
            cond_embeds = self.process_cond(self.embeds[i], input_x.size(0) // len(cond_or_uncond))
            h = self.ella(
                self.model_sampling.timestep(timestep_[0]),
                **cond_embeds,
            )
            if self.mode == APPLY_MODE_ELLA_ONLY:
                time_aware_encoder_hidden_states.append(h)
                continue
            if "clip_embeds" not in cond_embeds:
                time_aware_encoder_hidden_states.append(h)
                continue
            h = torch.concat([h, cond_embeds["clip_embeds"]], dim=1)
            time_aware_encoder_hidden_states.append(h)
        self.ella.to(self.offload_device)

        c["c_crossattn"] = torch.cat(time_aware_encoder_hidden_states, dim=0).to(_device)

        return apply_model(input_x, timestep_, **c)


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Apply Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class EllaAdvancedApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ella": ("ELLA",),
                "positive": (ELLA_EMBEDS_TYPE,),
                "negative": (ELLA_EMBEDS_TYPE,),
            },
            "optional": {
                "mode": ([APPLY_MODE_ELLA_AND_CLIP, APPLY_MODE_ELLA_ONLY],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_NAMES = ("model", "positive", "negative")
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "apply"
    CATEGORY = "ella/apply"

    def apply(
        self,
        model,
        ella,
        positive,
        negative,
        mode=APPLY_MODE_ELLA_AND_CLIP,
        start_at=0.0,
        end_at=1.0,
    ):
        model_clone = model.clone()
        model_sampling = model_clone.get_model_object("model_sampling")
        sigma_start = model_clone.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model_clone.get_model_object("model_sampling").percent_to_sigma(end_at)

        ella_proxy = EllaProxyUNet(
            ella=ella,
            model_sampling=model_sampling,
            positive={
                k[ELLA_EMBEDS_PREFIX_LEN:]: v.clone() for k, v in positive.items() if k.startswith(ELLA_EMBEDS_PREFIX)
            },
            negative={
                k[ELLA_EMBEDS_PREFIX_LEN:]: v.clone() for k, v in negative.items() if k.startswith(ELLA_EMBEDS_PREFIX)
            },
            mode=mode,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
        )

        model_clone.set_model_unet_function_wrapper(ella_proxy)
        # No matter how many tokens are text features, the ella output must be 64 tokens.
        _cond, _uncond = ella_proxy.prepare_conds()
        cond = [_cond, {k: v for k, v in positive.items() if not k.startswith(ELLA_EMBEDS_PREFIX)}]
        uncond = [_uncond, {k: v for k, v in negative.items() if not k.startswith(ELLA_EMBEDS_PREFIX)}]

        return (model_clone, [cond], [uncond])


class EllaApply(EllaAdvancedApply):
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


class ConcatConditionEllaEmbeds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "embeds": (ELLA_EMBEDS_TYPE,),
            }
        }

    RETURN_TYPES = (ELLA_EMBEDS_TYPE,)
    FUNCTION = "combine"

    CATEGORY = "ella/helper"

    def combine(self, cond, embeds):
        # only use batch 0
        # CONDITIONING: [[cond, {"pooled_output": pooled}]]
        clip_key = f"{ELLA_EMBEDS_PREFIX}clip_embeds"
        if clip_key in embeds:
            print("warning: there is already a clip embeds, the previous condition will be overwritten")
        return ({f"{ELLA_EMBEDS_PREFIX}clip_embeds": cond[0][0], **cond[0][1], **embeds},)


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Register
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "EllaApply": EllaApply,
    # "EllaAdvancedApply": EllaAdvancedApply,
    "T5TextEncode #ELLA": T5TextEncode,
    # Loaders
    "ELLALoader": ELLALoader,
    "T5TextEncoderLoader #ELLA": T5TextEncoderLoader,
    # Helpers
    "EllaCombineEmbeds": EllaCombineEmbeds,
    "ConditionToEllaEmbeds": ConditionToEllaEmbeds,
    "ConcatConditionEllaEmbeds": ConcatConditionEllaEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "EllaApply": "Apply ELLA",
    # "EllaAdvancedApply": "Apply ELLA Advanced",
    "T5TextEncode #ELLA": "T5 Text Encode #ELLA",
    # Loaders
    "ELLALoader": "Load ELLA Model",
    "T5TextEncoderLoader #ELLA": "Load T5 TextEncoder #ELLA",
    # Helpers
    "EllaCombineEmbeds": "ELLA Combine Embeds",
    "ConditionToEllaEmbeds": "Convert Condition to ELLA Embeds",
    "ConcatConditionEllaEmbeds": "Concat Condition & ELLA Embeds",
}
