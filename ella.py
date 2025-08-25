import logging
import os
from typing import Dict

import folder_paths
import torch
from comfy import model_management, samplers
from comfy.conds import CONDCrossAttn

from .model import ELLA, T5TextEmbedder

ELLA_TYPE = "ELLA"
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

# === device/dtype alignment helpers ===
def _infer_float_dtype_from_embeds(d: dict):
    import torch
    for v in d.values():
        if torch.is_tensor(v) and v.is_floating_point():
            return v.dtype
        if isinstance(v, (list, tuple)):
            for t in v:
                if torch.is_tensor(t) and t.is_floating_point():
                    return t.dtype
        if isinstance(v, dict):
            dt = _infer_float_dtype_from_embeds(v)
            if dt is not None:
                return dt
    return None

def _align_to_model_device_dtype(x, device, dtype):
    import torch
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.is_floating_point():
            return x.to(device=device, dtype=dtype, non_blocking=True)
        return x.to(device=device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_align_to_model_device_dtype(xx, device, dtype) for xx in x)
    if isinstance(x, dict):
        return {k: _align_to_model_device_dtype(v, device, dtype) for k, v in x.items()}
    return x

# === /helpers ===
def ella_encode(ella: ELLA, timesteps: torch.Tensor, embeds: dict):
    num_steps = len(timesteps) - 1
    # print(f"creating ELLA conds for {num_steps} timesteps")
    conds = []
    for i, timestep in enumerate(timesteps[:-1]):
        # Calculate start and end percentages based on the position of sigma in the batch
        start = i / num_steps  # Start percentage is calculated based on the index
        end = (i + 1) / num_steps  # End percentage is calculated based on the next index

        # cond_ella = ella(timestep, **embeds)
        # align dtype/device to ELLA model
        device = getattr(ella, "output_device", timesteps.device)
        want_dtype = _infer_float_dtype_from_embeds(embeds) or torch.float16
        _t = timestep.to(device=device, dtype=want_dtype)
        _embeds = _align_to_model_device_dtype(embeds, device, want_dtype)

        cond_ella = ella(_t, **_embeds)

        cond_ella_dict = {"start_percent": start, "end_percent": end}
        conds.append([cond_ella, cond_ella_dict])

    return conds


class EllaProxyUNet:
    def __init__(
        self,
        ella: ELLA,
        model_sampling,
        positive,
        negative,
        mode=APPLY_MODE_ELLA_ONLY,
        **kwargs,
    ) -> None:
        self.ella = ella
        self.model_sampling = model_sampling
        self.mode = mode
        if positive.keys() != negative.keys():
            raise ValueError("positive and negative embeds types must match")
        self.embeds = [positive, negative]

        for i in range(len(self.embeds)):
            for k in self.embeds[i]:
                self.embeds[i][k] = CONDCrossAttn(self.embeds[i][k])

    def process_cond(self, embeds: Dict[str, CONDCrossAttn], batch_size, **kwargs):
        # return {k: v.process_cond(batch_size, self.ella.output_device, **kwargs).cond for k, v in embeds.items()}
        out = {k: v.process_cond(batch_size, self.ella.output_device, **kwargs).cond for k, v in embeds.items()}
        # align floats to a common dtype inferred from outputs (or fallback fp16)
        want_dtype = _infer_float_dtype_from_embeds(out) or torch.float16
        return _align_to_model_device_dtype(out, self.ella.output_device, want_dtype)

    def prepare_conds(self):

        cond_embeds = self.process_cond(self.embeds[0], 1)
        want_dtype = _infer_float_dtype_from_embeds(cond_embeds) or torch.float16
        t999 = torch.tensor([999.0], device=self.ella.output_device, dtype=want_dtype)
        cond = self.ella(t999, **cond_embeds)

        uncond_embeds = self.process_cond(self.embeds[1], 1)
        # same dtype for consistency
        t999u = t999
        uncond = self.ella(t999u, **uncond_embeds)

        if self.mode == APPLY_MODE_ELLA_ONLY:
            return cond, uncond
        if "clip_embeds" not in cond_embeds or "clip_embeds" not in uncond_embeds:
            logging.warning("'clip_embeds' is required, fallback to 'ELLA ONLY' mode")
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

        time_aware_encoder_hidden_states = []
            # get the dtype of the target model from the cond-data of the first group
            # (process_cond has already aligned device to self.ella.output_device)   
        for idx, i in enumerate(cond_or_uncond):
            cond_embeds = self.process_cond(self.embeds[i], input_x.size(0) // len(cond_or_uncond))
            want_dtype = _infer_float_dtype_from_embeds(cond_embeds) or torch.float16

            # timestep from sampler can be on CPU and in fp32 - we will align it
            t_model = self.model_sampling.timestep(timestep_[0])
            t_model = t_model.to(device=self.ella.output_device, dtype=want_dtype)

            h = self.ella(t_model, **cond_embeds)

            if self.mode == APPLY_MODE_ELLA_ONLY or "clip_embeds" not in cond_embeds:
                time_aware_encoder_hidden_states.append(h)
            else:
                h = torch.concat([h, cond_embeds["clip_embeds"]], dim=1)
                time_aware_encoder_hidden_states.append(h)

        # build a batch and move it under the downstream-UNet device
        hidden = torch.cat(time_aware_encoder_hidden_states, dim=0)
        c["c_crossattn"] = hidden.to(_device)


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
                "ella": (ELLA_TYPE,),
                "positive": (ELLA_EMBEDS_TYPE,),
                "negative": (ELLA_EMBEDS_TYPE,),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"default": None}),
                "mode": ([APPLY_MODE_ELLA_AND_CLIP, APPLY_MODE_ELLA_ONLY],),
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
        sigmas=None,
        mode=APPLY_MODE_ELLA_AND_CLIP,
        **kwargs,
    ):
        model_clone = model.clone()
        model_sampling = model_clone.get_model_object("model_sampling")
        positive = {k[ELLA_EMBEDS_PREFIX_LEN:]: v for k, v in positive.items() if k.startswith(ELLA_EMBEDS_PREFIX)}
        negative = {k[ELLA_EMBEDS_PREFIX_LEN:]: v for k, v in negative.items() if k.startswith(ELLA_EMBEDS_PREFIX)}
        if sigmas is not None or "timesteps" in ella:
            timesteps = model_sampling.timestep(sigmas) if sigmas is not None else ella.get("timesteps", None)
            conds = ella_encode(ella["model"], timesteps, positive)
            unconds = ella_encode(ella["model"], timesteps, negative)
        else:
            conds, unconds = self.legacy_patch(ella["model"], positive, negative, mode, model_clone, model_sampling)

        return (model_clone, conds, unconds)

    def legacy_patch(self, ella, positive, negative, mode, model_clone, model_sampling):
        logging.warning(
            "`Apply ELLA` without `simgas` is deprecated and it will be removed in a future version. "
            "Add `sigmas` input link OR use `Set ELLA Timesteps` + `ELLA Encode` instead."
        )
        ella_proxy = EllaProxyUNet(
            ella=ella, model_sampling=model_sampling, positive=positive, negative=negative, mode=mode
        )

        model_clone.set_model_unet_function_wrapper(ella_proxy)
        # No matter how many tokens are text features, the ella output must be 64 tokens.
        _cond, _uncond = ella_proxy.prepare_conds()
        cond = [_cond, {k: v for k, v in positive.items() if not k.startswith(ELLA_EMBEDS_PREFIX)}]
        uncond = [_uncond, {k: v for k, v in negative.items() if not k.startswith(ELLA_EMBEDS_PREFIX)}]
        return [cond], [uncond]


class EllaApply(EllaAdvancedApply):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ella": (ELLA_TYPE,),
                "positive": (ELLA_EMBEDS_TYPE,),
                "negative": (ELLA_EMBEDS_TYPE,),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"default": None}),
            },
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
            },
            "optional": {
                "embeds": (ELLA_EMBEDS_TYPE, {"default": None}),
            },
        }

    RETURN_TYPES = (ELLA_EMBEDS_TYPE,)
    FUNCTION = "encode"

    CATEGORY = "ella/conditioning"

    def encode(self, text, text_encoder: dict, max_length=None, embeds=None, **kwargs):
        text_encoder_model = text_encoder["model"]
        cond = text_encoder_model(text, max_length=max_length)
        embeds = embeds.copy() if embeds is not None else {}
        embeds[f"{ELLA_EMBEDS_PREFIX}t5_embeds"] = cond
        return (embeds,)


class EllaEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ella": (ELLA_TYPE,),
                "embeds": (ELLA_EMBEDS_TYPE,),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "ella/conditioning"

    def encode(self, ella, embeds: dict, **kwargs):
        timesteps = ella.get("timesteps", None)
        if timesteps is None:
            raise ValueError("timesteps are required but not provided, use the 'Set ELLA Timesteps' node first.")
        embeds = {k[ELLA_EMBEDS_PREFIX_LEN:]: v for k, v in embeds.items() if k.startswith(ELLA_EMBEDS_PREFIX)}
        conds = ella_encode(ella["model"], timesteps, embeds)
        return (conds,)


class EllaTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ella": (ELLA_TYPE,),
                "text_encoder": ("T5_TEXT_ENCODER",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "clip": ("CLIP", {"default": None}),
                "text_clip": ("STRING", {"default":"", "multiline": True, "dynamicPrompts": True}),
            },
        }

    RETURN_NAMES = ("CONDITIONING", "CLIP CONDITIONING")
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    FUNCTION = "encode"

    CATEGORY = "ella/conditioning"

    def encode(self, ella, text_encoder, text, clip=None, text_clip="", **kwargs):
        text_encoder_model = text_encoder["model"]
        cond = text_encoder_model(text, max_length=None)
        embeds = {}
        embeds[f"{ELLA_EMBEDS_PREFIX}t5_embeds"] = cond

        timesteps = ella.get("timesteps", None)
        if timesteps is None:
            raise ValueError("timesteps are required but not provided, use the 'Set ELLA Timesteps' node first.")
        embeds = {k[ELLA_EMBEDS_PREFIX_LEN:]: v for k, v in embeds.items() if k.startswith(ELLA_EMBEDS_PREFIX)}
        ella_conds = ella_encode(ella["model"], timesteps, embeds)

        clip_conds = None
        if clip is None and text_clip:
            raise ValueError("text_clip needs a clip to encode")
        if clip is not None:
            tokens = clip.tokenize(text_clip)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            clip_conds = [[cond, {"pooled_output": pooled}]]

        if clip_conds is not None:
            return (self.concat(ella_conds, clip_conds), clip_conds)

        return (ella_conds, None)

    def concat(self, conditioning_to, conditioning_from):
        out = []
        cond_from = conditioning_from[0][0]

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from),1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return out

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
            },
        }

    RETURN_TYPES = (ELLA_TYPE,)
    FUNCTION = "load"
    CATEGORY = "ella/loaders"

    def load(self, name: str, **kwargs):
        ella_file = folder_paths.get_full_path("ella", name)
        if not ella_file:
            raise ValueError("ELLA ckpt not found")
        ella = ELLA(ella_file)
        return ({"model": ella, "file": ella_file},)


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
        if dtype == "auto":
            dtype = model_management.text_encoder_dtype(model_management.text_encoder_device())
        elif dtype == "FP16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        t5_encoder = T5TextEmbedder(t5_file, max_length=max_length or None, dtype=dtype)  # type: ignore
        return ({"model": t5_encoder, "file": t5_file},)


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
            logging.warning("because there are some same keys, one of them will be overwritten.")

        return ({**embeds, **embeds_add},)


class CombineClipEllaEmbeds:
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
            logging.warning("there is already a clip embeds, the previous condition will be overwritten")
        return ({f"{ELLA_EMBEDS_PREFIX}clip_embeds": cond[0][0], **cond[0][1], **embeds},)


# Referenced from comfy_extra.BasicScheduler
# Convert BasicScheduler's SIGMAS return into timesteps
class SetEllaTimesteps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ella": (ELLA_TYPE,),
                "scheduler": (samplers.SCHEDULER_NAMES,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"default": None}),
            },
        }

    RETURN_TYPES = (ELLA_TYPE,)
    CATEGORY = "ella/helper"

    FUNCTION = "set_timesteps"

    def set_timesteps(self, model, ella, scheduler, steps, denoise, sigmas=None):
        model_sampling = model.get_model_object("model_sampling")
        if sigmas is None:
            total_steps = steps
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                total_steps = int(steps / denoise)
            sigmas = samplers.calculate_sigmas(model_sampling, scheduler, total_steps).cpu()[-(steps + 1) :]
        timesteps = model_sampling.timestep(sigmas)
        return ({**ella, "timesteps": timesteps},)


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Register
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "EllaApply": EllaApply,
    "EllaEncode": EllaEncode,
    "T5TextEncode #ELLA": T5TextEncode,
    "EllaTextEncode": EllaTextEncode,
    # Loaders
    "ELLALoader": ELLALoader,
    "T5TextEncoderLoader #ELLA": T5TextEncoderLoader,
    # Helpers
    "EllaCombineEmbeds": EllaCombineEmbeds,
    "ConditionToEllaEmbeds": ConditionToEllaEmbeds,  # Deprecated, use Combine instead
    "ConcatConditionEllaEmbeds": CombineClipEllaEmbeds,  # Deprecated, use Combine instead
    "CombineClipEllaEmbeds": CombineClipEllaEmbeds,
    "SetEllaTimesteps": SetEllaTimesteps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "EllaApply": "Apply ELLA",
    "EllaEncode": "ELLA Encode",
    "T5TextEncode #ELLA": "T5 Text Encode #ELLA",
    "EllaTextEncode": "ELLA Text Encode",
    # Loaders
    "ELLALoader": "Load ELLA Model",
    "T5TextEncoderLoader #ELLA": "Load T5 TextEncoder #ELLA",
    # Helpers
    "EllaCombineEmbeds": "ELLA Combine Embeds",
    "ConditionToEllaEmbeds": "Convert Condition to ELLA Embeds(Deprecated, CombineClip instead)",
    "ConcatConditionEllaEmbeds": "Concat Condition & ELLA Embeds(Deprecated, CombineClip instead)",
    "CombineClipEllaEmbeds": "Combine CLIP & ELLA Embeds",
    "SetEllaTimesteps": "Set ELLA Timesteps",
}
