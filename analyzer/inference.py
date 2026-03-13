import os
import importlib.util
from contextlib import contextmanager
import threading
import types
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
try:
    from transformers.modeling_utils import local_torch_dtype
except Exception:
    from contextlib import contextmanager as _ctxmgr

    @_ctxmgr
    def local_torch_dtype(dtype, _name=None):
        yield

# =========================
# FastVLM (apple/FastVLM-0.5B)
# =========================
FASTVLM_ID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # model expects a special image token id

class FastVLMDescriber:
    def __init__(self):
        self.lock = threading.Lock()
        self.supports_multi_image = False
        self.tok = AutoTokenizer.from_pretrained(FASTVLM_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            FASTVLM_ID,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        # vision tower provides the image preprocessor
        self.img_processor = self.model.get_vision_tower().image_processor

    @torch.inference_mode()
    def describe(self, pil_img: Image.Image, prompt: str = "Give a short caption.", max_new_tokens: int = 96):
        # Guard: if prompt is blank, fallback
        prompt = (prompt or "").strip() or "Give a short caption."

        # Build chat with explicit <image> placeholder
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        rendered = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Split around the placeholder so we can inject the image token id
        pre, post = rendered.split("<image>", 1)
        pre_ids  = self.tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tok(post, return_tensors="pt", add_special_tokens=False).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)

        # Compose input ids and mask
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)

        # Image features
        px = self.img_processor(images=pil_img.convert("RGB"), return_tensors="pt")["pixel_values"]
        px = px.to(self.model.device, dtype=self.model.dtype)

        with self.lock:
            out = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=max_new_tokens,
            )
        text = self.tok.decode(out[0], skip_special_tokens=True).strip()
        # FINAL FALLBACK so history never shows empty
        return text or "(no caption generated)"


# =========================
# Qwen2.5-VL (Qwen/Qwen2.5-VL-3B-Instruct)
# =========================
from transformers import AutoProcessor
try:
    # Newer transformers
    from transformers import AutoModelForVision2Seq as _QwenVLAutoModel
except Exception:
    # transformers 5.0.0.dev0 uses image+text -> text naming
    from transformers import AutoModelForImageTextToText as _QwenVLAutoModel

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False

QWEN_VL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

class QwenVLDescriber:
    """
    Uses chat template with an image placeholder, which injects the special image token.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.supports_multi_image = False
        self.processor = AutoProcessor.from_pretrained(QWEN_VL_ID, trust_remote_code=True)

        quant_config = None
        if torch.cuda.is_available() and _BNB_AVAILABLE:
            # 4-bit quantization helps on 6GB GPUs (e.g., 1660 Ti)
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = _QwenVLAutoModel.from_pretrained(
            QWEN_VL_ID,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            quantization_config=quant_config,  # can be None if bitsandbytes not installed
        )

    @torch.inference_mode()
    def describe(self, pil_img: Image.Image, prompt: str = "Give a short caption.", max_new_tokens: int = 96):
        # Guard: if prompt is blank, fallback
        prompt = (prompt or "").strip() or "Give a short caption."

        # Build a chat turn that includes an IMAGE item; template inserts the image token
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Pass BOTH text (with image token) and image tensor
        inputs = self.processor(
            text=[chat_text],
            images=[pil_img.convert("RGB")],
            return_tensors="pt"
        )
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with self.lock:
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        return text or "(no caption generated)"


# =========================
# Mini-InternVL2-2B-DA-DriveLM
# =========================
MINI_INTERNVL2_DRIVELM_PATH = os.environ.get(
    "MINI_INTERNVL2_DRIVELM_PATH",
    "/home/vision/work/Mini-InternVL2-2B-DA-DriveLM",
)
MINI_INTERNVL2_DEVICE = os.environ.get("MINI_INTERNVL2_DEVICE", "cuda").lower()
MINI_INTERNVL2_DEVICE_MAP = os.environ.get("MINI_INTERNVL2_DEVICE_MAP", "auto")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class MiniInternVL2DriveLMDescriber:
    def __init__(self):
        self.lock = threading.Lock()
        self.supports_multi_image = True
        use_cuda = torch.cuda.is_available() and MINI_INTERNVL2_DEVICE != "cpu"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if self.device.type == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.image_size = 448
        self.max_num = 1
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        # This model's remote code calls .item() during __init__; avoid meta-tensor init.
        model_kwargs = dict(
            dtype=self.dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )
        if self.device.type == "cuda":
            if importlib.util.find_spec("flash_attn") is not None:
                model_kwargs["use_flash_attn"] = True
            # Avoid meta-tensor init issues with remote-code models unless user explicitly sets a map.
            if MINI_INTERNVL2_DEVICE_MAP and MINI_INTERNVL2_DEVICE_MAP not in ("auto", "balanced", "balanced_low_0"):
                model_kwargs["device_map"] = MINI_INTERNVL2_DEVICE_MAP
            else:
                model_kwargs["low_cpu_mem_usage"] = False

        @contextmanager
        def _disable_meta_init():
            # transformers 5 dev initializes on meta by default; this model's __init__ can't handle meta tensors.
            orig = PreTrainedModel.get_init_context

            @classmethod
            def _no_meta_init(cls, *args, **kwargs):
                # transformers 4.x and 5.x have different signatures; keep it minimal.
                dtype = None
                if args:
                    dtype = args[0] if hasattr(args[0], "device") or args[0] is not None else None
                if "dtype" in kwargs:
                    dtype = kwargs["dtype"]
                return [local_torch_dtype(dtype, cls.__name__)]

            PreTrainedModel.get_init_context = _no_meta_init
            try:
                yield
            finally:
                PreTrainedModel.get_init_context = orig

        @contextmanager
        def _patch_tied_weights_marker():
            # Some remote-code models don't define all_tied_weights_keys; guard the marker.
            orig = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)
            if orig is None:
                yield
                return

            def _safe_mark(self):
                keys = getattr(self, "all_tied_weights_keys", None)
                if keys is None:
                    keys = getattr(self, "_tied_weights_keys", None)
                if not keys:
                    return
                for tied_param in keys.keys():
                    if tied_param in self._weights_initialized:
                        continue
                    self._weights_initialized.add(tied_param)

            PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark
            try:
                yield
            finally:
                PreTrainedModel.mark_tied_weights_as_initialized = orig

        try:
            with _disable_meta_init(), _patch_tied_weights_marker():
                self.model = AutoModel.from_pretrained(MINI_INTERNVL2_DRIVELM_PATH, **model_kwargs)
        except TypeError:
            model_kwargs.pop("use_flash_attn", None)
            with _disable_meta_init(), _patch_tied_weights_marker():
                self.model = AutoModel.from_pretrained(MINI_INTERNVL2_DRIVELM_PATH, **model_kwargs)

        self.model = self.model.eval()
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
        elif "device_map" not in model_kwargs:
            # Ensure model lands on GPU when not using accelerate device_map
            self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MINI_INTERNVL2_DRIVELM_PATH,
            trust_remote_code=True,
            use_fast=False,
        )

        if hasattr(self.model, "language_model") and not hasattr(self.model.language_model, "generate"):
            base_cls = self.model.language_model.__class__
            if GenerationMixin not in base_cls.__mro__:
                self.model.language_model.__class__ = type(
                    f"{base_cls.__name__}WithGenerate",
                    (base_cls, GenerationMixin),
                    {},
                )
        if hasattr(self.model, "language_model") and getattr(self.model.language_model, "generation_config", None) is None:
            if getattr(self.model, "generation_config", None) is not None:
                self.model.language_model.generation_config = self.model.generation_config
            else:
                self.model.language_model.generation_config = GenerationConfig.from_model_config(
                    self.model.language_model.config
                )
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "prepare_inputs_for_generation"):
            orig_prepare = self.model.language_model.prepare_inputs_for_generation

            def _prepare_inputs_for_generation(
                _self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
            ):
                try:
                    if past_key_values is not None and past_key_values[0][0] is None:
                        past_key_values = None
                except Exception:
                    pass
                return orig_prepare(
                    input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )

            self.model.language_model.prepare_inputs_for_generation = types.MethodType(
                _prepare_inputs_for_generation, self.model.language_model
            )

    def _transform(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr)
        return (tensor - self._mean) / self._std

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.image_size * self.image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image: Image.Image, min_num=1, max_num=12, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height
        )

        target_width = int(self.image_size * target_aspect_ratio[0])
        target_height = int(self.image_size * target_aspect_ratio[1])
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height), resample=Image.BICUBIC)
        processed_images = []
        grid_w = target_width // self.image_size
        grid_h = target_height // self.image_size
        for i in range(blocks):
            box = (
                (i % grid_w) * self.image_size,
                (i // grid_w) * self.image_size,
                ((i % grid_w) + 1) * self.image_size,
                ((i // grid_w) + 1) * self.image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((self.image_size, self.image_size), resample=Image.BICUBIC))
        return processed_images

    def _load_pixel_values(self, image: Image.Image):
        images = self._dynamic_preprocess(image, use_thumbnail=True, max_num=self.max_num)
        pixel_values = [self._transform(img) for img in images]
        return torch.stack(pixel_values)

    @torch.inference_mode()
    def describe(self, pil_img: Image.Image, prompt: str = "Give a short caption.", max_new_tokens: int = 96):
        prompt = (prompt or "").strip() or "Give a short caption."
        generation_config = dict(max_new_tokens=max_new_tokens)

        if isinstance(pil_img, (list, tuple)):
            images = [img.convert("RGB") for img in pil_img if img is not None]
            if not images:
                return "(no image provided)"
            pixel_batches = [self._load_pixel_values(img) for img in images]
            num_patches_list = [pv.size(0) for pv in pixel_batches]
            pixel_values = torch.cat(pixel_batches, dim=0).to(self.device, dtype=self.dtype)
            prefix = "\n".join([f"Image-{i+1}: <image>" for i in range(len(images))])
            question = f"{prefix}\n{prompt}"
            with self.lock:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                )
        else:
            question = f"<image>\n{prompt}"
            pixel_values = self._load_pixel_values(pil_img.convert("RGB"))
            pixel_values = pixel_values.to(self.device, dtype=self.dtype)
            with self.lock:
                response = self.model.chat(
                    self.tokenizer, pixel_values, question, generation_config
                )
        return (response or "").strip() or "(no caption generated)"
