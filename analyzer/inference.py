import threading
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# FastVLM (apple/FastVLM-0.5B)
# =========================
FASTVLM_ID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # model expects a special image token id

class FastVLMDescriber:
    def __init__(self):
        self.lock = threading.Lock()
        self.tok = AutoTokenizer.from_pretrained(FASTVLM_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            FASTVLM_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
from transformers import AutoProcessor, AutoModelForVision2Seq  # correct class for VL models

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

        self.model = AutoModelForVision2Seq.from_pretrained(
            QWEN_VL_ID,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
