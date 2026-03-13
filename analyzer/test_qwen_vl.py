# test_qwen_vl.py
# Test Qwen/Qwen2.5-VL-3B-Instruct on an image or a single webcam frame.
# Requires: transformers>=4.45, accelerate>=0.34, (optional) bitsandbytes>=0.43, pillow, torch, opencv-python (if --camera)

import argparse
import sys
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import AutoModelForVision2Seq as _QwenVLAutoModel
except Exception:
    from transformers import AutoModelForImageTextToText as _QwenVLAutoModel

# Try to enable 4-bit on GPU (good for 6GB cards like 1660 Ti)
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_model(model_id: str, use_4bit: bool):
    print(f"[info] Loading {model_id} (4-bit={use_4bit and BNB_AVAILABLE and torch.cuda.is_available()}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    quant_config = None
    if use_4bit and BNB_AVAILABLE and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = _QwenVLAutoModel.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    return processor, model


def load_image(path: str, max_width: Optional[int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_width and img.width > max_width:
        ratio = max_width / float(img.width)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.BICUBIC)
    return img


def grab_camera_frame(index: int, max_width: Optional[int]) -> Image.Image:
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read from camera")
    if max_width and frame.shape[1] > max_width:
        s = max_width / frame.shape[1]
        frame = cv2.resize(frame, (int(frame.shape[1]*s), int(frame.shape[0]*s)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def build_chat_text(processor, prompt: str) -> str:
    """
    Use Qwen's chat template so the special <image> token is inserted.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},                # <-- this tells the template to add the image token
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_caption(processor, model, img: Image.Image, prompt: str, max_new_tokens: int) -> str:
    chat_text = build_chat_text(processor, prompt)
    # IMPORTANT: pass BOTH text (with image token inserted by template) and the image tensor
    inputs = processor(text=[chat_text], images=[img], return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser(description="Test Qwen/Qwen2.5-VL-3B-Instruct on an image or webcam.")
    ap.add_argument("--image", type=str, help="Path to an image file (jpg/png).")
    ap.add_argument("--camera", type=int, help="Camera index (e.g., 0) to grab one frame.")
    ap.add_argument("--prompt", type=str, default="Give a short caption.",
                    help="Instruction/prompt for the model.")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="Generation length.")
    ap.add_argument("--max-width", type=int, default=1280, help="Resize image width for speed/VRAM.")
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization.")
    ap.add_argument("--model-id", type=str, default=MODEL_ID, help="Override model id.")
    args = ap.parse_args()

    if not args.image and args.camera is None:
        ap.error("Provide either --image <path> or --camera <index>")

    try:
        processor, model = load_model(args.model_id, use_4bit=(not args.no_4bit))
    except Exception as e:
        print(f"[error] Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.image:
            img = load_image(args.image, args.max_width)
        else:
            img = grab_camera_frame(args.camera, args.max_width)
    except Exception as e:
        print(f"[error] Failed to get image: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[info] Prompt: {args.prompt}")
    print(f"[info] max_new_tokens={args.max_new_tokens}, image={img.width}x{img.height}")
    try:
        caption = run_caption(processor, model, img, args.prompt, args.max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        print("[oom] CUDA out of memory. Try --max-width 960 or --max-new-tokens 24.", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"[error] Inference failed: {e}", file=sys.stderr)
        sys.exit(4)

    print("\n=== MODEL OUTPUT ===")
    print(caption)


if __name__ == "__main__":
    main()
