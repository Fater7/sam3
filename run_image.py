import argparse
from pathlib import Path

import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 image segmentation.")
    parser.add_argument(
        "--image",
        default="assets/images/truck.jpg",
        help="Path to an input image.",
    )
    parser.add_argument(
        "--prompt",
        default="truck",
        help="Text prompt, for example: truck, person, dog.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Inference device. auto uses MPS when available.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional local checkpoint path. Omit to download from Hugging Face.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for returned masks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = pick_device(args.device)
    print(f"device: {device}")

    model = build_sam3_image_model(
        device="cpu",
        checkpoint_path=args.checkpoint,
        load_from_HF=args.checkpoint is None,
    )
    model = model.to(device)
    model.eval()

    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.threshold,
    )

    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    print(f"prompt: {args.prompt}")
    print(f"image: {image_path}")
    print(f"instances: {len(scores)}")
    print(f"masks shape: {tuple(masks.shape)}")
    print(f"boxes: {boxes.detach().cpu().tolist()}")
    print(f"scores: {scores.detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
