import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "element"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract game elements with SAM3.")
    parser.add_argument(
        "--input",
        default="/Users/bytedance/Desktop/game-element/input/1",
        help="Input directory containing context.json and scene images.",
    )
    parser.add_argument(
        "--output",
        default="/Users/bytedance/Desktop/game-element/output/sam3/1",
        help="Independent output directory for SAM3 results.",
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
        help="Optional local SAM3 checkpoint path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold.",
    )
    parser.add_argument(
        "--include-types",
        default="element,ui",
        help="Comma-separated element types to process. Use all to process everything.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["title", "description", "full"],
        default="title",
        help="Text prompt source for SAM3.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directory before running.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Keep only the top scored instances per element. 0 keeps all.",
    )
    return parser.parse_args()


def load_context(input_dir: Path) -> dict:
    context_path = input_dir / "context.json"
    with context_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def map_scene_images(input_dir: Path, scenes: list[dict]) -> dict[str, Path]:
    scene_images = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
    )
    local_scenes = [scene for scene in scenes if scene.get("elements")]
    return {
        scene["id"]: image_path
        for scene, image_path in zip(local_scenes, scene_images, strict=False)
    }


def element_prompt(element: dict, prompt_mode: str) -> str:
    title = element.get("englishTitle") or element.get("name") or ""
    title = title.replace("-", " ")
    description = element.get("description") or ""
    if prompt_mode == "title":
        return title.strip()
    if prompt_mode == "description":
        return description.strip()
    return f"{title}. {description}".strip()


def tensor_to_numpy_mask(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy()
    return np.squeeze(mask_np).astype(bool)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_overlay(image: Image.Image, mask: np.ndarray, box: list[float], path: Path) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask.astype(np.uint8) * 150), mode="L")
    color = Image.new("RGBA", base.size, (255, 64, 64, 0))
    color.putalpha(mask_img)
    overlay.alpha_composite(color)

    draw = ImageDraw.Draw(overlay)
    draw.rectangle(box, outline=(255, 32, 32, 255), width=4)
    base.alpha_composite(overlay)
    base.save(path)


def save_crop(image: Image.Image, mask: np.ndarray, box: list[float], path: Path) -> None:
    rgba = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    rgba.putalpha(alpha)

    left, top, right, bottom = [int(round(v)) for v in box]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, rgba.width)
    bottom = min(bottom, rgba.height)
    if left >= right or top >= bottom:
        rgba.save(path)
        return
    rgba.crop((left, top, right, bottom)).save(path)


def run() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = load_context(input_dir)
    scenes = context["gameContext"]["scenes"]
    scene_to_image = map_scene_images(input_dir, scenes)

    include_types = None
    if args.include_types != "all":
        include_types = {value.strip() for value in args.include_types.split(",")}

    device = pick_device(args.device)
    print(f"device: {device}")
    print(f"input: {input_dir}")
    print(f"output: {output_dir}")

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

    summary = {
        "input": str(input_dir),
        "device": device,
        "threshold": args.threshold,
        "includeTypes": sorted(include_types) if include_types is not None else "all",
        "promptMode": args.prompt_mode,
        "maxInstances": args.max_instances,
        "scenes": [],
    }

    for scene in scenes:
        scene_id = scene["id"]
        image_path = scene_to_image.get(scene_id)
        scene_summary = {
            "id": scene_id,
            "name": scene.get("name"),
            "imagePath": str(image_path) if image_path else None,
            "elements": [],
        }
        summary["scenes"].append(scene_summary)

        if image_path is None:
            print(f"skip scene without local image: {scene_id}")
            continue

        image = Image.open(image_path).convert("RGB")
        scene_dir = output_dir / "scenes" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, scene_dir / image_path.name)

        state = processor.set_image(image)
        print(f"scene: {scene_id} image={image_path.name}")

        for element in scene.get("elements", []):
            if include_types is not None and element.get("type") not in include_types:
                continue

            prompt = element_prompt(element, args.prompt_mode)
            element_slug = slugify(element.get("englishTitle") or element["id"])
            element_dir = scene_dir / element_slug
            element_dir.mkdir(parents=True, exist_ok=True)

            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(state=state, prompt=prompt)

            boxes = output["boxes"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            masks = output["masks"]
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            if args.max_instances > 0:
                order = order[: args.max_instances]

            element_summary = {
                "id": element["id"],
                "name": element.get("name"),
                "englishTitle": element.get("englishTitle"),
                "type": element.get("type"),
                "prompt": prompt,
                "instances": [],
            }
            scene_summary["elements"].append(element_summary)

            for output_index, source_index in enumerate(order):
                box = boxes[source_index]
                score = scores[source_index]
                mask = tensor_to_numpy_mask(masks[source_index])
                prefix = f"{output_index:02d}"
                mask_path = element_dir / f"{prefix}_mask.png"
                overlay_path = element_dir / f"{prefix}_overlay.png"
                crop_path = element_dir / f"{prefix}_crop.png"

                save_mask(mask, mask_path)
                save_overlay(image, mask, box, overlay_path)
                save_crop(image, mask, box, crop_path)

                element_summary["instances"].append(
                    {
                        "index": output_index,
                        "sourceIndex": source_index,
                        "score": float(score),
                        "box": [float(v) for v in box],
                        "mask": str(mask_path.relative_to(output_dir)),
                        "overlay": str(overlay_path.relative_to(output_dir)),
                        "crop": str(crop_path.relative_to(output_dir)),
                    }
                )

            with (element_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(element_summary, f, ensure_ascii=False, indent=2)

            print(
                f"  element: {element.get('englishTitle')} "
                f"instances={len(element_summary['instances'])}"
            )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"done: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
