# ─────────────────────────────────────────────
#  inference.py  –  Real-time terrain classification
# ─────────────────────────────────────────────
#
#  Three modes:
#  1. Single image:   python inference.py --image path/to/img.jpg --phase 2
#  2. Video file:     python inference.py --video path/to/video.mp4 --phase 2
#  3. Webcam:         python inference.py --webcam --phase 2
#
#  This script also measures FPS — critical for wearable robotics real-time use.
# ─────────────────────────────────────────────

import os
import sys
import time
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IMG_SIZE, MEAN, STD, GTOS_ROOT
from utils.dataset import get_dataloaders

# color palette for visualization (one color per class)
COLORS = [
    (255, 99,  71),  (60,  179, 113), (70,  130, 180), (255, 165,   0),
    (148,   0, 211), (255, 20,  147), (0,   206, 209), (139,  90,  43),
    (128, 128,   0), (64,  224, 208), (255, 215,   0), (0,   128, 128),
]


def parse_args():
    p = argparse.ArgumentParser(description="Terrain Classifier – Inference")
    p.add_argument("--checkpoint", type=str,  default=None,
                   help="Path to checkpoint (auto-selects best phase if not given)")
    p.add_argument("--phase",      type=int,  default=2, choices=[1, 2])
    p.add_argument("--data",       type=str,  default=GTOS_ROOT)

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",  type=str, help="Path to a single image")
    mode.add_argument("--video",  type=str, help="Path to video file")
    mode.add_argument("--webcam", action="store_true", help="Use webcam (index 0)")

    p.add_argument("--topk",      type=int,   default=3,  help="Show top-K predictions")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Confidence threshold for 'uncertain' label")
    return p.parse_args()


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@torch.no_grad()
def predict(model, image_pil, transform, device, topk=3):
    """
    Returns top-k (class_index, probability) pairs and inference time in ms.
    """
    tensor = transform(image_pil).unsqueeze(0).to(device)

    t0  = time.perf_counter()
    out = model(tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(out, dim=1)[0]
    topk_probs, topk_idx = probs.topk(topk)
    return list(zip(topk_idx.cpu().numpy(), topk_probs.cpu().numpy())), elapsed_ms


def predict_image(model, image_path, transform, device, class_names, args):
    img = Image.open(image_path).convert("RGB")
    results, ms = predict(model, img, transform, device, topk=args.topk)

    print(f"\n[Image] {os.path.basename(image_path)}")
    print(f"{'─'*40}")
    for rank, (idx, prob) in enumerate(results, 1):
        label = class_names[idx]
        bar   = "█" * int(prob * 30)
        print(f"  #{rank} {label:<20} {prob*100:5.1f}%  {bar}")

    top_idx, top_prob = results[0]
    status = "✓ Confident" if top_prob >= args.threshold else "⚠ Uncertain"
    print(f"\n  Prediction : {class_names[top_idx]}  ({status})")
    print(f"  Inference  : {ms:.1f} ms")


def predict_video(model, source, transform, device, class_names, args, is_webcam=False):
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Run: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0 if is_webcam else source)
    if not cap.isOpened():
        print(f"Cannot open {'webcam' if is_webcam else source}")
        return

    print("[Video] Press 'q' to quit")
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR (cv2) → RGB → PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results, ms = predict(model, img_pil, transform, device, topk=args.topk)

        # ── Overlay on frame ───────────────────────
        fps = 1000.0 / (ms + 1e-6)
        fps_history.append(fps)

        # background box
        cv2.rectangle(frame, (0, 0), (350, 30 + args.topk * 30), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        for rank, (idx, prob) in enumerate(results, 1):
            color = COLORS[idx % len(COLORS)]
            text  = f"#{rank} {class_names[idx]}: {prob*100:.1f}%"
            cv2.putText(frame, text, (10, 30 + rank * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (color[2], color[1], color[0]),  # BGR
                        2)

        cv2.imshow("Terrain Classifier – DRDO", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    avg_fps = np.mean(fps_history)
    print(f"\n[Video] Average FPS: {avg_fps:.1f} | Average latency: {1000/avg_fps:.1f} ms")


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"[Device] {device}")

    # load class names from dataset
    _, _, class_names, num_classes = get_dataloaders(args.data)

    # load model
    if args.phase == 1:
        from models.cnn_scratch import get_model
        default_ckpt = "checkpoints/phase1_best.pth"
        model = get_model(num_classes=num_classes).to(device)
    else:
        from models.mobilenet_transfer import get_model
        default_ckpt = "checkpoints/phase2_best.pth"
        model = get_model(num_classes=num_classes, mode="finetune").to(device)

    ckpt_path = args.checkpoint or default_ckpt
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Model] Phase {args.phase} | Loaded: {ckpt_path} | val_acc={ckpt['val_acc']:.2f}%")

    transform = build_transform()

    if args.image:
        predict_image(model, args.image, transform, device, class_names, args)
    elif args.video:
        predict_video(model, args.video, transform, device, class_names, args)
    elif args.webcam:
        predict_video(model, None, transform, device, class_names, args, is_webcam=True)


if __name__ == "__main__":
    main()
