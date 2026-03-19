#!/usr/bin/env python3
"""
SevSeg-YOLO Mask Visualization Tool
=====================================

Generate per-detection visualization with all MaskGenerator pipeline steps.
Output: 7-panel strip per detection (Crop | TopK | Fusion | GuidedFilter | Threshold | Mask | Overlay)

Usage:
    python tools/visualize_masks.py \
        --data data.yaml \
        --weights path/to/best.pt \
        --device 0
"""
import os, sys, argparse, glob, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "mask_vis"))
    args = parser.parse_args()

    import torch, yaml
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import ScoreDetect
    from sevseg_yolo.mask_generator import MaskGeneratorV2

    os.makedirs(args.out, exist_ok=True)

    # Data
    with open(args.data) as f:
        dcfg = yaml.safe_load(f)
    dr = dcfg.get("path", os.path.dirname(os.path.abspath(args.data)))
    if not os.path.isabs(dr):
        dr = os.path.join(os.path.dirname(os.path.abspath(args.data)), dr)
    vd = os.path.join(dr, dcfg.get("val", "images/val"))
    imgs = sorted(glob.glob(os.path.join(vd, "*.jpg")) + glob.glob(os.path.join(vd, "*.png")))

    # Model
    model = YOLO(args.weights)
    raw = model.model; raw.eval()
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    raw.to(dev)

    hooked = {}; hooks = []
    hooks.append(raw.model[2].register_forward_hook(
        lambda m, i, o: hooked.update({"layer2": o.detach().cpu().numpy()[0]})))
    for md in raw.modules():
        if isinstance(md, ScoreDetect):
            def make_hook():
                def _h(mod, inp, out):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        x = inp[0]
                        if isinstance(x, list) and len(x) >= 2:
                            hooked["p3"] = x[0].detach().cpu().numpy()[0]
                            hooked["p4"] = x[1].detach().cpu().numpy()[0]
                            if len(x) > 2:
                                hooked["p5"] = x[2].detach().cpu().numpy()[0]
                return _h
            hooks.append(md.register_forward_hook(make_hook()))
            break

    mg = MaskGeneratorV2(topk_channels=48, guided_radius=6, guided_eps=0.005,
                         adaptive_block=15, adaptive_C=-5, morph_close_k=7, morph_open_k=3)

    print(f"Processing {len(imgs)} images → {args.out}/")
    count = 0

    for ip in imgs:
        orig = cv2.imread(ip)
        if orig is None: continue
        oh, ow = orig.shape[:2]
        iname = os.path.splitext(os.path.basename(ip))[0]
        hooked.clear()

        preds = model(ip, conf=args.conf, imgsz=640, device=dev, verbose=False)
        if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0 or "p3" not in hooked:
            continue

        _, fh, fw = hooked["p3"].shape
        input_h, input_w = fh * 8, fw * 8

        for bi in range(len(preds[0].boxes)):
            bb = preds[0].boxes.xyxy[bi].cpu().numpy()
            sev = preds[0].boxes.data[bi, 6].item() * 10 if preds[0].boxes.data.shape[-1] >= 7 else 0
            conf = preds[0].boxes.conf[bi].item()
            cls_id = int(preds[0].boxes.cls[bi].item())

            bx1, by1 = max(0, int(bb[0])), max(0, int(bb[1]))
            bx2, by2 = min(ow, int(bb[2])), min(oh, int(bb[3]))
            rh, rw = by2 - by1, bx2 - bx1
            if rh < 5 or rw < 5: continue

            g = min(input_h / oh, input_w / ow)
            px = round((input_w - ow * g) / 2 - 0.1)
            py = round((input_h - oh * g) / 2 - 0.1)
            mx1, my1 = int(max(0, bb[0]*g+px)), int(max(0, bb[1]*g+py))
            mx2, my2 = int(min(input_w, bb[2]*g+px)), int(min(input_h, bb[3]*g+py))

            guide_crop = orig[by1:by2, bx1:bx2]
            bbox_model = [mx1, my1, mx2, my2]

            t0 = time.perf_counter()
            mask_raw = mg.generate(
                bbox=bbox_model, feat_layer2=hooked.get("layer2"),
                feat_p3_fpn=hooked["p3"], feat_p4_fpn=hooked["p4"],
                feat_p5_fpn=hooked.get("p5"), guide_crop=guide_crop,
                input_hw=(input_h, input_w))
            elapsed = (time.perf_counter() - t0) * 1000

            mask = cv2.resize(mask_raw, (rw, rh), interpolation=cv2.INTER_NEAREST)
            fill = float(mask.mean())

            # Visualization
            ph, pw = max(rh, 60), max(rw, 60)
            color = (0, 220, 0) if sev < 4 else (0, 200, 255) if sev < 7 else (0, 0, 255)

            p_crop = cv2.resize(guide_crop, (pw, ph))
            cv2.rectangle(p_crop, (1, 1), (pw-2, ph-2), color, 2)

            p_mask = cv2.cvtColor(cv2.resize((mask*255).astype(np.uint8), (pw, ph),
                                  interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

            p_overlay = cv2.resize(guide_crop, (pw, ph)).copy()
            mask_r = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
            if mask_r.sum() > 0:
                p_overlay[mask_r > 0] = (p_overlay[mask_r > 0].astype(float) * 0.4 +
                                         np.array(color, dtype=np.uint8) * 0.6).astype(np.uint8)

            lh = 20
            panels = [(f"Crop {rh}x{rw}", p_crop),
                       (f"Mask f={fill:.2f}", p_mask),
                       ("Overlay", p_overlay)]

            strips = []
            for name, panel in panels:
                lbar = np.full((lh, pw, 3), 35, dtype=np.uint8)
                cv2.putText(lbar, name, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220,220,220), 1)
                strips.append(np.vstack([lbar, panel]))

            sep = np.full((strips[0].shape[0], 2, 3), 80, dtype=np.uint8)
            row = strips[0]
            for s in strips[1:]:
                row = np.hstack([row, sep, s])

            title = np.full((22, row.shape[1], 3), 45, dtype=np.uint8)
            cv2.putText(title, f"{iname} b{bi} c{cls_id} sev={sev:.1f} conf={conf:.2f} {elapsed:.1f}ms",
                        (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

            cv2.imwrite(os.path.join(args.out, f"{iname}_b{bi}_s{sev:.1f}.jpg"),
                        np.vstack([title, row]), [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1

    for h in hooks: h.remove()
    print(f"Done: {count} visualizations → {args.out}/")


if __name__ == "__main__":
    main()
