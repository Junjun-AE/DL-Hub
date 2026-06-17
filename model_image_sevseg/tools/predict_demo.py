#!/usr/bin/env python3
"""
SevSeg-YOLO Inference Demo
============================

Usage:
    python tools/predict_demo.py --weights best.pt --source image.jpg
    python tools/predict_demo.py --weights best.pt --source images/ --save-dir outputs/
"""
import os, sys, argparse, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--save-dir", default="outputs")
    args = parser.parse_args()

    from sevseg_yolo import SevSegYOLO
    os.makedirs(args.save_dir, exist_ok=True)

    # 一行加载
    model = SevSegYOLO(args.weights, device=args.device, conf=args.conf)
    print(f"Loaded: {model}")

    # 获取图像列表
    if os.path.isdir(args.source):
        images = sorted(glob.glob(os.path.join(args.source, "*.jpg")) +
                        glob.glob(os.path.join(args.source, "*.png")))
    else:
        images = [args.source]

    for ip in images:
        iname = os.path.splitext(os.path.basename(ip))[0]

        # 一行推理
        result = model.predict(ip)

        # 一行可视化+保存
        out = os.path.join(args.save_dir, f"{iname}_result.jpg")
        result.visualize().save(out)

        # 打印结果
        for det in result:
            print(f"  {det}")

        print(f"  → {out} ({result.num_detections} detections)")

    print("Done!")

if __name__ == "__main__":
    main()
