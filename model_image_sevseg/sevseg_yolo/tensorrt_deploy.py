"""SevSeg-YOLO TensorRT deployment pipeline (§10).

Provides:
    - PCA feature preparation (offline, one-time)
    - TensorRT engine building (FP16/INT8)
    - Inference runner with score denormalization
    - Precision validation (PyTorch vs ONNX vs TRT)

Usage:
    from sevseg_yolo.tensorrt_deploy import (
        prepare_pca_matrices,
        build_trt_engine,
        SevSegInferenceEngine,
        validate_precision,
    )

    # Step 1: PCA preparation (offline)
    pca = prepare_pca_matrices(model, dataloader, variance_threshold=0.95)

    # Step 2: Export ONNX (see export.py)
    # Step 3: Build TensorRT engine
    build_trt_engine("model.onnx", "model.engine", fp16=True)

    # Step 4: Run inference
    engine = SevSegInferenceEngine("model.engine")
    results = engine.infer(image)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def detect_pca_knee(explained_variance_ratio, min_dim=4, max_dim=None):
    """Detect PCA knee point via second-derivative method (§18.3 / A5).

    Finds the elbow where adding more dimensions gives diminishing returns.

    Args:
        explained_variance_ratio: Per-component variance ratios.
        min_dim: Minimum dimensions to keep.
        max_dim: Maximum dimensions.

    Returns:
        Optimal number of PCA dimensions.
    """
    cumvar = np.cumsum(explained_variance_ratio)
    n = len(cumvar)
    if max_dim is None:
        max_dim = n
    if n <= min_dim:
        return n
    d1 = np.diff(cumvar)
    d2 = np.diff(d1)
    lo = max(0, min_dim - 1)
    hi = min(len(d2), max_dim - 1)
    if hi <= lo:
        return min_dim
    knee = lo + np.argmax(np.abs(d2[lo:hi]))
    dim = knee + 2
    while dim < n and cumvar[dim - 1] < 0.85:
        dim += 1
    return max(min_dim, min(dim, max_dim))


def prepare_pca_matrices(
    model: nn.Module,
    dataloader,
    variance_threshold: float = 0.95,
    max_samples: int = 500,
    use_knee_detection: bool = False,
) -> dict:
    """Compute PCA projection matrices from training set features (§10.4).

    Runs inference on a subset of the training set, collects P3/P4/P5
    feature maps, and computes PCA to retain specified variance.

    Args:
        model: Trained SevSeg-YOLO model.
        dataloader: Training dataloader.
        variance_threshold: Fraction of variance to retain (default 95%).
        max_samples: Max images to process for PCA computation.

    Returns:
        Dict with PCA parameters per scale:
        {'p3': {'W': ndarray(C, dim), 'mean': ndarray(C), 'dim': int}, ...}
    """
    model.eval()
    device = next(model.parameters()).device

    # Collect features via hooks
    features = {"p3": [], "p4": [], "p5": []}
    hooks = []

    # Find the detect head to hook its input
    for name, module in model.named_modules():
        from ultralytics.nn.modules.head import ScoreDetect
        if isinstance(module, ScoreDetect):
            # The score_head receives features list [P3, P4, P5]
            score_head = module.one2one_score_head if hasattr(module, "one2one_score_head") else module.score_head
            if score_head is not None:
                def make_hook():
                    def hook_fn(mod, inp, out):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            feat_list = inp[0]
                            if isinstance(feat_list, list) and len(feat_list) >= 3:
                                for i, key in enumerate(["p3", "p4", "p5"]):
                                    if i < len(feat_list):
                                        # Global average pool to reduce memory: (B, C, H, W) → (B, C)
                                        f = feat_list[i].detach()
                                        # Flatten spatial: (B, C, H*W) → transpose → (B*H*W, C)
                                        b, c, h, w = f.shape
                                        f_flat = f.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
                                        # Subsample to limit memory
                                        if f_flat.shape[0] > 5000:
                                            idx = np.random.choice(f_flat.shape[0], 5000, replace=False)
                                            f_flat = f_flat[idx]
                                        features[key].append(f_flat)
                    return hook_fn
                h = score_head.register_forward_hook(make_hook())
                hooks.append(h)
            break

    # Run inference to collect features
    n_processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if n_processed >= max_samples:
                break
            img = batch["img"].to(device).float() / 255.0
            model(img)
            n_processed += img.shape[0]

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute PCA for each scale
    pca_matrices = {}
    for key in ["p3", "p4", "p5"]:
        if not features[key]:
            continue
        all_feat = np.concatenate(features[key], axis=0)  # (N, C)

        # Center data
        mean = all_feat.mean(axis=0)
        centered = all_feat - mean

        # SVD for PCA
        # For large C, use randomized SVD if available
        try:
            from sklearn.decomposition import PCA as SklearnPCA
            pca = SklearnPCA(n_components=min(0.999, variance_threshold), svd_solver="full")
            pca.fit(centered)
            explained_var = pca.explained_variance_ratio_
            cumvar = np.cumsum(explained_var)
            if use_knee_detection:
                dim = detect_pca_knee(explained_var, min_dim=4, max_dim=min(64, len(explained_var)))
                W = pca.components_[:dim].T  # (C, dim)
            else:
                W = pca.components_.T  # (C, dim) — sklearn already selected by variance
                dim = W.shape[1]
        except ImportError:
            # Fallback: manual SVD
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            explained_var = (S ** 2) / (S ** 2).sum()
            cumvar = np.cumsum(explained_var)
            if use_knee_detection:
                dim = detect_pca_knee(explained_var, min_dim=4, max_dim=min(64, Vt.shape[0]))
            else:
                dim = int(np.searchsorted(cumvar, variance_threshold) + 1)
                dim = min(dim, Vt.shape[0])
            W = Vt[:dim].T  # (C, dim)

        pca_matrices[key] = {
            "W": W,
            "mean": mean,
            "dim": int(dim),
            "explained_variance_ratio": float(cumvar[dim - 1]) if "cumvar" in dir() else float(variance_threshold),
        }
        print(f"PCA {key}: {all_feat.shape[1]}ch → {dim}ch (variance: {pca_matrices[key]['explained_variance_ratio']:.3f})")

    return pca_matrices


def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    int8: bool = False,
    min_batch_size: int = 1,
    max_batch_size: int = 8,
    workspace_gb: float = 4.0,
) -> Path:
    """Build TensorRT engine from ONNX model (§10.1 Step 4).

    Args:
        onnx_path: Input ONNX model path.
        engine_path: Output TensorRT engine path.
        fp16: Enable FP16 precision.
        int8: Enable INT8 precision (requires calibration).
        min_batch_size: Minimum batch size for dynamic shapes.
        max_batch_size: Maximum batch size for dynamic shapes.
        workspace_gb: GPU workspace in GB.

    Returns:
        Path to the built engine file.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT not installed. Install with: pip install tensorrt")

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")
            # NOTE (FIX #10): Per-layer FP32 precision control (e.g., keeping Score Head
            # final Conv in FP32) is not implemented here. TensorRT's layer-level precision
            # API requires iterating network layers and calling layer.precision = trt.float32.
            # For score regression tasks sensitive to FP16 quantization, consider:
            #   1. Test FP16 vs FP32 score MAE difference first (§10.8)
            #   2. If score MAE degrades >10%, implement per-layer control in Phase 4

        if int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 enabled (calibration required)")

        # Dynamic shapes
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        # Min/Opt/Max shapes for dynamic batch
        # Assuming input is (B, 3, H, W) where H=W=imgsz
        _, c, h, w = input_tensor.shape
        if h <= 0 or w <= 0:
            h = w = 640  # default
        if c <= 0:
            c = 3

        profile.set_shape(input_name, (min_batch_size, c, h, w), (max(1, (min_batch_size + max_batch_size) // 2), c, h, w), (max_batch_size, c, h, w))
        config.add_optimization_profile(profile)

        # Build engine
        print(f"Building TensorRT engine (this may take a few minutes)...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        with open(engine_path, "wb") as f:
            f.write(serialized)

        print(f"TensorRT engine saved to {engine_path}")
        return engine_path


class SevSegInferenceEngine:
    """SevSeg-YOLO TensorRT inference engine (§10.2, §10.9).

    Wraps TensorRT engine execution with:
    - Automatic input preprocessing (resize, normalize)
    - Score denormalization ([0,1] → [0,10])
    - Batch inference support

    Args:
        engine_path: Path to TensorRT engine file.
        device_id: CUDA device ID.
    """

    def __init__(self, engine_path: str | Path, device_id: int = 0):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT inference requires: tensorrt pycuda")

        self.engine_path = Path(engine_path)
        self.device_id = device_id

        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

    def infer(
        self,
        images: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> list[dict]:
        """Run inference on preprocessed images.

        Args:
            images: Preprocessed images (B, 3, H, W) as float32 [0, 1].
            conf_threshold: Confidence threshold for filtering.

        Returns:
            List of dicts per image: {
                'boxes': (K, 4) [x1,y1,x2,y2],
                'scores': (K,) confidence,
                'classes': (K,) class indices,
                'severity': (K,) severity scores [0, 10],
            }
        """
        # This is a simplified interface; full TRT execution requires
        # buffer allocation and CUDA stream management.
        # Actual deployment would use the Ultralytics TRT engine utilities.
        raise NotImplementedError(
            "Full TRT inference requires pycuda buffer management. "
            "Use `yolo predict model=best.engine` or implement with "
            "Ultralytics engine utilities for production deployment."
        )

    @staticmethod
    def postprocess_detections(
        raw_output: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> list[dict]:
        """Post-process raw TRT output to structured results.

        Handles score denormalization: pred * 10 → [0, 10] scale.

        Args:
            raw_output: (B, K, 7) raw detection output from TRT engine.
                Columns: [x1, y1, x2, y2, conf, cls_idx, severity_01]
            conf_threshold: Min confidence to keep.

        Returns:
            List of dicts per image with denormalized severity scores.
        """
        results = []
        for i in range(raw_output.shape[0]):
            dets = raw_output[i]  # (K, 7)

            # Filter by confidence
            mask = dets[:, 4] >= conf_threshold
            dets = dets[mask]

            results.append({
                "boxes": dets[:, :4],
                "scores": dets[:, 4],
                "classes": dets[:, 5].astype(int),
                "severity": dets[:, 6] * 10.0,  # denormalize to [0, 10]
            })
        return results

    def infer_multi_camera(
        self,
        camera_images: dict,
        conf_threshold: float = 0.25,
    ) -> dict:
        """Run batch inference on images from multiple cameras (§10.7 / A6).

        Batches all camera images into a single forward pass for max GPU utilization.

        Args:
            camera_images: Dict {camera_id: preprocessed_image (3, H, W) float32}.
            conf_threshold: Confidence threshold.

        Returns:
            Dict {camera_id: list[detection_dict]}.
        """
        cam_ids = list(camera_images.keys())
        if not cam_ids:
            return {}
        batch = np.stack([camera_images[cid] for cid in cam_ids], axis=0)
        raw_output = self._forward_batch(batch)
        results = self.postprocess_detections(raw_output, conf_threshold=conf_threshold)
        return {cid: results[i] for i, cid in enumerate(cam_ids)}

    def _forward_batch(self, images: np.ndarray) -> np.ndarray:
        """Execute inference on batch of images.

        Supports two backends:
        1. pycuda + tensorrt: Direct TRT engine execution (if available)
        2. YOLO predict: Fallback using ultralytics YOLO inference

        Args:
            images: Preprocessed batch (B, 3, H, W) float32, normalized [0, 1].

        Returns:
            Raw detection output (B, K, 7) ndarray = [x1,y1,x2,y2,conf,cls,severity].
        """
        # Try pycuda path first (production TRT deployment)
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
            import tensorrt as trt

            if hasattr(self, '_trt_context') and self._trt_context is not None:
                batch_size = images.shape[0]
                d_input = cuda.mem_alloc(images.nbytes)
                cuda.memcpy_htod(d_input, images.astype(np.float32).ravel())

                output_shape = self._trt_context.get_tensor_shape(
                    self._trt_context.engine.get_tensor_name(1))
                output_size = int(np.prod(output_shape)) * 4
                d_output = cuda.mem_alloc(output_size)

                self._trt_context.execute_v2([int(d_input), int(d_output)])

                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh(output, d_output)
                return output
        except (ImportError, AttributeError):
            pass

        # Fallback: Use YOLO predict API
        try:
            from ultralytics import YOLO
            import torch

            if hasattr(self, 'engine_path') and self.engine_path:
                model = YOLO(self.engine_path)
            elif hasattr(self, 'onnx_path') and self.onnx_path:
                model = YOLO(self.onnx_path)
            else:
                raise RuntimeError("No engine or ONNX model path available")

            tensor_batch = torch.from_numpy(images).float()
            results = model.predict(tensor_batch, verbose=False)

            # BUG-NEW-2 fix: Pad to uniform (B, max_dets, 7) ndarray.
            # postprocess_detections expects raw_output.shape[0] for batch loop.
            per_image = []
            max_dets = 0
            for r in results:
                if r.boxes is not None and r.boxes.data.shape[0] > 0:
                    d = r.boxes.data.cpu().numpy()
                    per_image.append(d)
                    max_dets = max(max_dets, d.shape[0])
                else:
                    per_image.append(np.zeros((0, 7), dtype=np.float32))

            max_dets = max(max_dets, 1)
            padded = np.zeros((len(per_image), max_dets, 7), dtype=np.float32)
            for i, det in enumerate(per_image):
                if det.shape[0] > 0:
                    ncols = min(det.shape[1], 7)
                    padded[i, :det.shape[0], :ncols] = det[:, :ncols]
            return padded
        except Exception as e:
            raise RuntimeError(
                f"Batch inference failed: {e}. "
                f"Install pycuda for direct TRT execution, or use "
                f"`yolo predict model=best.engine` for production."
            )


def validate_precision(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    engine_path: Optional[str | Path] = None,
    dataloader=None,
    imgsz: int = 640,
    num_samples: int = 50,
) -> dict:
    """Validate precision alignment across PyTorch → ONNX → TRT (§10.8).

    Compares detection outputs and score predictions between formats.

    Args:
        pytorch_model: Reference PyTorch model.
        onnx_path: ONNX model path.
        engine_path: Optional TensorRT engine path.
        dataloader: Validation dataloader (or uses random input).
        imgsz: Input image size.
        num_samples: Number of samples to validate.

    Returns:
        Dict of precision metrics:
        {'onnx_max_diff': float, 'onnx_score_mae': float, 'trt_max_diff': float, ...}
    """
    results = {}
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device

    # Generate test inputs
    if dataloader is not None:
        test_inputs = []
        for batch in dataloader:
            img = batch["img"].to(device).float() / 255.0
            test_inputs.append(img)
            if sum(x.shape[0] for x in test_inputs) >= num_samples:
                break
        test_input = torch.cat(test_inputs, dim=0)[:num_samples]
    else:
        test_input = torch.randn(min(num_samples, 4), 3, imgsz, imgsz, device=device)

    # PyTorch reference
    with torch.no_grad():
        pt_outputs = pytorch_model(test_input)

    # ONNX comparison
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        ort_outputs = sess.run(None, {"images": test_input.cpu().numpy()})

        pt_np = pt_outputs.cpu().numpy() if isinstance(pt_outputs, torch.Tensor) else pt_outputs[0].cpu().numpy()
        onnx_np = ort_outputs[0]

        # Match shapes (handle different numbers of detections)
        min_k = min(pt_np.shape[1], onnx_np.shape[1])
        diff = np.abs(pt_np[:, :min_k] - onnx_np[:, :min_k])
        results["onnx_max_diff"] = float(diff.max())

        # Score-specific comparison (last column)
        if pt_np.shape[-1] >= 7 and onnx_np.shape[-1] >= 7:
            score_diff = np.abs(pt_np[:, :min_k, -1] - onnx_np[:, :min_k, -1])
            results["onnx_score_mae"] = float(score_diff.mean())
            results["onnx_score_max_diff"] = float(score_diff.max())

        print(f"ONNX validation: max_diff={results['onnx_max_diff']:.6f}, "
              f"score_mae={results.get('onnx_score_mae', 'N/A')}")
    except Exception as e:
        print(f"ONNX validation skipped: {e}")
        results["onnx_error"] = str(e)

    # TRT comparison (if engine provided)
    if engine_path is not None:
        results["trt_status"] = "not_implemented"
        print("TRT precision validation requires pycuda — skipped")

    return results


def estimate_vram(
    model: nn.Module,
    batch_size: int = 1,
    imgsz: int = 640,
    fp16: bool = True,
) -> dict:
    """Estimate VRAM usage for TRT deployment (§10.6).

    Args:
        model: SevSeg-YOLO model.
        batch_size: Batch size for inference.
        imgsz: Input image size.
        fp16: Whether using FP16 precision.

    Returns:
        Dict with estimated VRAM breakdown in MB.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if fp16 else 4

    # Model weights
    model_mb = total_params * bytes_per_param / (1024 ** 2)

    # Input tensor
    input_mb = batch_size * 3 * imgsz * imgsz * bytes_per_param / (1024 ** 2)

    # Intermediate activations (rough estimate: 2x model size per batch)
    activation_mb = model_mb * 2 * batch_size

    # Score Head overhead (minimal, ~1% of model)
    score_params = sum(p.numel() for n, p in model.named_parameters() if "score_head" in n)
    score_mb = score_params * bytes_per_param / (1024 ** 2)

    total_mb = model_mb + input_mb + activation_mb

    return {
        "model_weights_mb": round(model_mb, 1),
        "input_mb": round(input_mb, 1),
        "activation_mb": round(activation_mb, 1),
        "score_head_mb": round(score_mb, 1),
        "total_estimated_mb": round(total_mb, 1),
        "precision": "FP16" if fp16 else "FP32",
        "batch_size": batch_size,
    }


# §10.6 alias: design doc uses estimate_gpu_memory
estimate_gpu_memory = estimate_vram


def deploy_scoreyolo(
    model_path: str,
    output_dir: str,
    dataloader=None,
    imgsz: int = 640,
    fp16: bool = True,
    pca_variance: float = 0.95,
    max_batch_size: int = 8,
    validate: bool = True,
) -> dict:
    """End-to-end SevSeg-YOLO deployment pipeline (§10.9 / A3).

    Steps: Load model -> PCA (optional) -> ONNX export -> TRT build -> Validate.

    Args:
        model_path: Trained SevSeg-YOLO .pt weights.
        output_dir: Directory for all deployment artifacts.
        dataloader: Optional training dataloader for PCA.
        imgsz: Input image size.
        fp16: Enable FP16 for TRT.
        pca_variance: PCA variance threshold (0 to skip).
        max_batch_size: Max batch for TRT dynamic shapes.
        validate: Run precision validation after build.

    Returns:
        Dict with paths and validation results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"steps": []}

    # Step 1: Load model
    print(f"[1/5] Loading model from {model_path}...")
    try:
        from ultralytics import YOLO
        yolo = YOLO(model_path)
        model = yolo.model
        model.eval()
        results["steps"].append("load_model")
    except Exception as e:
        return {"error": f"Load failed: {e}", "steps": results["steps"]}

    # Step 2: PCA
    pca = None
    if dataloader and pca_variance > 0:
        print(f"[2/5] Computing PCA (variance={pca_variance})...")
        try:
            pca = prepare_pca_matrices(model, dataloader, variance_threshold=pca_variance)
            pca_path = output_dir / "pca_matrices.npz"
            save_dict = {}
            for k, v in pca.items():
                save_dict[f"{k}_W"] = v["W"]
                save_dict[f"{k}_mean"] = v["mean"]
            np.savez(str(pca_path), **save_dict)
            results["pca_path"] = str(pca_path)
            results["steps"].append("pca")
        except Exception as e:
            print(f"  PCA skipped: {e}")
    else:
        print("[2/5] PCA skipped")

    # Step 3: ONNX
    print("[3/5] Exporting ONNX...")
    onnx_path = output_dir / "model.onnx"
    try:
        from sevseg_yolo.export import export_scoreyolo_onnx
        export_scoreyolo_onnx(model, onnx_path, imgsz=imgsz, pca_matrices=pca)
        results["onnx_path"] = str(onnx_path)
        results["steps"].append("onnx_export")
    except Exception as e:
        return {"error": f"ONNX failed: {e}", "steps": results["steps"]}

    # Step 4: TRT
    print(f"[4/5] Building TRT engine (fp16={fp16})...")
    engine_path = output_dir / "model.engine"
    try:
        build_trt_engine(onnx_path, engine_path, fp16=fp16, max_batch_size=max_batch_size)
        results["engine_path"] = str(engine_path)
        results["steps"].append("trt_build")
    except Exception as e:
        print(f"  TRT skipped: {e}")

    # Step 5: Validate
    if validate:
        print("[5/5] Validating precision...")
        try:
            ep = engine_path if engine_path.exists() else None
            val = validate_precision(model, onnx_path, ep, imgsz=imgsz)
            results["validation"] = val
            results["steps"].append("validation")
        except Exception as e:
            print(f"  Validation skipped: {e}")
    else:
        print("[5/5] Validation skipped")

    print(f"\nDone: {len(results['steps'])}/5 steps -> {output_dir}")
    return results
