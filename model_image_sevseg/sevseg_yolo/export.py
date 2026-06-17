"""SevSeg-YOLO ONNX export and verification (§12).

Provides:
    - SevSegYOLOExport: Wrapper that embeds PCA transforms as 1x1 Conv for export
    - export_scoreyolo_onnx(): Export model to ONNX with score + optional PCA features
    - verify_onnx(): Validate ONNX output against PyTorch reference

Output nodes:
    - det_output: Detection results (B, K, 7) = [x1,y1,x2,y2,conf,cls,severity]
    - feat_p3/p4/p5 (optional): PCA-compressed feature maps for Mask Generator

Usage:
    from sevseg_yolo.export import export_scoreyolo_onnx

    export_scoreyolo_onnx(model, output_path="model.onnx", imgsz=640)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class PCAConvModule(nn.Module):
    """PCA transform implemented as frozen 1x1 Conv2d for ONNX export.

    Converts a PCA projection matrix W and mean vector into a Conv2d layer
    that can be serialized into the ONNX computation graph.

    The PCA transform is: y = W^T @ (x - mean) = W^T @ x - W^T @ mean
    Implemented as: Conv2d(weight=W^T, bias=-W^T @ mean)

    Args:
        W: PCA projection matrix (C_in, pca_dim).
        mean: Channel-wise mean vector (C_in,).
    """

    def __init__(self, W: torch.Tensor, mean: torch.Tensor):
        super().__init__()
        C_in, pca_dim = W.shape
        conv = nn.Conv2d(C_in, pca_dim, kernel_size=1, bias=True)
        # W^T as 1x1 Conv weight: (pca_dim, C_in, 1, 1)
        conv.weight.data = W.T.float().unsqueeze(-1).unsqueeze(-1)
        # bias = -W^T @ mean
        conv.bias.data = (-W.T @ mean).float()
        conv.requires_grad_(False)
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PCA projection.

        Args:
            x: Input feature map (B, C_in, H, W).

        Returns:
            PCA-projected features (B, pca_dim, H, W).
        """
        return self.conv(x)


class SevSegYOLOExport(nn.Module):
    """SevSeg-YOLO export wrapper with optional PCA feature embedding.

    Wraps the trained model and optionally embeds PCA transforms as frozen
    1x1 Conv layers so exported ONNX can directly output compressed features.

    If no PCA matrices are provided, only the detection+score output is exported.

    Args:
        model: Trained SevSeg-YOLO model.
        pca_matrices: Optional dict of PCA parameters per scale:
            {'p3': {'W': Tensor(C3, dim3), 'mean': Tensor(C3)}, ...}
    """

    def __init__(self, model: nn.Module, pca_matrices: dict | None = None):
        super().__init__()
        self.model = model
        self.model.eval()
        self._hooked_features = {}
        self._hooks = []

        # Build PCA conv modules
        self.pca_convs = nn.ModuleDict()
        if pca_matrices:
            for name, pca in pca_matrices.items():
                W = pca["W"] if isinstance(pca["W"], torch.Tensor) else torch.tensor(pca["W"])
                mean = pca["mean"] if isinstance(pca["mean"], torch.Tensor) else torch.tensor(pca["mean"])
                self.pca_convs[name] = PCAConvModule(W, mean)

        # Register forward hooks to capture P3/P4/P5 features
        if pca_matrices:
            self._register_feature_hooks()

    def _register_feature_hooks(self):
        """Register hooks on the model to capture intermediate feature maps.

        Hooks are registered on the ScoreDetect module's forward_head method
        to capture the input features (P3/P4/P5) BEFORE they are processed.
        This works even after fuse() because we hook the head module itself,
        not the score_head sub-module (which may be None after fuse).
        """
        from ultralytics.nn.modules.head import ScoreDetect
        scale_names = ["p3", "p4", "p5"]

        for module in self.model.modules():
            if isinstance(module, ScoreDetect):
                # Hook on ScoreDetect itself to capture input feature list x
                def make_head_hook():
                    def hook(mod, input_args, output):
                        # ScoreDetect.forward receives x: list[Tensor] (P3/P4/P5)
                        if isinstance(input_args, tuple) and len(input_args) > 0:
                            x = input_args[0]
                            if isinstance(x, list):
                                for i, f in enumerate(x):
                                    if i < len(scale_names):
                                        self._hooked_features[scale_names[i]] = f
                    return hook

                h = module.register_forward_hook(make_head_hook())
                self._hooks.append(h)

                # Also try score_head if available (pre-fuse, higher quality hook)
                if hasattr(module, "score_head") and module.score_head is not None:
                    def make_score_hook():
                        def hook(mod, input_args, output):
                            if isinstance(input_args, tuple) and len(input_args) > 0:
                                feat = input_args[0]
                                if isinstance(feat, list):
                                    for i, f in enumerate(feat):
                                        if i < len(scale_names):
                                            self._hooked_features[scale_names[i]] = f
                        return hook
                    h2 = module.score_head.register_forward_hook(make_score_hook())
                    self._hooks.append(h2)
                break

    def forward(self, x: torch.Tensor):
        """Forward pass for export.

        Returns:
            If no PCA: just detection output (B, K, 7)
            If PCA: (det_output, feat_p3_pca, feat_p4_pca, feat_p5_pca)
        """
        self._hooked_features.clear()
        y = self.model(x)  # (B, K, 7) in inference mode

        if self.pca_convs:
            pca_outputs = []
            for name in ["p3", "p4", "p5"]:
                if name in self._hooked_features and name in self.pca_convs:
                    pca_outputs.append(self.pca_convs[name](self._hooked_features[name]))
                else:
                    pca_outputs.append(None)
            # Filter None and return as tuple
            valid_outputs = [o for o in pca_outputs if o is not None]
            if valid_outputs:
                return (y, *valid_outputs)

        return y

    def __del__(self):
        for h in self._hooks:
            h.remove()


def export_scoreyolo_onnx(
    model: nn.Module,
    output_path: str | Path,
    imgsz: int = 640,
    opset: int = 17,
    pca_matrices: dict | None = None,
    dynamic_batch: bool = True,
):
    """Export SevSeg-YOLO to ONNX format (§12.2).

    Args:
        model: Trained SevSeg-YOLO model.
        output_path: Output ONNX file path.
        imgsz: Input image size.
        opset: ONNX opset version.
        pca_matrices: Optional PCA parameters for feature export.
        dynamic_batch: Enable dynamic batch size.
    """
    output_path = Path(output_path)
    export_model = SevSegYOLOExport(model, pca_matrices)
    export_model.eval()

    dummy = torch.randn(1, 3, imgsz, imgsz)

    # Determine output names
    output_names = ["det_output"]
    dynamic_axes = {"images": {0: "batch"}, "det_output": {0: "batch", 1: "num_dets"}}

    if pca_matrices:
        for name in ["feat_p3", "feat_p4", "feat_p5"]:
            output_names.append(name)
            dynamic_axes[name] = {0: "batch"}

    if not dynamic_batch:
        dynamic_axes = None

    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"SevSeg-YOLO ONNX exported to {output_path}")
    print(f"  Output nodes: {output_names}")
    return output_path


def verify_onnx(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    imgsz: int = 640,
    atol: float = 1e-4,
) -> bool:
    """Verify ONNX output matches PyTorch output (§12.3).

    Args:
        pytorch_model: Trained PyTorch model (eval mode).
        onnx_path: Path to exported ONNX model.
        imgsz: Input image size.
        atol: Absolute tolerance for comparison.

    Returns:
        True if all outputs pass.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping ONNX verification.")
        return False

    import numpy as np

    pytorch_model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)

    with torch.no_grad():
        pt_out = pytorch_model(dummy)

    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(None, {"images": dummy.numpy()})

    all_pass = True

    # Compare detection output
    if isinstance(pt_out, torch.Tensor):
        pt_np = pt_out.numpy()
    elif isinstance(pt_out, (list, tuple)):
        pt_np = pt_out[0].numpy() if isinstance(pt_out[0], torch.Tensor) else pt_out[0]
    else:
        pt_np = pt_out

    diff = np.abs(pt_np - ort_out[0]).max()
    status = "PASS" if diff < atol else "FAIL"
    print(f"Detection output: max_diff={diff:.6f} [{status}]")
    if status == "FAIL":
        all_pass = False

    # Score range validation: last column of det output should be in [0, 1]
    det_out = ort_out[0]
    if det_out.shape[-1] >= 7:
        score_col = det_out[..., -1]
        score_min, score_max = score_col.min(), score_col.max()
        score_ok = score_min >= -0.01 and score_max <= 1.01
        print(f"Score range: [{score_min:.4f}, {score_max:.4f}] [{'PASS' if score_ok else 'FAIL'}]")
        if not score_ok:
            all_pass = False

    return all_pass
