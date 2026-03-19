"""SevSeg-YOLO MaskGenerator V2 — Multi-scale EigenCAM + Edge-guided upsampling.

Improvements over V1:
    1. Top-K channel selection (variance-based) replaces full L2 norm
    2. Layer2 (stride=4) for n/s scales → 2x resolution vs P3
    3. Canny edge-guided upsampling → edge-aware without over-constraining boundaries
    4. Adaptive threshold replaces Otsu → robust to non-bimodal distributions
    5. Close-then-Open morphology → preserves mask integrity

Algorithm for edge-guided upsampling (Step 4):
    - Canny edge detection on the original image crop → binary edge map
    - Dilate edges (3x3) to create an "edge zone"
    - In edge zones: preserve the raw bilinear-upsampled activation (sharp boundaries)
    - In smooth zones: apply Gaussian blur to suppress noise
    - Result: clean mask edges that follow feature activation, not image texture

This approach outperforms cv2.ximgproc.guidedFilter for industrial defect scenarios
because defect boundaries are determined by learned features, not image texture edges.
Guided Filter tends to over-constrain mask edges to texture boundaries, which can
clip valid defect regions or introduce false edges in textured areas.

Constraints satisfied:
    - No mask annotations needed (pure post-processing)
    - No SAM/SAM2 (latency unacceptable)
    - <10ms per mask on CPU
    - Compatible with all 5 scales (n/s/m/l/x)
    - No opencv-contrib dependency (uses only opencv-python)

Usage:
    from sevseg_yolo.mask_generator_v2 import MaskGeneratorV2

    mg = MaskGeneratorV2()
    mask = mg.generate(
        bbox=[100, 50, 200, 150],
        feat_layer2=...,  # (C, 160, 160) or None
        feat_p3_fpn=...,  # (C, 80, 80)
        feat_p4_fpn=...,  # (C, 40, 40)
        feat_p5_fpn=...,  # (C, 20, 20) or None
        original_image=...,  # (H, W, 3) uint8
        input_hw=(640, 640),
    )
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class MaskGeneratorV2:
    """Multi-scale EigenCAM + edge-guided upsampling mask generator.

    Scale-adaptive feature selection:
        n/s (channels ≤ 128): Layer2(s=4) + P3(s=8) + P4(s=16)
        m/l/x (channels > 128): P3(s=8) + P4(s=16) + P5(s=32)

    Edge-guided upsampling (Canny-blend):
        Uses Canny edge detection on the original image crop to create an edge mask.
        Edge regions preserve sharp activation values; smooth regions get Gaussian blur.
        This approach is superior to Guided Filter for industrial defects because it
        lets learned features determine mask boundaries rather than image texture.

    Args:
        topk_channels: Number of highest-variance channels to select. Default 32.
        canny_low: Canny edge detector low threshold. Default 50.
        canny_high: Canny edge detector high threshold. Default 150.
        edge_dilate_k: Edge dilation kernel size. Default 3.
        smooth_ksize: Gaussian blur kernel size for smooth regions. Default 5.
        smooth_sigma: Gaussian blur sigma for smooth regions. Default 2.0.
        adaptive_block: Adaptive threshold block size (odd). Default 11.
        adaptive_C: Adaptive threshold bias (negative → more foreground). Default -3.
        morph_close_k: Close kernel size (fill holes). Default 5.
        morph_open_k: Open kernel size (remove noise). Default 3.
        min_adaptive_size: Min bbox side for adaptive threshold; below uses Otsu. Default 15.
        channel_threshold: Max channels for Layer2 usage (above → skip Layer2). Default 128.
    """

    def __init__(
        self,
        topk_channels: int = 32,
        canny_low: int = 50,
        canny_high: int = 150,
        edge_dilate_k: int = 3,
        smooth_ksize: int = 5,
        smooth_sigma: float = 2.0,
        adaptive_block: int = 11,
        adaptive_C: float = -3,
        morph_close_k: int = 5,
        morph_open_k: int = 3,
        min_adaptive_size: int = 15,
        channel_threshold: int = 128,
        # Legacy params (accepted but ignored for backward compatibility)
        guided_radius: int = 4,
        guided_eps: float = 0.01,
    ):
        if cv2 is None:
            raise ImportError("MaskGeneratorV2 requires opencv-python")
        self.topk_channels = topk_channels
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.edge_dilate_k = edge_dilate_k
        self.smooth_ksize = smooth_ksize
        self.smooth_sigma = smooth_sigma
        self.adaptive_block = adaptive_block
        self.adaptive_C = adaptive_C
        self.morph_close_k = morph_close_k
        self.morph_open_k = morph_open_k
        self.min_adaptive_size = min_adaptive_size
        self.channel_threshold = channel_threshold

    # ── Activation ──────────────────────────────────────

    def _topk_l2_activation(self, feat: np.ndarray) -> np.ndarray:
        """Top-K channel selection + L2 norm activation.

        Select K channels with highest spatial variance, then compute L2 norm
        only on those channels. High-variance channels are most discriminative.

        Args:
            feat: (C, H, W) feature map crop.

        Returns:
            (H, W) normalized activation in [0, 1].
        """
        C, H, W = feat.shape
        K = min(self.topk_channels, C)

        # Channel variance: for each channel, compute spatial variance
        var_per_ch = feat.reshape(C, -1).var(axis=1)  # (C,)

        # Select top-K channels by variance
        topk_idx = np.argpartition(var_per_ch, -K)[-K:]
        feat_selected = feat[topk_idx]  # (K, H, W)

        # L2 norm on selected channels
        activation = np.sqrt(np.sum(feat_selected ** 2, axis=0))  # (H, W)

        # Normalize to [0, 1]
        a_min, a_max = activation.min(), activation.max()
        if a_max > a_min:
            return ((activation - a_min) / (a_max - a_min)).astype(np.float32)
        return np.zeros((H, W), dtype=np.float32)

    # ── Upsampling ──────────────────────────────────────

    def _edge_guided_upsample(self, low_res: np.ndarray, guide_crop: np.ndarray,
                              target_h: int, target_w: int) -> np.ndarray:
        """Edge-aware upsample using Canny edge-guided blending.

        Strategy:
            1. Bilinear upsample the low-res activation to target size
            2. Detect edges in the original image crop (Canny)
            3. Dilate edges to create an "edge zone"
            4. Edge zone: keep raw upsampled values (preserves sharp boundaries)
            5. Smooth zone: apply Gaussian blur (suppresses activation noise)

        This is superior to cv2.ximgproc.guidedFilter for industrial defects:
        - Guided Filter forces mask edges to follow image texture edges, which can
          clip valid defect regions or create false boundaries in textured areas
        - Edge-guided blending lets learned feature activations determine the mask
          shape, while only using image edges to decide where to keep sharpness

        Args:
            low_res: (h, w) float32 activation in [0, 1].
            guide_crop: (H, W, 3) uint8 original image crop at bbox location.
            target_h: Output height (bbox height).
            target_w: Output width (bbox width).

        Returns:
            (target_h, target_w) float32 activation in [0, 1].
        """
        up = cv2.resize(low_res, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        guide_resized = cv2.resize(guide_crop, (target_w, target_h))

        # Canny edge detection → binary edge map
        edges = cv2.Canny(guide_resized, self.canny_low, self.canny_high)
        edges_f = edges.astype(np.float32) / 255.0

        # Dilate edges to create wider "edge zone"
        dilate_kernel = np.ones((self.edge_dilate_k, self.edge_dilate_k), np.uint8)
        edges_d = cv2.dilate(edges_f, dilate_kernel)

        # Gaussian blur for smooth regions
        ksize = self.smooth_ksize if self.smooth_ksize % 2 == 1 else self.smooth_ksize + 1
        smoothed = cv2.GaussianBlur(up, (ksize, ksize), self.smooth_sigma)

        # Blend: edge zone keeps raw values (sharp), smooth zone gets blurred (clean)
        result = up * edges_d + smoothed * (1.0 - edges_d)

        return np.clip(result, 0.0, 1.0)

    # ── Crop utility ────────────────────────────────────

    @staticmethod
    def _crop_feat(feat: np.ndarray, bbox, input_hw) -> np.ndarray | None:
        """Map bbox to feature map coordinates and crop."""
        C, fh, fw = feat.shape
        ih, iw = input_hw
        sx, sy = fw / iw, fh / ih
        fx1 = max(0, int(bbox[0] * sx))
        fy1 = max(0, int(bbox[1] * sy))
        fx2 = min(fw, int(bbox[2] * sx) + 1)
        fy2 = min(fh, int(bbox[3] * sy) + 1)
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        return feat[:, fy1:fy2, fx1:fx2]

    # ── Main generate ───────────────────────────────────

    def generate(
        self,
        bbox: list | tuple | np.ndarray,
        feat_layer2: np.ndarray | None,
        feat_p3_fpn: np.ndarray,
        feat_p4_fpn: np.ndarray,
        feat_p5_fpn: np.ndarray | None = None,
        original_image: np.ndarray = None,
        input_hw: tuple[int, int] = (640, 640),
        guide_crop: np.ndarray = None,
    ) -> np.ndarray:
        """Generate pixel-level binary mask for a single detection.

        Args:
            bbox: [x1, y1, x2, y2] in model input space coordinates.
            feat_layer2: Backbone layer 2, (C, H, W). None if m/l/x.
            feat_p3_fpn: FPN P3, (C, H, W), stride=8.
            feat_p4_fpn: FPN P4, (C, H, W), stride=16.
            feat_p5_fpn: FPN P5, (C, H, W), stride=32. None if n/s.
            original_image: (H, W, 3) uint8, deprecated — use guide_crop instead.
            input_hw: Model input size (H, W), inferred from P3 if not given.
            guide_crop: (crop_H, crop_W, 3) uint8, original image crop at bbox
                location in original coordinates. Used as Guided Filter guide.
                This is the recommended way to pass the guide image.

        Returns:
            Binary mask (bbox_h, bbox_w) uint8 {0, 1}.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_h, bbox_w = y2 - y1, x2 - x1
        if bbox_h <= 0 or bbox_w <= 0:
            return np.zeros((max(bbox_h, 1), max(bbox_w, 1)), dtype=np.uint8)

        # ── Step 1: Scale-adaptive feature selection ──
        # Decide whether to use Layer2 based on channel count
        use_layer2 = (feat_layer2 is not None and feat_layer2.shape[0] <= self.channel_threshold)

        if use_layer2:
            features = [feat_layer2, feat_p3_fpn, feat_p4_fpn]
            weights = [0.35, 0.35, 0.30]
        elif feat_p5_fpn is not None:
            features = [feat_p3_fpn, feat_p4_fpn, feat_p5_fpn]
            weights = [0.45, 0.35, 0.20]
        else:
            features = [feat_p3_fpn, feat_p4_fpn]
            weights = [0.55, 0.45]

        # ── Step 2: Crop + Top-K activation per scale ──
        activations = []
        for feat in features:
            crop = self._crop_feat(feat, bbox, input_hw)
            if crop is not None and crop.shape[1] >= 2 and crop.shape[2] >= 2:
                activations.append(self._topk_l2_activation(crop))
            else:
                activations.append(None)

        valid = [(a, w) for a, w in zip(activations, weights) if a is not None]
        if not valid:
            return np.zeros((bbox_h, bbox_w), dtype=np.uint8)

        # ── Step 3: Upsample to highest resolution + weighted fusion ──
        target_h, target_w = valid[0][0].shape
        fused = np.zeros((target_h, target_w), dtype=np.float32)
        w_sum = 0.0
        for act, w in valid:
            resized = cv2.resize(act, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            fused += w * resized
            w_sum += w
        if w_sum > 0:
            fused /= w_sum

        # ── Step 4: Edge-guided upsample to bbox size ──
        # Priority: guide_crop (correctly cropped from original image) > original_image
        if guide_crop is not None and guide_crop.size > 0:
            fused = self._edge_guided_upsample(fused, guide_crop, bbox_h, bbox_w)
        elif original_image is not None:
            img_h, img_w = original_image.shape[:2]
            cy1 = max(0, min(y1, img_h))
            cy2 = max(0, min(y2, img_h))
            cx1 = max(0, min(x1, img_w))
            cx2 = max(0, min(x2, img_w))
            img_crop = original_image[cy1:cy2, cx1:cx2]
            if img_crop.size > 0:
                fused = self._edge_guided_upsample(fused, img_crop, bbox_h, bbox_w)
            else:
                fused = cv2.resize(fused, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
        else:
            fused = cv2.resize(fused, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)

        # ── Step 5: Adaptive binarization ──
        fused_u8 = (np.clip(fused, 0, 1) * 255).astype(np.uint8)
        if min(bbox_h, bbox_w) >= self.min_adaptive_size:
            blk = min(self.adaptive_block, (min(bbox_h, bbox_w) // 2) * 2 - 1)
            blk = max(blk, 3)
            mask = cv2.adaptiveThreshold(
                fused_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blk, int(self.adaptive_C))
        else:
            _, mask = cv2.threshold(fused_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── Step 6: Morphology (close then open) ──
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_close_k, self.morph_close_k))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_open_k, self.morph_open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        return (mask > 127).astype(np.uint8)

    def generate_batch(
        self,
        bboxes: np.ndarray,
        feat_layer2: np.ndarray | None,
        feat_p3_fpn: np.ndarray,
        feat_p4_fpn: np.ndarray,
        feat_p5_fpn: np.ndarray | None = None,
        original_image: np.ndarray = None,
        input_hw: tuple[int, int] = (640, 640),
    ) -> list[np.ndarray]:
        """Generate masks for multiple detections."""
        return [
            self.generate(bbox, feat_layer2, feat_p3_fpn, feat_p4_fpn,
                          feat_p5_fpn, original_image, input_hw)
            for bbox in bboxes
        ]
