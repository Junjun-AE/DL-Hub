"""SevSeg-YOLO MaskGenerator V3 — Configurable improvements over V2.

Same coordinate system as V2 (model input space → model.py resizes to orig).
Same API as V2 (drop-in replacement, no change to model.py needed).

Configurable improvements (all can be toggled on/off for A/B testing):

  Direction 1 — Better signal (make defects more visible in activation):
    [A] Channel selection: "variance" (V2) vs "bimodal" (V3)
        bimodal: pick channels where bbox region has two distinct value clusters
    [B] Channel weighting: "equal" (V2 L2) vs "contrast" (V3)
        contrast: weight each channel by its internal contrast ratio
    [C] Normalization: "minmax" (V2) vs "percentile" (V3)
        percentile: clip to [2%, 98%] before normalizing, robust to outliers

  Direction 2 — Better extraction (sharper mask edges):
    [D] Upsampling guidance: "canny" (V2) vs "sobel" (V3)
        sobel: continuous gradient magnitude instead of binary edges
    [E] Post-threshold refinement: off (V2) vs "gradient_snap" (V3)
        gradient_snap: align mask edges to nearest gradient peak in original image

Usage in GUI: user picks options, app.py passes them as constructor params.
"""

from __future__ import annotations
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class MaskGeneratorV3:
    """Configurable mask generator with V2-compatible API.

    All parameters have V2-equivalent defaults, so MaskGeneratorV3() with no
    args behaves identically to V2.

    Args:
        topk_channels: Number of channels to select. Default 48.
        channel_select: "variance" (V2) or "bimodal" (V3 improved).
        channel_weight: "equal" (V2 L2 norm) or "contrast" (V3 weighted).
        normalize_mode: "minmax" (V2) or "percentile" (V3 robust).
        upsample_guide: "canny" (V2) or "sobel" (V3 continuous gradient).
        gradient_snap: Enable post-threshold edge alignment. Default False.
        adaptive_block: Adaptive threshold block size. Default 15.
        adaptive_C: Adaptive threshold constant. Default -5.
        morph_close_k: Close kernel. Default 7.
        morph_open_k: Open kernel. Default 3.
        min_adaptive_size: Min bbox side for adaptive threshold. Default 15.
        channel_threshold: Max channels for Layer2 usage. Default 128.
    """

    def __init__(
        self,
        topk_channels: int = 48,
        channel_select: str = "variance",
        channel_weight: str = "equal",
        normalize_mode: str = "minmax",
        upsample_guide: str = "canny",
        gradient_snap: bool = False,
        adaptive_block: int = 15,
        adaptive_C: float = -5,
        morph_close_k: int = 7,
        morph_open_k: int = 3,
        min_adaptive_size: int = 15,
        channel_threshold: int = 128,
        # V2 legacy params (used when upsample_guide="canny")
        canny_low: int = 50,
        canny_high: int = 150,
        edge_dilate_k: int = 3,
        smooth_ksize: int = 5,
        smooth_sigma: float = 2.0,
        # Legacy compat (ignored)
        guided_radius: int = 4,
        guided_eps: float = 0.01,
    ):
        if cv2 is None:
            raise ImportError("MaskGeneratorV3 requires opencv-python")

        self.topk_channels = topk_channels
        self.channel_select = channel_select
        self.channel_weight = channel_weight
        self.normalize_mode = normalize_mode
        self.upsample_guide = upsample_guide
        self.gradient_snap = gradient_snap
        self.adaptive_block = adaptive_block
        self.adaptive_C = adaptive_C
        self.morph_close_k = morph_close_k
        self.morph_open_k = morph_open_k
        self.min_adaptive_size = min_adaptive_size
        self.channel_threshold = channel_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.edge_dilate_k = edge_dilate_k
        self.smooth_ksize = smooth_ksize
        self.smooth_sigma = smooth_sigma

    # ════════════════════════════════════════════════════════
    # Step 2: Activation — Channel selection + combination
    # ════════════════════════════════════════════════════════

    def _compute_activation(self, feat: np.ndarray) -> np.ndarray:
        """Compute (H, W) activation from (C, H, W) feature crop.

        Dispatches to selected channel_select + channel_weight method.

        Returns:
            (H, W) float32 in [0, 1].
        """
        C, H, W = feat.shape
        if C == 0 or H == 0 or W == 0:
            return np.zeros((max(H, 1), max(W, 1)), dtype=np.float32)

        K = min(self.topk_channels, C)

        # ── Channel selection ──
        if self.channel_select == "bimodal":
            selected_idx, scores = self._select_bimodal(feat, K)
        else:
            selected_idx, scores = self._select_variance(feat, K)

        feat_sel = feat[selected_idx]  # (K, H, W)

        # ── Channel combination ──
        if self.channel_weight == "contrast":
            activation = self._combine_contrast(feat_sel, scores)
        else:
            activation = self._combine_l2(feat_sel)

        # ── Normalization ──
        if self.normalize_mode == "percentile":
            return self._normalize_percentile(activation)
        else:
            return self._normalize_minmax(activation)

    def _select_variance(self, feat, K):
        """V2 original: select by spatial variance."""
        C = feat.shape[0]
        var_per_ch = feat.reshape(C, -1).var(axis=1)
        idx = np.argpartition(var_per_ch, -K)[-K:]
        return idx, var_per_ch[idx]

    def _select_bimodal(self, feat, K):
        """V3: select channels with strongest bimodal separation in bbox.

        For each channel, compute the gap between the mean of top-30% pixels
        and bottom-30% pixels. Channels where this gap is large have clear
        defect/normal separation within the bbox.
        """
        C, H, W = feat.shape
        flat = feat.reshape(C, -1)  # (C, H*W)
        n = flat.shape[1]

        if n < 4:
            return self._select_variance(feat, K)

        n30 = max(1, n * 3 // 10)
        # Partition: bottom-30% and top-30% means per channel
        sorted_flat = np.sort(flat, axis=1)
        low_mean = sorted_flat[:, :n30].mean(axis=1)    # (C,)
        high_mean = sorted_flat[:, -n30:].mean(axis=1)   # (C,)
        bimodal_gap = high_mean - low_mean                # (C,)

        idx = np.argpartition(bimodal_gap, -K)[-K:]
        return idx, bimodal_gap[idx]

    def _combine_l2(self, feat_sel):
        """V2 original: L2 norm across channels (equal weight)."""
        return np.sqrt(np.sum(feat_sel ** 2, axis=0))

    def _combine_contrast(self, feat_sel, scores):
        """V3: weighted combination by contrast score.

        Channels with higher contrast scores contribute more.
        """
        K = feat_sel.shape[0]
        if scores.sum() < 1e-8:
            return self._combine_l2(feat_sel)

        weights = scores / scores.sum()  # normalize to sum=1
        # Weighted L2: sqrt(sum(w_i * f_i^2))
        weighted_sq = np.zeros_like(feat_sel[0], dtype=np.float64)
        for i in range(K):
            weighted_sq += weights[i] * (feat_sel[i].astype(np.float64) ** 2)
        return np.sqrt(weighted_sq).astype(np.float32)

    def _normalize_minmax(self, act):
        """V2 original: min-max to [0, 1]."""
        a_min, a_max = act.min(), act.max()
        if a_max > a_min:
            return ((act - a_min) / (a_max - a_min)).astype(np.float32)
        return np.zeros_like(act, dtype=np.float32)

    def _normalize_percentile(self, act, lo=2, hi=98):
        """V3: percentile-clipped normalization, robust to outliers."""
        p_lo = np.percentile(act, lo)
        p_hi = np.percentile(act, hi)
        if p_hi > p_lo + 1e-8:
            result = np.clip((act - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            return result.astype(np.float32)
        return self._normalize_minmax(act)

    # ════════════════════════════════════════════════════════
    # Step 4: Upsampling guidance
    # ════════════════════════════════════════════════════════

    def _upsample_canny(self, low_res, guide_crop, target_h, target_w):
        """V2 original: Canny binary edge guidance."""
        up = cv2.resize(low_res, (target_w, target_h),
                        interpolation=cv2.INTER_LINEAR)
        guide_resized = cv2.resize(guide_crop, (target_w, target_h))
        edges = cv2.Canny(guide_resized, self.canny_low, self.canny_high)
        edges_f = edges.astype(np.float32) / 255.0
        dilate_k = np.ones((self.edge_dilate_k, self.edge_dilate_k), np.uint8)
        edges_d = cv2.dilate(edges_f, dilate_k)
        ksize = self.smooth_ksize | 1  # ensure odd
        smoothed = cv2.GaussianBlur(up, (ksize, ksize), self.smooth_sigma)
        result = up * edges_d + smoothed * (1.0 - edges_d)
        return np.clip(result, 0.0, 1.0)

    def _upsample_sobel(self, low_res, guide_crop, target_h, target_w):
        """V3: Sobel continuous gradient guidance.

        Uses gradient magnitude (continuous 0-1) instead of binary edges.
        Combines image gradient and activation gradient for dual guidance.
        """
        up = cv2.resize(low_res, (target_w, target_h),
                        interpolation=cv2.INTER_LINEAR)
        guide_resized = cv2.resize(guide_crop, (target_w, target_h))

        # Image gradient
        gray = cv2.cvtColor(guide_resized, cv2.COLOR_BGR2GRAY) \
            if guide_resized.ndim == 3 else guide_resized
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        img_grad = np.sqrt(gx**2 + gy**2)
        g_max = img_grad.max()
        img_grad = (img_grad / g_max).astype(np.float32) if g_max > 0 else img_grad

        # Activation gradient
        up_u8 = (np.clip(up, 0, 1) * 255).astype(np.uint8)
        ax = cv2.Sobel(up_u8, cv2.CV_32F, 1, 0, ksize=3)
        ay = cv2.Sobel(up_u8, cv2.CV_32F, 0, 1, ksize=3)
        act_grad = np.sqrt(ax**2 + ay**2)
        a_max = act_grad.max()
        act_grad = (act_grad / a_max).astype(np.float32) if a_max > 0 else act_grad

        # Combined weight: max of both gradients, slightly blurred
        edge_w = np.maximum(img_grad, act_grad)
        edge_w = cv2.GaussianBlur(edge_w, (3, 3), 0.5)
        edge_w = np.clip(edge_w, 0.0, 1.0)

        # Smooth zones
        ksize = self.smooth_ksize | 1
        smoothed = cv2.GaussianBlur(up, (ksize, ksize), self.smooth_sigma)

        # Continuous blend
        return np.clip(up * edge_w + smoothed * (1.0 - edge_w), 0.0, 1.0)

    # ════════════════════════════════════════════════════════
    # Step 5 extension: Gradient snap refinement
    # ════════════════════════════════════════════════════════

    def _gradient_snap(self, mask_u8, guide_crop, target_h, target_w):
        """V3: Snap mask edges to nearest image gradient peaks.

        After initial binarization, the mask edges may not align perfectly
        with the actual defect boundary in the original image. This step
        uses the image gradient to pull mask edges to the nearest strong
        gradient (= true edge in original image).
        """
        guide_resized = cv2.resize(guide_crop, (target_w, target_h))
        gray = cv2.cvtColor(guide_resized, cv2.COLOR_BGR2GRAY) \
            if guide_resized.ndim == 3 else guide_resized

        # Compute gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        g_max = grad_mag.max()
        if g_max > 0:
            grad_mag = (grad_mag / g_max * 255).astype(np.uint8)
        else:
            return mask_u8

        # Find strong gradient regions (potential edges)
        _, strong_edges = cv2.threshold(grad_mag, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find mask contours
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_u8

        # For each contour, dilate slightly then AND with strong edges
        # This pulls the contour to align with image edges
        refined = np.zeros_like(mask_u8)

        # Dilate mask slightly to reach nearby edges
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(mask_u8, dilate_k)

        # Core = original mask eroded slightly
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        core = cv2.erode(mask_u8, erode_k)

        # Border zone = dilated - core
        border = cv2.subtract(dilated, core)

        # In border zone, keep pixels where image has strong gradient
        border_refined = cv2.bitwise_and(border, strong_edges)

        # Final = core + refined border
        refined = cv2.bitwise_or(core, border_refined)

        return refined

    # ════════════════════════════════════════════════════════
    # Crop utility (same as V2)
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _crop_feat(feat, bbox, input_hw):
        """Map bbox to feature map coordinates and crop (same as V2)."""
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

    # ════════════════════════════════════════════════════════
    # Main generate (same structure as V2)
    # ════════════════════════════════════════════════════════

    def generate(
        self,
        bbox,
        feat_layer2=None,
        feat_p3_fpn=None,
        feat_p4_fpn=None,
        feat_p5_fpn=None,
        original_image=None,
        input_hw=(640, 640),
        guide_crop=None,
    ) -> np.ndarray:
        """Generate mask — same API as V2, same coordinate system.

        Input bbox is in MODEL INPUT SPACE (640x640 with letterbox).
        Output mask is (bbox_h, bbox_w) in model input space.
        model.py will resize to original image bbox size.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_h, bbox_w = y2 - y1, x2 - x1
        if bbox_h <= 0 or bbox_w <= 0:
            return np.zeros((max(bbox_h, 1), max(bbox_w, 1)), dtype=np.uint8)

        # ── Step 1: Scale-adaptive feature selection (same as V2) ──
        use_layer2 = (feat_layer2 is not None and
                      feat_layer2.shape[0] <= self.channel_threshold)
        if use_layer2:
            features = [feat_layer2, feat_p3_fpn, feat_p4_fpn]
            weights = [0.35, 0.35, 0.30]
        elif feat_p5_fpn is not None:
            features = [feat_p3_fpn, feat_p4_fpn, feat_p5_fpn]
            weights = [0.45, 0.35, 0.20]
        else:
            features = [feat_p3_fpn, feat_p4_fpn]
            weights = [0.55, 0.45]

        # ── Step 2: Crop + activation per scale (V3 configurable) ──
        activations = []
        for feat in features:
            crop = self._crop_feat(feat, bbox, input_hw)
            if crop is not None and crop.shape[1] >= 2 and crop.shape[2] >= 2:
                activations.append(self._compute_activation(crop))
            else:
                activations.append(None)

        valid = [(a, w) for a, w in zip(activations, weights) if a is not None]
        if not valid:
            return np.zeros((bbox_h, bbox_w), dtype=np.uint8)

        # ── Step 3: Multi-scale fusion (same as V2) ──
        target_h, target_w = valid[0][0].shape
        fused = np.zeros((target_h, target_w), dtype=np.float32)
        w_sum = 0.0
        for act, w in valid:
            resized = cv2.resize(act, (target_w, target_h),
                                 interpolation=cv2.INTER_LINEAR)
            fused += w * resized
            w_sum += w
        if w_sum > 0:
            fused /= w_sum

        # ── Step 4: Guided upsample (V3 configurable) ──
        if guide_crop is not None and guide_crop.size > 0:
            if self.upsample_guide == "sobel":
                fused = self._upsample_sobel(fused, guide_crop, bbox_h, bbox_w)
            else:
                fused = self._upsample_canny(fused, guide_crop, bbox_h, bbox_w)
        elif original_image is not None:
            ih, iw2 = original_image.shape[:2]
            cy1, cy2 = max(0, min(y1, ih)), max(0, min(y2, ih))
            cx1, cx2 = max(0, min(x1, iw2)), max(0, min(x2, iw2))
            img_crop = original_image[cy1:cy2, cx1:cx2]
            if img_crop.size > 0:
                if self.upsample_guide == "sobel":
                    fused = self._upsample_sobel(fused, img_crop, bbox_h, bbox_w)
                else:
                    fused = self._upsample_canny(fused, img_crop, bbox_h, bbox_w)
            else:
                fused = cv2.resize(fused, (bbox_w, bbox_h),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            fused = cv2.resize(fused, (bbox_w, bbox_h),
                               interpolation=cv2.INTER_LINEAR)

        # ── Step 5: Adaptive binarization (same as V2) ──
        fused_u8 = (np.clip(fused, 0, 1) * 255).astype(np.uint8)
        if min(bbox_h, bbox_w) >= self.min_adaptive_size:
            blk = min(self.adaptive_block,
                      (min(bbox_h, bbox_w) // 2) * 2 - 1)
            blk = max(blk, 3)
            mask = cv2.adaptiveThreshold(
                fused_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blk, int(self.adaptive_C))
        else:
            _, mask = cv2.threshold(
                fused_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── Step 5.5: Gradient snap refinement (V3 optional) ──
        if self.gradient_snap and guide_crop is not None and guide_crop.size > 0:
            mask = self._gradient_snap(mask, guide_crop, bbox_h, bbox_w)

        # ── Step 6: Morphology (same as V2) ──
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_close_k, self.morph_close_k))
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_open_k, self.morph_open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        return (mask > 127).astype(np.uint8)

    def generate_batch(self, bboxes, feat_layer2, feat_p3_fpn, feat_p4_fpn,
                       feat_p5_fpn=None, original_image=None,
                       input_hw=(640, 640)) -> list:
        """Generate masks for multiple detections (same as V2)."""
        return [self.generate(b, feat_layer2, feat_p3_fpn, feat_p4_fpn,
                              feat_p5_fpn, original_image, input_hw)
                for b in bboxes]
