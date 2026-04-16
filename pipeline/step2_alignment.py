# ─────────────────────────────────────────────
#  step2_alignment.py
#  • Register t2 onto t1 coordinate space
#  • Uses ORB + Lowe's ratio test + partial affine (4-DOF)
#  • Falls back to phase correlation (translation only)
# ─────────────────────────────────────────────

import cv2
import numpy as np
import matplotlib.pyplot as plt


def align_images(img_t1: np.ndarray, img_t2: np.ndarray,
                 debug: bool = False, save_path: str = None):
    """
    Align img_t2 to img_t1 using ORB keypoints + partial affine transform.

    Uses Lowe's ratio test (0.75) to filter matches before RANSAC, then fits
    a 4-DOF similarity (translation + rotation + uniform scale).  This avoids
    the stretching / perspective warp that a full homography can introduce
    when features are unevenly distributed.

    Returns:
      img_t2_aligned : warped t2 image in t1 coordinate space
      M              : 2×3 affine matrix (t2 → t1)
    """
    print("\n[Alignment] Computing registration t2 → t1")

    gray1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img_t2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("  [alignment] Not enough keypoints – falling back to translation")
        result = _fallback_translation(img_t1, img_t2, debug)
        if save_path:
            _save_landmark_image(img_t1, img_t2, result[0], [], [], [], None,
                                 save_path, method="phase-correlation fallback")
        return result

    # ── Lowe's ratio test: keep only unambiguous matches ──────────────────
    # kNN (k=2): for each descriptor find the two closest neighbours.
    # A match is "good" only when the best distance is clearly smaller
    # than the second-best (ratio < 0.75).  This removes many wrong pairs
    # that a simple nearest-neighbour or cross-check would keep.
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    print(f"  [alignment] {len(kp1)} / {len(kp2)} keypoints, "
          f"{len(good)} good matches (Lowe's ratio test)")

    if len(good) < 8:
        print("  [alignment] Too few good matches – falling back to translation")
        result = _fallback_translation(img_t1, img_t2, debug)
        if save_path:
            _save_landmark_image(img_t1, img_t2, result[0], kp1, kp2, good, None,
                                 save_path, method="phase-correlation fallback")
        return result

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # ── Partial affine (4-DOF): translation + rotation + uniform scale ────
    # A full homography (8-DOF) can warp / stretch the image when features
    # are clustered in one region.  estimateAffinePartial2D fits only the
    # four degrees of freedom a hand-held re-photo actually introduces.
    M, mask = cv2.estimateAffinePartial2D(pts2, pts1,
                                          method=cv2.RANSAC,
                                          ransacReprojThreshold=5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  [alignment] Partial-affine inliers: {inliers}/{len(good)}")

    if M is None or inliers < 6:
        print("  [alignment] Affine fit failed – falling back to translation")
        result = _fallback_translation(img_t1, img_t2, debug)
        if save_path:
            _save_landmark_image(img_t1, img_t2, result[0], kp1, kp2, good, mask,
                                 save_path, method="phase-correlation fallback")
        return result

    h, w = img_t1.shape[:2]
    img_t2_aligned = cv2.warpAffine(
        cv2.cvtColor(img_t2, cv2.COLOR_RGB2BGR),
        M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    img_t2_aligned = cv2.cvtColor(img_t2_aligned, cv2.COLOR_BGR2RGB)

    if save_path:
        _save_landmark_image(img_t1, img_t2, img_t2_aligned,
                             kp1, kp2, good, mask, save_path,
                             method=f"ORB partial-affine ({inliers} inliers)")
    if debug:
        _show_alignment_debug(img_t1, img_t2, img_t2_aligned, kp1, kp2, good, mask)

    return img_t2_aligned, M


def _save_landmark_image(img_t1, img_t2, img_t2_aligned,
                          kp1, kp2, good, mask, save_path, method=""):
    """
    Save a 3-panel QC image showing:
      1. Checkerboard blend before alignment
      2. Checkerboard blend after alignment
      3. Feature matches with inliers (green) and outliers (red)
    Always saved to disk regardless of debug flag.
    """
    import matplotlib
    matplotlib.use('Agg')

    h, w = img_t1.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor('#111111')

    # ── Checkerboard panels ──
    tile = max(40, min(h, w) // 20)
    checker = np.zeros((h, w), dtype=bool)
    for r in range(0, h, tile * 2):
        for c in range(0, w, tile * 2):
            checker[r:r+tile, c:c+tile] = True
            checker[r+tile:r+2*tile, c+tile:c+2*tile] = True

    # Resize t2 to t1 size for before-blend if needed
    t2_resized = cv2.resize(img_t2, (w, h)) if img_t2.shape[:2] != (h, w) else img_t2

    blend_before        = img_t1.copy()
    blend_before[checker] = t2_resized[checker]

    blend_after         = img_t1.copy()
    blend_after[checker] = img_t2_aligned[checker]

    axes[0].imshow(blend_before)
    axes[0].set_title("Before alignment\n(checkerboard: t1 vs t2)",
                       color='white', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(blend_after)
    axes[1].set_title("After alignment\n(should look like one image)",
                       color='white', fontsize=11)
    axes[1].axis('off')

    # ── Match visualisation ──
    if kp1 and kp2 and good:
        if mask is not None:
            inlier_m  = [good[i] for i in range(len(good)) if mask[i]]
            outlier_m = [good[i] for i in range(len(good)) if not mask[i]]
        else:
            inlier_m  = good[:30]
            outlier_m = []

        # Draw inliers green, outliers red
        match_img = cv2.drawMatches(
            cv2.cvtColor(img_t1, cv2.COLOR_RGB2BGR), kp1,
            cv2.cvtColor(t2_resized, cv2.COLOR_RGB2BGR), kp2,
            inlier_m[:40], None,
            matchColor=(0, 220, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        if outlier_m:
            match_img = cv2.drawMatches(
                cv2.cvtColor(img_t1, cv2.COLOR_RGB2BGR), kp1,
                cv2.cvtColor(t2_resized, cv2.COLOR_RGB2BGR), kp2,
                outlier_m[:20], match_img,
                matchColor=(0, 0, 220),
                flags=(cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS |
                       cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
            )
        axes[2].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(
            f"Landmarks used  [{method}]\n"
            f"Green = inliers ({len(inlier_m)})  "
            f"Red = outliers ({len(outlier_m)})",
            color='white', fontsize=11)
    else:
        axes[2].imshow(img_t1)
        axes[2].set_title(f"No feature matches\n[{method}]",
                           color='white', fontsize=11)
    axes[2].axis('off')

    plt.suptitle(
        "Alignment QC — check that the 'after' checkerboard looks like one image.\n"
        "If it looks wrong, the alignment failed and t2 positions will need more correction.",
        color='white', fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [alignment] Landmark QC saved → {save_path}")


def _fallback_translation(img_t1: np.ndarray, img_t2: np.ndarray,
                           debug: bool = False):
    """Phase-correlation based translation-only alignment."""
    h1, w1 = img_t1.shape[:2]
    h2, w2 = img_t2.shape[:2]

    gray1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img_t2, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Phase correlation requires same size — resize gray2 to gray1 size if needed
    if gray1.shape != gray2.shape:
        print(f"  [alignment] Size mismatch ({w1}x{h1} vs {w2}x{h2}) – resizing t2 for phase correlation")
        gray2_resized = cv2.resize(gray2, (w1, h1))
    else:
        gray2_resized = gray2

    shift, _ = cv2.phaseCorrelate(gray1, gray2_resized)
    dx, dy = -shift[0], -shift[1]
    print(f"  [alignment] Phase correlation shift: dx={dx:.1f}, dy={dy:.1f}")

    H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)

    # Warp t2 into t1 coordinate space (resize first, then translate)
    img_t2_bgr = cv2.cvtColor(img_t2, cv2.COLOR_RGB2BGR)
    if (h2, w2) != (h1, w1):
        img_t2_bgr = cv2.resize(img_t2_bgr, (w1, h1))

    img_t2_aligned = cv2.warpAffine(img_t2_bgr, H[:2], (w1, h1))
    img_t2_aligned = cv2.cvtColor(img_t2_aligned, cv2.COLOR_BGR2RGB)
    return img_t2_aligned, H


def _show_alignment_debug(img_t1, img_t2, img_t2_aligned, kp1, kp2, good, mask):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Checkerboard blend before/after
    h, w = img_t1.shape[:2]
    tile = 80
    checker = np.zeros((h, w), dtype=bool)
    for r in range(0, h, tile * 2):
        for c in range(0, w, tile * 2):
            checker[r:r+tile, c:c+tile] = True
            checker[r+tile:r+2*tile, c+tile:c+2*tile] = True

    blend_before = img_t1.copy()
    blend_before[checker] = img_t2[checker]

    blend_after = img_t1.copy()
    blend_after[checker] = img_t2_aligned[checker]

    axes[0].imshow(blend_before)
    axes[0].set_title("Before alignment (checkerboard)")
    axes[0].axis("off")

    axes[1].imshow(blend_after)
    axes[1].set_title("After alignment (checkerboard)")
    axes[1].axis("off")

    # Match visualization
    if mask is not None:
        inlier_matches = [good[i] for i in range(len(good)) if mask[i]]
    else:
        inlier_matches = good[:20]

    match_img = cv2.drawMatches(
        cv2.cvtColor(img_t1, cv2.COLOR_RGB2BGR), kp1,
        cv2.cvtColor(img_t2, cv2.COLOR_RGB2BGR), kp2,
        inlier_matches[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    axes[2].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Feature matches ({len(inlier_matches)} inliers shown)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    from step1_preprocess import load_image, detect_plate_roi
    p1 = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/IMG_1714.JPG"
    p2 = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/uploads/IMG_1728.JPG"

    img1 = load_image(p1)
    img2 = load_image(p2)
    x1,y1,w1,h1 = detect_plate_roi(img1)
    x2,y2,w2,h2 = detect_plate_roi(img2)
    crop1 = img1[y1:y1+h1, x1:x1+w1]
    crop2 = img2[y2:y2+h2, x2:x2+w2]

    aligned, M = align_images(crop1, crop2, debug=True)
    print("M =\n", M)
