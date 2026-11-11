# Fixed GT-selected qualitative evidence

27 unique images (108 method-image comparisons), all four predefined top8 cohorts, fixed final epoch100 seed0. Main preview uses each cohort's original rank1; small/occluded rank1 is the same image and is intentionally repeated. Threshold0.25, class-aware IoU0.50, no threshold optimization or extra NMS.

Blue solid prediction = matched; orange solid = unmatched prediction; magenta dashed GT = missed; gray dotted = ignored; cyan = GT. These are thresholded diagnostic matches, not official AP or a full error taxonomy. Retains duplicate, background, localization/classification failures rather than suppressing them.

Gallery pages show original full scenes, deterministic GT-only512px crops and RGB residual feature energy. Heatmap scalar normalization, shared limits, clipping fractions, provenance hashes, exact crop and matching rules are in source-selection-manifest.json. Feature energy is descriptive and cannot establish causal AP gains. Image adjustments: no brightness, contrast, gamma, denoising or retouch.

PNG is an electronic review image; PDF/SVG retain editable vector overlays. The large full gallery is supplementary audit material; the main preview requires final manuscript resizing before submission. The raw photos remain only within these derived scientific panels and are not repackaged as a dataset.
