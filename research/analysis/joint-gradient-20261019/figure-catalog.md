# Figure catalog

## Figure01: Native gradient variance screening

Files: figures/figure-01-native-variance-screen.{pdf,svg,png}. Purpose: show whether the fixed unit-coefficient control reduces conditional query variance. Two panels are the Online100 and Mix25 frozen heads. x: image scale0.85/1.15; y: ratio of pooled residual-gradient energy to pooled exact-gradient energy, log scale. Panel y-ranges differ and are explicitly labeled. Legend: Transport, Blank-fill, Rec, Blank+true-content oracle, True-padding+content oracle. Open markers identify oracle arms; dashed y=1 is the query-only reference. Bars: pointwise95% filename-prefix bootstrap sensitivity intervals,2000 draws,128 images,46 prefix groups, one trained seed/head.

Notice: deployable controls all lie above1; spatial oracles lie below1. The diagram motivates stopping this unit-coefficient candidate and examining the available representation, not claiming optimizer/AP failure or oracle deployability. Query-only is a variance comparator, not a measured equal-cost detector baseline. Differences between panels cannot be assessed from plotted height because y-ranges differ.

## Figure02: Joint spatial restoration

Files: figures/figure-02-joint-spatial-restoration.{pdf,svg,png}. Purpose: distinguish joint recovery from single-region restoration and identify head dependence. Four panels: two heads by two scales. x: eight intervention arms; y: mean cosine of P3 localization parameter gradient to exact under fixed hard exact assignments. Identical y-range across panels. Blue dots: paired image means averaged across128 images. Error bars use the same pointwise filename-bootstrap sensitivity scheme. Exact is a self-reference with cosine1; Blank+content uses true content features.

Notice: Mix25 shrinkage joint restoration reaches0.947 while cheap fill+true content reaches0.650; Online100 cheap fill+true content is weaker than true content alone. Enlargement is mostly resolved in these diagnostic coordinates by restoring content. Changes in natural-region token count, energy, boundary and content are not controlled. This plot is not AP, does not establish a unique causal attribution, and cannot justify using oracle data for a cheap deployment claim.

All figures are analysis artifacts, not a finalized manuscript layout. Statistical definitions and interpretive limitations must accompany any reuse.
