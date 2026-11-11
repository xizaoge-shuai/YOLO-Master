# SKU110K full preparation recovery — 2026-11-10

The complete 12,196,152,718-byte author tar was downloaded and all archived files were streamed/verified. The first indexing attempt stopped on train CSV line 71 because the previous validator rejected out-of-image boxes. That failure and all original data remain intact.

The author CSVGenerator._read_annotations explicitly applies boundary handling; the copied preparation version follows that exact function, checked for all 1,730,996 rows against a pinned primary-source AST. Every original CSV box and source line is retained in the manifest. Adjusted objects: train22,469;val1,565;test7,876, including equality-at-edge cases. No object or sample is dropped. Full counts: train8,219/val588/test2,936; objects1,208,482/90,968/431,546.

A complete strict-Pillow RGB decode found five damaged official JPEGs: train4222/5822/882/924 and test274. All raw bytes remain unchanged. A second complete11743-image decode with process-local ImageFile.LOAD_TRUNCATED_IMAGES=True passed; this is a disclosed tolerant-decoder protocol, not byte repair or a claim of pixel identity with OpenCV. The same setting applies to every new SKU baseline and candidate arm. Existing VOC/VisDrone source and process state are unaffected.

New SKU-only wrappers pin the reviewed1366-file VOC source anchor plus the two finished wrapper files and two decoder reports. New unrelated scripts do not enter the frozen snapshot. The new supervisor retains the existing two-slot GPU0 memory gate and baseline-first12-case matrix. No GPU smoke, subset training, or new GPU worker was launched during this repair; root reviews/commits before the new formal job starts.

Evaluation uses author-loader-compatible boundary annotations and fixed final epoch. It must be labelled as our compatible COCO AP, not a quoted original paper table result. Shared GPU timings remain observational.

Primary source: https://raw.githubusercontent.com/eg4000/SKU110K_CVPR19/master/object_detector_retinanet/keras_retinanet/preprocessing/csv_generator.py

See contracts, preparation seal, preserved failure and launch records in this directory. Decoder scripts are evidence of full data checks; they do not train models or import Torch.
