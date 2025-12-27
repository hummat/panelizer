# Tools Analysis: Comic Panel Detection

A comparative analysis of existing comic panel detection tools for informing Panelizer design.

---

## Kumiko

### Summary
OpenCV-based panel detection using contour detection and line segment analysis.

### Pipeline
```
1. Grayscale conversion
2. Sobel edge detection (horizontal + vertical gradients)
3. Binary threshold (fixed at 100)
4. Contour detection (cv.findContours)
5. Line segment detection (LSD - Line Segment Detector)
6. Post-processing:
   - Group small panels (close-by clustering)
   - Split panels (detect internal edges)
   - Exclude small panels
   - Merge panels (handle speech bubbles)
   - De-overlap panels
   - Expand panels to gutters
   - Group big panels
7. Grid-based reading order (LTR/RTL)
```

### Key Features
- **Reading order**: Custom `__lt__` operator implements row-major sorting with LTR/RTL support
- **Panel expansion**: Expands panels to neighbor edges or page boundaries
- **Polygon splitting**: Recursive splitting for panels with internal edges
- **Small panel handling**: Clusters adjacent tiny panels, excludes too-small ones
- **Segment-based refinement**: Uses detected line segments to validate splits

### Limitations
- Fixed thresholds (e.g., `cv2.threshold(..., 100, 255, ...)`)
- No confidence estimation
- Fails on borderless panels (no gutters)
- Art bleeding into gutters causes issues
- Requires tuning for different comic styles

### Output
- Bounding boxes (x, y, w, h)
- Polygon contours available but bboxes used for ordering

### License
MIT (inferred from repo)

### Reference
https://github.com/njean42/kumiko

---

## C.A.P.E (Comic Analysis and Panel Extraction)

### Summary
OpenCV-based panel detector with Electron-based editor UI. Semi-abandoned (~2015).

### Pipeline
```
1. Resize (1/3 original size)
2. Grayscale conversion
3. Detect dark borders via corner histogram
4. Border padding (handles unclosed panels)
5. Threshold (binary for white/black borders)
6. Dilation/erosion iteration search (binary search for optimal)
7. Contour detection (cv.RETR_TREE)
8. Large panel re-evaluation:
   - Polygon approximation (cv.approxPolyDP)
   - If > 4 points: graph-based subgraph detection
9. Area filtering (minAllowedPanelSize)
10. Grid-based sorting (gridY, gridX)
```

### Key Features
- **Dark border detection**: Histogram-based classification
- **Binary search tuning**: Iterates erosion to find optimal panel count
- **Large panel splitting**: Graph-based separation of merged panels
- **Layout validation**: Checks total panel area vs page area
- **`.cpanel` format**: Custom JSON metadata with version, bbox, polygon shape

### Limitations
- Fixed resize (1/3) loses detail
- Binary search iteration range hardcoded (0-4)
- No confidence estimation
- Target panel count hardcoded (GOOD_PANELS_GOAL = 5)
- Semi-abandoned, Python 2.7 code

### Output
```json
{
  "version": 2,
  "panels": [
    {
      "box": {"x": 10, "y": 20, "w": 300, "h": 400},
      "shape": [{"x": 10, "y": 20}, ...]  // polygon points
    }
  ]
}
```

### License
Apache 2.0

### Reference
https://github.com/CodeMinion/C.A.P.E

---

## best-comic-panel-detection (YOLOv12)

### Summary
YOLOv12x model fine-tuned on comic panels for bbox detection.

### Architecture
- **Base model**: YOLOv12x (extra-large variant)
- **Framework**: PyTorch + Ultralytics
- **Transfer learning**: Fine-tuned from COCO pre-trained checkpoint

### Performance Metrics
| Metric | Value |
|---------|-------|
| mAP50 | 0.991 |
| mAP50-95 | 0.985 |

Near-perfect precision/recall on validation data.

### Training Configuration
- Image size: 640x640
- Batch size: 16
- Optimizer: AdamW (lr=0.002)
- Epochs: 200
- Early stopping: 100 epochs patience
- Classes: Comic Panel (single class)

### Key Features
- **Fast inference**: YOLO is optimized for real-time detection
- **Confidence scores**: Native to YOLO output
- **Easy integration**: Ultralytics API is drop-in
- **Bounding box only**: No polygon segmentation

### Limitations
- Bbox output (not polygons for irregular panels)
- Inference requires GPU for speed
- Model size: YOLOv12x is large (~100MB+)
- No ordering provided
- Training data not publicly available

### Usage Example
```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict(image_path)

for result in results:
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        conf = box.conf.item()
```

### License
Apache 2.0

### Reference
https://huggingface.co/mosesb/best-comic-panel-detection

---

## segment-anything-comic (SAM-comic)

### Summary
SAM (Segment Anything Model) fine-tuned for comic panel polygon segmentation.

### Architecture
- **Base model**: Segment Anything Model (Meta, arXiv:2304.02643)
- **Fine-tuned**: For predicting comic frame polygon segmentations
- **Output**: Polygon masks (not bboxes)

### Key Features
- **Polygon segmentation**: Handles irregular panel shapes
- **Prompt-based**: SAM accepts point/box prompts (unclear if fine-tune modifies this)
- **General purpose**: SAM can segment arbitrary objects

### Limitations
- Slow inference: SAM is computationally expensive
- No ordering provided
- No confidence scores native to SAM (requires post-processing)
- No public performance metrics
- Training methodology not documented (see CS_580_Project.pdf)
- Research-grade code, not production-ready

### Implications for Panelizer
- **When to use**: Only for irregular panels where bbox is insufficient
- **Cost**: Likely slower than YOLO, may be overkill for most pages
- **Trade-off**: Polygon accuracy vs inference speed

### License
Apache 2.0 (inherited from SAM)

### Reference
https://github.com/Vrroom/segment-anything-comic

---

## DeepPanel

### Summary
U-Net CNN trained for comic panel segmentation, optimized for mobile.

### Architecture
- **Network**: U-Net (encoder-decoder CNN for segmentation)
- **Framework**: TensorFlow
- **Output type**: Segmentation mask (semantic segmentation)
- **Target labels**: 3 classes
  - Background (blue)
  - Border (red)
  - Panel content (green)

### Performance
- **Training dataset**: 550 pages (private)
- **Test dataset**: 83 pages
- **Inference time**: < 1 second (optimized for mobile)
- **Accuracy**: High (graphs provided, no numbers cited)

### Mobile Optimization
- Dedicated Android library: https://github.com/pedrovgs/DeepPanelAndroid
- Dedicated iOS library: https://github.com/pedrovgs/DeepPaneliOS
- Designed to simulate human reading behavior

### Key Features
- **Deep learning only**: No OpenCV preprocessing
- **Semantic segmentation**: Per-pixel classification
- **Panel grouping**: "Simulates human reading behavior" to group related panels
- **Supervised**: Requires labeled training data (mask format specified)

### Limitations
- Private training data (not available)
- Requires retraining for new datasets
- Mobile-first (may not be optimal for desktop/batch)
- TensorFlow dependency
- No ordering provided
- Semi-active project

### Training Data Format
```
dataset/
├── training/
│   ├── raw/              # .jpg comic pages
│   └── segmentation_mask/  # .png masks (3 colors)
├── test/
│   ├── raw/
│   └── segmentation_mask/
```

Mask colors:
- Full RGB blue (0, 0, 255) for background
- Full RGB red (255, 0, 0) for border
- Full RGB green (0, 255, 0) for panel content

### License
Apache 2.0

### Reference
https://github.com/pedrovgs/DeepPanel

---

## Comparative Summary

| Tool | Method | Output | Speed | Confidence | Ordering | License |
|-------|---------|--------|--------|------------|---------|
| Kumiko | CV (Sobel + contours) | Fast | No | Yes (grid) | MIT? |
| C.A.P.E | CV (threshold + contours) | Fast | No | Yes (grid) | Apache 2.0 |
| YOLOv12 | Deep learning | Fast | Yes | No | Apache 2.0 |
| SAM-comic | Deep learning | Slow | No | No | Apache 2.0 |
| DeepPanel | Deep learning (U-Net) | Medium (<1s) | No | No | Apache 2.0 |

## Recommendations for Panelizer

### Stage 1 (CV)
- **Primary reference**: Kumiko (more sophisticated than C.A.P.E)
- **Adopt**: Sobel + LSD pipeline, segment-based splitting, panel expansion
- **Add**: Confidence estimation (currently missing from all CV tools)

### Stage 2 (ML)
- **Primary**: YOLOv12 (best performance metrics, fastest ML)
- **Fallback**: SAM-comic (only for irregular panels where bbox is insufficient)
- **Skip**: DeepPanel (requires retraining, private data)

### Reading Order
- All tools use heuristics (row-major grid-based)
- None provide learned ordering
- VLM approach may be more sophisticated than existing heuristics

### Confidence Calibration
- Only YOLO provides native confidence
- CV tools need custom estimation (contour clarity, gutter strength, layout regularity)

### Data Format
- C.A.P.E's `.cpanel` format is well-specified (version 2)
- Consider compatibility for override import/export
