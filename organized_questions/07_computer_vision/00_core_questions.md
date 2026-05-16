 Core Computer Vision Questions

## Fundamentals

### Question 1
- [ ] Done
**What is computer vision and how does it relate to human vision?**

**Answer:**

Computer vision is a field of AI that enables machines to interpret and understand visual information from images or videos, mimicking how humans perceive and process visual data. It extracts meaningful information from pixels to make decisions or take actions.

**Core Concepts:**
- **Human vision:** Brain processes light signals from retina → extracts edges, textures, shapes → recognizes objects and scenes
- **Computer vision:** Camera captures pixels → algorithms extract features → neural networks recognize patterns → outputs predictions
- **Hierarchy:** Both systems process information hierarchically (low-level edges → mid-level parts → high-level objects)

**Key Differences:**

| Aspect | Human Vision | Computer Vision |
|--------|-------------|-----------------|
| Input | Continuous light signals | Discrete pixel arrays |
| Processing | Parallel, biological neurons | Sequential/parallel, digital |
| Learning | Few examples needed | Often needs large datasets |
| Robustness | Handles novel situations well | Can fail on edge cases |

**Practical Relevance:**
- Autonomous vehicles (scene understanding)
- Medical imaging (diagnosis assistance)
- Surveillance systems (anomaly detection)
- Augmented reality (spatial understanding)

**Interview Tip:** Emphasize that CV attempts to replicate human vision capabilities but uses fundamentally different mechanisms (mathematical transformations vs. biological processes).

---

### Question 2
- [ ] Done
**Describe the key components of a computer vision system.**

**Answer:**

A computer vision system consists of interconnected components: image acquisition (camera/sensor), preprocessing (noise removal, normalization), feature extraction (edges, textures, learned features), model/algorithm (CNN, detector), and output/decision module that produces final predictions or actions.

**Core Components:**

1. **Image Acquisition**
   - Camera/sensor captures raw image data
   - Considerations: resolution, frame rate, lighting conditions

2. **Preprocessing**
   - Noise reduction (Gaussian blur, median filter)
   - Normalization (pixel value scaling to [0,1] or standardization)
   - Resizing, color space conversion (RGB→BGR, HSV)

3. **Feature Extraction**
   - Traditional: SIFT, HOG, edge detectors
   - Deep learning: CNN layers automatically learn features

4. **Model/Algorithm**
   - Classification: CNNs (ResNet, EfficientNet)
   - Detection: YOLO, Faster R-CNN
   - Segmentation: U-Net, Mask R-CNN

5. **Post-processing**
   - NMS for detection, CRF for segmentation
   - Thresholding, filtering predictions

6. **Output/Decision**
   - Class labels, bounding boxes, masks, actions

**Pipeline Flow:**
```
Camera → Preprocessing → Feature Extraction → Model Inference → Post-processing → Output
```

**Practical Relevance:** Understanding this pipeline helps debug issues at each stage and optimize bottlenecks in production systems.

---

### Question 3
- [ ] Done
**What is the difference between image processing and computer vision?**

**Answer:**

Image processing transforms images into other images (input: image → output: image), focusing on enhancement, filtering, and manipulation. Computer vision extracts understanding and meaning from images (input: image → output: decision/information) to interpret visual content like humans do.

**Core Differences:**

| Aspect | Image Processing | Computer Vision |
|--------|-----------------|-----------------|
| **Goal** | Transform/enhance images | Understand/interpret images |
| **Output** | Modified image | Labels, coordinates, decisions |
| **Examples** | Sharpening, denoising, resizing | Object detection, classification |
| **Level** | Low-level operations | High-level understanding |
| **Human analogy** | Adjusting brightness on TV | Recognizing who's on TV |

**Image Processing Tasks:**
- Filtering (blur, sharpen, edge detection)
- Geometric transformations (rotation, scaling)
- Color manipulation (histogram equalization)
- Compression (JPEG encoding)

**Computer Vision Tasks:**
- Object detection and recognition
- Image classification
- Semantic segmentation
- Pose estimation

**Relationship:**
- Image processing is often a **preprocessing step** for computer vision
- CV systems use image processing techniques internally
- Both operate on pixel data but with different end goals

**Interview Tip:** Image processing = pixel manipulation; Computer vision = scene understanding. They complement each other in real pipelines.

---

## Image Analysis & Processing

### Question 4
- [ ] Done
**How does edge detection work in image analysis?**

**Answer:**

Edge detection identifies boundaries in images by detecting discontinuities in pixel intensity. It uses gradient-based operators (Sobel, Prewitt) or second-derivative methods (Laplacian, Canny) to find locations where intensity changes rapidly, indicating object boundaries or texture changes.

**Core Concepts:**
- **Edge:** Sharp change in intensity (high gradient magnitude)
- **Gradient:** Rate of change in intensity across pixels
- **Direction:** Edges have orientation perpendicular to gradient direction

**Mathematical Formulation:**

Gradient magnitude and direction:
$$G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$

Sobel operators:
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I$$

**Common Edge Detectors:**

| Method | Approach | Pros/Cons |
|--------|----------|-----------|
| Sobel | First derivative | Simple, sensitive to noise |
| Canny | Multi-stage (smooth→gradient→NMS→hysteresis) | Best results, more complex |
| Laplacian | Second derivative | Detects edges, very noise-sensitive |

**Canny Edge Detection Steps:**
1. Gaussian smoothing to reduce noise
2. Compute gradient magnitude and direction
3. Non-maximum suppression (thin edges)
4. Hysteresis thresholding (connect edges)

**Python Example:**
```python
import cv2
import numpy as np

# Load grayscale image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel edges
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)

# Canny edges (recommended)
canny_edges = cv2.Canny(img, threshold1=50, threshold2=150)
```

**Practical Relevance:** Edge detection is used for object boundary detection, lane detection in autonomous vehicles, and as preprocessing for contour extraction.

---

### Question 5
- [ ] Done
**What are the common image preprocessing steps in a computer vision pipeline?**

**Answer:**

Common preprocessing steps include: resizing (standardize dimensions), normalization (scale pixel values), noise reduction (Gaussian/median filtering), color space conversion, data augmentation (flips, rotations), and histogram equalization (contrast enhancement). These prepare raw images for model consumption.

**Core Preprocessing Steps:**

1. **Resizing/Rescaling**
   - Match model input dimensions (e.g., 224×224 for ResNet)
   - Maintain aspect ratio with padding if needed

2. **Normalization**
   - Scale to [0, 1]: `pixel / 255.0`
   - Standardize: `(pixel - mean) / std` (ImageNet: mean=[0.485, 0.456, 0.406])

3. **Noise Reduction**
   - Gaussian blur: smooths image, reduces high-frequency noise
   - Median filter: removes salt-and-pepper noise

4. **Color Space Conversion**
   - RGB → BGR (OpenCV default)
   - RGB → Grayscale (reduce dimensionality)
   - RGB → HSV (color-based segmentation)

5. **Data Augmentation**
   - Geometric: flip, rotate, crop, scale
   - Photometric: brightness, contrast, saturation
   - Advanced: mixup, cutout, mosaic

6. **Histogram Equalization**
   - Enhances contrast in low-contrast images
   - CLAHE for adaptive local enhancement

**Python Example:**
```python
import cv2
import numpy as np

# Load image
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
img_resized = cv2.resize(img, (224, 224))

# Normalize to [0, 1]
img_normalized = img_resized / 255.0

# Standardize (ImageNet stats)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_standardized = (img_normalized - mean) / std

# Denoise
img_denoised = cv2.GaussianBlur(img, (5, 5), 0)
```

**Interview Tip:** Always mention normalization matching the pretrained model's training distribution. Mismatched normalization is a common bug.

---

### Question 6
- [ ] Done
**How does image resizing affect model performance?**

**Answer:**

Image resizing affects model performance through information loss (downscaling removes fine details), aspect ratio distortion (squeezing can confuse models), and distribution shift (resized images differ from training data). Larger input sizes generally improve accuracy but increase computational cost quadratically.

**Core Concepts:**

**Effects of Downscaling:**
- Loss of fine-grained details and small objects
- Small objects may become undetectable
- Faster inference, lower memory usage

**Effects of Upscaling:**
- Introduces interpolation artifacts
- No new information added
- Increased compute without accuracy gain

**Aspect Ratio Handling:**

| Method | Approach | Trade-off |
|--------|----------|-----------|
| Stretch | Force to target size | Distorts objects |
| Crop | Cut to target ratio | Loses content |
| Pad | Add borders | Wastes compute on padding |
| Letterbox | Pad to maintain ratio | Best balance |

**Resolution vs. Accuracy Trade-off:**
- Higher resolution → better small object detection
- Computational cost scales as O(n²) with image dimension
- Common sizes: 224×224 (classification), 640×640 (YOLO)

**Python Example:**
```python
import cv2

img = cv2.imread('image.jpg')

# Method 1: Stretch (may distort)
resized_stretch = cv2.resize(img, (224, 224))

# Method 2: Letterbox (maintain aspect ratio)
def letterbox(img, target_size=224):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    # Pad to target size
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

resized_letterbox = letterbox(img, 224)
```

**Interview Tip:** For detection tasks, mention letterboxing to preserve aspect ratios. For classification, match the pretrained model's training resolution.

---

### Question 7
- [ ] Done
**What are some techniques to reduce noise in an image?**

**Answer:**

Noise reduction techniques include: Gaussian blur (smooths by averaging neighbors), median filter (replaces with median value, good for salt-and-pepper noise), bilateral filter (edge-preserving smoothing), and deep learning denoisers (learned filters). Choice depends on noise type and edge preservation requirements.

**Core Techniques:**

| Filter | Best For | Edge Preservation |
|--------|----------|-------------------|
| Gaussian Blur | General smoothing | Poor |
| Median Filter | Salt-and-pepper noise | Moderate |
| Bilateral Filter | General noise | Good |
| Non-local Means | General noise | Very Good |

**Mathematical Formulation:**

Gaussian filter:
$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

Bilateral filter (combines spatial and intensity):
$$I_{filtered}(x) = \frac{1}{W} \sum_{x_i \in \Omega} I(x_i) \cdot f_s(||x_i - x||) \cdot f_r(|I(x_i) - I(x)|)$$

Where $f_s$ = spatial kernel, $f_r$ = range (intensity) kernel

**Noise Types:**
- **Gaussian noise:** Random intensity variations (use Gaussian blur)
- **Salt-and-pepper:** Random black/white pixels (use median filter)
- **Speckle:** Multiplicative noise (use bilateral or adaptive filters)

**Python Example:**
```python
import cv2

img = cv2.imread('noisy_image.jpg')

# Gaussian Blur (kernel size must be odd)
gaussian = cv2.GaussianBlur(img, (5, 5), sigmaX=0)

# Median Filter (good for salt-and-pepper)
median = cv2.medianBlur(img, 5)

# Bilateral Filter (preserves edges)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Non-local Means (best quality, slower)
nlm = cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7)
```

**Interview Tip:** Median filter is best for salt-and-pepper noise; bilateral filter when edge preservation matters.

---

### Question 8
- [ ] Done
**Explain how image augmentation can improve the performance of a vision model.**

**Answer:**

Image augmentation artificially expands training data by applying transformations (flips, rotations, color jittering) to create varied versions of images. This improves model generalization by exposing it to diverse conditions, reduces overfitting, and makes models invariant to transformations they may encounter in real-world deployment.

**Core Concepts:**
- **Data Diversity:** Model sees more variations without collecting new data
- **Regularization Effect:** Prevents memorization of training samples
- **Invariance Learning:** Model learns to ignore irrelevant variations

**Types of Augmentation:**

| Category | Techniques | Purpose |
|----------|------------|---------|
| Geometric | Flip, rotate, crop, scale | Position/orientation invariance |
| Photometric | Brightness, contrast, saturation | Lighting invariance |
| Noise | Gaussian noise, blur | Robustness to noise |
| Advanced | Mixup, CutMix, Mosaic | Better generalization |

**Common Augmentations:**
- **Horizontal Flip:** Doubles data, useful for most objects
- **Random Crop:** Forces model to recognize partial objects
- **Color Jitter:** Robustness to lighting variations
- **Cutout/Random Erasing:** Occlusion robustness

**Python Example:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Apply to image
augmented = transform(image=image)
augmented_image = augmented['image']
```

**Interview Tip:** 
- Only use augmentations that make semantic sense (don't flip text, digits)
- Augmentation is applied during training only, not inference
- For detection/segmentation, transform bounding boxes/masks too

---

### Question 9
- [ ] Done
**Discuss the concept of color spaces and their importance in image processing.**

**Answer:**

Color spaces are mathematical models that represent colors as tuples of numbers. Different color spaces (RGB, HSV, LAB) organize color information differently, making certain tasks easier. RGB is standard for displays; HSV separates color (hue) from intensity; LAB is perceptually uniform for color comparison.

**Core Color Spaces:**

| Color Space | Components | Best For |
|-------------|------------|----------|
| **RGB** | Red, Green, Blue | Display, neural networks |
| **BGR** | Blue, Green, Red | OpenCV default |
| **HSV** | Hue, Saturation, Value | Color-based segmentation |
| **LAB** | Lightness, A, B | Color comparison, correction |
| **Grayscale** | Intensity | Edge detection, reduce complexity |
| **YCbCr** | Luma, Chroma | Video compression |

**Why Color Spaces Matter:**

1. **HSV for Color Segmentation:**
   - Hue separates color from brightness
   - Easy to threshold specific colors (e.g., detect red objects)

2. **LAB for Color Matching:**
   - Perceptually uniform: equal distances = equal perceived differences
   - Used in color transfer and histogram matching

3. **Grayscale for Efficiency:**
   - Reduces 3 channels → 1 channel
   - Edge detection, document analysis

**Python Example:**
```python
import cv2

# Load RGB image (OpenCV loads as BGR)
img_bgr = cv2.imread('image.jpg')

# Convert to different color spaces
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Color segmentation in HSV (detect red)
lower_red = (0, 100, 100)
upper_red = (10, 255, 255)
mask = cv2.inRange(img_hsv, lower_red, upper_red)
```

**Interview Tip:** Use HSV when you need to isolate colors regardless of lighting; use RGB/BGR for deep learning models.

---

## Segmentation

### Question 10
- [ ] Done
**Explain the concept of image segmentation in computer vision.**

**Answer:**

Image segmentation partitions an image into meaningful regions by assigning a label to every pixel. It goes beyond classification (one label per image) to provide pixel-level understanding. Types include semantic segmentation (class per pixel), instance segmentation (separate objects), and panoptic segmentation (combines both).

**Core Concepts:**

| Type | Output | Example |
|------|--------|---------|
| **Semantic** | Class label per pixel | All cars = same label |
| **Instance** | Separate each object | Car1, Car2, Car3 |
| **Panoptic** | Both semantic + instance | Background classes + instances |

**Segmentation Approaches:**

1. **Traditional Methods:**
   - Thresholding (binary segmentation)
   - Region growing (expand from seeds)
   - Watershed (treat gradients as topography)

2. **Deep Learning Methods:**
   - **FCN:** Fully convolutional for dense prediction
   - **U-Net:** Encoder-decoder with skip connections
   - **DeepLab:** Atrous convolutions for multi-scale
   - **Mask R-CNN:** Detection + instance masks

**Mathematical View:**
Given image $I$ with pixels $p$, segmentation produces label map $L$:
$$L(p) = \arg\max_c P(class = c | p, I)$$

**Python Example:**
```python
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

# Load pretrained model
model = deeplabv3_resnet50(pretrained=True)
model.eval()

# Preprocess image (add batch dim, normalize)
# input_tensor shape: [1, 3, H, W]

with torch.no_grad():
    output = model(input_tensor)['out']  # [1, num_classes, H, W]
    pred_mask = output.argmax(dim=1)     # [1, H, W] - class per pixel
```

**Practical Relevance:**
- Medical imaging (tumor segmentation)
- Autonomous driving (road, pedestrians, vehicles)
- Satellite imagery (land use classification)

---

### Question 11
- [ ] Done
**What is the difference between semantic and instance segmentation?**

**Answer:**

Semantic segmentation assigns a class label to every pixel but treats all instances of a class identically (all cars get same label). Instance segmentation differentiates individual objects of the same class (car1, car2, car3 as separate entities). Instance = semantic + object detection.

**Core Differences:**

| Aspect | Semantic Segmentation | Instance Segmentation |
|--------|----------------------|----------------------|
| **Output** | Class label per pixel | Object ID + class per pixel |
| **Same-class objects** | Merged together | Separated individually |
| **Architecture** | FCN, U-Net, DeepLab | Mask R-CNN, YOLACT |
| **Use case** | Scene parsing, road segmentation | Counting, tracking objects |

**Visual Example:**
```
Image: Two cars and background

Semantic:     [Background=0, Car=1, Car=1]  (both cars = label 1)
Instance:     [Background=0, Car_1=1, Car_2=2]  (separate IDs)
```

**Architectures:**

**Semantic Segmentation:**
- Fully Convolutional Networks (FCN)
- U-Net (encoder-decoder)
- DeepLabv3+ (atrous convolutions)

**Instance Segmentation:**
- Mask R-CNN (Faster R-CNN + mask head)
- YOLACT (single-stage)
- SOLOv2 (segment objects by locations)

**Panoptic Segmentation:**
- Combines both: instance labels for "things" (countable objects)
- Semantic labels for "stuff" (uncountable: sky, road)

**Practical Relevance:**
- **Semantic:** Autonomous driving road parsing, medical tissue classification
- **Instance:** Object counting, multi-object tracking, robotics grasping

**Interview Tip:** Instance segmentation is harder because it requires detecting and separating overlapping objects of the same class.

---

### Question 12
- [ ] Done
**Explain the Fully Convolutional Network (FCN) and its role in segmentation.**

**Answer:**

FCN replaces fully connected layers in classification CNNs with convolutional layers, enabling dense pixel-wise predictions for any input size. It pioneered end-to-end segmentation by using upsampling (transposed convolutions) to restore spatial resolution after feature extraction, producing output masks matching input dimensions.

**Core Concepts:**

**Key Innovation:**
- Classification CNNs: Conv → FC → single label
- FCN: Conv → Conv (1×1) → Upsample → pixel-wise labels

**Architecture Components:**

1. **Encoder (Downsampling):**
   - Pretrained backbone (VGG, ResNet)
   - Extracts hierarchical features
   - Reduces spatial resolution

2. **1×1 Convolution:**
   - Replaces FC layers
   - Converts feature maps to class scores

3. **Decoder (Upsampling):**
   - Transposed convolution / bilinear interpolation
   - Restores original resolution
   - Skip connections for fine details

**FCN Variants:**
- **FCN-32s:** Single 32× upsample (coarse)
- **FCN-16s:** Combines pool4 + 2× upsample of FCN-32s
- **FCN-8s:** Adds pool3, finest predictions

**Mathematical View:**
$$\text{Output} = \text{Upsample}(\text{Conv}_{1\times1}(\text{Encoder}(I)))$$

**Python Example:**
```python
import torch
import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder (simplified)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /4
        )
        # 1x1 conv for classification
        self.classifier = nn.Conv2d(128, num_classes, 1)
        # Upsample back to original size
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
    
    def forward(self, x):
        features = self.encoder(x)
        scores = self.classifier(features)
        output = self.upsample(scores)
        return output  # [B, num_classes, H, W]
```

**Practical Relevance:** FCN is the foundation for all modern segmentation networks (U-Net, DeepLab). Understanding it is essential.

---

## Feature Extraction & Detection

### Question 13
- [ ] Done
**What are feature descriptors, and why are they important in computer vision?**

**Answer:**

Feature descriptors are compact numerical representations that describe distinctive image regions (keypoints). They encode local patterns like edges, gradients, and textures in a way that's invariant to transformations. They enable matching, recognition, and tracking by comparing descriptor vectors instead of raw pixels.

**Core Concepts:**

**Two-Stage Process:**
1. **Detection:** Find interesting keypoints (corners, blobs)
2. **Description:** Compute descriptor vector for each keypoint

**Properties of Good Descriptors:**
- **Distinctiveness:** Different regions produce different descriptors
- **Invariance:** Robust to rotation, scale, illumination
- **Compactness:** Low-dimensional for efficient matching
- **Repeatability:** Same point detected across views

**Common Feature Descriptors:**

| Descriptor | Dimension | Invariance | Speed |
|------------|-----------|------------|-------|
| SIFT | 128 | Scale, rotation | Slow |
| SURF | 64 | Scale, rotation | Medium |
| ORB | 256 (binary) | Rotation | Fast |
| BRIEF | 256 (binary) | None | Very Fast |

**Matching Process:**
1. Extract keypoints and descriptors from both images
2. For each descriptor in image1, find nearest neighbor in image2
3. Apply ratio test to filter false matches
4. Use RANSAC to remove outliers

**Python Example:**
```python
import cv2

# Load images
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Create ORB detector (fast, patent-free)
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and compute descriptors
kp1, desc1 = orb.detectAndCompute(img1, None)
kp2, desc2 = orb.detectAndCompute(img2, None)

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

**Practical Relevance:** Image stitching, visual SLAM, object recognition, augmented reality marker tracking.

---

### Question 14
- [ ] Done
**Explain the Scale-Invariant Feature Transform (SIFT) algorithm.**

**Answer:**

SIFT detects keypoints that are invariant to scale and rotation, then computes 128-dimensional descriptors for each. It builds a scale-space pyramid using Difference of Gaussians (DoG) to find keypoints, assigns dominant orientations, and describes local gradients in orientation histograms.

**Algorithm Steps:**

**1. Scale-Space Extrema Detection:**
- Build Gaussian pyramid at multiple scales: $L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)$
- Compute DoG: $D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)$
- Find local extrema (keypoint candidates) in 3×3×3 neighborhood

**2. Keypoint Localization:**
- Refine location using Taylor expansion
- Reject low-contrast points and edge responses (using Hessian ratio)

**3. Orientation Assignment:**
- Compute gradient magnitude and orientation in region around keypoint
- Build histogram of orientations (36 bins)
- Assign dominant orientation → rotation invariance

**4. Descriptor Generation:**
- Take 16×16 window around keypoint
- Divide into 4×4 grid of cells
- Compute 8-bin orientation histogram per cell
- Result: 4×4×8 = **128-dimensional vector**
- Normalize descriptor for illumination invariance

**Mathematical Formulation:**
Gradient magnitude and orientation:
$$m(x,y) = \sqrt{(L_{x+1,y} - L_{x-1,y})^2 + (L_{x,y+1} - L_{x,y-1})^2}$$
$$\theta(x,y) = \arctan\left(\frac{L_{x,y+1} - L_{x,y-1}}{L_{x+1,y} - L_{x-1,y}}\right)$$

**Python Example:**
```python
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img, None)

# descriptors shape: (num_keypoints, 128)
print(f"Found {len(keypoints)} keypoints")
print(f"Descriptor shape: {descriptors.shape}")
```

**Interview Tip:** SIFT is patented (use ORB for free alternative). Key insight: DoG approximates Laplacian of Gaussian for scale-space blob detection.

---

### Question 15
- [ ] Done
**Describe how the Histogram of Oriented Gradients (HOG) descriptor works.**

**Answer:**

HOG captures local shape information by computing gradient orientations in dense grid cells and accumulating them into histograms. It divides the image into cells, computes oriented gradient histograms per cell, normalizes across blocks of cells, and concatenates into a feature vector for object detection.

**Algorithm Steps:**

**1. Preprocessing:**
- Convert to grayscale (optional)
- Resize to fixed dimensions (e.g., 64×128 for pedestrians)
- Gamma correction for lighting normalization

**2. Gradient Computation:**
- Compute horizontal and vertical gradients: $G_x = I * [-1, 0, 1]$, $G_y = I * [-1, 0, 1]^T$
- Magnitude: $|G| = \sqrt{G_x^2 + G_y^2}$
- Orientation: $\theta = \arctan(G_y / G_x)$

**3. Cell Histograms:**
- Divide image into small cells (e.g., 8×8 pixels)
- Create histogram of 9 orientation bins (0°-180°, unsigned)
- Vote weighted by gradient magnitude

**4. Block Normalization:**
- Group cells into overlapping blocks (e.g., 2×2 cells)
- L2-normalize each block for illumination invariance
- Slide block across image

**5. Feature Vector:**
- Concatenate all normalized block histograms
- For 64×128 image: 7×15 blocks × 4 cells × 9 bins = **3780 features**

**Python Example:**
```python
import cv2
from skimage.feature import hog

img = cv2.imread('person.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 128))

# Compute HOG features
features, hog_image = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm='L2-Hys'
)

print(f"HOG feature vector length: {len(features)}")
```

**Practical Relevance:** HOG + SVM was the classic pedestrian detection method (Dalal & Triggs, 2005). Now largely replaced by deep learning but still useful for understanding gradient-based features.

---

### Question 16
- [ ] Done
**Compare different image feature extraction methods.**

**Answer:**

Feature extraction methods range from handcrafted (SIFT, HOG, ORB) to learned (CNN features). Handcrafted methods are interpretable and work without training data but lack flexibility. CNN features are learned from data, capture hierarchical patterns, and dominate modern vision tasks.

**Comparison Table:**

| Method | Type | Dimension | Speed | Invariance | Best For |
|--------|------|-----------|-------|------------|----------|
| **SIFT** | Keypoint | 128 | Slow | Scale, rotation | Matching, stitching |
| **SURF** | Keypoint | 64 | Medium | Scale, rotation | Real-time matching |
| **ORB** | Keypoint | 256 (binary) | Fast | Rotation | Mobile/embedded |
| **HOG** | Dense | ~3780 | Medium | - | Object detection |
| **LBP** | Dense | 256 | Fast | Illumination | Texture, faces |
| **CNN** | Learned | 512-2048 | GPU-dependent | Learned | General-purpose |

**Handcrafted vs. Learned:**

| Aspect | Handcrafted | Deep Learning |
|--------|-------------|---------------|
| Design | Human-engineered | Data-driven |
| Training | Not needed | Requires large dataset |
| Interpretability | High | Low |
| Flexibility | Limited | High |
| Performance | Good on specific tasks | State-of-the-art |

**When to Use What:**
- **SIFT/ORB:** Image matching, homography estimation, SLAM
- **HOG:** When deep learning isn't available, embedded systems
- **CNN features:** Classification, detection, general feature extraction

**Python Example - CNN Feature Extraction:**
```python
import torch
from torchvision import models, transforms

# Load pretrained ResNet
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract features
with torch.no_grad():
    features = model(input_tensor)  # [1, 2048, 1, 1]
    features = features.flatten()   # [2048]
```

**Interview Tip:** CNN features from pretrained models (ImageNet) often outperform handcrafted features even for tasks not related to classification.

---

### Question 17
- [ ] Done
**What are Haar Cascades, and how are they used for object detection?**

**Answer:**

Haar Cascades are a machine learning-based object detection method using simple rectangular features (Haar-like features) and a cascade of AdaBoost classifiers. They scan images at multiple scales, quickly rejecting non-object regions in early cascade stages, making real-time detection possible (Viola-Jones framework).

**Core Concepts:**

**Haar-like Features:**
- Simple rectangular filters that compute intensity differences
- Types: edge, line, and center-surround features
- Fast computation using **integral images**

**Integral Image:**
$$II(x,y) = \sum_{x' \leq x, y' \leq y} I(x', y')$$
Any rectangular sum computed in O(1) using 4 corner lookups.

**Cascade Architecture:**
```
Image Region → Stage 1 → Stage 2 → Stage 3 → ... → Detection
                  ↓          ↓          ↓
              Reject     Reject     Reject
```
- Each stage is a strong classifier (AdaBoost ensemble)
- Early stages reject obvious negatives quickly
- Later stages handle harder cases

**Algorithm Steps:**
1. Compute integral image for fast feature calculation
2. Slide detection window across image at multiple scales
3. For each window, pass through cascade stages
4. If all stages pass, mark as detection
5. Apply non-maximum suppression to merge overlapping boxes

**Python Example:**
```python
import cv2

# Load pretrained Haar Cascade (face detection)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load image
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # Scale pyramid step
    minNeighbors=5,    # Robustness threshold
    minSize=(30, 30)   # Minimum object size
)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

**Practical Relevance:** Haar Cascades are fast and work on CPU, suitable for embedded systems. However, deep learning (YOLO, SSD) provides better accuracy for complex detection tasks.

---

### Question 18
- [ ] Done
**What's the significance of depth perception in computer vision applications?**

**Answer:**

Depth perception enables 3D scene understanding by estimating distance from camera to objects. It's crucial for robotics (navigation, grasping), autonomous vehicles (obstacle avoidance), AR/VR (spatial mapping), and 3D reconstruction. Methods include stereo vision, structured light, ToF sensors, and monocular depth estimation.

**Core Concepts:**

**Depth Acquisition Methods:**

| Method | Principle | Range | Accuracy |
|--------|-----------|-------|----------|
| **Stereo Vision** | Triangulation from 2 cameras | Medium | Moderate |
| **Structured Light** | Project patterns, analyze distortion | Short | High |
| **Time-of-Flight (ToF)** | Measure light travel time | Medium | High |
| **LiDAR** | Laser scanning | Long | Very High |
| **Monocular (DL)** | CNN learns depth from single image | Any | Moderate |

**Stereo Vision Math:**
$$Z = \frac{f \cdot B}{d}$$
Where: $Z$ = depth, $f$ = focal length, $B$ = baseline, $d$ = disparity

**Applications:**

1. **Autonomous Vehicles:**
   - Obstacle detection and distance estimation
   - Path planning and collision avoidance

2. **Robotics:**
   - Object grasping and manipulation
   - SLAM (Simultaneous Localization and Mapping)

3. **AR/VR:**
   - Scene reconstruction
   - Object occlusion handling

4. **Medical:**
   - Surgical navigation
   - 3D organ modeling

**Python Example - Monocular Depth Estimation:**
```python
import torch

# Load MiDaS depth estimation model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Estimate depth
input_batch = transform(img)
with torch.no_grad():
    depth = model(input_batch)  # Relative depth map
```

**Interview Tip:** RGB-D sensors (Kinect, RealSense) combine color and depth, providing rich 3D information for indoor applications.

---

### Question 19
- [ ] Done
**Explain the challenges of object recognition in varied lighting and orientations.**

**Answer:**

Varied lighting causes appearance changes (shadows, highlights, color shifts) while different orientations alter object shape in images. These variations make features inconsistent between training and test images. Solutions include data augmentation, illumination-invariant features, and rotation-equivariant networks.

**Lighting Challenges:**

| Issue | Effect | Solution |
|-------|--------|----------|
| Shadows | Hide features, create false edges | Shadow removal, HDR imaging |
| Overexposure | Loss of detail in bright regions | Histogram equalization, HDR |
| Color shifts | Different color appearance | Color normalization, white balance |
| Low light | Noise, reduced contrast | Denoising, low-light enhancement |

**Orientation Challenges:**

| Issue | Effect | Solution |
|-------|--------|----------|
| Rotation | Features at different angles | Rotation augmentation, rotation-invariant features |
| Scale | Objects appear larger/smaller | Multi-scale detection, scale augmentation |
| Viewpoint | 3D shape changes | 3D-aware models, multiple training views |
| Deformation | Non-rigid shape changes | Deformable convolutions, spatial transformer |

**Solutions:**

1. **Data Augmentation:**
   - Random brightness, contrast, saturation
   - Rotation, flipping, perspective transforms

2. **Normalization:**
   - Histogram equalization for lighting
   - Instance normalization in neural networks

3. **Robust Features:**
   - SIFT/ORB: rotation/scale invariant
   - CNN: learns invariance from data

4. **Specialized Architectures:**
   - Spatial Transformer Networks
   - Group equivariant CNNs

**Python Example:**
```python
import albumentations as A

# Augmentation for lighting and orientation robustness
transform = A.Compose([
    # Lighting variations
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    
    # Orientation variations
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.3),
])
```

**Interview Tip:** Heavy augmentation during training makes models robust at inference. Test-time augmentation (TTA) can also improve predictions.

---

## Neural Networks & Deep Learning

### Question 20
- [ ] Done
**Discuss the role of convolutional neural networks (CNNs) in computer vision.**

**Answer:**

CNNs are the foundation of modern computer vision, automatically learning hierarchical features from raw pixels through convolutional layers. They capture spatial patterns (edges → textures → parts → objects) via local receptive fields and weight sharing, achieving state-of-the-art results in classification, detection, and segmentation.

**Core Concepts:**

**Why CNNs for Vision:**
- **Local connectivity:** Exploit spatial structure of images
- **Weight sharing:** Same filter applied across image → parameter efficiency
- **Translation equivariance:** Detect features regardless of position
- **Hierarchical learning:** Low-level → high-level features automatically

**Feature Hierarchy:**
```
Input → Conv1 (edges) → Conv2 (textures) → Conv3 (parts) → Conv4 (objects)
```

**Key Components:**

| Layer | Purpose |
|-------|---------|
| Convolution | Extract local features |
| Activation (ReLU) | Non-linearity |
| Pooling | Downsample, increase invariance |
| Batch Normalization | Stabilize training |
| Fully Connected | Classification head |

**Impact on Computer Vision:**

| Task | Before CNNs | With CNNs |
|------|-------------|-----------|
| Classification | SIFT + SVM (~74% ImageNet) | ResNet (~96% ImageNet) |
| Detection | HOG + DPM | YOLO, Faster R-CNN |
| Segmentation | Graph cuts | U-Net, DeepLab |

**Python Example:**
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 32, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 32, H/2, W/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 64, H/4, W/4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),  # For 224x224 input
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Interview Tip:** CNNs learn what features to extract (end-to-end), unlike traditional methods where features are hand-designed.

---

### Question 21
- [ ] Done
**How do CNNs differ from traditional neural networks in architecture?**

**Answer:**

CNNs use convolutional layers with local receptive fields and shared weights instead of fully connected layers. This exploits image spatial structure, dramatically reduces parameters (e.g., 3×3 filter vs. connecting every pixel), and provides translation equivariance. Traditional MLPs treat inputs as flat vectors, ignoring spatial relationships.

**Key Architectural Differences:**

| Aspect | Traditional NN (MLP) | CNN |
|--------|---------------------|-----|
| Input | Flattened vector | 2D/3D tensor (H×W×C) |
| Connectivity | Fully connected | Local (receptive field) |
| Weights | Unique per connection | Shared (same filter everywhere) |
| Parameters | Very high | Much fewer |
| Spatial info | Lost | Preserved |

**Parameter Comparison:**

For 224×224×3 input to 1000 hidden units:
- **MLP:** 224 × 224 × 3 × 1000 = **150 million parameters**
- **CNN (3×3 conv, 64 filters):** 3 × 3 × 3 × 64 = **1,728 parameters**

**Mathematical Formulation:**

**Fully Connected:**
$$y_j = \sigma\left(\sum_{i=1}^{n} w_{ij} x_i + b_j\right)$$

**Convolution:**
$$y_{i,j} = \sigma\left(\sum_{m}\sum_{n} w_{m,n} \cdot x_{i+m, j+n} + b\right)$$

**Key CNN Properties:**
1. **Local Connectivity:** Each neuron connected to small region
2. **Weight Sharing:** Same filter applied everywhere
3. **Translation Equivariance:** Shifting input shifts output
4. **Hierarchical Features:** Stacked convolutions = larger receptive field

**Visual Comparison:**
```
MLP:        [Input: 50176] → [FC: 1000] → [FC: 10]
             Every pixel connected to every neuron

CNN:        [Input: 224×224×3] → [Conv: 64 3×3 filters] → [Pool] → [Conv] → ...
             Local filters slide across spatial dimensions
```

**Interview Tip:** Weight sharing in CNNs provides built-in regularization and translation equivariance, making them ideal for vision tasks.

---

### Question 22
- [ ] Done
**Explain the purpose of pooling layers in a CNN.**

**Answer:**

Pooling layers downsample feature maps by summarizing regions (max or average), reducing spatial dimensions while retaining important features. This decreases computational cost, provides translation invariance (small shifts don't change pooled output), and increases receptive field without adding parameters.

**Core Concepts:**

**Types of Pooling:**

| Type | Operation | Use Case |
|------|-----------|----------|
| Max Pooling | Take maximum value | Most common, preserves strong features |
| Average Pooling | Take average value | Smoother, global context |
| Global Average Pooling | Average entire feature map | Replace FC layer at end |

**Mathematical Formulation:**

For 2×2 pooling with stride 2:
$$\text{MaxPool}(x)_{i,j} = \max(x_{2i,2j}, x_{2i+1,2j}, x_{2i,2j+1}, x_{2i+1,2j+1})$$

**Benefits:**
1. **Dimensionality Reduction:** H×W → H/2 × W/2 (with 2×2 pool, stride 2)
2. **Translation Invariance:** Small shifts produce same pooled output
3. **Computational Efficiency:** Fewer pixels to process downstream
4. **Larger Receptive Field:** Each output pixel "sees" more of input

**Example:**
```
Input 4×4:          MaxPool 2×2:
[1  2  5  6]        
[3  4  7  8]   →    [4   8]
[9  10 13 14]       [12  16]
[11 12 15 16]
```

**Python Example:**
```python
import torch
import torch.nn as nn

# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global average pooling (common before final classification)
global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: [B, C, 1, 1]

x = torch.randn(1, 64, 224, 224)  # [B, C, H, W]
out_max = max_pool(x)              # [1, 64, 112, 112]
out_global = global_avg_pool(x)    # [1, 64, 1, 1]
```

**Interview Tip:** Modern architectures often use strided convolutions instead of pooling (more learnable). Global Average Pooling replaced FC layers in networks like ResNet.

---

### Question 23
- [ ] Done
**What is transfer learning, and when would you apply it in computer vision?**

**Answer:**

Transfer learning uses a model pretrained on a large dataset (like ImageNet) as a starting point for a new task. The pretrained weights capture general visual features (edges, textures, shapes) that transfer to new domains. Apply it when you have limited data, similar domain to pretrain data, or need faster training.

**Core Concepts:**

**Why It Works:**
- Early CNN layers learn general features (edges, corners)
- Later layers learn task-specific features
- General features transfer across tasks

**Transfer Learning Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Feature Extraction** | Freeze pretrained layers, train only new head | Very small dataset, similar domain |
| **Fine-tuning** | Unfreeze some/all layers, train with low LR | Medium dataset, related domain |
| **Full Training** | Use pretrained weights as initialization | Large dataset |

**When to Apply:**
- Limited labeled data (< 10,000 images)
- Similar visual domain to pretrain dataset
- Faster convergence needed
- Resource constraints

**When NOT to Apply:**
- Vastly different domain (e.g., ImageNet → X-ray)
- Sufficient data to train from scratch
- Privacy concerns with pretrained models

**Python Example:**
```python
import torch
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Strategy 1: Feature extraction (freeze backbone)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for new task
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Strategy 2: Fine-tuning (unfreeze later layers)
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Use lower learning rate for pretrained layers
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

**Interview Tip:** Start with frozen backbone, train head, then gradually unfreeze layers from top to bottom if more data is available.

---

### Question 24
- [ ] Done
**Discuss using pre-trained models in computer vision.**

**Answer:**

Pre-trained models are neural networks trained on large datasets (ImageNet, COCO) whose learned weights can be reused. They provide strong feature extractors out-of-the-box, reduce training time/data requirements, and often outperform training from scratch. Common sources: torchvision, timm, TensorFlow Hub.

**Core Concepts:**

**Popular Pre-trained Models:**

| Model | Parameters | Top-1 Acc | Use Case |
|-------|------------|-----------|----------|
| ResNet-50 | 25M | 76.1% | General backbone |
| EfficientNet-B0 | 5M | 77.1% | Efficient/mobile |
| ViT-B/16 | 86M | 81.8% | High accuracy |
| MobileNet-V3 | 5.4M | 75.2% | Mobile/edge |
| CLIP | 400M | Zero-shot | Multi-modal |

**Using Pre-trained Models:**

1. **As Feature Extractor:**
   - Remove classification head
   - Use features for downstream tasks

2. **Fine-tuning:**
   - Adapt to new dataset/classes
   - Train with small learning rate

3. **Zero-shot (CLIP-style):**
   - Use without task-specific training

**Model Sources:**
- `torchvision.models` - PyTorch official
- `timm` - Extensive model zoo (1000+ models)
- `transformers` - Vision Transformers
- TensorFlow Hub, Keras Applications

**Python Example:**
```python
import timm
import torch

# List available models
available = timm.list_models('*efficientnet*')

# Load pretrained model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)

# Use for inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)  # [B, 10]

# Feature extraction (no classification head)
feature_extractor = timm.create_model('resnet50', pretrained=True, 
                                       num_classes=0)  # Removes head
features = feature_extractor(input_tensor)  # [B, 2048]
```

**Common Pitfalls:**
- Wrong input normalization (must match training preprocessing)
- Wrong input size (some models require specific dimensions)
- Not matching RGB vs BGR channel order

**Interview Tip:** Always check the model's expected preprocessing (mean, std, size). Mismatched normalization is a silent killer.

---

## Object Detection & Classification

### Question 25
- [ ] Done
**What's the difference between object detection and image classification?**

**Answer:**

Image classification assigns a single label to the entire image (what is in the image?). Object detection identifies multiple objects, providing both class labels AND bounding box locations (what objects are where?). Detection = classification + localization for potentially multiple instances.

**Core Differences:**

| Aspect | Classification | Detection |
|--------|---------------|-----------|
| **Output** | Single class label | Multiple (class, bbox) pairs |
| **Localization** | No | Yes (bounding boxes) |
| **Multiple objects** | No (or multi-label) | Yes |
| **Complexity** | Lower | Higher |
| **Examples** | "This is a cat" | "Cat at (x1,y1,x2,y2), Dog at (...)" |

**Output Comparison:**

```
Classification: {class: "dog", confidence: 0.95}

Detection: [
    {class: "dog", confidence: 0.95, bbox: [100, 50, 200, 150]},
    {class: "cat", confidence: 0.87, bbox: [300, 100, 400, 250]},
    {class: "car", confidence: 0.92, bbox: [50, 200, 150, 280]}
]
```

**Architectures:**

| Task | Architectures |
|------|---------------|
| Classification | ResNet, EfficientNet, ViT |
| Detection | YOLO, Faster R-CNN, SSD, DETR |

**Detection Components:**
1. **Backbone:** Feature extraction (ResNet, CSPDarknet)
2. **Neck:** Feature aggregation (FPN, PANet)
3. **Head:** Predict boxes + classes

**Python Example - Classification vs Detection:**
```python
# Classification
from torchvision.models import resnet50
classifier = resnet50(pretrained=True)
# Output: [batch, 1000] class probabilities

# Detection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
detector = fasterrcnn_resnet50_fpn(pretrained=True)
detector.eval()
# Output: list of {boxes: [N, 4], labels: [N], scores: [N]}
```

**Interview Tip:** Detection is harder because it requires predicting a variable number of outputs (boxes) per image, unlike classification's fixed output size.

---

### Question 26
- [ ] Done
**What algorithms can you use for real-time object detection?**

**Answer:**

Real-time detectors prioritize speed while maintaining accuracy. Single-stage detectors (YOLO, SSD) are fastest as they predict in one pass. YOLO variants dominate real-time detection, with YOLOv8/v10 achieving >100 FPS. For edge devices, use MobileNet-SSD, YOLO-Nano, or apply quantization/pruning.

**Real-Time Detectors:**

| Model | Speed (FPS) | mAP | Architecture |
|-------|-------------|-----|--------------|
| YOLOv8-n | ~200 | 37.3 | Single-stage, anchor-free |
| YOLOv8-s | ~130 | 44.9 | Single-stage, anchor-free |
| YOLOv10 | ~150 | 46.3 | NMS-free |
| SSD-MobileNet | ~60 | 21.0 | Single-stage |
| RT-DETR | ~100 | 53.1 | Transformer-based |

**Single-Stage vs Two-Stage:**

| Aspect | Single-Stage (YOLO, SSD) | Two-Stage (Faster R-CNN) |
|--------|-------------------------|--------------------------|
| Speed | Fast (real-time) | Slower |
| Accuracy | Good | Better |
| Pipeline | One forward pass | Region proposal + classification |
| Use case | Real-time, edge | High accuracy requirements |

**Optimization Techniques:**

1. **Model Architecture:**
   - Use efficient backbones (MobileNet, CSPNet)
   - Depthwise separable convolutions

2. **Model Compression:**
   - Quantization (FP32 → INT8)
   - Pruning (remove unnecessary weights)
   - Knowledge distillation

3. **Inference Optimization:**
   - TensorRT, ONNX Runtime
   - Batch inference
   - Hardware-specific optimizations

**Python Example:**
```python
from ultralytics import YOLO

# Load YOLOv8 nano (fastest)
model = YOLO('yolov8n.pt')

# Inference on image
results = model('image.jpg')

# Real-time video inference
results = model.predict(source=0, stream=True)  # 0 = webcam
for r in results:
    boxes = r.boxes
    # Process boxes in real-time
```

**Interview Tip:** YOLO series is the go-to for real-time detection. Mention FPS targets (30 FPS for video, 60+ for games) and trade-offs with accuracy.

---

### Question 27
- [ ] Done
**Explain the YOLO approach to object detection.**

**Answer:**

YOLO (You Only Look Once) frames detection as a single regression problem. It divides the image into a grid, and each cell predicts bounding boxes and class probabilities directly in one forward pass. This unified architecture enables real-time detection by eliminating the region proposal step of two-stage detectors.

**Core Concepts:**

**Key Idea:**
- Divide image into S×S grid
- Each cell predicts B bounding boxes + confidence + C class probabilities
- Single forward pass → all predictions simultaneously

**YOLO Output:**
Each cell predicts: $(x, y, w, h, confidence, class_1, class_2, ..., class_C)$
- $(x, y)$: Box center relative to grid cell
- $(w, h)$: Box dimensions relative to image
- $confidence = P(object) \times IoU$

**YOLO Evolution:**

| Version | Key Innovation |
|---------|---------------|
| YOLOv1 | Original grid-based approach |
| YOLOv2 | Anchor boxes, batch norm, multi-scale |
| YOLOv3 | Multi-scale predictions, FPN-like |
| YOLOv4 | CSPDarknet, mosaic augmentation |
| YOLOv5 | PyTorch, easier training |
| YOLOv8 | Anchor-free, decoupled head |
| YOLOv10 | NMS-free, end-to-end |

**Architecture (YOLOv8):**
```
Input → Backbone (CSPDarknet) → Neck (PANet) → Detection Heads
                                                ├─ P3 (small objects)
                                                ├─ P4 (medium objects)
                                                └─ P5 (large objects)
```

**Loss Function:**
$$\mathcal{L} = \lambda_{coord} \mathcal{L}_{box} + \mathcal{L}_{obj} + \lambda_{cls} \mathcal{L}_{cls}$$

**Python Example:**
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8m.pt')

# Train on custom data
model.train(data='dataset.yaml', epochs=100, imgsz=640)

# Inference
results = model.predict('image.jpg', conf=0.5)

for r in results:
    boxes = r.boxes.xyxy    # [N, 4] - x1, y1, x2, y2
    scores = r.boxes.conf   # [N] - confidence
    classes = r.boxes.cls   # [N] - class indices
```

**Interview Tip:** YOLO's speed comes from single-pass prediction. The trade-off is slightly lower accuracy on small objects compared to two-stage detectors.

---

### Question 28
- [ ] Done
**Discuss the R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN).**

**Answer:**

The R-CNN family uses a two-stage approach: first generate region proposals, then classify each region. R-CNN was slow (per-region CNN), Fast R-CNN shared computation via ROI pooling, and Faster R-CNN integrated proposal generation with Region Proposal Network (RPN), achieving near real-time speeds.

**Evolution:**

| Model | Region Proposals | Feature Sharing | Speed |
|-------|-----------------|-----------------|-------|
| R-CNN | Selective Search (external) | None | ~47s/image |
| Fast R-CNN | Selective Search (external) | CNN features shared | ~2s/image |
| Faster R-CNN | RPN (learned) | Full end-to-end | ~0.2s/image |

**R-CNN (2014):**
```
Image → Selective Search (2000 regions) → CNN per region → SVM classifier
```
- Slow: CNN runs 2000 times per image
- Not end-to-end

**Fast R-CNN (2015):**
```
Image → CNN (once) → ROI Pooling → FC layers → Class + BBox
```
- Key: Compute CNN features once, share across regions
- ROI Pooling: Extract fixed-size features from any region

**Faster R-CNN (2016):**
```
Image → CNN → RPN (proposals) → ROI Pooling → Class + BBox
           ↓
    Feature Map (shared)
```
- Region Proposal Network: 3×3 conv + anchors
- Anchors: predefined boxes at multiple scales/ratios
- Fully end-to-end trainable

**Faster R-CNN Components:**
1. **Backbone:** ResNet, VGG (feature extraction)
2. **RPN:** Predicts objectness + refines anchor boxes
3. **ROI Pooling/Align:** Extract fixed features from proposals
4. **Detection Head:** Final classification + bbox regression

**Python Example:**
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference (input: list of tensors [C, H, W])
images = [torch.randn(3, 800, 600)]
predictions = model(images)

# predictions[0]: {'boxes': [N,4], 'labels': [N], 'scores': [N]}
```

**Interview Tip:** Know the bottleneck each version solved: R-CNN's repeated CNN → Fast R-CNN's external proposals → Faster R-CNN's learned RPN.

---

### Question 29
- [ ] Done
**Compare one-stage vs. two-stage detectors for object detection.**

**Answer:**

Two-stage detectors (Faster R-CNN) first generate region proposals, then classify them—achieving higher accuracy but slower speed. One-stage detectors (YOLO, SSD) directly predict boxes and classes in a single pass—faster but traditionally less accurate. Modern one-stage detectors have closed the accuracy gap.

**Core Comparison:**

| Aspect | One-Stage (YOLO, SSD) | Two-Stage (Faster R-CNN) |
|--------|----------------------|--------------------------|
| **Pipeline** | Single forward pass | Region proposal + classification |
| **Speed** | Fast (real-time) | Slower |
| **Accuracy** | Good (catching up) | Higher |
| **Small Objects** | Challenging | Better (with FPN) |
| **Complexity** | Simpler | More components |
| **Training** | End-to-end | Can be end-to-end |

**Architecture Comparison:**

**One-Stage:**
```
Image → Backbone → Neck → Prediction Heads → NMS → Detections
                   (directly predicts boxes + classes)
```

**Two-Stage:**
```
Image → Backbone → RPN (proposals) → ROI Align → Head → Detections
                   ↓
              Region Proposals (stage 1)
                               ↓
                        Classification (stage 2)
```

**When to Choose:**

| Scenario | Choice |
|----------|--------|
| Real-time applications | One-stage (YOLO) |
| Maximum accuracy needed | Two-stage (Faster R-CNN) |
| Edge/mobile deployment | One-stage (MobileNet-SSD) |
| Small object detection | Two-stage with FPN |
| Instance segmentation | Two-stage (Mask R-CNN) |

**Modern Developments:**
- **DETR:** Transformer-based, no NMS, competitive accuracy
- **YOLOv8/v10:** One-stage matching two-stage accuracy
- **RT-DETR:** Real-time transformer detector

**Interview Tip:** The gap is closing. YOLOv8 achieves accuracy comparable to Faster R-CNN while being 10× faster. Choice depends on latency requirements.

---

### Question 30
- [ ] Done
**How do image recognition models deal with occlusion?**

**Answer:**

Models handle occlusion through: data augmentation (random erasing, cutout), part-based representations (recognize visible parts independently), attention mechanisms (focus on non-occluded regions), and multi-scale features (different levels capture different object portions). Robust training on occluded examples is key.

**Core Strategies:**

**1. Data Augmentation:**
| Technique | Description |
|-----------|-------------|
| Random Erasing | Randomly mask rectangular regions |
| Cutout | Zero out random square patches |
| GridMask | Structured grid-based masking |
| Copy-paste | Paste objects creating occlusion |

**2. Architectural Solutions:**
- **Attention Mechanisms:** Focus on visible, discriminative parts
- **Part-based Models:** Detect object parts independently
- **Multi-scale Features:** FPN captures both local and global context
- **Deformable Convolutions:** Adapt receptive field to visible regions

**3. Training Strategies:**
- Include heavily occluded samples in training data
- Hard negative mining for occluded cases
- Self-supervised pre-training on masked images (MAE)

**4. Detection-Specific:**
- Soft-NMS handles overlapping/occluded objects better
- Repulsion loss pushes apart nearby detections
- CrowdDet for crowded scenes

**Python Example - Occlusion Augmentation:**
```python
import albumentations as A

# Augmentations for occlusion robustness
transform = A.Compose([
    A.CoarseDropout(
        max_holes=8, 
        max_height=32, 
        max_width=32,
        fill_value=0,
        p=0.5
    ),
    A.RandomGridShuffle(grid=(3, 3), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply augmentation
augmented = transform(image=image)
```

**Interview Tip:** Mention that CNNs are inherently robust to moderate occlusion due to hierarchical feature learning. Training with Cutout/Random Erasing explicitly improves this.

---

## Advanced Techniques

### Question 31
- [ ] Done
**What is pose estimation, and what are its applications?**

**Answer:**

Pose estimation detects human body keypoints (joints like wrists, elbows, shoulders) from images or video to understand body posture and movement. It outputs 2D or 3D coordinates of keypoints and their connections (skeleton). Applications include fitness tracking, gesture recognition, sports analysis, and AR.

**Core Concepts:**

**Types of Pose Estimation:**

| Type | Description | Output |
|------|-------------|--------|
| **Single-person** | One person in image | N keypoints (x, y) |
| **Multi-person** | Multiple people | Keypoints per person |
| **2D Pose** | Keypoints in image plane | (x, y) coordinates |
| **3D Pose** | Keypoints in 3D space | (x, y, z) coordinates |

**Approaches:**

**Top-Down:**
1. Detect all people (bounding boxes)
2. Apply single-person pose estimator to each box
- Pros: Higher accuracy
- Cons: Speed depends on number of people

**Bottom-Up:**
1. Detect all keypoints in image
2. Group keypoints into person instances
- Pros: Speed independent of people count
- Cons: Grouping is challenging

**Common Models:**
- **OpenPose:** Bottom-up, Part Affinity Fields
- **HRNet:** High-Resolution Net, maintains resolution
- **MediaPipe Pose:** Real-time, mobile-friendly
- **ViTPose:** Transformer-based, state-of-the-art

**Applications:**
- Fitness apps (form correction)
- Sports analytics (motion analysis)
- Gaming (gesture control)
- Healthcare (gait analysis, rehabilitation)
- AR/VR (avatar animation)

**Python Example:**
```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Process image
image = cv2.imread('person.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# Draw keypoints
if results.pose_landmarks:
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Access individual keypoints
    for landmark in results.pose_landmarks.landmark:
        x, y = landmark.x, landmark.y  # Normalized coordinates
```

**Interview Tip:** MediaPipe is fastest for real-time. HRNet/ViTPose for accuracy. Know the difference between top-down and bottom-up approaches.

---

### Question 32
- [ ] Done
**How does optical flow contribute to understanding motion in videos?**

**Answer:**

Optical flow estimates per-pixel motion vectors between consecutive frames, showing how each pixel moves from frame t to t+1. It captures apparent motion patterns, enabling motion segmentation, video stabilization, action recognition, and tracking. Dense flow provides motion for every pixel; sparse flow tracks selected points.

**Core Concepts:**

**Optical Flow Output:**
- For each pixel $(x, y)$: motion vector $(u, v)$
- $u$ = horizontal displacement, $v$ = vertical displacement
- Visualized as color-coded flow field (hue = direction, saturation = magnitude)

**Brightness Constancy Assumption:**
$$I(x, y, t) = I(x + u, y + v, t + 1)$$

**Optical Flow Constraint Equation:**
$$I_x u + I_y v + I_t = 0$$

Where $I_x, I_y$ = spatial gradients, $I_t$ = temporal gradient

**Types:**

| Type | Description | Methods |
|------|-------------|---------|
| **Sparse** | Track specific points | Lucas-Kanade |
| **Dense** | Flow for every pixel | Farneback, RAFT |

**Classical Methods:**
- **Lucas-Kanade:** Sparse, assumes local constancy
- **Horn-Schunck:** Dense, global smoothness constraint
- **Farneback:** Dense, polynomial expansion

**Deep Learning Methods:**
- **FlowNet/FlowNet2:** CNN-based, fast
- **RAFT:** Recurrent All-pairs Field Transforms, state-of-the-art
- **PWC-Net:** Pyramid, warping, cost volume

**Applications:**
- Video object tracking
- Action recognition
- Video stabilization
- Autonomous vehicle motion estimation
- Video interpolation

**Python Example:**
```python
import cv2
import numpy as np

# Read two consecutive frames
frame1 = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)

# Dense optical flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(
    frame1, frame2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# flow shape: (H, W, 2) -> [u, v] per pixel
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
```

**Interview Tip:** Dense optical flow is computationally expensive. For real-time applications, use sparse flow (Lucas-Kanade) or efficient deep models (RAFT-Small).

---

### Question 33
- [ ] Done
**Explain how CNNs can be used for human activity recognition in video data.**

**Answer:**

CNNs extract spatial features from video frames, combined with temporal modeling for activity recognition. Approaches include: 2D CNNs on individual frames + temporal aggregation (LSTM/pooling), 3D CNNs (spatiotemporal convolutions), two-stream networks (RGB + optical flow), and video transformers for long-range dependencies.

**Core Approaches:**

**1. 2D CNN + Temporal Aggregation:**
```
Frame 1 → CNN → Features ─┐
Frame 2 → CNN → Features ─┼→ Temporal Model (LSTM/Pool) → Class
Frame 3 → CNN → Features ─┘
```
- Extract per-frame features with pretrained CNN
- Aggregate with LSTM, temporal pooling, or attention

**2. 3D CNN (Spatiotemporal Convolutions):**
```
Video Clip [T, H, W, C] → 3D Conv layers → 3D Pool → FC → Class
```
- C3D, I3D: 3D kernels capture both space and time
- Computationally expensive

**3. Two-Stream Networks:**
```
RGB frames → Spatial CNN ─┐
                          ├→ Fusion → Class
Optical flow → Motion CNN ┘
```
- Separate streams for appearance and motion
- Complementary information

**4. Video Transformers:**
- ViViT, Video Swin: Attention across space and time
- Handle long-range temporal dependencies

**Model Comparison:**

| Model | Input | Temporal | Pros |
|-------|-------|----------|------|
| 2D CNN + LSTM | Frames | Sequential | Simple, pretrained weights |
| C3D/I3D | Clips | 3D conv | End-to-end, captures motion |
| Two-Stream | RGB + Flow | Parallel | Strong motion features |
| ViViT | Patches | Attention | Long-range, SOTA |

**Python Example:**
```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class ActivityRecognizer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 2D CNN backbone (remove FC)
        self.cnn = resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()
        
        # Temporal modeling
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, video):
        # video: [B, T, C, H, W]
        B, T = video.shape[:2]
        
        # Extract frame features
        frames = video.view(B * T, *video.shape[2:])
        features = self.cnn(frames)  # [B*T, 2048]
        features = features.view(B, T, -1)  # [B, T, 2048]
        
        # Temporal aggregation
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])  # Last timestep
        return out
```

**Interview Tip:** I3D inflated from ImageNet weights is a strong baseline. For efficiency, use 2D CNN with temporal pooling. For SOTA, use Video Swin or ViViT.

---

### Question 34
- [ ] Done
**What are Generative Adversarial Networks (GANs) and their role in computer vision?**

**Answer:**

GANs consist of two neural networks—generator (creates fake images) and discriminator (distinguishes real from fake)—trained adversarially. The generator learns to produce realistic images to fool the discriminator. In CV, GANs enable image synthesis, super-resolution, style transfer, data augmentation, and image-to-image translation.

**Core Concepts:**

**Architecture:**
```
Random Noise (z) → Generator → Fake Image
                                    ↓
                              Discriminator ← Real Image
                                    ↓
                              Real or Fake?
```

**Training Dynamics:**
- **Generator:** Minimize $\log(1 - D(G(z)))$ → fool discriminator
- **Discriminator:** Maximize $\log(D(x)) + \log(1 - D(G(z)))$ → distinguish real/fake

**Minimax Objective:**
$$\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

**Computer Vision Applications:**

| Application | GAN Type |
|-------------|----------|
| Image synthesis | StyleGAN, BigGAN |
| Super-resolution | SRGAN, ESRGAN |
| Image-to-image | Pix2Pix, CycleGAN |
| Data augmentation | Conditional GANs |
| Inpainting | DeepFill, LaMa |
| Face aging | Age-cGAN |

**Key Variants:**
- **DCGAN:** Convolutional architecture, stable training
- **StyleGAN:** High-quality faces, style control
- **CycleGAN:** Unpaired image translation
- **Conditional GAN:** Control output with labels

**Python Example:**
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),  # 28x28 image
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
```

**Interview Tip:** GANs are being replaced by diffusion models for image generation (better quality, stable training), but remain important for image-to-image tasks.

---

### Question 35
- [ ] Done
**Discuss few-shot learning and its challenges in computer vision.**

**Answer:**

Few-shot learning trains models to recognize new classes from very few examples (1-5 images per class). It addresses the data scarcity problem by learning to learn—using meta-learning, metric learning, or transfer learning. Challenges include overfitting to limited samples, domain shift, and evaluating generalization properly.

**Core Concepts:**

**Problem Setting:**
- **N-way K-shot:** Classify among N classes, K examples each
- **Support set:** Few labeled examples for new classes
- **Query set:** Images to classify

**Approaches:**

| Method | Idea |
|--------|------|
| **Metric Learning** | Learn embedding space, classify by similarity |
| **Meta-Learning** | Learn to adapt quickly from few examples |
| **Transfer Learning** | Pretrain on large data, fine-tune on few samples |
| **Data Augmentation** | Generate synthetic examples |

**Metric Learning (Siamese, Prototypical):**
- Learn embeddings where same-class samples are close
- Classify query by nearest neighbor in embedding space

**Meta-Learning (MAML, Reptile):**
- Train on many few-shot tasks
- Learn initialization that adapts quickly

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Overfitting | Few samples → easy to memorize |
| Domain shift | Support set may differ from deployment |
| Intra-class variation | Few samples don't cover class diversity |
| Evaluation | Standard benchmarks may not reflect real use |

**Python Example - Prototypical Networks:**
```python
import torch
import torch.nn.functional as F

def prototypical_loss(embeddings, labels, n_way, k_shot, n_query):
    """
    embeddings: [n_way * (k_shot + n_query), embed_dim]
    """
    # Reshape
    support = embeddings[:n_way * k_shot]  # Support set
    query = embeddings[n_way * k_shot:]     # Query set
    
    # Compute prototypes (class mean)
    support = support.view(n_way, k_shot, -1)
    prototypes = support.mean(dim=1)  # [n_way, embed_dim]
    
    # Compute distances from query to prototypes
    dists = torch.cdist(query, prototypes)  # [n_query, n_way]
    
    # Classification: nearest prototype
    log_probs = F.log_softmax(-dists, dim=1)
    return log_probs  # Use cross-entropy loss
```

**Interview Tip:** Prototypical Networks are a simple, strong baseline. For real applications, pretrained models (CLIP, DINO) with linear probing often outperform specialized few-shot methods.

---

### Question 36
- [ ] Done
**Explain zero-shot learning in image recognition.**

**Answer:**

Zero-shot learning recognizes classes never seen during training by leveraging auxiliary information (text descriptions, attributes, or embeddings). Models learn a shared embedding space where images and class descriptions can be compared. CLIP exemplifies this—matching images to text descriptions enables classification without class-specific training.

**Core Concepts:**

**Problem Setting:**
- **Seen classes:** Used for training
- **Unseen classes:** Never seen, must recognize at test time
- **Auxiliary info:** Text, attributes, or semantic embeddings describe classes

**How It Works:**
```
Training: Learn to map images → shared embedding space
          Learn to map text/attributes → same space

Inference: Image → embedding
           Compare with all class embeddings (including unseen)
           Predict nearest class
```

**Approaches:**

| Method | Auxiliary Information |
|--------|----------------------|
| Attribute-based | Binary attributes per class |
| Text-based (CLIP) | Natural language descriptions |
| Semantic embeddings | Word2Vec, class hierarchies |

**CLIP Architecture:**
```
Image → Image Encoder → Image Embedding
Text  → Text Encoder  → Text Embedding
                            ↓
              Contrastive matching (cosine similarity)
```

**Python Example - CLIP Zero-Shot:**
```python
import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load image
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)

# Define classes (including unseen)
text = clip.tokenize(["a dog", "a cat", "a bird", "a car"]).to(device)

# Zero-shot classification
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Compute similarity
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    predicted_class = similarity.argmax().item()
```

**Challenges:**
- Unseen classes must be describable in auxiliary space
- Domain gap between training and test classes
- Bias toward seen classes in generalized zero-shot learning

**Interview Tip:** CLIP represents a paradigm shift—massive web-scale training enables strong zero-shot transfer. Combine with prompting for best results.

---

### Question 37
- [ ] Done
**How can reinforcement learning be applied to problems in computer vision?**

**Answer:**

RL in CV uses visual input as state and learns policies for sequential decision-making tasks. Applications include active perception (where to look next), visual navigation, robotic manipulation, attention mechanisms (where to attend), and adversarial training. The agent learns to take actions that maximize cumulative visual task reward.

**Core Concepts:**

**RL Framework Applied to Vision:**
```
State: Visual input (image/video)
Action: Where to look, how to move, what to attend
Reward: Task-specific (navigation success, recognition accuracy)
Policy: CNN/ViT that maps images to actions
```

**Applications:**

| Application | State | Action | Reward |
|-------------|-------|--------|--------|
| Visual Navigation | Camera view | Move/turn | Reach goal |
| Active Object Recognition | Current view | Move camera | Correct classification |
| Hard Attention | Image | Select region | Recognition accuracy |
| Image Editing | Current image | Apply edit | User preference |
| Video Summarization | Video frame | Include/skip | Summary quality |

**Key Architectures:**

**1. Visual Navigation (PointGoal):**
```
RGB + Depth → CNN → Policy (move forward, turn left/right, stop)
```

**2. Visual Attention (Hard):**
- Agent decides where to look (glimpse location)
- Recurrent Attention Model (RAM)

**3. Neural Architecture Search:**
- RL agent designs CNN architecture
- Reward = validation accuracy

**Challenges:**
- Sample inefficiency (vision models are expensive)
- Sparse rewards in visual tasks
- Sim-to-real gap for robotics

**Python Example - Simple Visual RL:**
```python
import gym
import torch
import torch.nn as nn

class VisualPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # CNN encoder for visual input
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 9 * 9, 256)
        self.policy = nn.Linear(256, action_dim)
    
    def forward(self, state):
        # state: [B, C, H, W] image
        x = self.cnn(state)
        x = torch.relu(self.fc(x))
        action_probs = torch.softmax(self.policy(x), dim=-1)
        return action_probs
```

**Interview Tip:** RL in CV is computationally expensive but powerful for active perception and embodied AI. Simulation (Habitat, AI2-THOR) is crucial for training.

---

### Question 38
- [ ] Done
**What are Siamese networks and where are they applicable?**

**Answer:**

Siamese networks use two identical subnetworks (shared weights) to process two inputs and compare their embeddings. They learn similarity metrics rather than class labels, making them ideal for verification tasks (same/different), one-shot learning, and tracking. The network outputs a similarity score between input pairs.

**Core Concepts:**

**Architecture:**
```
Input A → Encoder → Embedding A ─┐
                                  ├→ Distance/Similarity → Same/Different
Input B → Encoder → Embedding B ─┘
    (shared weights)
```

**Key Properties:**
- **Weight Sharing:** Same network for both inputs
- **Metric Learning:** Learns to compare, not classify
- **Generalization:** Can handle unseen classes

**Loss Functions:**

**Contrastive Loss:**
$$L = (1-y) \frac{1}{2} D^2 + y \frac{1}{2} \max(0, m - D)^2$$
Where $y=1$ for different, $D$ = distance, $m$ = margin

**Triplet Loss:**
$$L = \max(0, D(a, p) - D(a, n) + m)$$
Push positive pairs closer, negatives apart

**Applications:**

| Application | Input Pair | Output |
|-------------|-----------|--------|
| Face verification | Two face images | Same person? |
| Signature verification | Two signatures | Match? |
| One-shot recognition | Query, support image | Same class? |
| Visual tracking | Current frame, target | Similarity map |
| Duplicate detection | Two images | Duplicate? |

**Python Example:**
```python
import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256)
        )
    
    def forward_one(self, x):
        return self.encoder(x)
    
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = torch.pairwise_distance(emb1, emb2)
    loss = (1 - label) * dist.pow(2) + \
           label * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()
```

**Interview Tip:** Siamese networks are great when you need to compare inputs rather than classify them, especially with limited training examples per class.

---

## Model Optimization & Evaluation

### Question 39
- [ ] Done
**How do you handle overfitting in a computer vision model?**

**Answer:**

Handle overfitting through: data augmentation (most effective in CV), regularization (dropout, weight decay), early stopping, transfer learning from pretrained models, reducing model complexity, and using more training data. For CV specifically, augmentation is crucial since images have natural transformations that preserve semantics.

**Core Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Data Augmentation** | Create varied training samples | Always, first choice |
| **Dropout** | Random neuron deactivation | FC layers |
| **Weight Decay** | L2 regularization | Always |
| **Early Stopping** | Stop when val loss increases | Always |
| **Transfer Learning** | Use pretrained weights | Limited data |
| **Reduce Model Size** | Fewer parameters | Clear overfitting |
| **Batch Normalization** | Regularization effect | Standard practice |

**Data Augmentation (Most Important for CV):**
- Geometric: flip, rotate, crop, scale
- Photometric: brightness, contrast, color
- Advanced: Mixup, CutMix, AutoAugment

**Regularization Techniques:**

$$L_{total} = L_{task} + \lambda ||w||_2^2 \quad \text{(Weight Decay)}$$

**Signs of Overfitting:**
- Training accuracy >> Validation accuracy
- Training loss decreasing, validation loss increasing
- Gap widens with more epochs

**Python Example:**
```python
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. Data Augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# 2. Model with Dropout
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout before FC
            nn.Linear(128 * 54 * 54, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

# 3. Weight Decay in Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 4. Early Stopping
best_val_loss = float('inf')
patience_counter = 0
for epoch in range(epochs):
    val_loss = validate(model)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

**Interview Tip:** In CV, data augmentation is often more effective than regularization. Combine augmentation with transfer learning for small datasets.

---

### Question 40
- [ ] Done
**What are some common metrics to evaluate a computer vision system's performance?**

**Answer:**

Metrics depend on the task: Classification uses accuracy, precision, recall, F1-score, and AUC-ROC. Detection uses mAP (mean Average Precision) and IoU. Segmentation uses IoU/Dice coefficient and pixel accuracy. Generation uses FID and Inception Score. Choose metrics aligned with business requirements.

**Metrics by Task:**

**Classification:**

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN) / Total | Balanced classes |
| Precision | TP / (TP+FP) | Minimize false positives |
| Recall | TP / (TP+FN) | Minimize false negatives |
| F1-Score | 2·P·R / (P+R) | Balance P and R |
| AUC-ROC | Area under ROC | Binary classification |

**Object Detection:**

| Metric | Description |
|--------|-------------|
| IoU | Intersection over Union of boxes |
| mAP | Mean AP across classes |
| mAP@0.5 | AP at IoU threshold 0.5 |
| mAP@0.5:0.95 | Average across IoU thresholds |

**Segmentation:**

| Metric | Formula |
|--------|---------|
| Pixel Accuracy | Correct pixels / Total pixels |
| IoU (Jaccard) | Intersection / Union |
| Dice Coefficient | 2·Intersection / (Pred + GT) |
| Boundary F1 | F1 score on boundary pixels |

**Image Generation:**

| Metric | Description |
|--------|-------------|
| FID | Fréchet Inception Distance (lower = better) |
| IS | Inception Score (higher = better) |
| LPIPS | Perceptual similarity |

**Python Example:**
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Classification metrics
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

# IoU for segmentation
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

# Dice coefficient
def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return 2 * intersection / (pred_mask.sum() + gt_mask.sum())
```

**Interview Tip:** Always clarify what metric matters for the business. High recall for medical diagnosis (don't miss disease), high precision for spam (don't block good emails).

---

### Question 41
- [ ] Done
**Discuss the importance of cross-validation for a vision model.**

**Answer:**

Cross-validation provides robust performance estimates by training and evaluating on multiple data splits, reducing variance from a single train-test split. For vision models with limited data, it maximizes use of available samples. K-fold CV is standard; stratified splits maintain class balance. However, it's expensive for large models.

**Core Concepts:**

**Why Cross-Validation:**
- Single train-test split can be lucky/unlucky
- Provides confidence intervals for performance
- Uses all data for both training and validation
- Detects overfitting to specific data splits

**Types of Cross-Validation:**

| Type | Description | Use Case |
|------|-------------|----------|
| K-Fold | Split into K parts, rotate test set | Standard |
| Stratified K-Fold | Maintain class proportions | Imbalanced classes |
| Leave-One-Out | K = N (expensive) | Very small datasets |
| Group K-Fold | Keep groups together | Patient data, video frames |

**K-Fold Process:**
```
Data: [Fold1, Fold2, Fold3, Fold4, Fold5]

Iteration 1: Train on [2,3,4,5], Test on [1]
Iteration 2: Train on [1,3,4,5], Test on [2]
...
Final: Average metrics across all folds
```

**CV for Vision Models:**

**When to Use:**
- Small datasets (< 10,000 images)
- Model selection and hyperparameter tuning
- Comparing architectures fairly

**When to Skip:**
- Large datasets (ImageNet-scale)
- Expensive models (training time prohibitive)
- Use single validation set instead

**Python Example:**
```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Stratified K-Fold (maintains class balance)
X = np.array(image_paths)
y = np.array(labels)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model on this fold
    model = create_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    val_acc = model.evaluate(X_val, y_val)
    fold_results.append(val_acc)
    print(f"Fold {fold+1}: {val_acc:.4f}")

print(f"Mean: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
```

**Interview Tip:** For deep learning, full K-fold is often impractical. Use single validation split for training, K-fold for final model comparison or small datasets.

---

### Question 42
- [ ] Done
**Explain how Intersection over Union (IoU) works for object detection models.**

**Answer:**

IoU measures overlap between predicted and ground-truth bounding boxes by computing intersection area divided by union area. It ranges from 0 (no overlap) to 1 (perfect match). IoU is used to determine true/false positives at various thresholds (typically 0.5) and to calculate mAP. It's also used as a loss function variant.

**Mathematical Formulation:**

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Intersection Area}}{\text{Union Area}}$$

**Computing IoU:**
```
Box A: (x1_a, y1_a, x2_a, y2_a)
Box B: (x1_b, y1_b, x2_b, y2_b)

Intersection:
  x1_i = max(x1_a, x1_b)
  y1_i = max(y1_a, y1_b)
  x2_i = min(x2_a, x2_b)
  y2_i = min(y2_a, y2_b)
  area_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

Union:
  area_a = (x2_a - x1_a) * (y2_a - y1_a)
  area_b = (x2_b - x1_b) * (y2_b - y1_b)
  area_u = area_a + area_b - area_i

IoU = area_i / area_u
```

**IoU Thresholds:**

| Threshold | Meaning |
|-----------|---------|
| IoU > 0.5 | PASCAL VOC standard |
| IoU > 0.75 | Stricter matching |
| mAP@0.5:0.95 | Average across 0.5, 0.55, ..., 0.95 (COCO) |

**IoU Loss Variants:**

| Loss | Improvement |
|------|-------------|
| IoU Loss | $L = 1 - IoU$ |
| GIoU | Handles non-overlapping boxes |
| DIoU | Considers center distance |
| CIoU | Adds aspect ratio penalty |

**Python Example:**
```python
import numpy as np

def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# Example
pred_box = [100, 100, 200, 200]
gt_box = [120, 120, 220, 220]
print(f"IoU: {compute_iou(pred_box, gt_box):.3f}")
```

**Interview Tip:** IoU = 0.5 is "good enough" match; IoU > 0.75 is "precise" match. COCO mAP averages across thresholds for comprehensive evaluation.

---

## Real-World Applications

### Question 43
- [ ] Done
**How would you design a vision system for automatic license plate recognition?**

**Answer:**

ALPR pipeline: (1) Detect vehicle → (2) Localize license plate region → (3) Segment characters → (4) Recognize characters via OCR. Use YOLO/SSD for detection, perspective correction for tilted plates, and CNN+CTC or transformer OCR. Handle variable plate formats, lighting conditions, and motion blur.

**System Design:**

```
Input Image → Vehicle Detection → Plate Localization → Preprocessing 
                                                            ↓
                              Output ← Post-processing ← OCR Recognition
```

**Pipeline Components:**

| Stage | Method | Output |
|-------|--------|--------|
| Vehicle Detection | YOLO, Faster R-CNN | Vehicle bounding boxes |
| Plate Localization | YOLO, edge detection | Plate region |
| Preprocessing | Deskew, binarization | Clean plate image |
| OCR | CRNN+CTC, TrOCR | Character sequence |
| Post-processing | Regex, format validation | Validated plate number |

**Key Challenges:**

| Challenge | Solution |
|-----------|----------|
| Variable plate sizes | Multi-scale detection |
| Perspective distortion | Perspective transform, spatial transformer |
| Low light / night | IR cameras, image enhancement |
| Motion blur | Deblurring, high frame rate |
| Different formats | Region-specific training data |
| Occlusion | Multiple frames, tracking |

**Design Considerations:**
- **Speed:** Real-time processing for traffic systems
- **Accuracy:** >95% character accuracy typical requirement
- **Edge deployment:** Optimize for embedded hardware

**Python Example - Simplified Pipeline:**
```python
import cv2
from ultralytics import YOLO
import easyocr

class LicensePlateRecognizer:
    def __init__(self):
        self.plate_detector = YOLO('plate_detector.pt')
        self.ocr = easyocr.Reader(['en'])
    
    def preprocess_plate(self, plate_img):
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    
    def recognize(self, image):
        # Detect plate
        results = self.plate_detector(image)
        plates = []
        
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_img = image[y1:y2, x1:x2]
            
            # Preprocess
            processed = self.preprocess_plate(plate_img)
            
            # OCR
            text = self.ocr.readtext(processed, detail=0)
            plates.append(''.join(text).upper())
        
        return plates
```

**Interview Tip:** Discuss real-world constraints: camera placement, lighting, speed limits, privacy regulations. Mention tracking for temporal voting across frames.

---

### Question 44
- [ ] Done
**Outline the computer vision technologies involved in autonomous vehicle navigation.**

**Answer:**

Autonomous vehicles use multi-sensor fusion: cameras for object detection/lane recognition, LiDAR for 3D mapping, radar for velocity estimation. CV tasks include object detection (vehicles, pedestrians), lane detection, traffic sign recognition, depth estimation, SLAM, and motion prediction. Redundancy across sensors ensures safety.

**Sensor Suite:**

| Sensor | Purpose | Strengths |
|--------|---------|-----------|
| Cameras | Object detection, lane lines, signs | Rich semantic info, low cost |
| LiDAR | 3D point cloud, distance | Accurate depth, weather robust |
| Radar | Velocity, distance | Works in fog/rain |
| Ultrasonic | Short-range obstacles | Parking assistance |
| GPS/IMU | Localization | Global positioning |

**Computer Vision Tasks:**

| Task | Method | Output |
|------|--------|--------|
| Object Detection | YOLO, Faster R-CNN | Bounding boxes + classes |
| Semantic Segmentation | DeepLab, SegFormer | Pixel-wise road/lane/object |
| Lane Detection | LaneNet, SCNN | Lane polynomials |
| Depth Estimation | Stereo, monocular DL | Depth map |
| 3D Object Detection | PointPillars, BEVFusion | 3D bounding boxes |
| Traffic Sign Recognition | CNN classifier | Sign type |
| Motion Prediction | Trajectron++, VectorNet | Future trajectories |
| SLAM | ORB-SLAM, visual odometry | Localization + mapping |

**System Architecture:**
```
Sensors → Perception → Sensor Fusion → Planning → Control → Actuators
           ↓
      Detection, Segmentation, Tracking
           ↓
      3D Scene Understanding
```

**Key Challenges:**

| Challenge | Solution |
|-----------|----------|
| Adverse weather | Multi-sensor fusion, radar |
| Edge cases | Extensive simulation, rare event training |
| Real-time processing | Efficient models, TensorRT |
| Safety | Redundancy, uncertainty estimation |

**Python Example - Bird's Eye View Perception:**
```python
import torch
import torch.nn as nn

class BEVPerception(nn.Module):
    """Simplified Bird's Eye View perception"""
    def __init__(self):
        super().__init__()
        # Image backbone (multi-camera)
        self.img_backbone = ResNet50()
        
        # Transform to BEV
        self.img_to_bev = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            # View transformation (simplified)
        )
        
        # LiDAR backbone
        self.lidar_backbone = PointPillars()
        
        # Fusion
        self.fusion = nn.Conv2d(512, 256, 3, padding=1)
        
        # Detection head
        self.detection_head = nn.Conv2d(256, 10, 1)  # Classes + boxes
    
    def forward(self, images, lidar):
        img_feat = self.img_backbone(images)
        img_bev = self.img_to_bev(img_feat)
        
        lidar_bev = self.lidar_backbone(lidar)
        
        fused = self.fusion(torch.cat([img_bev, lidar_bev], dim=1))
        return self.detection_head(fused)
```

**Interview Tip:** Emphasize sensor fusion and redundancy for safety. Mention specific datasets (nuScenes, KITTI, Waymo) and the importance of handling edge cases.

---

### Question 45
- [ ] Done
**Propose an approach for medical image analysis using computer vision.**

**Answer:**

Medical image analysis uses specialized CV models for diagnosis assistance. Approach: (1) Domain-specific preprocessing (DICOM handling, windowing), (2) Pretrained models fine-tuned on medical data, (3) U-Net variants for segmentation, (4) Explainability via Grad-CAM. Address class imbalance, require regulatory compliance, and provide uncertainty estimates.

**System Design:**

```
DICOM Images → Preprocessing → Augmentation → Model → Post-processing → Clinical Output
                    ↓                             ↓
              Normalization                  Explainability
              Windowing                      Uncertainty
```

**Medical CV Tasks:**

| Task | Application | Methods |
|------|-------------|---------|
| Classification | Disease diagnosis | CNN, ViT with attention |
| Segmentation | Tumor delineation | U-Net, nnU-Net |
| Detection | Lesion finding | Faster R-CNN, YOLO |
| Registration | Image alignment | VoxelMorph, ANTs |
| Reconstruction | MRI acceleration | Deep unfolding, diffusion |

**Key Considerations:**

| Aspect | Requirement |
|--------|-------------|
| Data privacy | HIPAA compliance, federated learning |
| Class imbalance | Focal loss, oversampling |
| Explainability | Grad-CAM, attention maps |
| Uncertainty | Dropout inference, ensembles |
| Validation | Multi-site testing, clinical trials |
| Regulatory | FDA approval, CE marking |

**Architecture Choice:**
- **nnU-Net:** Self-configuring, SOTA for segmentation
- **DenseNet:** Good for classification, fewer parameters
- **3D Models:** For volumetric data (CT, MRI)

**Python Example - Medical Image Classification:**
```python
import torch
import torch.nn as nn
from torchvision import models
import pydicom

class MedicalImageClassifier:
    def __init__(self, num_classes):
        # Pretrained backbone
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, num_classes)
        
    def preprocess_dicom(self, dicom_path):
        # Load DICOM
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(float)
        
        # Windowing (for CT)
        if hasattr(dcm, 'WindowCenter'):
            center = dcm.WindowCenter
            width = dcm.WindowWidth
            img = np.clip(img, center - width/2, center + width/2)
        
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min())
        
        # Convert to 3-channel
        img = np.stack([img, img, img], axis=0)
        return torch.tensor(img, dtype=torch.float32)
    
    def predict_with_uncertainty(self, x, n_samples=10):
        """Monte Carlo Dropout for uncertainty"""
        self.model.train()  # Enable dropout
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = torch.softmax(self.model(x), dim=1)
                predictions.append(pred)
        
        mean_pred = torch.stack(predictions).mean(dim=0)
        uncertainty = torch.stack(predictions).std(dim=0)
        return mean_pred, uncertainty
```

**Interview Tip:** Medical AI requires explainability (clinicians need to trust), uncertainty (know when to refer to human), and rigorous validation (multiple hospitals, diverse populations).

---

### Question 46
- [ ] Done
**Discuss the use of computer vision in retail for product recognition and tracking.**

**Answer:**

Retail CV enables automated checkout, inventory management, and analytics. Systems use object detection for product recognition from shelf images, barcode/label reading, customer tracking for behavior analysis, and planogram compliance checking. Challenges include high SKU counts, similar products, and varying lighting conditions.

**Retail CV Applications:**

| Application | CV Task | Business Value |
|-------------|---------|----------------|
| Smart checkout | Product detection | Reduced wait times |
| Inventory counting | Object detection + counting | Stock accuracy |
| Planogram compliance | Object detection + matching | Shelf optimization |
| Customer analytics | Person tracking, pose | Store layout improvement |
| Loss prevention | Anomaly detection | Shrinkage reduction |
| Price tag validation | OCR | Pricing accuracy |

**System Architecture:**
```
Cameras → Edge Processing → Detection/Recognition → Cloud Analytics → Actions
              ↓
        Real-time inference
        Low latency required
```

**Technical Challenges:**

| Challenge | Solution |
|-----------|----------|
| High SKU count (10,000+) | Hierarchical classification, retrieval |
| Similar products | Fine-grained recognition |
| Varying lighting | Robust augmentation, domain adaptation |
| Occlusion on shelves | Multi-view cameras |
| Real-time requirements | Edge AI, quantized models |
| Privacy concerns | On-device processing, no face storage |

**Key Technologies:**
- **Product Detection:** YOLO, EfficientDet
- **Product Identification:** Image retrieval, embedding matching
- **Barcode/OCR:** Specialized readers, TrOCR
- **People Tracking:** DeepSORT, ByteTrack

**Python Example - Product Recognition System:**
```python
from ultralytics import YOLO
import faiss
import numpy as np

class ProductRecognitionSystem:
    def __init__(self):
        self.detector = YOLO('product_detector.pt')
        self.embedder = load_embedding_model()
        
        # Product database (embeddings + metadata)
        self.product_db = self.load_product_database()
        self.index = self.build_faiss_index()
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        embeddings = np.array(self.product_db['embeddings'])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    
    def recognize_products(self, shelf_image):
        # Detect products
        detections = self.detector(shelf_image)
        
        products = []
        for box in detections[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            product_crop = shelf_image[y1:y2, x1:x2]
            
            # Get embedding
            embedding = self.embedder(product_crop)
            
            # Search in database
            D, I = self.index.search(embedding.reshape(1, -1), k=1)
            
            if D[0][0] < threshold:
                product_id = self.product_db['ids'][I[0][0]]
                products.append({
                    'bbox': [x1, y1, x2, y2],
                    'product_id': product_id,
                    'confidence': 1 - D[0][0]
                })
        
        return products
```

**Interview Tip:** Discuss the scale challenge (thousands of SKUs) and the need for image retrieval rather than pure classification. Mention edge deployment for privacy and latency.

---

### Question 47
- [ ] Done
**How might AR applications benefit from advances in computer vision?**

**Answer:**

AR relies heavily on CV for understanding the real world: SLAM for device localization, plane detection for surface anchoring, object recognition for context, hand/face tracking for interaction, and depth estimation for occlusion. Advances in real-time CV enable more immersive, stable, and interactive AR experiences.

**CV Technologies Enabling AR:**

| CV Technology | AR Application |
|---------------|----------------|
| SLAM | Device tracking, world mapping |
| Plane Detection | Placing virtual objects on surfaces |
| Depth Estimation | Realistic occlusion handling |
| Object Detection | Object-anchored AR content |
| Hand Tracking | Gesture-based interaction |
| Face Tracking | AR filters, face effects |
| Semantic Segmentation | Environment understanding |
| Pose Estimation | Body-based AR effects |

**How CV Advances Benefit AR:**

| Advance | AR Benefit |
|---------|------------|
| Real-time processing | Smoother, lower latency |
| Better depth estimation | More realistic occlusion |
| Improved SLAM | More stable tracking |
| Neural rendering (NeRF) | Photorealistic virtual objects |
| Efficient models | On-device AR, longer battery |
| 3D reconstruction | Persistent AR worlds |

**AR Pipeline:**
```
Camera → Tracking/SLAM → Scene Understanding → Rendering → Display
             ↓                   ↓
        Camera pose        Planes, objects, depth
             ↓                   ↓
        Place content      Handle occlusion
```

**Key AR Platforms:**
- **ARKit (Apple):** LiDAR + CV for depth, plane detection
- **ARCore (Google):** Motion tracking, environmental understanding
- **Meta Quest:** Hand tracking, spatial understanding
- **HoloLens:** Full 3D mapping, spatial anchors

**Python Example - Simple AR Marker Detection:**
```python
import cv2
import numpy as np

class SimpleARSystem:
    def __init__(self):
        # ArUco marker dictionary
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Camera calibration (intrinsics)
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros(5)
    
    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        return corners, ids
    
    def get_marker_pose(self, corners, marker_size=0.05):
        """Get rotation and translation of marker"""
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, self.camera_matrix, self.dist_coeffs
        )
        return rvecs, tvecs
    
    def render_cube(self, frame, rvec, tvec, size=0.05):
        """Render virtual cube on marker"""
        # Define cube vertices
        cube_points = np.float32([
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
            [0, 0, -size], [size, 0, -size], [size, size, -size], [0, size, -size]
        ])
        
        # Project 3D points to 2D
        img_points, _ = cv2.projectPoints(
            cube_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        # Draw cube edges
        img_points = img_points.reshape(-1, 2).astype(int)
        # ... draw lines between points
        return frame
```

**Interview Tip:** AR success depends on robust tracking (SLAM) and understanding scene geometry (depth, planes). Discuss trade-offs between quality and real-time performance on mobile devices.

---



---

# --- Image Classification Questions (from 07_computer_vision/01_image_classification) ---

# Image Classification - Theory Questions

## Question 1
- [ ] Done
**How do you handle class imbalance in image classification datasets using techniques beyond simple oversampling?**

**Answer:**

Handling class imbalance in image classification requires sophisticated techniques beyond basic oversampling to avoid overfitting and ensure robustness.

### Advanced Techniques for Class Imbalance

#### 1. **Loss Function Modifications**

```python
import torch
import torch.nn as nn
import numpy as np

# Focal Loss - reduces well-classified example contribution
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

# Class-Balanced Loss with effective number of samples
def class_balanced_loss(labels, logits, samples_per_class, beta=0.9999):
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(samples_per_class)
    weights = torch.tensor(weights, dtype=torch.float32)
    return nn.functional.cross_entropy(logits, labels, weight=weights)
```

#### 2. **Sampling Strategies**

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Class-Balanced Sampling** | Sample each class equally | Simple, effective | May overfit rare classes |
| **Square-Root Sampling** | Sample proportional to √(class_freq) | Balanced trade-off | Requires tuning |
| **Progressively Balanced** | Gradually shift from instance-balanced to class-balanced | Learns both representation and decision boundary | More complex training |
| **Meta-learning Reweighting** | Learn sample weights via meta-gradient | Adaptive | Computational overhead |

#### 3. **Data-Level Approaches**

```python
from torchvision import transforms

# Advanced augmentation for minority classes
minority_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)),
    transforms.RandomErasing(p=0.5),
])

# Mixup between minority and majority classes
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam
```

#### 4. **Two-Stage Training (Decoupled Training)**
1. **Stage 1**: Train backbone with instance-balanced sampling (learns good representations)
2. **Stage 2**: Freeze backbone, retrain classifier with class-balanced sampling (learns balanced decision boundary)

> **Interview Tip:** Mention the decoupled training approach as it's state-of-the-art. The key insight is that representation learning benefits from natural distribution while classifier learning benefits from balanced distribution. Also discuss how Focal Loss addresses the easy-negative problem in imbalanced settings.

---

## Question 2
- [ ] Done
**What are the trade-offs between using pre-trained models versus training from scratch for domain-specific image classification?**

**Answer:**

The decision between pre-trained models and training from scratch depends on dataset size, domain similarity, computational resources, and task requirements.

### Decision Framework

| Factor | Use Pre-trained | Train from Scratch |
|--------|----------------|-------------------|
| **Dataset size** | < 10K images | > 100K+ images |
| **Domain similarity** | Similar to ImageNet (natural images) | Very different (medical, satellite, microscopy) |
| **Compute budget** | Limited | Abundant |
| **Time constraints** | Tight deadlines | Flexible timeline |
| **Label quality** | Noisy labels (pre-trained features are robust) | Clean, well-annotated |

### Pre-trained Model Approach (Transfer Learning)

```python
import torch
import torchvision.models as models

# Strategy 1: Feature Extraction (freeze backbone)
model = models.efficientnet_b0(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[1] = torch.nn.Linear(1280, num_classes)

# Strategy 2: Fine-tuning (unfreeze progressively)
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, num_classes)

# Discriminative learning rates
optimizer = torch.optim.Adam([
    {'params': model.features[:4].parameters(), 'lr': 1e-5},   # Early layers
    {'params': model.features[4:].parameters(), 'lr': 1e-4},   # Mid layers
    {'params': model.classifier.parameters(), 'lr': 1e-3},     # Classifier
])
```

### Training from Scratch Considerations

```python
# When domain is very different (e.g., spectrograms, medical images)
# Use domain-specific architecture design
class DomainSpecificNet(torch.nn.Module):
    def __init__(self, in_channels=1, num_classes=10):  # grayscale medical
        super().__init__()
        # Custom channel handling for non-RGB inputs
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # ... custom architecture
        )
```

### Hybrid Approach: Self-Supervised Pre-training

When ImageNet pre-training doesn't match your domain:
1. Collect large unlabeled domain data
2. Pre-train with contrastive learning (SimCLR, MoCo, DINO)
3. Fine-tune on labeled task data

> **Interview Tip:** Never say "always use pre-trained" or "always train from scratch." Demonstrate understanding of the trade-offs and mention the hybrid self-supervised approach for domains far from ImageNet (medical, satellite, industrial). Also mention that even for very different domains, ImageNet pre-trained low-level features (edges, textures) often transfer well.

---

## Question 3
- [ ] Done
**How do you implement and evaluate data augmentation strategies that preserve class-relevant features?**

**Answer:**

Data augmentation must be carefully designed to introduce meaningful variation while preserving the features that define each class.

### Principles of Class-Preserving Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentations based on task-specific invariances
def get_augmentation_pipeline(task='general'):
    if task == 'medical':
        # Medical: preserve spatial relationships, allow intensity changes
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])
    elif task == 'fine_grained':
        # Fine-grained: preserve discriminative parts, mild geometric changes
        return A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            A.GaussNoise(var_limit=(10, 30), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif task == 'texture':
        # Texture: preserve texture patterns, allow geometric transforms
        return A.Compose([
            A.RandomCrop(224, 224),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # NO color jitter - texture depends on color patterns
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
```

### Evaluation Framework

```python
# Measure augmentation effectiveness
def evaluate_augmentation(model, train_loader, val_loader, augmented_loader):
    metrics = {}
    # 1. Compare val accuracy with/without augmentation
    metrics['baseline_acc'] = evaluate(model, val_loader)
    metrics['augmented_acc'] = evaluate(model, val_loader)  # after training with aug
    
    # 2. Check augmentation doesn't change class distribution
    # 3. Visualize augmented samples to verify class preservation
    # 4. Measure feature space impact with t-SNE/UMAP
    return metrics
```

### Key Evaluation Considerations

| Metric | What It Measures | Threshold |
|--------|-----------------|-----------|
| **Val accuracy improvement** | Augmentation effectiveness | Should improve |
| **Class confusion matrix change** | Label preservation | Should not increase off-diagonal |
| **Feature space visualization** | Cluster separation | Clusters should remain distinct |
| **Augmentation diversity score** | Transformation variety | Higher is better |

> **Interview Tip:** Emphasize that augmentation should reflect real-world variations the model will encounter. Reference AutoAugment/RandAugment for automated policy search, and mention that domain expertise is crucial—e.g., vertical flips make sense for satellite but not for digit recognition.

---

## Question 4
- [ ] Done
**In multi-label image classification, how do you handle label correlation and dependency structures?**

**Answer:**

Multi-label classification with label correlations requires specialized architectures, loss functions, and training strategies that model the dependencies between labels.

### Approaches to Handle Label Correlation

#### 1. **Graph-Based Label Modeling (GCN)**

```python
import torch
import torch.nn as nn
import numpy as np

class LabelGCN(nn.Module):
    """Graph Convolutional Network for modeling label dependencies"""
    def __init__(self, num_labels, feature_dim, adj_matrix):
        super().__init__()
        self.adj = torch.tensor(adj_matrix, dtype=torch.float32)  # label co-occurrence
        self.gc1 = nn.Linear(feature_dim, 256)
        self.gc2 = nn.Linear(256, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_labels)
    
    def forward(self, x):
        # x: image features from backbone
        label_features = torch.relu(self.adj @ self.gc1.weight.T)
        label_features = self.adj @ self.gc2(label_features)
        # Combine with image features
        logits = (x @ label_features.T)
        return logits

# Build adjacency matrix from label co-occurrence
def build_adj_matrix(labels, threshold=0.4):
    num_labels = labels.shape[1]
    adj = np.zeros((num_labels, num_labels))
    for i in range(num_labels):
        for j in range(num_labels):
            co_occur = ((labels[:, i] == 1) & (labels[:, j] == 1)).sum()
            count_i = (labels[:, i] == 1).sum()
            adj[i, j] = co_occur / max(count_i, 1)
    adj[adj < threshold] = 0
    return adj
```

#### 2. **Loss Functions for Multi-Label**

```python
# Asymmetric Loss - handles positive/negative imbalance per label
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, x, y):
        xs_pos = x
        xs_neg = 1 - x
        
        # Basic CE
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        
        # Asymmetric focusing
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        pt_pos = xs_pos * y
        pt_neg = xs_neg * (1 - y)
        
        los_pos = los_pos * ((1 - pt_pos) ** self.gamma_pos)
        los_neg = los_neg * ((1 - pt_neg) ** self.gamma_neg)
        
        return -(los_pos + los_neg).mean()
```

#### 3. **Label Dependency Structures**

| Method | Approach | Best For |
|--------|----------|----------|
| **Classifier Chains** | Sequential prediction, each label conditions on previous | Strong pairwise dependencies |
| **Label GCN** | Graph models label relationships | Co-occurrence patterns |
| **Transformer Decoder** | Cross-attention between labels | Complex dependencies |
| **Conditional Random Fields** | Joint probability modeling | Structured output |

> **Interview Tip:** Highlight that ignoring label correlations (binary cross-entropy per label) works as a strong baseline but misses dependencies. The choice between explicit (GCN, CRF) and implicit (transformer) modeling depends on dataset size and label structure complexity.

---

## Question 5
- [ ] Done
**What techniques do you use to improve model interpretability in medical image classification applications?**

**Answer:**

Model interpretability in medical image classification is critical for clinical adoption, regulatory compliance, and building trust with healthcare professionals.

### Interpretability Techniques

#### 1. **Gradient-Based Attribution Methods**

```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """Gradient-weighted Class Activation Mapping for medical images"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_image, target_class=None):
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear')
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().detach().numpy()
```

#### 2. **Clinical Interpretability Requirements**

| Requirement | Technique | Clinical Benefit |
|-------------|-----------|-----------------|
| **Lesion localization** | Grad-CAM, Attention maps | Shows where model "looks" |
| **Feature importance** | SHAP, LIME | Explains which features drive decision |
| **Confidence calibration** | Temperature scaling, MC Dropout | Reliable probability estimates |
| **Counterfactual explanation** | GAN-based generation | "What would need to change?" |
| **Concept-based explanation** | TCAV | "Model uses texture, not shape" |

#### 3. **Uncertainty Estimation for Clinical Safety**

```python
# Monte Carlo Dropout for uncertainty
def mc_dropout_predict(model, image, n_samples=50):
    model.train()  # Keep dropout active
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = torch.softmax(model(image), dim=1)
        predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)  # Epistemic uncertainty
    entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1)
    
    return {
        'prediction': mean_pred.argmax(dim=1),
        'confidence': mean_pred.max(dim=1).values,
        'uncertainty': uncertainty,
        'entropy': entropy
    }
```

> **Interview Tip:** Emphasize the regulatory aspect—FDA/CE requires explainability for AI medical devices. Mention that GradCAM alone is insufficient; clinical validation requires radiologist agreement studies comparing AI attention with expert-annotated regions of interest.

---

## Question 6
- [ ] Done
**How do you design curriculum learning strategies for progressively training image classifiers?**

**Answer:**

Curriculum learning trains models on progressively harder examples, mimicking how humans learn—from simple to complex concepts.

### Implementation Strategy

```python
import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler

class CurriculumSampler(Sampler):
    """Samples easy examples first, gradually introduces harder ones"""
    def __init__(self, difficulty_scores, epoch, max_epochs, strategy='linear'):
        self.scores = difficulty_scores
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.strategy = strategy
    
    def __iter__(self):
        # Calculate proportion of data to use
        if self.strategy == 'linear':
            fraction = min(1.0, 0.3 + 0.7 * (self.epoch / self.max_epochs))
        elif self.strategy == 'root':
            fraction = min(1.0, np.sqrt(self.epoch / self.max_epochs))
        
        # Select easiest examples up to fraction
        n_samples = int(len(self.scores) * fraction)
        sorted_indices = np.argsort(self.scores)[:n_samples]
        np.random.shuffle(sorted_indices)
        return iter(sorted_indices.tolist())
    
    def __len__(self):
        fraction = min(1.0, 0.3 + 0.7 * (self.epoch / self.max_epochs))
        return int(len(self.scores) * fraction)

# Difficulty scoring methods
def compute_difficulty_scores(model, dataset):
    """Score samples by loss (higher loss = harder)"""
    model.eval()
    scores = []
    loader = DataLoader(dataset, batch_size=64)
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')
            scores.extend(losses.cpu().numpy())
    return np.array(scores)
```

### Difficulty Metrics

| Method | Description | Best For |
|--------|-------------|----------|
| **Loss-based** | Model loss on each sample | General purpose |
| **Prediction confidence** | Model's softmax confidence | Classification |
| **Data complexity** | K-NN or C-score | Dataset analysis |
| **Human annotation** | Expert difficulty ratings | Medical, specialized |
| **Self-paced** | Let model determine its own curriculum | Adaptive |

### Training Loop with Curriculum

```python
def train_with_curriculum(model, dataset, epochs=100):
    # Phase 1: Score sample difficulty with simple model
    scores = compute_difficulty_scores(model, dataset)
    
    for epoch in range(epochs):
        sampler = CurriculumSampler(scores, epoch, epochs)
        loader = DataLoader(dataset, batch_size=32, sampler=sampler)
        
        for images, labels in loader:
            loss = train_step(model, images, labels)
        
        # Optionally re-score difficulty periodically
        if epoch % 10 == 0:
            scores = compute_difficulty_scores(model, dataset)
```

> **Interview Tip:** Mention that curriculum learning is especially useful for noisy datasets (easy examples tend to be clean) and when training on very large datasets (efficient use of compute). Reference SPL (Self-Paced Learning) as a fully automatic alternative.

---

## Question 7
- [ ] Done
**What are the considerations for deploying image classification models on edge devices with memory constraints?**

**Answer:**

Deploying image classifiers on edge devices requires balancing accuracy with strict memory, compute, and latency constraints.

### Key Considerations

| Constraint | Typical Limit | Optimization |
|------------|--------------|--------------|
| **Model size** | 5-50 MB | Quantization, pruning |
| **RAM** | 256 MB - 2 GB | Reduce batch size, streaming |
| **Compute** | No GPU, limited CPU/NPU | Efficient architectures |
| **Latency** | 10-100 ms | Model distillation, TensorRT |
| **Power** | Battery-constrained | Reduce FLOPs |

### Architecture Selection

```python
import torch
import torchvision.models as models

# Efficient architectures ranked by accuracy/efficiency trade-off
efficient_models = {
    'MobileNetV3-Small': {'params': '2.5M', 'flops': '56M', 'top1': 67.4},
    'MobileNetV3-Large': {'params': '5.4M', 'flops': '219M', 'top1': 75.2},
    'EfficientNet-B0':   {'params': '5.3M', 'flops': '390M', 'top1': 77.1},
    'ShuffleNetV2-1.0':  {'params': '2.3M', 'flops': '146M', 'top1': 69.4},
}

# Post-training quantization
model = models.mobilenet_v3_small(pretrained=True)
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)

# Export for mobile deployment
example_input = torch.randn(1, 3, 224, 224)
scripted_model = torch.jit.trace(quantized_model, example_input)
scripted_model._save_for_lite_interpreter("model.ptl")  # PyTorch Mobile
```

### Optimization Pipeline

```
Full Model (ResNet-50, 25M params)
    ↓ Knowledge Distillation
Compact Model (MobileNetV3, 5M params)
    ↓ Structured Pruning (remove 30% channels)
Pruned Model (3.5M params)
    ↓ INT8 Quantization (4x compression)
Quantized Model (~1MB)
    ↓ Export to ONNX/TFLite/CoreML
Deployment-Ready Model
```

> **Interview Tip:** Know the full pipeline: architecture choice → distillation → pruning → quantization → runtime optimization. Mention ONNX Runtime, TensorRT, CoreML, and TFLite as deployment frameworks. Quantization alone often gives 4x size reduction and 2-3x speedup with <1% accuracy loss.

---

## Question 8
- [ ] Done
**How do you handle fine-grained image classification where inter-class differences are minimal?**

**Answer:**

Fine-grained classification identifies subtle differences between visually similar subcategories (e.g., bird species, car models, plant diseases).

### Key Challenges and Solutions

#### 1. **Part-Based Feature Learning**

```python
import torch
import torch.nn as nn

class BilinearPooling(nn.Module):
    """Bilinear pooling captures pairwise feature interactions for fine-grained recognition"""
    def __init__(self, feature_dim=512, num_classes=200):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Bilinear attention pooling
        self.attention = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 2048]
        # Spatial features before global pooling for attention
        return self.fc(features)

class PartAttentionNet(nn.Module):
    """Multi-attention network that discovers discriminative parts"""
    def __init__(self, num_parts=8, num_classes=200):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        
        # Multiple attention heads for different parts
        self.part_attentions = nn.ModuleList([
            nn.Conv2d(2048, 1, 1) for _ in range(num_parts)
        ])
        self.classifier = nn.Linear(2048 * num_parts, num_classes)
```

#### 2. **Strategies Comparison**

| Technique | Approach | Best For |
|-----------|----------|----------|
| **Bilinear pooling** | Second-order feature statistics | Texture-based distinctions |
| **Part attention** | Discover discriminative regions | Part-based recognition (birds, faces) |
| **Contrastive learning** | Push apart similar classes | Large-scale fine-grained |
| **Jigsaw/Destruction** | Force model to learn all parts | Prevents shortcut learning |
| **Web data mining** | Augment with web-scraped images | Limited labeled data |

> **Interview Tip:** Fine-grained classification differs from standard classification because inter-class variance is small while intra-class variance is large (same bird species in different poses). Mention that attention mechanisms and part-based approaches force the model to focus on discriminative details rather than global features.

---

## Question 9
- [ ] Done
**What approaches work best for few-shot image classification in novel domains?**

**Answer:**

Few-shot image classification aims to classify images from novel classes with only 1-5 labeled examples per class.

### Core Approaches

#### 1. **Meta-Learning (Learning to Learn)**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot classification"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # Feature extractor
    
    def forward(self, support_images, support_labels, query_images, n_way):
        # Encode all images
        support_features = self.backbone(support_images)  # [N*K, D]
        query_features = self.backbone(query_images)       # [Q, D]
        
        # Compute class prototypes (mean of support features per class)
        prototypes = []
        for c in range(n_way):
            mask = (support_labels == c)
            prototypes.append(support_features[mask].mean(dim=0))
        prototypes = torch.stack(prototypes)  # [N, D]
        
        # Classify queries by nearest prototype
        distances = torch.cdist(query_features, prototypes)
        logits = -distances  # Negative distance as logits
        return logits

# Episode-based training
def create_episode(dataset, n_way=5, k_shot=1, n_query=15):
    classes = random.sample(dataset.classes, n_way)
    support, query = [], []
    for cls in classes:
        samples = random.sample(dataset.class_to_samples[cls], k_shot + n_query)
        support.extend(samples[:k_shot])
        query.extend(samples[k_shot:])
    return support, query
```

#### 2. **Approach Comparison**

| Method | Mechanism | N-way 5-shot (miniImageNet) |
|--------|-----------|----------------------------|
| **Prototypical Networks** | Nearest class prototype | ~68% |
| **MAML** | Fast adaptation via meta-gradients | ~63% |
| **Matching Networks** | Attention-based comparison | ~65% |
| **CLIP (zero-shot)** | Vision-language alignment | ~70%+ |
| **DINO + kNN** | Self-supervised features + kNN | ~72% |

#### 3. **Modern Approach: Foundation Models**

```python
# Using CLIP for zero/few-shot classification
import clip

model, preprocess = clip.load("ViT-B/32")

# Zero-shot: classify using text descriptions
text_prompts = [f"a photo of a {cls}" for cls in class_names]
text_features = model.encode_text(clip.tokenize(text_prompts))
image_features = model.encode_image(preprocess(image))
similarity = (image_features @ text_features.T)
prediction = similarity.argmax()
```

> **Interview Tip:** Modern few-shot classification has shifted from meta-learning to leveraging large pre-trained models (CLIP, DINO, DINOv2). A simple kNN on DINOv2 features often outperforms complex meta-learning methods. Mention this practical insight.

---

## Question 10
- [ ] Done
**How do you implement active learning strategies to reduce annotation costs in image classification?**

**Answer:**

Active learning reduces annotation costs by strategically selecting the most informative samples for labeling.

### Active Learning Pipeline

```python
import torch
import numpy as np
from torch.utils.data import DataLoader

class ActiveLearningPipeline:
    def __init__(self, model, unlabeled_pool, labeled_set, budget_per_round=100):
        self.model = model
        self.unlabeled_pool = unlabeled_pool
        self.labeled_set = labeled_set
        self.budget = budget_per_round
    
    def uncertainty_sampling(self):
        """Select samples with highest prediction uncertainty"""
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for images, _ in DataLoader(self.unlabeled_pool, batch_size=64):
                probs = torch.softmax(self.model(images), dim=1)
                # Entropy-based uncertainty
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                uncertainties.extend(entropy.cpu().numpy())
        
        # Select top-k most uncertain
        indices = np.argsort(uncertainties)[-self.budget:]
        return indices
    
    def diversity_sampling(self, features):
        """Core-set selection for diverse sampling"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.budget)
        kmeans.fit(features)
        # Select sample closest to each cluster center
        indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features - center, axis=1)
            indices.append(np.argmin(distances))
        return indices
    
    def badge_sampling(self, features, gradients):
        """BADGE: combines uncertainty and diversity via gradient embeddings"""
        # Use gradient embeddings for k-means++ initialization
        from sklearn.cluster import kmeans_plusplus
        centers, _ = kmeans_plusplus(gradients, n_clusters=self.budget)
        return centers
```

### Strategy Comparison

| Strategy | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **Uncertainty (Entropy)** | High entropy samples | Simple, effective | Biased toward decision boundary |
| **Margin sampling** | Smallest margin between top-2 | Targets ambiguous samples | Ignores diversity |
| **Core-set** | Maximize coverage in feature space | Good diversity | Ignores uncertainty |
| **BADGE** | Gradient embeddings + diversity | Both uncertainty + diversity | Computationally expensive |
| **BALD** | Bayesian disagreement | Theoretically grounded | Requires MC Dropout |

> **Interview Tip:** The key insight is that uncertainty alone leads to redundant selections (many similar uncertain samples). The best methods combine uncertainty with diversity (BADGE, BatchBALD). In practice, a simple uncertainty + k-means diversity hybrid often performs as well as complex methods.

---

## Question 11
- [ ] Done
**What are the best practices for handling noisy labels in large-scale image classification datasets?**

**Answer:**

Noisy labels are a critical problem in large-scale image classification where annotation errors can significantly degrade model performance.

### Approaches to Handle Label Noise

#### 1. **Noise-Robust Loss Functions**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricCrossEntropy(nn.Module):
    """SCE: Robust to label noise by combining CE and Reverse CE"""
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
    
    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels)
        pred = F.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, self.num_classes).float()
        rce = -(pred * torch.log(one_hot + 1e-4)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce

class GeneralizedCrossEntropy(nn.Module):
    """GCE: Noise-robust with truncated loss"""
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q
    
    def forward(self, logits, labels):
        pred = F.softmax(logits, dim=1)
        pred_correct = pred.gather(1, labels.view(-1, 1)).squeeze()
        loss = (1 - pred_correct ** self.q) / self.q
        return loss.mean()
```

#### 2. **Sample Selection Methods**

```python
class CoTeaching:
    """Co-Teaching: Two networks select clean samples for each other"""
    def __init__(self, model1, model2, forget_rate=0.2):
        self.model1 = model1
        self.model2 = model2
        self.forget_rate = forget_rate
    
    def train_step(self, images, labels, epoch, total_epochs):
        # Forward both models
        logits1 = self.model1(images)
        logits2 = self.model2(images)
        
        loss1 = F.cross_entropy(logits1, labels, reduction='none')
        loss2 = F.cross_entropy(logits2, labels, reduction='none')
        
        # Select small-loss samples (likely clean)
        n_keep = int(len(labels) * (1 - self.forget_rate))
        
        # Model 1 selects for Model 2 and vice versa
        _, idx1 = torch.topk(loss1, n_keep, largest=False)
        _, idx2 = torch.topk(loss2, n_keep, largest=False)
        
        clean_loss1 = F.cross_entropy(logits1[idx2], labels[idx2])
        clean_loss2 = F.cross_entropy(logits2[idx1], labels[idx1])
        
        return clean_loss1, clean_loss2
```

#### 3. **Methods Comparison**

| Method | Type | Noise Rate Tolerance | Complexity |
|--------|------|---------------------|------------|
| **Label Smoothing** | Regularization | Low (< 20%) | Simple |
| **Mixup** | Data augmentation | Low-Medium | Simple |
| **SCE/GCE** | Robust loss | Medium (20-40%) | Simple |
| **Co-Teaching** | Sample selection | High (up to 50%) | Two networks |
| **DivideMix** | Combined approach | Very high (up to 80%) | Complex |

> **Interview Tip:** Mention the "memorization effect"—DNNs learn clean patterns first and noisy patterns later. This is why loss-based selection works (small-loss samples are likely clean). DivideMix is state-of-the-art, combining sample selection with semi-supervised learning on rejected samples.

---

## Question 12
- [ ] Done
**How do you design ensemble methods that balance accuracy and computational efficiency?**

**Answer:**

Ensemble methods combine multiple models to improve accuracy, but computational cost must be carefully managed for practical deployment.

### Ensemble Strategies

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientEnsemble:
    """Ensemble with accuracy-efficiency trade-offs"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0/len(models)] * len(models)
    
    def predict_averaging(self, x):
        """Simple probability averaging"""
        probs = []
        for model, w in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                prob = F.softmax(model(x), dim=1)
            probs.append(w * prob)
        return torch.stack(probs).sum(dim=0)
    
    def predict_stacking(self, x, meta_model):
        """Learned combination via meta-model"""
        features = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                features.append(model(x))
        combined = torch.cat(features, dim=1)
        return meta_model(combined)

# Snapshot Ensemble - multiple models from single training run
class SnapshotEnsemble:
    def __init__(self, model, lr_schedule, n_snapshots=5):
        self.model = model
        self.snapshots = []
        self.n_snapshots = n_snapshots
    
    def cosine_annealing_lr(self, epoch, T, lr_max=0.1):
        return lr_max / 2 * (1 + np.cos(np.pi * (epoch % T) / T))
```

### Efficiency Strategies

| Method | Models | Cost | Diversity |
|--------|--------|------|-----------|
| **Simple average** | N separate models | N× inference | High |
| **Snapshot ensemble** | Checkpoints from 1 training | 1× train, N× inference | Medium |
| **Stochastic Weight Avg** | 1 averaged model | 1× inference | Low |
| **MC Dropout** | 1 model, N forward passes | 1× train, N× inference | Medium |
| **Knowledge distillation** | 1 student model | 1× inference | Captures ensemble |

> **Interview Tip:** For production, distill the ensemble into a single model using knowledge distillation—you get most of the ensemble's accuracy with single-model inference cost. Snapshot ensembles are the best free lunch (multiple models from one training run).

---

## Question 13
- [ ] Done
**What techniques help with domain adaptation when deploying image classifiers to new environments?**

**Answer:**

Domain adaptation transfers classification knowledge from a source domain (with labels) to a target domain (with different distribution, few/no labels).

### Key Techniques

#### 1. **Feature Alignment Methods**

```python
import torch
import torch.nn as nn

class DANNClassifier(nn.Module):
    """Domain-Adversarial Neural Network"""
    def __init__(self, backbone, num_classes, domain_dim=256):
        super().__init__()
        self.backbone = backbone
        self.class_head = nn.Linear(2048, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(2048, domain_dim),
            nn.ReLU(),
            nn.Linear(domain_dim, 2)  # source vs target
        )
    
    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        class_output = self.class_head(features)
        
        # Gradient reversal for domain confusion
        reversed_features = GradientReversal.apply(features, alpha)
        domain_output = self.domain_head(reversed_features)
        
        return class_output, domain_output

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# MMD (Maximum Mean Discrepancy) loss for distribution alignment
def mmd_loss(source_features, target_features, kernel='rbf'):
    ss = torch.mm(source_features, source_features.t())
    tt = torch.mm(target_features, target_features.t())
    st = torch.mm(source_features, target_features.t())
    return ss.mean() + tt.mean() - 2 * st.mean()
```

#### 2. **Adaptation Strategies**

| Method | Label Requirement | Key Idea |
|--------|-------------------|----------|
| **DANN** | Unsupervised | Adversarial domain confusion |
| **MMD** | Unsupervised | Statistical distribution matching |
| **Self-training** | Semi-supervised | Pseudo-labels on target |
| **CLIP adaptation** | Zero-shot | Vision-language transfer |
| **Test-time adaptation** | Unsupervised | Adapt at inference time |

> **Interview Tip:** Domain adaptation is crucial for real-world deployment where training and deployment distributions differ. Mention that modern foundation models (CLIP, DINOv2) have inherent domain robustness, reducing the need for explicit adaptation in many cases.

---

## Question 14
- [ ] Done
**How do you handle hierarchical classification where categories have parent-child relationships?**

**Answer:**

Hierarchical classification handles categories organized in a tree structure (e.g., Animal → Dog → Poodle), ensuring predictions respect the taxonomy.

### Implementation Approaches

#### 1. **Hierarchy-Aware Architecture**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalClassifier(nn.Module):
    """Multi-level classifier respecting category hierarchy"""
    def __init__(self, backbone, hierarchy):
        super().__init__()
        self.backbone = backbone
        self.hierarchy = hierarchy  # dict: {level: num_classes}
        
        # Separate heads per level
        self.heads = nn.ModuleDict()
        feature_dim = 2048
        for level, n_classes in hierarchy.items():
            self.heads[level] = nn.Linear(feature_dim, n_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = {}
        for level, head in self.heads.items():
            outputs[level] = head(features)
        return outputs

# Hierarchical loss with consistency enforcement
def hierarchical_loss(outputs, targets, hierarchy_matrix, alpha=0.5):
    total_loss = 0
    for level, logits in outputs.items():
        ce_loss = F.cross_entropy(logits, targets[level])
        total_loss += ce_loss
    
    # Consistency: child prediction must be consistent with parent
    parent_probs = F.softmax(outputs['coarse'], dim=1)
    child_probs = F.softmax(outputs['fine'], dim=1)
    
    # Map child probs to parent level and enforce consistency
    mapped_child = child_probs @ hierarchy_matrix  # [B, n_parent]
    consistency_loss = F.kl_div(
        mapped_child.log(), parent_probs, reduction='batchmean'
    )
    
    return total_loss + alpha * consistency_loss
```

#### 2. **Strategy Comparison**

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Flat classification** | Ignore hierarchy | Simple | Allows invalid predictions |
| **Multi-head** | Separate head per level | Explicit hierarchy | More parameters |
| **Conditional** | Predict coarse → fine | Ensures consistency | Error propagation |
| **Label embedding** | Embed labels in hierarchy-aware space | Flexible | Complex training |

> **Interview Tip:** Highlight that hierarchical classification naturally enables graceful degradation—when uncertain about fine-grained class, the model can still make correct coarse predictions. This is valuable in production where "I know it's a dog but unsure which breed" is better than a wrong specific prediction.

---

## Question 15
- [ ] Done
**What are effective strategies for handling images with multiple objects during classification?**

**Answer:**

Images with multiple objects pose unique challenges for classification, requiring the model to handle multi-object scenes effectively.

### Strategies

#### 1. **Multi-Label Classification**

```python
import torch
import torch.nn as nn

class MultiObjectClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return torch.sigmoid(logits)  # Independent per-class probability

# Binary Cross Entropy for multi-label
criterion = nn.BCEWithLogitsLoss()

# With class-specific thresholds
def predict_with_thresholds(model, image, thresholds):
    probs = model(image)
    predictions = (probs > thresholds).int()
    return predictions
```

#### 2. **Attention-Based Approaches**

```python
class MultiObjectAttention(nn.Module):
    """Attend to different objects separately"""
    def __init__(self, backbone, num_classes, num_objects=5):
        super().__init__()
        self.backbone = backbone
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_objects)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(2048, num_classes) for _ in range(num_objects)
        ])
    
    def forward(self, x):
        feature_map = self.backbone.extract_features(x)  # [B, 2048, H, W]
        predictions = []
        for attn, cls in zip(self.attention_heads, self.classifiers):
            mask = attn(feature_map)  # [B, 1, H, W]
            attended = (feature_map * mask).mean(dim=[2, 3])  # [B, 2048]
            predictions.append(cls(attended))
        return predictions
```

| Approach | When to Use |
|----------|------------|
| **Multi-label + BCE** | Known set of possible classes |
| **Multi-attention** | Need to localize each object |
| **Detection → Classification** | Need precise object-level predictions |
| **Set prediction (DETR-style)** | Variable number of objects |

> **Interview Tip:** Distinguish between "what objects are in this image?" (multi-label classification) vs "where and what are the objects?" (detection). For interviews, mention that the choice depends on whether spatial information is needed.

---

## Question 16
- [ ] Done
**How do you implement and evaluate self-supervised pre-training for image classification?**

**Answer:**

Self-supervised pre-training learns visual representations from unlabeled images, then transfers to classification with limited labels.

### Key Methods

#### 1. **Contrastive Learning (SimCLR)**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        return z1, z2

def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, -1e9)
    
    return F.cross_entropy(sim_matrix, labels)
```

#### 2. **Self-Distillation (DINO)**

```python
# DINO: Self-distillation with no labels
class DINO:
    def __init__(self, student, teacher, center_momentum=0.996):
        self.student = student
        self.teacher = teacher  # EMA of student
    
    def train_step(self, images, global_crops, local_crops):
        # Teacher processes global crops
        teacher_out = self.teacher(global_crops)
        # Student processes all crops
        student_out = self.student(torch.cat([global_crops, local_crops]))
        # Cross-entropy between student and teacher (centering + sharpening)
        loss = self.dino_loss(student_out, teacher_out)
        return loss
```

### Method Comparison

| Method | Mechanism | ImageNet Linear Probe |
|--------|-----------|----------------------|
| **SimCLR** | Contrastive pairs | 71.7% |
| **MoCo v3** | Momentum contrastive | 76.7% |
| **BYOL** | Self-distillation (no negatives) | 74.3% |
| **DINO** | Self-distillation + ViT | 77.3% |
| **DINOv2** | Curated data + ViT-g | 83.5% |
| **MAE** | Masked autoencoder | 75.8% |

> **Interview Tip:** DINOv2 features are currently the strongest general-purpose visual features. For practical classification, DINOv2 backbone frozen + linear probe often beats fine-tuned supervised models, especially with limited labels. Mention this as the modern practical approach.

---

## Question 17
- [ ] Done
**What approaches work best for classifying images with significant viewpoint variations?**

**Answer:**

Viewpoint variation is a fundamental challenge in image classification where the same object looks very different from different angles.

### Key Approaches

```python
import torch
import torchvision.transforms as T

# 1. Data augmentation for viewpoint robustness
viewpoint_augmentation = T.Compose([
    T.RandomPerspective(distortion_scale=0.3, p=0.5),
    T.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
])

# 2. Multi-view learning: aggregate features from multiple views
class MultiViewClassifier(torch.nn.Module):
    def __init__(self, backbone, num_classes, n_views=4):
        super().__init__()
        self.backbone = backbone
        self.aggregator = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=2048, nhead=8), num_layers=2
        )
        self.classifier = torch.nn.Linear(2048, num_classes)
    
    def forward(self, views):
        # views: [B, N_views, C, H, W]
        B, N = views.shape[:2]
        features = []
        for i in range(N):
            feat = self.backbone(views[:, i])  # [B, 2048]
            features.append(feat)
        features = torch.stack(features, dim=1)  # [B, N, 2048]
        aggregated = self.aggregator(features).mean(dim=1)  # [B, 2048]
        return self.classifier(aggregated)
```

### Strategies

| Method | Description | Effectiveness |
|--------|-------------|---------------|
| **Augmentation** | Random perspective/affine | Good baseline |
| **Multi-view fusion** | Combine multiple viewpoints | Best when multi-view data available |
| **Equivariant networks** | Built-in rotation/transformation equivariance | Theoretically elegant |
| **3D-aware features** | Learn 3D structure from 2D images | Best for strong viewpoint changes |
| **Self-supervised pre-training** | Learn viewpoint-invariant features | Modern best practice |

> **Interview Tip:** DINOv2 and MAE learn viewpoint-invariant features through self-supervised training with aggressive cropping. Mention that equivariant neural networks (E(2)-CNNs) provide guaranteed equivariance but are more complex to implement.

---

## Question 18
- [ ] Done
**How do you handle temporal consistency in video-based image classification?**

**Answer:**

Video-based classification must maintain consistent predictions across frames while leveraging temporal information.

### Approaches

```python
import torch
import torch.nn as nn

class TemporalClassifier(nn.Module):
    """Combines per-frame features with temporal modeling"""
    def __init__(self, backbone, num_classes, seq_len=16):
        super().__init__()
        self.backbone = backbone  # Per-frame feature extractor
        self.temporal = nn.LSTM(2048, 512, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, video):
        # video: [B, T, C, H, W]
        B, T = video.shape[:2]
        frames = video.view(B * T, *video.shape[2:])
        features = self.backbone(frames).view(B, T, -1)  # [B, T, 2048]
        temporal_out, _ = self.temporal(features)
        # Use last hidden state or average
        output = temporal_out[:, -1, :]
        return self.classifier(output)

# Temporal consistency via exponential moving average
class TemporalSmoothing:
    def __init__(self, alpha=0.7, num_classes=10):
        self.alpha = alpha
        self.prev_probs = None
    
    def smooth(self, current_probs):
        if self.prev_probs is None:
            self.prev_probs = current_probs
        else:
            self.prev_probs = self.alpha * self.prev_probs + (1 - self.alpha) * current_probs
        return self.prev_probs
```

### Video Classification Architectures

| Architecture | Temporal Modeling | Efficiency |
|-------------|-------------------|------------|
| **2D CNN + LSTM** | Sequential | Moderate |
| **3D CNN (C3D, I3D)** | Spatiotemporal convolutions | High compute |
| **SlowFast** | Dual-pathway (high/low frame rate) | Good trade-off |
| **TimeSformer** | Temporal attention | High accuracy |
| **VideoMAE** | Masked autoencoder for video | State-of-the-art |

> **Interview Tip:** For classification consistency in video, distinguish between temporal modeling (learning from motion/dynamics) and temporal smoothing (post-processing for stability). Production systems often use a simple exponential moving average over frame-level predictions.

---

## Question 19
- [ ] Done
**What techniques are most effective for handling classification under varying illumination conditions?**

**Answer:**

Illumination variation is a persistent challenge where lighting changes alter pixel values significantly without changing the actual object.

### Techniques

#### 1. **Preprocessing for Illumination Robustness**

```python
import cv2
import numpy as np

def illumination_normalization(image):
    """Multi-technique illumination normalization pipeline"""
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. Retinex-based normalization
    def single_scale_retinex(img, sigma=300):
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex = np.log10(img.astype(float) + 1) - np.log10(blur.astype(float) + 1)
        return retinex
    
    # 3. Illumination-invariant color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log_chromaticity = np.log(image.astype(float) + 1) - np.log(gray[:,:,np.newaxis].astype(float) + 1)
    
    return result_clahe

# Augmentation for illumination robustness
import albumentations as A
illumination_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
    A.RandomGamma(gamma_limit=(60, 140), p=0.5),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5),
])
```

#### 2. **Model-Level Solutions**

| Approach | How It Helps |
|----------|-------------|
| **Color normalization layers** | Learnable normalization in first layer |
| **Batch/Instance Normalization** | Reduces internal covariate shift |
| **Domain randomization** | Train with extreme augmentations |
| **Self-supervised features** | DINO/MAE learn illumination-invariant features |

> **Interview Tip:** The most practical approach combines aggressive augmentation during training with preprocessing normalization (CLAHE) at inference. Models pre-trained on diverse data (ImageNet-22K, LAION) are inherently more robust to illumination changes.

---

## Question 20
- [ ] Done
**How do you implement cost-sensitive learning when misclassification costs vary across classes?**

**Answer:**

Cost-sensitive learning adjusts the training process when different classification errors have different real-world costs.

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class CostSensitiveLoss(nn.Module):
    """Cross-entropy weighted by misclassification cost matrix"""
    def __init__(self, cost_matrix):
        super().__init__()
        # cost_matrix[i][j] = cost of predicting j when true label is i
        self.cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32)
    
    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        # Expected cost for each sample
        costs = self.cost_matrix[labels]  # [B, num_classes]
        expected_cost = (probs * costs).sum(dim=1)
        return expected_cost.mean()

# Example: Medical diagnosis cost matrix
# Classes: 0=Healthy, 1=Benign, 2=Malignant
cost_matrix = [
    [0, 1, 1],    # True Healthy: low cost for any error
    [2, 0, 1],    # True Benign: higher cost to miss as Healthy
    [10, 5, 0],   # True Malignant: very high cost to miss!
]

# Alternative: Class-weighted cross-entropy
weights = torch.tensor([1.0, 2.0, 10.0])  # Higher weight for critical classes
criterion = nn.CrossEntropyLoss(weight=weights)

# Threshold adjustment at inference
def cost_sensitive_prediction(probs, cost_matrix):
    expected_costs = probs @ torch.tensor(cost_matrix).T
    return expected_costs.argmin(dim=1)  # Minimize expected cost
```

### Application Scenarios

| Domain | Critical Error | Approach |
|--------|---------------|----------|
| **Medical diagnosis** | False negative (missing disease) | High cost for FN |
| **Fraud detection** | False negative (missing fraud) | Asymmetric costs |
| **Autonomous driving** | False negative (missing pedestrian) | Safety-critical weighting |
| **Manufacturing QC** | False positive (rejecting good items) | Economic cost modeling |

> **Interview Tip:** Cost-sensitive learning is essential for real-world deployment where not all errors are equal. Mention that you can combine cost-sensitive training (modified loss) with cost-sensitive inference (adjusted decision thresholds) for comprehensive cost optimization.

---

## Question 21
- [ ] Done
**What are the best practices for handling high-resolution images in classification pipelines?**

**Answer:**

High-resolution images contain more information but require specialized handling to avoid memory issues and maintain important details.

### Strategies

```python
import torch
import torch.nn as nn

# 1. Multi-scale processing
class MultiScaleClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifiers = nn.ModuleList([
            nn.Linear(2048, num_classes) for _ in range(3)  # 3 scales
        ])
        self.fusion = nn.Linear(num_classes * 3, num_classes)
    
    def forward(self, image):
        # Process at multiple resolutions
        scales = [224, 384, 512]
        scale_outputs = []
        for i, size in enumerate(scales):
            resized = nn.functional.interpolate(image, size=(size, size))
            features = self.backbone(resized)
            scale_outputs.append(self.classifiers[i](features))
        
        combined = torch.cat(scale_outputs, dim=1)
        return self.fusion(combined)

# 2. Tiled processing for very high resolution
def tile_process(model, image, tile_size=512, overlap=64):
    """Process image in overlapping tiles for memory efficiency"""
    H, W = image.shape[2:]
    features = []
    for y in range(0, H, tile_size - overlap):
        for x in range(0, W, tile_size - overlap):
            tile = image[:, :, y:y+tile_size, x:x+tile_size]
            if tile.shape[2] < tile_size or tile.shape[3] < tile_size:
                tile = nn.functional.pad(tile, (0, tile_size-tile.shape[3], 0, tile_size-tile.shape[2]))
            feat = model.extract_features(tile)
            features.append(feat)
    
    # Aggregate tile features
    return torch.stack(features).mean(dim=0)
```

### Best Practices

| Strategy | Memory | Quality | Speed |
|----------|--------|---------|-------|
| **Downsample to 224** | Low | Loses fine details | Fast |
| **Multi-scale** | Medium | Good balance | Medium |
| **Tiled processing** | Low | Preserves details | Slower |
| **Progressive resizing** | Medium | Curriculum-style | Medium |
| **ViT with large patches** | Medium | Can handle larger inputs | Medium |

> **Interview Tip:** Modern ViTs (DINOv2, SigLIP) can process higher resolutions by interpolating position embeddings, making them more flexible than CNNs for high-res inputs. Mention that for production, progressive resizing during training (start small, increase resolution) is highly effective.

---

## Question 22
- [ ] Done
**How do you design multi-scale feature extraction for objects of varying sizes?**

**Answer:**

Multi-scale feature extraction is essential when objects of interest vary significantly in size within and across images.

### Implementation

```python
import torch
import torch.nn as nn

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
    
    def forward(self, features):
        # features: list of feature maps [C2, C3, C4, C5] from backbone
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = nn.functional.interpolate(laterals[i], scale_factor=2)
            laterals[i-1] += upsampled
        
        # Anti-aliasing convolution
        outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outputs

# Multi-scale classification with pooling
class MultiScalePool(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(s) for s in [1, 2, 4]
        ])
        self.classifier = nn.Linear(256 * (1 + 4 + 16), num_classes)
    
    def forward(self, feature_map):
        pooled = []
        for pool in self.pools:
            pooled.append(pool(feature_map).flatten(1))
        return self.classifier(torch.cat(pooled, dim=1))
```

> **Interview Tip:** FPN is the standard multi-scale approach. For classification specifically, SPP (Spatial Pyramid Pooling) or multi-scale test-time augmentation (resize to multiple scales and average predictions) are common and effective.

---

## Question 23
- [ ] Done
**What techniques help with classifying images under different lighting conditions?**

**Answer:**

This question overlaps with Q19 (varying illumination). The core techniques include color space transformations, histogram equalization (CLAHE), aggressive brightness/contrast augmentation, and using illumination-invariant features.

### Additional Techniques for Lighting Robustness

```python
# Color constancy preprocessing
def white_balance(image):
    """Gray World assumption for color constancy"""
    result = image.copy().astype(np.float32)
    avg = result.mean(axis=(0, 1))
    gray_avg = avg.mean()
    for c in range(3):
        result[:, :, c] *= gray_avg / (avg[c] + 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)

# Shadow-invariant features
def shadow_removal(image):
    """Remove shadows using illumination-reflectance separation"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    # Estimate illumination with large Gaussian blur
    illumination = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=50)
    # Reflectance = Original / Illumination
    reflectance = l_channel.astype(float) / (illumination.astype(float) + 1)
    lab[:, :, 0] = np.clip(reflectance * 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

> **Interview Tip:** In production, combine preprocessing (white balance, CLAHE) with augmentation (random brightness/contrast/gamma) during training. Self-supervised pre-training on diverse data naturally learns lighting-invariant features.

---

## Question 24
- [ ] Done
**How do you implement knowledge distillation to compress large classification models?**

**Answer:**

Knowledge distillation compresses a large teacher model into a smaller student model while retaining most of the teacher's accuracy.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soft loss: match teacher's soft predictions
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)
        
        # Hard loss: standard cross-entropy with ground truth
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# Training loop
def distill(teacher, student, train_loader, epochs=100):
    teacher.eval()
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            student_logits = student(images)
            loss = criterion(student_logits, teacher_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Distillation Variants

| Method | What's Transferred | Compression |
|--------|-------------------|-------------|
| **Logit distillation** | Output probabilities | High (~4x) |
| **Feature distillation** | Intermediate features | Higher quality |
| **Attention distillation** | Attention maps | Preserves focus patterns |
| **Self-distillation** | Same architecture, different capacity | Regularization |
| **Ensemble distillation** | Multiple teachers → one student | Best quality |

> **Interview Tip:** The temperature parameter T is key—higher T produces softer probability distributions that carry more information about inter-class relationships (dark knowledge). The optimal α and T depend on the teacher-student capacity gap.

---

## Question 25
- [ ] Done
**What approaches work best for zero-shot image classification using semantic embeddings?**

**Answer:**

Zero-shot classification recognizes classes never seen during training by leveraging semantic embeddings that bridge visual and textual representations.

### Core Approaches

```python
import torch
import clip
from PIL import Image

# 1. CLIP-based zero-shot classification
def zero_shot_classify(image_path, class_names, model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name)
    
    # Encode image
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    image_features = model.encode_image(image)
    
    # Encode class descriptions
    prompts = [f"a photo of a {cls}" for cls in class_names]
    text_tokens = clip.tokenize(prompts)
    text_features = model.encode_text(text_tokens)
    
    # Cosine similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    
    return {cls: sim.item() for cls, sim in zip(class_names, similarity[0])}

# 2. Improved prompts for better zero-shot performance
prompt_templates = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of many {}.",
    "a close-up photo of a {}.",
    "a photo of a {} in the wild.",
]

def ensemble_zero_shot(image, class_names, model, preprocess):
    image_features = model.encode_image(preprocess(image).unsqueeze(0))
    
    all_text_features = []
    for cls in class_names:
        class_features = []
        for template in prompt_templates:
            text = clip.tokenize(template.format(cls))
            feat = model.encode_text(text)
            class_features.append(feat)
        # Average across templates
        all_text_features.append(torch.stack(class_features).mean(dim=0))
    
    text_features = torch.cat(all_text_features)
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    return similarity
```

### Semantic Embedding Methods

| Method | Embedding Source | Accuracy |
|--------|-----------------|----------|
| **CLIP** | Contrastive image-text training | Best overall |
| **SigLIP** | Sigmoid loss variant of CLIP | Slightly better efficiency |
| **Attribute-based** | Human-defined attributes | Interpretable |
| **Word2Vec/GloVe** | Word embeddings | Limited by text quality |
| **LLM descriptions** | GPT-4 generated class descriptions | Strong with good prompts |

> **Interview Tip:** CLIP and SigLIP have revolutionized zero-shot classification. For interviews, know that prompt engineering significantly impacts zero-shot accuracy (up to 5-10% improvement), and prompt ensembling (averaging across multiple templates) is standard practice.

---

## Question 26
- [ ] Done
**How do you handle classification of images with artistic or stylistic variations?**

**Answer:**

Classifying images with artistic or stylistic variations (paintings, sketches, cartoons vs. photos) requires domain-invariant feature learning.

### Approaches

```python
import torch
import torch.nn as nn

# Style-invariant feature extraction
class StyleDomainNet(nn.Module):
    def __init__(self, backbone, num_classes, num_domains=6):
        super().__init__()
        self.backbone = backbone
        # Instance Normalization removes style while preserving content
        self.style_norm = nn.InstanceNorm2d(2048, affine=True)
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        features = self.backbone.extract_features(x)
        # Remove style information
        normalized = self.style_norm(features)
        pooled = normalized.mean(dim=[2, 3])
        return self.classifier(pooled)
```

### Domain Generalization Strategies

| Strategy | Description | Effectiveness |
|----------|-------------|---------------|
| **Instance normalization** | Removes domain-specific statistics | Good baseline |
| **Domain randomization** | Random style augmentation | Robust |
| **CLIP features** | Pre-trained on diverse visual data | Best zero-shot |
| **DomainBed methods** | Formal domain generalization | Research-grade |
| **Style transfer augmentation** | Generate training variants via NST | Creative, effective |

> **Interview Tip:** Instance normalization's key insight is that style is encoded in feature statistics (mean/variance) while content is in the spatial structure. By normalizing statistics, you get style-invariant features. CLIP naturally handles style variations due to its diverse training data.

---

## Question 27
- [ ] Done
**What are effective methods for handling classification in the presence of adversarial attacks?**

**Answer:**

Adversarial attacks exploit model vulnerabilities by adding imperceptible perturbations that cause misclassification. Defense requires multi-layered strategies.

### Attack and Defense Implementation

```python
import torch
import torch.nn.functional as F

# FGSM Attack
def fgsm_attack(model, image, label, epsilon=0.03):
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    loss.backward()
    
    perturbation = epsilon * image.grad.sign()
    adversarial_image = image + perturbation
    return torch.clamp(adversarial_image, 0, 1)

# PGD Attack (stronger)
def pgd_attack(model, image, label, epsilon=0.03, alpha=0.01, steps=10):
    adv_image = image.clone()
    for _ in range(steps):
        adv_image.requires_grad = True
        output = model(adv_image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        
        adv_image = adv_image + alpha * adv_image.grad.sign()
        perturbation = torch.clamp(adv_image - image, -epsilon, epsilon)
        adv_image = torch.clamp(image + perturbation, 0, 1).detach()
    return adv_image

# Adversarial Training (primary defense)
def adversarial_training_step(model, images, labels, optimizer, epsilon=0.03):
    # Generate adversarial examples
    adv_images = pgd_attack(model, images, labels, epsilon)
    
    # Train on both clean and adversarial
    clean_loss = F.cross_entropy(model(images), labels)
    adv_loss = F.cross_entropy(model(adv_images), labels)
    
    total_loss = 0.5 * clean_loss + 0.5 * adv_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Defense Methods

| Defense | Type | Robustness | Clean Accuracy |
|---------|------|-----------|----------------|
| **Adversarial training** | Training-time | Strong | ~5% drop |
| **Input preprocessing** | Runtime | Weak alone | Maintained |
| **Certified defense** | Provable | Guaranteed bound | Significant drop |
| **Ensemble adversarial** | Training-time | Broad robustness | ~3% drop |
| **Randomized smoothing** | Certified + practical | Good | ~8% drop |

> **Interview Tip:** There's a fundamental trade-off between clean accuracy and adversarial robustness. Adversarial training is the most practical defense but reduces clean accuracy. Mention that real-world attacks often don't need to be imperceptible—physical adversarial patches are a bigger practical threat.

---

## Question 28
- [ ] Done
**How do you implement progressive learning for continuously expanding class sets?**

**Answer:**

Progressive learning (also called class-incremental or continual learning) enables a classifier to learn new classes over time without forgetting old ones.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EWC(nn.Module):
    """Elastic Weight Consolidation - prevents catastrophic forgetting"""
    def __init__(self, model, importance=1000):
        super().__init__()
        self.model = model
        self.importance = importance
        self.saved_params = {}
        self.fisher_info = {}
    
    def compute_fisher(self, dataloader):
        self.model.train()
        self.fisher_info = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        for images, labels in dataloader:
            self.model.zero_grad()
            output = self.model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher_info[n] += p.grad.data ** 2
        
        # Normalize
        for n in self.fisher_info:
            self.fisher_info[n] /= len(dataloader)
        
        # Save current parameters
        self.saved_params = {n: p.data.clone() for n, p in self.model.named_parameters()}
    
    def ewc_loss(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher_info:
                loss += (self.fisher_info[n] * (p - self.saved_params[n]) ** 2).sum()
        return self.importance * loss
```

### Continual Learning Strategies

| Method | Mechanism | Memory | Forgetting |
|--------|-----------|--------|-----------|
| **EWC** | Regularize important weights | Params only | Moderate |
| **Replay buffer** | Store exemplars from old tasks | Small buffer | Low |
| **Knowledge distillation** | Distill old model's knowledge | Old model | Low |
| **Dynamic architecture** | Expand network for new classes | Growing | None |
| **Prompt tuning** | Add new prompts per task (ViT) | Prompts only | Very low |

> **Interview Tip:** The key challenge is the stability-plasticity dilemma: the model must be stable enough to retain old knowledge but plastic enough to learn new concepts. Modern approaches using prompt-based learning with frozen ViT backbones (L2P, DualPrompt) are state-of-the-art.

---

## Question 29
- [ ] Done
**What techniques help with cross-modal classification using both visual and textual features?**

**Answer:**

Cross-modal classification combines visual features with textual features (captions, metadata, tags) for more robust classification.

### Implementation

```python
import torch
import torch.nn as nn

class MultiModalClassifier(nn.Module):
    def __init__(self, vision_dim=2048, text_dim=768, num_classes=100):
        super().__init__()
        # Visual encoder (e.g., ResNet)
        self.vision_proj = nn.Linear(vision_dim, 512)
        # Text encoder (e.g., BERT)
        self.text_proj = nn.Linear(text_dim, 512)
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        # Cross-attention for interaction
        self.cross_attn = nn.MultiheadAttention(512, 8)
    
    def forward(self, image_features, text_features):
        v = self.vision_proj(image_features)
        t = self.text_proj(text_features)
        
        # Cross-attention: visual attends to text
        attended_v, _ = self.cross_attn(v.unsqueeze(0), t.unsqueeze(0), t.unsqueeze(0))
        attended_v = attended_v.squeeze(0)
        
        # Late fusion
        combined = torch.cat([attended_v, t], dim=1)
        return self.fusion(combined)
```

### Fusion Strategies

| Strategy | Method | When to Use |
|----------|--------|-------------|
| **Early fusion** | Concatenate raw features | Small models |
| **Late fusion** | Average predictions | Independent modalities |
| **Cross-attention** | Modalities attend to each other | Rich interactions |
| **Bilinear fusion** | Outer product of features | Compact interactions |
| **CLIP-style** | Contrastive alignment | Zero-shot capability |

> **Interview Tip:** CLIP and BLIP models natively provide aligned vision-language features. For practical multi-modal classification, using CLIP's joint embedding space often outperforms building custom fusion architectures.

---

## Question 30
- [ ] Done
**How do you design evaluation protocols that account for real-world deployment scenarios?**

**Answer:**

Evaluation protocols must reflect real-world deployment conditions, going beyond standard train/val/test splits.

### Comprehensive Evaluation Framework

```python
def comprehensive_evaluation(model, test_sets):
    results = {}
    
    # 1. Standard metrics
    results['accuracy'] = compute_accuracy(model, test_sets['standard'])
    results['top5'] = compute_topk_accuracy(model, test_sets['standard'], k=5)
    
    # 2. Robustness evaluation
    results['corruption_robustness'] = evaluate_corruptions(model, test_sets['corrupted'])
    results['adversarial_robustness'] = evaluate_adversarial(model, test_sets['standard'])
    
    # 3. Distribution shift
    results['ood_detection'] = evaluate_ood(model, test_sets['ood'])
    results['domain_shift'] = evaluate_domain_shift(model, test_sets['shifted'])
    
    # 4. Fairness
    results['demographic_parity'] = evaluate_fairness(model, test_sets['demographic'])
    
    # 5. Efficiency
    results['latency_ms'] = measure_latency(model)
    results['flops'] = count_flops(model)
    results['model_size_mb'] = count_parameters(model) * 4 / 1e6
    
    # 6. Calibration
    results['ece'] = expected_calibration_error(model, test_sets['standard'])
    
    return results
```

### Key Evaluation Dimensions

| Dimension | Metrics | Why It Matters |
|-----------|---------|----------------|
| **Accuracy** | Top-1, Top-5, per-class | Basic performance |
| **Robustness** | Corruption accuracy, adversarial accuracy | Real-world reliability |
| **Calibration** | ECE, reliability diagram | Trustworthy probabilities |
| **Efficiency** | FLOPs, latency, memory | Deployment feasibility |
| **Fairness** | Demographic parity, equalized odds | Ethical deployment |
| **OOD detection** | AUROC, FPR@95%TPR | Safety critical |

> **Interview Tip:** Always mention that accuracy alone is insufficient. Production models need calibrated confidence scores (for rejecting uncertain predictions), robustness to common corruptions, and fairness across demographic groups. Reference ImageNet-C, ImageNet-R, and ImageNet-A as standard robustness benchmarks.

---

## Question 31
- [ ] Done
**What approaches work best for classifying images with cultural or geographical variations?**

**Answer:**

Cultural and geographical variations include differences in architecture, vegetation, food, fashion, and daily objects that vary across regions.

### Approaches

```python
# Geo-aware training with balanced regional sampling
class GeoBalancedSampler:
    def __init__(self, dataset, regions):
        self.region_indices = {r: [] for r in regions}
        for idx, (_, meta) in enumerate(dataset):
            self.region_indices[meta['region']].append(idx)
    
    def sample(self, batch_size):
        per_region = batch_size // len(self.region_indices)
        indices = []
        for region, idxs in self.region_indices.items():
            sampled = np.random.choice(idxs, per_region, replace=True)
            indices.extend(sampled)
        return indices
```

### Key Strategies

| Strategy | Description |
|----------|-------------|
| **Diverse training data** | Include data from multiple geographic regions |
| **Geo-balanced sampling** | Ensure equal representation of regions |
| **Foundation models (CLIP)** | Trained on global web data, inherently diverse |
| **Region-specific fine-tuning** | Adapt base model per deployment region |
| **Cultural metadata** | Include geographic/cultural context as features |

> **Interview Tip:** GeoDE and Dollar Street datasets help evaluate geographic fairness. CLIP models perform better across cultures than ImageNet-trained models due to broader training data. Mention that biased training data (predominantly Western) is a systemic issue requiring deliberate data curation.

---

## Question 32
- [ ] Done
**How do you handle classification of synthetic or generated images?**

**Answer:**

Classifying synthetic/generated images (from GANs, diffusion models, etc.) presents unique challenges—both classifying the content and detecting whether the image is synthetic.

### Approaches

```python
import torch
import torch.nn as nn

class SyntheticImageClassifier(nn.Module):
    """Classify content of both real and synthetic images"""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.content_head = nn.Linear(2048, num_classes)  # What's in the image
        self.authenticity_head = nn.Linear(2048, 2)       # Real vs synthetic
    
    def forward(self, x):
        features = self.backbone(x)
        content = self.content_head(features)
        authenticity = self.authenticity_head(features)
        return content, authenticity

# Training on mixed real/synthetic data
def mixed_training(model, real_loader, synthetic_loader, optimizer):
    for real_batch, synth_batch in zip(real_loader, synthetic_loader):
        # Mix real and synthetic
        images = torch.cat([real_batch[0], synth_batch[0]])
        content_labels = torch.cat([real_batch[1], synth_batch[1]])
        auth_labels = torch.cat([
            torch.zeros(len(real_batch[0])),    # 0 = real
            torch.ones(len(synth_batch[0]))      # 1 = synthetic
        ]).long()
        
        content_out, auth_out = model(images)
        loss = F.cross_entropy(content_out, content_labels) + F.cross_entropy(auth_out, auth_labels)
        loss.backward()
        optimizer.step()
```

### Key Considerations

| Challenge | Solution |
|-----------|----------|
| **GAN artifacts** | Frequency domain analysis, spectral features |
| **Diffusion model outputs** | Content-based rather than artifact-based |
| **Domain gap** | Train on diverse synthetic sources |
| **Evolving generators** | Continual learning / regular retraining |

> **Interview Tip:** Synthetic image detection is increasingly important for misinformation detection. Key insight: GAN images have telltale frequency-domain artifacts (checkerboard patterns from transposed convolutions), while diffusion model images are harder to detect due to iterative refinement.

---

## Question 33
- [ ] Done
**What techniques are effective for classifying images with varying aspect ratios and compositions?**

**Answer:**

Aspect ratio and composition variations require models to handle images of different proportions and layouts.

### Techniques

```python
import torch
import torchvision.transforms as T

# Aspect-ratio preserving approaches
# 1. Letterboxing/padding
def letterbox_resize(image, target_size=224):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    # Pad to square
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return padded

# 2. Multi-crop evaluation
def multi_crop_predict(model, image, crops=10):
    # Center + 4 corners + horizontal flips
    predictions = []
    for crop_fn in get_crop_functions(crops):
        cropped = crop_fn(image)
        pred = model(cropped)
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)

# 3. Dynamic input resolution (ViT)
# ViTs can handle different resolutions by interpolating position embeddings
```

### Best Practices

| Strategy | How It Works |
|----------|-------------|
| **Letterboxing** | Preserve ratio, pad to square |
| **Random crop + resize** | Training augmentation |
| **Multi-crop TTA** | Test-time augmentation |
| **Aspect ratio bucketing** | Group similar ratios in batches |
| **Flexible ViT** | Interpolate position embeddings |

> **Interview Tip:** Naive center-cropping or stretching loses information or introduces distortion. Aspect ratio bucketing (used in Stable Diffusion training) groups similar-ratio images for efficient batching without distortion.

---

## Question 34
- [ ] Done
**How do you implement uncertainty quantification in image classification predictions?**

**Answer:**

Uncertainty quantification tells us how confident the model is in its predictions—essential for safety-critical applications.

### Methods

```python
import torch
import torch.nn.functional as F

# 1. MC Dropout
class MCDropoutClassifier(torch.nn.Module):
    def __init__(self, backbone, num_classes, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(2048, num_classes)
    
    def predict_with_uncertainty(self, x, n_samples=50):
        self.train()  # Keep dropout active
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                feat = self.backbone(x)
                feat = self.dropout(feat)
                pred = F.softmax(self.fc(feat), dim=1)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0).sum(dim=1)   # Model uncertainty
        aleatoric = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1)  # Data uncertainty
        
        return mean_pred, epistemic, aleatoric

# 2. Temperature Scaling (post-hoc calibration)
class TemperatureScaling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature

# 3. Deep Ensemble
def ensemble_uncertainty(models, x):
    predictions = [F.softmax(model(x), dim=1) for model in models]
    predictions = torch.stack(predictions)
    mean = predictions.mean(dim=0)
    variance = predictions.var(dim=0).sum(dim=1)
    return mean, variance
```

### Comparison

| Method | Epistemic | Aleatoric | Cost |
|--------|-----------|-----------|------|
| **MC Dropout** | Yes | Approximate | N forward passes |
| **Deep Ensemble** | Yes | Approximate | N models |
| **Temperature Scaling** | No | No | Calibration only |
| **Evidential Deep Learning** | Yes | Yes | Single pass |

> **Interview Tip:** Distinguish epistemic (model uncertainty, reducible with more data) from aleatoric (inherent data noise, irreducible). Deep ensembles provide the best uncertainty estimates but at high cost. MC Dropout is the practical go-to.

---

## Question 35
- [ ] Done
**What are the best practices for handling privacy-preserving image classification?**

**Answer:**

Privacy-preserving classification protects sensitive data while training and deploying image classification models.

### Techniques

```python
import torch
import numpy as np

# 1. Differential Privacy with Opacus
# pip install opacus
from opacus import PrivacyEngine

def train_with_dp(model, train_loader, epochs=10, epsilon=1.0):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=1e-5,
        epochs=epochs,
        max_grad_norm=1.0,
    )
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
    
    print(f"Final epsilon: {privacy_engine.get_epsilon(delta=1e-5)}")

# 2. Federated Learning sketch
class FederatedClassifier:
    def __init__(self, global_model, n_clients):
        self.global_model = global_model
        self.n_clients = n_clients
    
    def federated_round(self, client_data_loaders):
        client_updates = []
        for loader in client_data_loaders:
            local_model = copy.deepcopy(self.global_model)
            # Train locally
            train_local(local_model, loader, epochs=5)
            client_updates.append(local_model.state_dict())
        
        # Aggregate (FedAvg)
        avg_state = {}
        for key in client_updates[0]:
            avg_state[key] = torch.stack([u[key].float() for u in client_updates]).mean(0)
        self.global_model.load_state_dict(avg_state)
```

### Privacy Methods

| Method | Protection Level | Accuracy Impact |
|--------|-----------------|----------------|
| **Differential Privacy** | Mathematical guarantee | 5-15% drop |
| **Federated Learning** | Data stays local | Minimal |
| **Secure Aggregation** | Encrypted updates | None |
| **Model watermarking** | IP protection | None |

> **Interview Tip:** Know the trade-off between privacy guarantee (epsilon) and accuracy. Smaller epsilon = stronger privacy = lower accuracy. Federated learning is practical for large-scale deployment with data privacy requirements (healthcare, finance).

---

## Question 36
- [ ] Done
**How do you design architectures that handle both common and rare class instances effectively?**

**Answer:**

Handling both common and rare classes (long-tailed distribution) requires architectures that don't ignore the tail.

### Approaches

```python
import torch
import torch.nn as nn

class DecoupledTraining:
    """Decoupled training: representation learning + balanced classifier"""
    
    def stage1_representation(self, model, loader, epochs=90):
        """Train with instance-balanced sampling (natural distribution)"""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for epoch in range(epochs):
            for images, labels in loader:
                loss = nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
    
    def stage2_classifier(self, model, loader, epochs=10):
        """Retrain classifier with class-balanced sampling"""
        # Freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
        balanced_loader = make_class_balanced_loader(loader)
        
        for epoch in range(epochs):
            for images, labels in balanced_loader:
                loss = nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()

# Learnable weight scaling (tau-normalization)
class TauNormClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=False)
        self.tau = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Normalize both features and weights
        x_norm = nn.functional.normalize(x, dim=1)
        w_norm = nn.functional.normalize(self.fc.weight, dim=1)
        return self.tau * (x_norm @ w_norm.T)
```

### Methods for Long-Tailed Classification

| Method | Type | Effectiveness |
|--------|------|---------------|
| **Decoupled training** | Two-stage | State-of-the-art |
| **Class-balanced loss** | Loss reweighting | Good baseline |
| **Mixup** | Augmentation | Helps tail classes |
| **Feature augmentation** | Generate tail features | Advanced |
| **Tau-normalization** | Post-hoc calibration | Simple, effective |

> **Interview Tip:** The decoupled training discovery was key: representation learning benefits from the natural (imbalanced) distribution, while classifier learning benefits from balanced training. This simple two-stage approach matches or beats complex methods.

---

## Question 37
- [ ] Done
**What approaches work best for real-time image classification in streaming applications?**

**Answer:**

Real-time classification in streaming applications must maintain low latency while processing continuous image streams.

### Architecture and Optimization

```python
import torch
import time
from collections import deque

class StreamingClassifier:
    def __init__(self, model, device='cuda', batch_size=8, buffer_size=32):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        
        # Optimize with TorchScript and CUDA
        self.model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(device))
        
        # Warm up GPU
        with torch.no_grad():
            for _ in range(10):
                self.model(torch.randn(batch_size, 3, 224, 224).to(device))
    
    @torch.inference_mode()
    def classify_frame(self, frame):
        # Preprocess
        tensor = preprocess(frame).unsqueeze(0).to(self.device)
        
        # Inference with CUDA streams for overlap
        output = self.model(tensor)
        probs = torch.softmax(output, dim=1)
        
        # Temporal smoothing
        self.buffer.append(probs)
        smoothed = torch.stack(list(self.buffer)).mean(dim=0)
        
        return smoothed.argmax(dim=1), smoothed.max(dim=1).values

# Latency benchmarking
def benchmark_model(model, input_size=(1, 3, 224, 224), n_runs=100):
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size).to(device)
    
    # Warm up
    for _ in range(10):
        model(dummy)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        model(dummy)
    torch.cuda.synchronize()
    
    avg_ms = (time.time() - start) / n_runs * 1000
    return avg_ms
```

### Optimization Techniques

| Technique | Speedup | Accuracy Impact |
|-----------|---------|----------------|
| **TorchScript/ONNX** | 1.5-3x | None |
| **FP16 inference** | 2x on GPU | <0.1% |
| **INT8 quantization** | 3-4x on CPU | <1% |
| **TensorRT** | 3-5x on NVIDIA | <0.5% |
| **Batched inference** | Linear with batch | None |

> **Interview Tip:** Real-time means different things: 30 FPS for video (33ms limit), or sub-100ms for API responses. Know the optimization stack: model architecture → quantization → runtime optimization (TensorRT/ONNX Runtime) → hardware (GPU/NPU/dedicated accelerator).

---

## Question 38
- [ ] Done
**How do you handle classification of images with metadata or contextual information?**

**Answer:**

Leveraging metadata (GPS coordinates, timestamps, camera settings, text tags) alongside images can significantly improve classification.

### Implementation

```python
import torch
import torch.nn as nn

class MetadataAwareClassifier(nn.Module):
    def __init__(self, backbone, num_classes, metadata_dim=32):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        
        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, image, metadata):
        img_features = self.backbone(image)
        meta_features = self.metadata_encoder(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)

# Metadata feature engineering
def encode_metadata(exif_data):
    features = []
    # Temporal: hour of day (cyclical encoding)
    hour = exif_data.get('hour', 12)
    features.extend([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)])
    # GPS: latitude, longitude (normalized)
    features.extend([exif_data.get('lat', 0)/90, exif_data.get('lon', 0)/180])
    # Camera: focal length, ISO (normalized)
    features.append(exif_data.get('focal_length', 50) / 500)
    features.append(np.log(exif_data.get('iso', 100) + 1) / 10)
    return np.array(features, dtype=np.float32)
```

| Metadata Type | Use Case | Encoding |
|---------------|----------|----------|
| **GPS** | Geographic context | Normalized lat/lon |
| **Timestamp** | Temporal patterns | Cyclical sin/cos |
| **Camera settings** | Image quality context | Log-normalized |
| **Text tags** | Semantic context | BERT embeddings |
| **Weather** | Environmental context | One-hot / embedding |

> **Interview Tip:** Metadata is often "free" information that's ignored. Simple late fusion (concatenate metadata features with image features) works surprisingly well. The key is proper encoding—cyclical features for time, embeddings for categorical metadata.

---

## Question 39
- [ ] Done
**What techniques help with robust classification under dataset shift?**

**Answer:**

Dataset shift occurs when training and deployment data distributions differ—a common real-world problem.

### Types and Solutions

```python
import torch
import torch.nn as nn

# 1. Test-Time Training (TTT) / Test-Time Adaptation
class TestTimeAdaptation:
    def __init__(self, model, lr=0.001):
        self.model = model
        # Only adapt normalization layers
        self.params = [p for n, p in model.named_parameters() if 'bn' in n or 'norm' in n]
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
    
    def adapt(self, test_batch, n_steps=10):
        self.model.train()
        for _ in range(n_steps):
            # Self-supervised objective (entropy minimization)
            output = self.model(test_batch)
            probs = torch.softmax(output, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            
            self.optimizer.zero_grad()
            entropy.backward()
            self.optimizer.step()
        
        self.model.eval()
        return self.model(test_batch)

# 2. Domain-Invariant Features
class CORAL:
    """Correlation Alignment for domain adaptation"""
    @staticmethod
    def coral_loss(source_features, target_features):
        d = source_features.size(1)
        cs = torch.cov(source_features.T)
        ct = torch.cov(target_features.T)
        loss = (cs - ct).pow(2).sum() / (4 * d * d)
        return loss
```

### Dataset Shift Types

| Shift Type | Description | Solution |
|------------|-------------|----------|
| **Covariate shift** | Input distribution changes | Domain adaptation |
| **Label shift** | Class proportions change | Prior adjustment |
| **Concept drift** | P(Y|X) changes over time | Continual learning |
| **Domain shift** | Different environments | Domain generalization |

> **Interview Tip:** Test-time adaptation (TENT, TTT) is a practical approach that adapts batch normalization statistics at inference time. It requires no labeled target data and can handle gradual shifts. Mention it as a modern practical solution.

---

## Question 40
- [ ] Done
**How do you implement fairness-aware training to reduce classification bias?**

**Answer:**

Fairness-aware training ensures classification models don't discriminate based on protected attributes (race, gender, age).

### Implementation

```python
import torch
import torch.nn as nn

class FairnessRegularizedClassifier(nn.Module):
    def __init__(self, backbone, num_classes, num_groups):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)
        self.adversary = nn.Linear(2048, num_groups)  # Predict protected attribute
    
    def forward(self, x):
        features = self.backbone(x)
        class_output = self.classifier(features)
        # Adversarial: try to remove group information from features
        group_output = self.adversary(GradientReversal.apply(features, 1.0))
        return class_output, group_output

def fairness_loss(class_output, group_output, labels, groups, lambda_fair=1.0):
    task_loss = nn.functional.cross_entropy(class_output, labels)
    fairness_loss = nn.functional.cross_entropy(group_output, groups)
    return task_loss + lambda_fair * fairness_loss

# Fairness metrics
def compute_fairness_metrics(predictions, labels, groups):
    metrics = {}
    for g in groups.unique():
        mask = groups == g
        metrics[f'accuracy_group_{g}'] = (predictions[mask] == labels[mask]).float().mean()
    
    accuracies = list(metrics.values())
    metrics['demographic_parity_gap'] = max(accuracies) - min(accuracies)
    metrics['equalized_odds_gap'] = compute_equalized_odds(predictions, labels, groups)
    return metrics
```

### Fairness Metrics

| Metric | Definition | Goal |
|--------|------------|------|
| **Demographic Parity** | P(Ŷ=1) equal across groups | Equal prediction rates |
| **Equalized Odds** | TPR and FPR equal across groups | Equal error rates |
| **Calibration** | P(Y=1|Ŷ=p) equal across groups | Equal reliability |
| **Individual Fairness** | Similar individuals get similar predictions | Consistency |

> **Interview Tip:** There's a mathematical impossibility result—you can't satisfy all fairness criteria simultaneously (except in degenerate cases). Know the trade-offs and mention that the choice depends on the application context and regulatory requirements.

---

## Question 41
- [ ] Done
**What are effective strategies for handling classification of compressed or low-quality images?**

**Answer:**

Compressed and low-quality images (JPEG artifacts, low resolution, noise) are common in real-world deployment.

### Approaches

```python
import albumentations as A

# Simulate compression artifacts during training
compression_augmentation = A.Compose([
    A.ImageCompression(quality_lower=20, quality_upper=95, p=0.5),
    A.Downscale(scale_min=0.25, scale_max=0.75, p=0.3),
    A.GaussNoise(var_limit=(10, 80), p=0.3),
    A.Blur(blur_limit=7, p=0.3),
    A.ISONoise(p=0.2),
])

# Quality-aware classifier
class QualityAwareClassifier(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.quality_estimator = torch.nn.Linear(2048, 1)  # Predict quality score
        self.classifier = torch.nn.Linear(2048 + 1, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        quality = torch.sigmoid(self.quality_estimator(features))
        # Quality-conditioned classification
        combined = torch.cat([features, quality], dim=1)
        return self.classifier(combined), quality
```

> **Interview Tip:** The most effective strategy is training with simulated degradations (JPEG compression, blur, noise, low resolution) so the model learns to be robust. This is simpler and more effective than building a separate preprocessing/enhancement pipeline.

---

## Question 42
- [ ] Done
**How do you design multi-task learning frameworks that share classification knowledge?**

**Answer:**

Multi-task learning shares representations across related classification tasks to improve all tasks simultaneously.

### Implementation

```python
import torch
import torch.nn as nn

class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone, task_configs):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, n_classes in task_configs.items():
            self.heads[task_name] = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, n_classes)
            )
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = {task: head(features) for task, head in self.heads.items()}
        return outputs

# Dynamic task weighting
class UncertaintyWeighting(nn.Module):
    """Learn task weights automatically via uncertainty"""
    def __init__(self, n_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total
```

### Task Relationship Management

| Relationship | Strategy | Example |
|-------------|----------|---------|
| **Complementary** | Shared backbone, separate heads | Species + Habitat |
| **Auxiliary** | Auxiliary task improves main | Depth estimation helps segmentation |
| **Conflicting** | Gradient surgery / separate paths | Different-scale tasks |

> **Interview Tip:** Multi-task learning can hurt performance if tasks conflict (negative transfer). Use gradient-based approaches like GradNorm or PCGrad to detect and handle conflicting gradients. The key design question is which tasks share which layers.

---

## Question 43
- [ ] Done
**What approaches work best for classifying images in specialized domains like satellite imagery?**

**Answer:**

Satellite imagery classification involves unique challenges: very large images, specific spectral bands, geographic patterns.

### Specialized Techniques

```python
import torch
import torch.nn as nn

# Handle multispectral input (satellite images have more than RGB)
class MultispectralClassifier(nn.Module):
    def __init__(self, in_channels=13, num_classes=10):  # Sentinel-2 has 13 bands
        super().__init__()
        # Adapt first conv layer for multispectral input
        backbone = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        
        # Initialize: copy RGB weights, random for other bands
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv.weight
            nn.init.kaiming_normal_(self.conv1.weight[:, 3:])
        
        backbone.conv1 = self.conv1
        self.backbone = backbone
        self.backbone.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Tile-based processing for large satellite images
def classify_satellite_image(model, image, tile_size=256, overlap=32):
    H, W = image.shape[1:]
    predictions = torch.zeros(H, W)
    counts = torch.zeros(H, W)
    
    for y in range(0, H - tile_size + 1, tile_size - overlap):
        for x in range(0, W - tile_size + 1, tile_size - overlap):
            tile = image[:, y:y+tile_size, x:x+tile_size]
            pred = model(tile.unsqueeze(0))
            predictions[y:y+tile_size, x:x+tile_size] += pred.squeeze()
            counts[y:y+tile_size, x:x+tile_size] += 1
    
    return predictions / counts.clamp(min=1)
```

### Key Considerations

| Aspect | Challenge | Solution |
|--------|-----------|----------|
| **Image size** | 10K × 10K pixels | Tiled processing |
| **Spectral bands** | 13+ channels | Adapt first conv layer |
| **Scale** | Large geographic areas | Multi-scale training |
| **Data sources** | Sentinel-2, Landsat | Sensor-specific normalization |
| **Temporal** | Time series available | Temporal aggregation |

> **Interview Tip:** Satellite classification differs from natural image classification mainly in input (multispectral, very high resolution) and task (land use, crop type, change detection). Foundation models like SatMAE and SkySense are specifically pre-trained for remote sensing.

---

## Question 44
- [ ] Done
**How do you handle classification with evolving class definitions over time?**

**Answer:**

Evolving class definitions require models that gracefully handle changes in taxonomy, merging/splitting of classes over time.

### Approaches

```python
import torch
import torch.nn as nn

class FlexibleClassifier(nn.Module):
    """Supports dynamic class addition/merging/splitting"""
    def __init__(self, backbone, initial_classes=10, embed_dim=512):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Linear(2048, embed_dim)
        # Class prototypes instead of fixed linear layer
        self.class_prototypes = nn.Parameter(torch.randn(initial_classes, embed_dim))
    
    def forward(self, x):
        features = self.projector(self.backbone(x))
        features = nn.functional.normalize(features, dim=1)
        prototypes = nn.functional.normalize(self.class_prototypes, dim=1)
        logits = features @ prototypes.T * 10  # Temperature-scaled cosine similarity
        return logits
    
    def add_class(self, new_class_images):
        """Add new class from examples"""
        with torch.no_grad():
            features = self.projector(self.backbone(new_class_images))
            new_prototype = features.mean(dim=0, keepdim=True)
        self.class_prototypes = nn.Parameter(
            torch.cat([self.class_prototypes.data, new_prototype])
        )
    
    def merge_classes(self, class_indices):
        """Merge multiple classes into one"""
        merged = self.class_prototypes.data[class_indices].mean(dim=0, keepdim=True)
        keep = [i for i in range(len(self.class_prototypes)) if i not in class_indices]
        self.class_prototypes = nn.Parameter(
            torch.cat([self.class_prototypes.data[keep], merged])
        )
```

> **Interview Tip:** Prototype-based classifiers (using cosine similarity to class prototypes) are more flexible than fixed linear layers for evolving taxonomies. You can add/remove/merge classes by simply manipulating prototypes without retraining the entire model.

---

## Question 45
- [ ] Done
**What techniques are most effective for explaining classification decisions to end users?**

**Answer:**

Explaining classification decisions to end users requires methods that produce human-understandable explanations.

### Techniques

```python
import lime
from lime import lime_image
import shap

# 1. LIME explanation
def explain_with_lime(model, image, class_names):
    explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(images):
        tensor = preprocess_batch(images)
        with torch.no_grad():
            return torch.softmax(model(tensor), dim=1).numpy()
    
    explanation = explainer.explain_instance(
        image, predict_fn, top_labels=3, num_samples=1000
    )
    
    # Visual explanation
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5
    )
    return temp, mask

# 2. Counterfactual explanation
def generate_counterfactual(model, image, target_class, steps=100, lr=0.01):
    """What minimal change would make the model predict target_class?"""
    perturbed = image.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([perturbed], lr=lr)
    
    for _ in range(steps):
        output = model(perturbed)
        target_loss = -output[0, target_class]
        similarity_loss = ((perturbed - image) ** 2).sum()
        loss = target_loss + 0.1 * similarity_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    difference = (perturbed - image).abs()
    return perturbed.detach(), difference.detach()
```

### Explanation Types

| Method | Output | Best For |
|--------|--------|----------|
| **Grad-CAM** | Heatmap | "Where did it look?" |
| **LIME** | Highlighted regions | "Which parts mattered?" |
| **SHAP** | Feature importance | "How much did each feature contribute?" |
| **Counterfactual** | Modified image | "What would need to change?" |
| **Concept-based** | Semantic concepts | "What concepts drove the decision?" |

> **Interview Tip:** Different stakeholders need different explanations: engineers need Grad-CAM/SHAP for debugging, end users need simple "the model focused on X" explanations, and regulators need comprehensive audit trails. Design explanation systems for your specific audience.

---

## Question 46
- [ ] Done
**How do you implement online learning for classification models that adapt to new data?**

**Answer:**

Online learning enables classification models to continuously learn from new data arriving in a stream.

### Implementation

```python
import torch
import torch.nn as nn
from collections import deque

class OnlineLearningClassifier:
    def __init__(self, model, lr=0.001, buffer_size=1000):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
    
    def update(self, new_images, new_labels, replay_ratio=0.5):
        """Update model with new data + replay buffer"""
        self.model.train()
        
        # Add to replay buffer
        for img, lbl in zip(new_images, new_labels):
            self.replay_buffer.append((img, lbl))
        
        # Mix new data with replay
        batch_images = [new_images]
        batch_labels = [new_labels]
        
        if len(self.replay_buffer) > 0:
            n_replay = int(len(new_images) * replay_ratio)
            replay_indices = np.random.choice(len(self.replay_buffer), n_replay)
            replay_imgs = torch.stack([self.replay_buffer[i][0] for i in replay_indices])
            replay_lbls = torch.tensor([self.replay_buffer[i][1] for i in replay_indices])
            batch_images.append(replay_imgs)
            batch_labels.append(replay_lbls)
        
        all_images = torch.cat(batch_images)
        all_labels = torch.cat(batch_labels)
        
        loss = nn.functional.cross_entropy(self.model(all_images), all_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### Strategies

| Method | Forgetting Prevention | Memory |
|--------|----------------------|--------|
| **Experience replay** | Store and revisit old data | Buffer needed |
| **EWC** | Regularize important weights | Parameters |
| **Progressive nets** | Freeze old, add new capacity | Growing model |
| **Adapter tuning** | Add small adapters per task | Small overhead |

> **Interview Tip:** Online learning's main challenge is catastrophic forgetting. The experience replay buffer is the simplest and most effective approach. For production, combine a replay buffer with periodic full retraining when enough new data accumulates.

---

## Question 47
- [ ] Done
**What are the considerations for classification in federated learning scenarios?**

**Answer:**

Federated learning for classification trains models across distributed data without sharing raw images.

### Core Concepts

```python
import torch
import copy

class FederatedAveraging:
    """FedAvg: the standard federated learning algorithm"""
    
    def __init__(self, global_model, n_clients, client_fraction=0.1):
        self.global_model = global_model
        self.n_clients = n_clients
        self.client_fraction = client_fraction
    
    def server_round(self, client_loaders, local_epochs=5):
        # Select random subset of clients
        n_selected = max(1, int(self.n_clients * self.client_fraction))
        selected = np.random.choice(self.n_clients, n_selected, replace=False)
        
        client_weights = []
        client_sizes = []
        
        for client_id in selected:
            # Send global model to client
            local_model = copy.deepcopy(self.global_model)
            
            # Local training
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
            for epoch in range(local_epochs):
                for images, labels in client_loaders[client_id]:
                    loss = torch.nn.functional.cross_entropy(local_model(images), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            client_weights.append(local_model.state_dict())
            client_sizes.append(len(client_loaders[client_id].dataset))
        
        # Weighted aggregation
        total_size = sum(client_sizes)
        new_state = {}
        for key in client_weights[0]:
            new_state[key] = sum(
                w[key] * (s / total_size) for w, s in zip(client_weights, client_sizes)
            )
        
        self.global_model.load_state_dict(new_state)
```

### Federated Learning Challenges

| Challenge | Solution |
|-----------|----------|
| **Non-IID data** | FedProx, SCAFFOLD |
| **Communication cost** | Gradient compression, fewer rounds |
| **Privacy** | Differential privacy, secure aggregation |
| **Heterogeneous devices** | Asynchronous updates |

> **Interview Tip:** Federated learning is especially important for healthcare (HIPAA) and finance (data privacy). The main challenge is non-IID data (each client has different class distributions). FedProx adds a proximal term to keep local models close to the global model.

---

## Question 48
- [ ] Done
**How do you design robust evaluation metrics for imbalanced classification problems?**

**Answer:**

Robust evaluation metrics for imbalanced classification go beyond accuracy to capture performance across all classes.

### Comprehensive Metrics

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, matthews_corrcoef
)
import numpy as np

def comprehensive_imbalanced_metrics(y_true, y_pred, y_probs):
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = (y_true == y_pred).mean()
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['macro_f1'] = report['macro avg']['f1-score']
    metrics['weighted_f1'] = report['weighted avg']['f1-score']
    
    # Ranking metrics (use probabilities)
    metrics['macro_auroc'] = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
    metrics['macro_auprc'] = average_precision_score(y_true, y_probs, average='macro')
    
    # Class-specific performance
    for cls in np.unique(y_true):
        mask = y_true == cls
        metrics[f'class_{cls}_recall'] = (y_pred[mask] == cls).mean()
    
    return metrics
```

### Metric Selection Guide

| Metric | When to Use | Imbalance-Robust |
|--------|-------------|-----------------|
| **Accuracy** | Balanced datasets only | No |
| **Balanced Accuracy** | General imbalanced | Yes |
| **Macro F1** | Equal class importance | Yes |
| **Weighted F1** | Proportional importance | Partially |
| **MCC** | Binary, highly imbalanced | Yes |
| **AUPRC** | Rare positive class | Yes |
| **AUROC** | Ranking quality | Partially |

> **Interview Tip:** Never report only accuracy on imbalanced data. Matthews Correlation Coefficient (MCC) is the most informative single metric for binary imbalanced classification. For multi-class, balanced accuracy + macro F1 + per-class recall provide a complete picture.

---

## Question 49
- [ ] Done
**What approaches work best for classifying images with multiple annotation sources?**

**Answer:**

Multiple annotation sources (different annotators, automated tools, crowdsourcing) create label inconsistencies that must be reconciled.

### Approaches

```python
import numpy as np
import torch

class MultiAnnotatorLearning:
    def __init__(self, n_annotators, n_classes):
        self.n_annotators = n_annotators
        self.n_classes = n_classes
    
    def dawid_skene(self, annotations, max_iter=50):
        """Dawid-Skene algorithm for estimating true labels from multiple annotators"""
        n_samples = len(annotations)
        
        # Initialize with majority voting
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = [a for a in annotations[i] if a is not None]
            labels[i] = max(set(votes), key=votes.count) if votes else 0
        
        for _ in range(max_iter):
            # E-step: estimate annotator confusion matrices
            confusion = np.zeros((self.n_annotators, self.n_classes, self.n_classes))
            for i, annots in enumerate(annotations):
                for j, label in enumerate(annots):
                    if label is not None:
                        confusion[j, labels[i], label] += 1
            
            # Normalize
            confusion = confusion / confusion.sum(axis=2, keepdims=True).clip(1e-8)
            
            # M-step: re-estimate labels using confusion matrices
            for i in range(n_samples):
                posteriors = np.ones(self.n_classes)
                for j, label in enumerate(annotations[i]):
                    if label is not None:
                        posteriors *= confusion[j, :, label]
                labels[i] = posteriors.argmax()
        
        return labels, confusion
```

### Strategies

| Method | Approach | Quality |
|--------|----------|---------|
| **Majority voting** | Simple counting | Baseline |
| **Dawid-Skene** | Model annotator reliability | Good |
| **CROWDLAB** | Deep learning + annotator modeling | State-of-the-art |
| **Soft labels** | Average label distributions | Preserves uncertainty |

> **Interview Tip:** In practice, annotator quality varies significantly. The Dawid-Skene algorithm or its modern deep learning variants (learning from crowds) jointly learn the true label and each annotator's confusion matrix. This is far superior to majority voting.

---

## Question 50
- [ ] Done
**How do you handle classification optimization when training data and deployment data differ significantly?**

**Answer:**

Handling differences between training data and deployment data is critical for reliable production classification systems.

### Strategies

```python
import torch
from torch.utils.data import DataLoader

class ProductionOptimizer:
    """Handles train/deploy distribution mismatch"""
    
    def __init__(self, model):
        self.model = model
    
    def update_bn_stats(self, deployment_data, num_batches=100):
        """Update BatchNorm statistics with deployment data distribution"""
        self.model.train()
        loader = DataLoader(deployment_data, batch_size=64, shuffle=True)
        
        # Reset BN statistics
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.running_mean.zero_()
                module.running_var.fill_(1)
                module.num_batches_tracked.zero_()
        
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                if i >= num_batches:
                    break
                self.model(images)
        
        self.model.eval()
    
    def importance_weighted_training(self, train_loader, deploy_loader, density_estimator):
        """Re-weight training samples to match deployment distribution"""
        for images, labels in train_loader:
            # Estimate importance weights
            train_density = density_estimator.log_prob(images)
            deploy_density = density_estimator.log_prob_target(images)
            weights = torch.exp(deploy_density - train_density).clamp(0.1, 10)
            
            output = self.model(images)
            loss = (weights * torch.nn.functional.cross_entropy(output, labels, reduction='none')).mean()
            loss.backward()
```

### Mitigation Approaches

| Approach | When to Use |
|----------|-------------|
| **BN statistics update** | Different image statistics |
| **Importance weighting** | Covariate shift |
| **Test-time augmentation** | Reduce prediction variance |
| **Domain randomization** | Known deployment variations |
| **Monitoring + retraining** | Gradual drift |

> **Interview Tip:** The simplest production fix is often updating BatchNorm running statistics with a small sample of deployment data. This handles many common distribution shifts (different cameras, lighting, preprocessing) without any retraining.

---


---

# --- Object Detection Questions (from 07_computer_vision/02_object_detection) ---

# Object Detection (YOLO, R-CNN) - Theory Questions

## Question 1
- [ ] Done
**How does YOLOv10's architecture differ from previous versions, and what specific improvements does it offer?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the key innovations in YOLOv10's end-to-end object detection that eliminate NMS post-processing?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement and tune the anchor-free detection mechanism in modern YOLO versions?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What strategies work best for training YOLO models on custom datasets with limited annotations?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you optimize YOLO inference speed for real-time applications while maintaining accuracy?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What are the trade-offs between single-stage (YOLO, SSD) and two-stage (Faster R-CNN) detectors?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you handle multi-scale object detection in YOLO using Feature Pyramid Networks (FPN)?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What techniques help improve YOLO's performance on small object detection?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement data augmentation strategies specific to object detection tasks?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What are the best practices for handling class imbalance in object detection datasets?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you design loss functions that balance localization and classification in YOLO?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What approaches work best for detecting objects with extreme aspect ratios using YOLO?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement and evaluate object tracking using YOLO-based detection?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What techniques help with detecting partially occluded objects in YOLO models?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle domain adaptation when deploying YOLO models to new environments?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What are the considerations for training YOLO on datasets with dense object arrangements?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you implement attention mechanisms to improve YOLO's feature extraction?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What strategies work best for reducing false positives in YOLO detection results?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you design ensemble methods combining different detection architectures?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What techniques help with detecting objects under varying lighting conditions?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement active learning for efficient annotation of detection datasets?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What approaches work best for fine-tuning pre-trained YOLO models on domain-specific data?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle detection of objects with similar appearances but different classes?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What are the best practices for optimizing YOLO models for edge deployment?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement hard negative mining to improve YOLO training efficiency?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What techniques help with detecting objects in cluttered or complex backgrounds?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you design evaluation metrics that accurately reflect detection performance?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What approaches work best for detecting objects across different scales in the same image?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle temporal consistency in video object detection using YOLO?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What techniques help with detecting objects with deformable shapes using YOLO?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement knowledge distillation for compressing large detection models?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What strategies work best for detecting objects in adverse weather conditions?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle detection of objects with inter-class confusion using YOLO?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What approaches work best for few-shot object detection in novel categories?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement uncertainty quantification in YOLO detection predictions?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What techniques help with detecting objects in high-resolution images efficiently?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you design architectures that handle both common and rare object classes?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What approaches work best for detecting objects with significant pose variations?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle detection in scenarios with heavy occlusion or crowding?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What techniques help with detecting objects across different camera viewpoints?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you implement online learning for detection models that adapt to new classes?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What strategies work best for detecting objects in specialized domains like medical imaging?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you handle detection with limited computational resources on mobile devices?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What approaches work best for detecting objects with temporal appearance changes?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you design robust training procedures for noisy or weakly supervised detection data?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What techniques help with detecting objects in images with varying quality and resolution?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you implement fairness-aware detection to reduce bias across different groups?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches work best for detecting objects in synthetic or artificially generated images?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you handle detection optimization when balancing precision and recall requirements?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What techniques help with explaining detection decisions and improving model interpretability?**
**Answer:** _To be filled_

---


---

# --- Instance Segmentation Questions (from 07_computer_vision/03_instance_segmentation) ---

# Instance Segmentation (Mask R-CNN) - Theory Questions

## Question 1
- [ ] Done
**How does Mask R-CNN's architecture balance object detection and pixel-level segmentation accuracy?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the key differences between ROIPool and ROIAlign, and why is ROIAlign crucial for segmentation?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement and optimize the mask head in Mask R-CNN for different object types?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What strategies work best for training Mask R-CNN on datasets with incomplete mask annotations?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you handle class imbalance in instance segmentation when some classes are rare?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What techniques help improve Mask R-CNN's performance on small objects or fine details?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement data augmentation that preserves both bounding boxes and mask accuracy?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What are the trade-offs between segmentation accuracy and inference speed in Mask R-CNN?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you design loss functions that effectively balance detection and segmentation objectives?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What approaches work best for handling overlapping instances in dense object arrangements?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you implement multi-scale training and testing for Mask R-CNN?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What techniques help with segmenting objects that have complex or irregular shapes?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you handle domain adaptation when applying Mask R-CNN to new visual domains?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What strategies work best for fine-tuning pre-trained Mask R-CNN models on custom datasets?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you implement active learning strategies for efficient mask annotation?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What approaches help with segmenting transparent or reflective objects using Mask R-CNN?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you optimize Mask R-CNN for real-time applications without significant accuracy loss?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What techniques help with handling mask annotation noise and inconsistencies?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you design evaluation metrics that properly assess instance segmentation quality?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What approaches work best for segmenting objects with significant pose or viewpoint changes?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement knowledge distillation for compressing Mask R-CNN models?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What techniques help with segmenting objects in cluttered or complex backgrounds?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle temporal consistency in video instance segmentation?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies work best for segmenting objects with deformable or articulated parts?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement uncertainty quantification in instance segmentation predictions?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches help with segmenting objects under varying lighting conditions?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you design architectures that handle both common and rare instance classes?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What techniques work best for segmenting objects with inter-class visual similarity?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle segmentation of objects with partial occlusion or truncation?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What approaches work best for few-shot instance segmentation in novel categories?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement panoptic segmentation by combining instance and semantic segmentation?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What techniques help with segmenting objects across different scales in the same image?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle segmentation in scenarios with heavy object crowding?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies work best for segmenting objects in specialized domains like medical imaging?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you optimize mask quality while maintaining computational efficiency?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches help with segmenting objects that undergo significant deformation?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement online learning for segmentation models adapting to new classes?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques work best for segmenting objects in adverse weather or lighting conditions?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle segmentation with limited GPU memory or computational resources?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What approaches work best for segmenting objects with fuzzy or ambiguous boundaries?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you design robust training procedures for noisy segmentation datasets?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What techniques help with explaining segmentation decisions to domain experts?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you implement fairness-aware segmentation to reduce bias across different groups?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What approaches work best for segmenting objects in synthetic or artificially generated scenes?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you handle segmentation quality assessment in the absence of ground truth masks?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What techniques help with segmenting objects that have significant appearance variations?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you implement efficient inference pipelines for large-scale segmentation applications?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches work best for segmenting objects with hierarchical part-whole relationships?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you handle segmentation optimization when balancing mask quality and detection accuracy?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What techniques help with integrating instance segmentation into larger computer vision pipelines?**
**Answer:** _To be filled_

---


---

# --- Semantic Segmentation Questions (from 07_computer_vision/04_semantic_segmentation) ---

# Semantic Segmentation (U-Net, DeepLab) - Theory Questions

## Question 1
- [ ] Done
**How does U-Net's encoder-decoder architecture with skip connections improve segmentation accuracy?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the key innovations in DeepLabv3+ that enhance boundary delineation in segmentation?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement and optimize atrous convolutions for multi-scale feature extraction?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What strategies work best for handling class imbalance in semantic segmentation datasets?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you design loss functions that emphasize boundary accuracy in segmentation tasks?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What techniques help improve segmentation performance on small or thin objects?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement data augmentation strategies that preserve spatial relationships?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What approaches work best for handling multi-class segmentation with hierarchical categories?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you optimize U-Net architectures for medical image segmentation applications?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What techniques help with segmenting objects under varying lighting or contrast conditions?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you implement domain adaptation for segmentation models across different imaging modalities?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies work best for handling noisy or inconsistent segmentation annotations?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you design evaluation metrics that properly assess segmentation quality?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What approaches help with segmenting scenes with significant depth variations?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you implement active learning for efficient segmentation annotation?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What techniques work best for real-time semantic segmentation applications?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you handle segmentation of objects with fuzzy or ambiguous boundaries?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What strategies help with segmenting rare classes in highly imbalanced datasets?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you implement uncertainty quantification in segmentation predictions?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What approaches work best for handling segmentation across different image resolutions?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you design architectures that efficiently process high-resolution images?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What techniques help with segmenting objects that undergo significant deformation?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle temporal consistency in video semantic segmentation?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies work best for segmenting objects in specialized domains like satellite imagery?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement knowledge distillation for compressing segmentation models?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches help with segmenting objects with significant appearance variations?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you handle segmentation in scenarios with partial occlusion or overlapping objects?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What techniques work best for few-shot segmentation in novel semantic categories?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you implement online learning for segmentation models adapting to new environments?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies help with segmenting objects across different camera viewpoints?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you design robust training procedures for weakly supervised segmentation?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What approaches work best for segmenting objects in adverse weather conditions?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle segmentation with limited computational resources or memory?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies work best for segmenting objects with complex internal structures?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle segmentation quality assessment without perfect ground truth?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches help with segmenting objects that have contextual dependencies?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement efficient inference for large-scale segmentation applications?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques work best for segmenting objects with significant scale variations?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you design architectures that handle both coarse and fine-grained segmentation?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies help with segmenting objects in synthetic or artificially generated images?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you handle segmentation optimization when balancing accuracy and efficiency?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches work best for segmenting objects with temporal appearance changes?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you implement multi-task learning that combines segmentation with other vision tasks?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help with segmenting objects across different imaging sensors or modalities?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you handle segmentation in federated learning scenarios with distributed data?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies work best for segmenting objects with inter-annotator disagreement?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you design evaluation protocols that reflect real-world deployment scenarios?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches help with integrating semantic segmentation into robotics or autonomous systems?**
**Answer:** _To be filled_

---


---

# --- Image Super-Resolution Questions (from 07_computer_vision/05_image_super_resolution) ---

# Image Super-Resolution - Theory Questions

## Question 1
- [ ] Done
**How do you choose between single-image and multi-frame super-resolution approaches for different applications?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the trade-offs between PSNR optimization and perceptual quality in super-resolution models?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement and evaluate generative adversarial networks for photo-realistic super-resolution?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What techniques help with preserving fine details and textures during upscaling processes?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you handle super-resolution for images with different degradation types (blur, noise, compression)?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What strategies work best for real-time super-resolution in video streaming applications?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement attention mechanisms to focus on important image regions during upscaling?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What approaches help with handling diverse content types (text, faces, natural scenes) in super-resolution?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you design loss functions that balance fidelity and perceptual quality?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What techniques work best for super-resolution of images with repetitive patterns or textures?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you implement domain-specific super-resolution for specialized applications like medical imaging?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies help with handling super-resolution across different upscaling factors?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you evaluate super-resolution quality when ground truth high-resolution images aren't available?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What approaches work best for super-resolution of images with motion blur or camera shake?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you implement efficient architectures for mobile or edge device deployment?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What techniques help with preserving semantic content during aggressive upscaling?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you handle super-resolution for images with mixed resolution regions?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What strategies work best for batch processing large collections of images for super-resolution?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you implement uncertainty quantification to assess super-resolution confidence?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What approaches help with handling super-resolution of compressed or artifact-laden images?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you design training procedures that generalize well to unseen degradation types?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What techniques work best for super-resolution of images with complex lighting conditions?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle super-resolution optimization for specific downstream tasks?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies help with preserving important visual features during upscaling?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement progressive super-resolution for extremely high upscaling factors?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches work best for super-resolution of images with geometric distortions?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you handle super-resolution in scenarios with limited computational resources?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What techniques help with maintaining temporal consistency in video super-resolution?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you design evaluation metrics that align with human perceptual preferences?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies work best for super-resolution of images from different camera sensors?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement knowledge distillation for compressing super-resolution models?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What approaches help with handling super-resolution of synthetic or artificially generated images?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle super-resolution quality control and automatic failure detection?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What techniques work best for super-resolution of images with significant noise levels?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement online learning for super-resolution models adapting to new content types?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies help with super-resolution of images captured under extreme conditions?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you design architectures that handle both natural and artistic image content?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches work best for super-resolution with privacy-preserving requirements?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you handle super-resolution optimization when training and deployment hardware differ?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques help with explaining super-resolution decisions to end users?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you implement fairness-aware super-resolution to avoid bias across different image types?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies work best for super-resolution of historical or archival images?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you handle super-resolution in federated learning scenarios with distributed data?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches help with combining super-resolution with other image enhancement tasks?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you implement efficient batch processing pipelines for large-scale super-resolution?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques work best for super-resolution of images with cultural or artistic significance?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you handle super-resolution quality assessment in production environments?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies help with adapting super-resolution models to emerging image formats?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you design robust training procedures for diverse and noisy training datasets?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches work best for integrating super-resolution into broader image processing workflows?**
**Answer:** _To be filled_

---


---

# --- OCR Questions (from 07_computer_vision/06_ocr) ---

# Optical Character Recognition (OCR) - Theory Questions

## Question 1
- [ ] Done
**How do modern transformer-based OCR models like TrOCR improve upon traditional CNN-RNN approaches?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the key challenges in implementing OCR for handwritten text versus printed text?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you handle OCR for documents with complex layouts, tables, and mixed content types?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What techniques work best for OCR in multilingual documents with different scripts and writing directions?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you implement preprocessing steps to improve OCR accuracy on low-quality or degraded images?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What strategies help with OCR performance on documents with varying fonts, sizes, and styles?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you design OCR systems that handle both structured forms and unstructured documents?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What approaches work best for real-time OCR in mobile applications with computational constraints?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement post-processing techniques to correct OCR errors using language models?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What techniques help with OCR accuracy for documents captured under poor lighting conditions?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you handle OCR for documents with watermarks, stamps, or overlapping text?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies work best for OCR on historical documents with faded or damaged text?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement OCR systems that maintain document formatting and layout information?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What approaches help with OCR accuracy for specialized domains like legal or medical documents?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you design evaluation metrics that properly assess OCR quality for different applications?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What techniques work best for OCR on documents with mixed languages within the same line?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you handle OCR confidence scoring and uncertainty quantification?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What strategies help with OCR performance on documents with complex mathematical notation?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you implement active learning for improving OCR models with minimal annotation effort?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What approaches work best for OCR on documents with non-standard or artistic fonts?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you handle OCR for documents with varying text orientations and skew angles?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What techniques help with OCR accuracy on documents with background patterns or textures?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you implement knowledge distillation for compressing large OCR models?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies work best for OCR on documents captured with different camera angles?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you handle OCR quality control and automatic error detection in production systems?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches help with OCR for documents with security features like microtext?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you implement OCR systems that preserve document authenticity and prevent tampering?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What techniques work best for OCR on documents with varying resolution and image quality?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle OCR for documents with mixed content (text, images, graphics)?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies help with OCR performance on documents with fade, stains, or physical damage?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement domain adaptation for OCR models across different document types?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What approaches work best for OCR on documents requiring high accuracy for compliance purposes?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle OCR in scenarios with privacy constraints and sensitive information?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What techniques help with OCR accuracy for documents with unconventional layouts?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement online learning for OCR models adapting to new document formats?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies work best for OCR on documents with time-sensitive processing requirements?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle OCR optimization when balancing accuracy and processing speed?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches help with OCR for documents in specialized industries like banking or insurance?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement efficient batch processing pipelines for large-scale OCR applications?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques work best for OCR on documents with multi-column layouts?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle OCR quality assessment when ground truth transcriptions aren't available?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies help with OCR for documents captured using different imaging technologies?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you implement fairness-aware OCR to avoid bias across different languages or scripts?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches work best for OCR integration with document management systems?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you handle OCR for documents with legal or regulatory requirements for accuracy?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help with explaining OCR decisions and building user trust?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement robust error handling for OCR systems in production environments?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies work best for OCR on documents with varying paper types and textures?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle OCR adaptation to emerging document formats and standards?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches help with combining OCR with other document analysis tasks for comprehensive processing?**
**Answer:** _To be filled_

---


---

# --- 3D Reconstruction Questions (from 07_computer_vision/07_3d_reconstruction) ---

# 3D Reconstruction (NeRF, Gaussian Splatting) - Theory Questions

## Question 1
- [ ] Done
**How does 3D Gaussian Splatting achieve real-time rendering compared to NeRF's neural network approach?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the key advantages of Gaussian Splatting's rasterization method over NeRF's volumetric rendering?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement and optimize the view-dependent color representation using spherical harmonics?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What techniques help with handling dynamic scenes and moving objects in NeRF reconstructions?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you design training procedures for NeRF that generalize well to novel viewpoints?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What strategies work best for handling sparse or unevenly distributed camera viewpoints in 3D reconstruction?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement multi-resolution training for NeRF to capture both coarse and fine details?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What approaches help with reducing training time while maintaining reconstruction quality?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you handle 3D reconstruction for scenes with varying lighting conditions?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What techniques work best for reconstructing scenes with reflective or transparent materials?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you implement uncertainty quantification in neural radiance field predictions?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies help with handling large-scale scenes that exceed memory limitations?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you design loss functions that balance photometric consistency and geometric accuracy?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What approaches work best for reconstructing scenes with limited texture or repetitive patterns?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle 3D reconstruction quality assessment when ground truth geometry isn't available?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What techniques help with preserving fine details during neural scene representation?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you implement efficient rendering pipelines for real-time 3D scene visualization?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What strategies work best for handling occlusion and visibility in complex 3D scenes?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle 3D reconstruction for scenes captured under different weather conditions?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What approaches help with combining multiple reconstruction techniques for improved results?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement progressive training strategies for complex 3D scenes?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What techniques work best for reconstructing scenes with significant depth variations?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle 3D reconstruction optimization for specific downstream applications?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies help with preserving semantic information during neural scene representation?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement knowledge distillation for compressing 3D reconstruction models?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches work best for reconstructing scenes with moving objects or people?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you handle 3D reconstruction in scenarios with limited computational resources?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What techniques help with explaining reconstruction quality and confidence to end users?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you implement domain adaptation for 3D reconstruction across different environments?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies work best for reconstructing historical or cultural heritage scenes?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you handle 3D reconstruction with privacy constraints for sensitive locations?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What approaches help with maintaining temporal consistency in dynamic scene reconstruction?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you implement efficient data collection strategies for optimal 3D reconstruction?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What techniques work best for reconstructing scenes with extreme lighting conditions?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle 3D reconstruction quality control in production environments?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies help with combining 3D reconstruction with other computer vision tasks?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement online learning for reconstruction models adapting to new scenes?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches work best for reconstructing scenes captured with different camera systems?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you handle 3D reconstruction for scenes requiring high geometric accuracy?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques help with preserving important visual features during scene compression?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you implement fairness-aware reconstruction to avoid bias across different scene types?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies work best for reconstructing scenes for virtual or augmented reality applications?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you handle 3D reconstruction optimization when balancing quality and rendering speed?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches help with reconstructing scenes with complex material properties?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you implement multi-view consistency constraints in neural reconstruction?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques work best for reconstructing scenes with indoor and outdoor mixed environments?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you handle 3D reconstruction in federated learning scenarios with distributed capture?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies help with reconstructing scenes captured across different time periods?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you design evaluation protocols that reflect real-world reconstruction requirements?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches work best for integrating 3D reconstruction into robotics or autonomous navigation systems?**
**Answer:** _To be filled_

---


---

# --- Video Tracking Questions (from 07_computer_vision/08_video_tracking) ---

# Video Tracking - Theory Questions

## Question 1
- [ ] Done
**How do you implement multi-object tracking that maintains identity consistency across long sequences?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the trade-offs between detection-based and correlation-based tracking approaches?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you handle tracking objects through occlusions and temporary disappearances?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What techniques work best for tracking objects with significant appearance changes over time?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you implement real-time tracking systems with computational efficiency constraints?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What strategies help with tracking objects in crowded scenes with frequent interactions?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you design tracking systems that handle camera motion and viewpoint changes?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What approaches work best for tracking objects across different scales and resolutions?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement uncertainty quantification in tracking predictions and associations?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What techniques help with handling tracking failures and automatic recovery mechanisms?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you handle multi-camera tracking with non-overlapping fields of view?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies work best for tracking objects with deformable shapes or articulated motion?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement online learning for tracking models adapting to new object appearances?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What approaches help with tracking objects in scenarios with similar-looking distractors?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you design evaluation metrics that properly assess tracking performance?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What techniques work best for tracking objects with intermittent visibility?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you handle tracking optimization when balancing accuracy and computational speed?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What strategies help with tracking objects across different lighting and weather conditions?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you implement knowledge distillation for compressing tracking models?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What approaches work best for tracking objects with non-rigid motion patterns?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you handle tracking quality assessment and confidence scoring?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What techniques help with explaining tracking decisions and predicted trajectories?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you implement active learning for improving tracking models with minimal annotation?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies work best for tracking objects in specialized domains like sports or surveillance?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you handle tracking in scenarios with limited computational resources?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches help with maintaining tracking consistency across video cuts or scene changes?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you implement fairness-aware tracking to avoid bias across different object types?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What techniques work best for tracking objects with varying motion dynamics?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle tracking for objects that split, merge, or change in number?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies help with tracking objects across different camera perspectives?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement efficient data association algorithms for multi-object scenarios?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What approaches work best for tracking objects with partial visibility or truncation?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle tracking in videos with varying frame rates or temporal resolution?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What techniques help with tracking objects that undergo significant pose changes?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement robust initialization procedures for tracking in challenging scenarios?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies work best for tracking objects in adverse weather or environmental conditions?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle tracking optimization for specific downstream applications?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches help with combining tracking with other video analysis tasks?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement online adaptation for tracking models in changing environments?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques work best for tracking objects with complex interaction patterns?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle tracking quality control in production video processing systems?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies help with tracking objects across different video formats and codecs?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you implement efficient batch processing for large-scale video tracking applications?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches work best for tracking objects with temporal appearance patterns?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you handle tracking in federated learning scenarios with distributed video data?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help with preserving privacy while maintaining tracking accuracy?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement robust error handling for tracking systems in real-world deployments?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies work best for tracking objects in synthetic or artificially generated videos?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle tracking adaptation to emerging video technologies and formats?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches help with integrating tracking into broader video understanding pipelines?**
**Answer:** _To be filled_

---


---

# --- Style Transfer Questions (from 07_computer_vision/09_style_transfer) ---

# Style Transfer - Theory Questions

## Question 1
- [ ] Done
**How do you balance content preservation and style adoption in neural style transfer models?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the key differences between optimization-based and feed-forward style transfer approaches?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement real-time style transfer for video processing applications?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What techniques help with handling multiple styles simultaneously in a single model?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you design loss functions that capture both perceptual and artistic quality?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What strategies work best for style transfer with limited computational resources?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you handle style transfer for images with different content types (portraits, landscapes, objects)?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What approaches help with preserving important semantic content during style transformation?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement user control mechanisms for adjusting style transfer intensity?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What techniques work best for handling style transfer across different image resolutions?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you design evaluation metrics that assess both technical and artistic quality?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies help with style transfer for specialized domains like medical or satellite imagery?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you handle style transfer quality control and automatic failure detection?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What approaches work best for adapting style transfer to new artistic styles with minimal examples?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you implement knowledge distillation for compressing style transfer models?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What techniques help with maintaining temporal consistency in video style transfer?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you handle style transfer for images with complex compositions or multiple objects?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What strategies work best for style transfer that preserves facial features and identity?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you implement uncertainty quantification in style transfer predictions?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What approaches help with explaining style transfer decisions to users?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you handle style transfer optimization when balancing quality and processing speed?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What techniques work best for style transfer with privacy-preserving requirements?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you implement online learning for style transfer models adapting to new styles?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies help with style transfer for images captured under different conditions?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you handle style transfer in scenarios with copyright or intellectual property concerns?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches work best for combining multiple artistic techniques in single transformations?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you implement efficient batch processing pipelines for large-scale style transfer?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What techniques help with style transfer that maintains image metadata and EXIF information?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle style transfer quality assessment when ground truth isn't available?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies work best for style transfer in interactive or real-time applications?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement fairness-aware style transfer to avoid bias across different image types?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What approaches help with style transfer for images with cultural or historical significance?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle style transfer adaptation to emerging artistic movements or techniques?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What techniques work best for style transfer with specific color palette constraints?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement robust error handling for style transfer in production environments?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies help with combining style transfer with other image enhancement tasks?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle style transfer for images with varying lighting and exposure conditions?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches work best for style transfer that preserves important visual details?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement domain adaptation for style transfer across different image domains?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques help with style transfer quality consistency across different input types?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle style transfer in federated learning scenarios with distributed style data?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies work best for style transfer with memory-efficient architectures?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you implement progressive style transfer for extremely detailed transformations?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches help with integrating style transfer into broader creative workflows?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you handle style transfer optimization for specific artistic or commercial requirements?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques work best for style transfer that adapts to user preferences over time?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement robust training procedures for diverse and challenging style datasets?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies help with style transfer for emerging image formats and technologies?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle style transfer quality benchmarking across different model architectures?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches work best for combining traditional and neural approaches in style transfer systems?**
**Answer:** _To be filled_

---


---

# --- Facial Recognition Questions (from 07_computer_vision/10_facial_recognition) ---

# Facial Recognition - Theory Questions

## Question 1
- [ ] Done
**How do you implement face recognition systems that work reliably across different ethnicities and demographics?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the key privacy considerations when deploying facial recognition in public spaces?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you handle face recognition for individuals wearing masks, glasses, or other accessories?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What techniques help with face recognition under varying lighting conditions and poses?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you design face recognition systems that are robust to aging and appearance changes?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What strategies work best for liveness detection to prevent spoofing attacks?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement face recognition that maintains accuracy across different camera qualities?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What approaches help with handling face recognition in crowded or cluttered environments?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you design evaluation protocols that assess fairness across different demographic groups?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What techniques work best for face recognition with limited enrollment samples per person?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you handle face recognition optimization for real-time applications?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies help with face recognition across different scales and resolutions?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement uncertainty quantification in face recognition predictions?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What approaches work best for face recognition in challenging environmental conditions?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle face recognition quality control and confidence scoring?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What techniques help with explaining face recognition decisions for transparency?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you implement active learning for improving face recognition with minimal annotation?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What strategies work best for face recognition in specialized applications like access control?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle face recognition with privacy-preserving techniques like federated learning?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What approaches help with face recognition across different facial expressions and emotions?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement knowledge distillation for compressing face recognition models?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What techniques work best for face recognition with temporal consistency in video streams?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle face recognition for individuals with facial hair or makeup changes?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies help with face recognition across different camera angles and viewpoints?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement robust face detection as a preprocessing step for recognition?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches work best for face recognition in low-light or infrared imaging?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you handle face recognition quality assessment and performance monitoring?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What techniques help with face recognition that adapts to new individuals over time?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you implement fairness-aware training to reduce recognition bias?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies work best for face recognition with computational efficiency constraints?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you handle face recognition in scenarios with multiple faces per image?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What approaches help with face recognition across different cultural or stylistic contexts?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you implement secure storage and processing of facial recognition data?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What techniques work best for face recognition with occlusion or partial visibility?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle face recognition adaptation to emerging imaging technologies?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What strategies help with combining face recognition with other biometric modalities?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement robust error handling for face recognition in production systems?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What approaches work best for face recognition with regulatory compliance requirements?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you handle face recognition optimization for specific deployment scenarios?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques help with face recognition that preserves user privacy and anonymity?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you implement online learning for face recognition systems in dynamic environments?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What strategies work best for face recognition in forensic or investigative applications?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you handle face recognition quality benchmarking across different algorithms?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What approaches help with integrating face recognition into broader security systems?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you implement robust training procedures for diverse facial recognition datasets?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques work best for face recognition with emerging privacy regulations?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you handle face recognition adaptation to new demographic groups or populations?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What strategies help with face recognition in challenging deployment environments?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you design evaluation protocols that reflect real-world recognition scenarios?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What approaches work best for combining traditional and deep learning methods in face recognition?**
**Answer:** _To be filled_

---

