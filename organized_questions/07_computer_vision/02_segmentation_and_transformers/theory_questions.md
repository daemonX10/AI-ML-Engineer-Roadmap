# Segmentation & Vision Transformers - Interview Questions

## Semantic Segmentation

### Question 1
- [ ] Done
**Explain U-Net's encoder-decoder architecture with skip connections. Why are skip connections crucial for segmentation?**

**Answer:**

U-Net uses an encoder (contracting path) to capture context and a decoder (expanding path) to enable precise localization. Skip connections directly connect encoder layers to decoder layers at the same resolution, preserving fine-grained spatial information that would otherwise be lost during downsampling.

**Architecture:**

```
Encoder (Contracting)         Decoder (Expanding)
    [Input 572×572]
         ↓ Conv
    [64×568×568] ─────────→ [Concat] → [64×388×388]
         ↓ Pool                ↑
    [128×280×280] ────────→ [Concat] → [128×196×196]
         ↓ Pool                ↑
    [256×136×136] ────────→ [Concat] → [256×100×100]
         ↓ Pool                ↑
    [512×64×64] ──────────→ [Concat] → [512×52×52]
         ↓ Pool                ↑
    [1024×28×28] ─────────────→ UpConv
         (Bottleneck)
```

**Why Skip Connections Matter:**

| Without Skip Connections | With Skip Connections |
|-------------------------|----------------------|
| Only semantic info from bottleneck | Semantic + spatial info |
| Blurry boundaries | Sharp boundaries |
| Poor small object recovery | Better fine details |
| Localization errors | Precise localization |

**Key Insight:**
- Encoder: "What" (high-level features, semantic meaning)
- Decoder: "Where" (precise localization)
- Skip connections: Combine both for accurate segmentation

**Python Implementation:**
```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._double_conv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.decoder.append(self._double_conv(feature * 2, feature))
        
        # Final output
        self.final = nn.Conv2d(features[0], out_channels, 1)
    
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Decoder with skip connections
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)  # Skip connection
            x = self.decoder[idx + 1](x)  # Conv
        
        return self.final(x)
```

**Interview Tip:** Skip connections solve the information bottleneck problem—high-level features alone can't recover precise boundaries. The concatenation (not addition) doubles channels, giving the decoder rich information from multiple scales.

---

### Question 2
- [ ] Done
**What are the key innovations in DeepLabv3+ (atrous convolutions, ASPP, encoder-decoder) for boundary delineation?**

**Answer:**

DeepLabv3+ combines three key innovations: atrous (dilated) convolutions for multi-scale context without resolution loss, ASPP (Atrous Spatial Pyramid Pooling) for capturing objects at multiple scales, and an encoder-decoder structure for sharp boundary recovery.

**Key Innovations:**

| Component | Purpose | Benefit |
|-----------|---------|---------|
| Atrous Convolution | Expand receptive field | No resolution loss |
| ASPP | Multi-scale feature extraction | Handle varying object sizes |
| Encoder-Decoder | Low-level feature fusion | Sharp boundaries |

**Atrous Convolution:**

Standard convolution vs. atrous with rate r:
$$y[i] = \sum_k x[i + r \cdot k] \cdot w[k]$$

- Rate r=1: Standard convolution
- Rate r=2: Skips every other pixel, 2× receptive field
- Rate r=4: 4× receptive field, same parameters

**ASPP Module:**

Parallel atrous convolutions at different rates:
```
        Input Features
       /    |    |    \
   1×1   3×3   3×3   3×3   Image
   Conv  r=6  r=12  r=18   Pooling
       \    |    |    /
         Concatenate
             ↓
          1×1 Conv
             ↓
        ASPP Output
```

**DeepLabv3+ Architecture:**
```
Image → Backbone → ASPP → Decoder → Segmentation
              ↓              ↑
         Low-level     ← 1×1 Conv (reduce channels)
         Features
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        
        # 1×1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Atrous convolutions at different rates
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, 
                         dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
        # Image-level features (global context)
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Fusion
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Multi-scale features
        features = [self.conv1(x)]
        for atrous_conv in self.atrous_convs:
            features.append(atrous_conv(x))
        
        # Global features
        global_feat = self.image_pooling(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', 
                                     align_corners=False)
        features.append(global_feat)
        
        # Concatenate and project
        return self.project(torch.cat(features, dim=1))

class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ with encoder-decoder"""
    def __init__(self, backbone, num_classes, low_level_channels=256):
        super().__init__()
        self.backbone = backbone
        self.aspp = ASPP(2048, 256)
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Backbone features
        low_level, high_level = self.backbone(x)  # C2 and C5
        
        # ASPP on high-level features
        aspp_out = self.aspp(high_level)
        aspp_out = F.interpolate(aspp_out, size=low_level.shape[2:], 
                                  mode='bilinear', align_corners=False)
        
        # Process low-level features
        low_level = self.low_level_conv(low_level)
        
        # Concatenate and decode
        x = torch.cat([aspp_out, low_level], dim=1)
        x = self.decoder(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        return x
```

**Comparison with Earlier Versions:**

| Version | Key Feature | Limitation Fixed |
|---------|-------------|-----------------|
| DeepLabv1 | Atrous convolution + CRF | Resolution loss |
| DeepLabv2 | ASPP | Multi-scale objects |
| DeepLabv3 | Improved ASPP + batch norm | Training stability |
| DeepLabv3+ | Encoder-decoder | Boundary sharpness |

**Interview Tip:** DeepLabv3+'s key insight is combining dense prediction (ASPP) with sharp boundaries (decoder). The low-level features provide edge information while ASPP provides semantic context. This is why it excels at boundary delineation.

---

### Question 3
- [ ] Done
**Compare Dice loss vs. Cross-entropy for segmentation. When would you use focal loss or Lovász loss?**

**Answer:**

Cross-entropy treats pixels independently; Dice loss optimizes global overlap. Use Dice for imbalanced classes, focal loss for hard examples, and Lovász loss for direct IoU optimization. Often combine multiple losses for best results.

**Loss Function Comparison:**

| Loss | Optimizes | Strengths | Weaknesses |
|------|-----------|-----------|------------|
| Cross-Entropy | Per-pixel accuracy | Stable, well-understood | Ignores class imbalance |
| Dice | Overlap (F1 score) | Handles imbalance | Unstable for small objects |
| Focal | Hard examples | Reduces easy example weight | Hyperparameter sensitive |
| Lovász | IoU directly | Matches eval metric | Complex, slower |

**Mathematical Formulations:**

**Cross-Entropy:**
$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{p}_{i,c})$$

**Dice Loss:**
$$\mathcal{L}_{Dice} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

**Focal Loss:**
$$\mathcal{L}_{Focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

**Lovász-Softmax:**
Directly optimizes IoU using Lovász extension (submodular set function optimization).

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / \
               (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation"""
    def __init__(self, num_classes, smooth=1.0, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = weight  # Class weights
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2)
        
        dice_per_class = []
        for c in range(self.num_classes):
            p = pred[:, c].flatten()
            g = target_one_hot[:, c].float().flatten()
            
            intersection = (p * g).sum()
            dice = (2. * intersection + self.smooth) / \
                   (p.sum() + g.sum() + self.smooth)
            dice_per_class.append(dice)
        
        if self.weight is not None:
            dice_per_class = [d * w for d, w in zip(dice_per_class, self.weight)]
        
        return 1 - torch.stack(dice_per_class).mean()

class FocalLoss(nn.Module):
    """Focal Loss for hard example mining"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

class LovaszSoftmax(nn.Module):
    """Lovász-Softmax loss for IoU optimization"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        C = pred.shape[1]
        
        losses = []
        for c in range(C):
            fg = (target == c).float()
            errors = (fg - pred[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg.flatten()[perm]
            
            # Lovász extension
            gts = fg_sorted.sum()
            intersection = gts - fg_sorted.cumsum(0)
            union = gts + (1 - fg_sorted).cumsum(0)
            jaccard = 1 - intersection / union
            
            # Gradient of Jaccard
            jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
            losses.append((errors_sorted * jaccard).sum())
        
        return torch.stack(losses).mean()

class CombinedLoss(nn.Module):
    """Combine multiple losses"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5, focal_weight=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(num_classes=21)
        self.focal = FocalLoss()
        
        self.weights = {
            'ce': ce_weight,
            'dice': dice_weight,
            'focal': focal_weight
        }
    
    def forward(self, pred, target):
        loss = 0
        if self.weights['ce'] > 0:
            loss += self.weights['ce'] * self.ce(pred, target)
        if self.weights['dice'] > 0:
            loss += self.weights['dice'] * self.dice(pred, target)
        if self.weights['focal'] > 0:
            loss += self.weights['focal'] * self.focal(pred, target)
        return loss
```

**When to Use Each:**

| Scenario | Recommended Loss |
|----------|------------------|
| Balanced classes | Cross-Entropy |
| Imbalanced (medical) | Dice or Dice + CE |
| Many small objects | Focal Loss |
| Maximize mIoU | Lovász-Softmax |
| General purpose | CE + Dice (0.5 each) |

**Interview Tip:** Dice loss directly optimizes overlap but can be unstable when target is empty. Cross-entropy is stable but ignores imbalance. Best practice: combine CE + Dice for stability and imbalance handling. Use Lovász when IoU is the evaluation metric.

---

### Question 4
- [ ] Done
**How do atrous/dilated convolutions capture multi-scale context without losing resolution?**

**Answer:**

Atrous convolutions insert gaps (zeros) between kernel weights, expanding the receptive field without increasing parameters or reducing spatial resolution. This allows capturing context at multiple scales while maintaining dense predictions.

**Standard vs. Atrous Convolution:**

| Aspect | Standard 3×3 | Atrous 3×3 (rate=2) |
|--------|-------------|---------------------|
| Receptive field | 3×3 | 5×5 |
| Parameters | 9 | 9 |
| Output resolution | Same (with padding) | Same |
| Context captured | Local | Larger |

**How It Works:**

Standard 3×3 kernel:
```
[x x x]     Receptive field: 3×3
[x x x]     Spacing: 1
[x x x]
```

Atrous 3×3 with rate=2:
```
[x . x . x]     Receptive field: 5×5
[. . . . .]     Spacing: 2
[x . x . x]     (dots are skipped pixels)
[. . . . .]
[x . x . x]
```

**Mathematical Definition:**
$$y[i] = \sum_{k=1}^{K} x[i + r \cdot k] \cdot w[k]$$

Where:
- $r$ = dilation rate
- $K$ = kernel size
- Effective kernel size = $K + (K-1) \times (r-1)$

**Multi-Scale Context:**
```
        Input Feature Map
       /        |        \
  Atrous     Atrous     Atrous
  rate=1     rate=2     rate=4
  (3×3)      (5×5)      (9×9)
       \        |        /
         Concatenate
              ↓
       Multi-scale Features
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtrousConv(nn.Module):
    """Single atrous convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MultiScaleAtrous(nn.Module):
    """Multi-scale context using atrous convolutions"""
    def __init__(self, in_channels, out_channels, rates=[1, 2, 4, 8]):
        super().__init__()
        self.branches = nn.ModuleList([
            AtrousConv(in_channels, out_channels, dilation=r) for r in rates
        ])
        self.fusion = nn.Conv2d(out_channels * len(rates), out_channels, 1)
    
    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(features, dim=1))

# Visualization of receptive fields
def visualize_receptive_field(kernel_size=3, dilation=1):
    """Show effective receptive field"""
    effective_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    
    grid = torch.zeros(effective_size, effective_size)
    center = effective_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            pos_i = i * dilation
            pos_j = j * dilation
            grid[pos_i, pos_j] = 1
    
    return grid

# Example: Compare receptive fields
for rate in [1, 2, 4]:
    rf = visualize_receptive_field(3, rate)
    print(f"Rate {rate}: {rf.shape[0]}×{rf.shape[0]} effective receptive field")
```

**Resolution Preservation:**
```python
# Standard approach: pooling reduces resolution
standard = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1),  # H×W
    nn.MaxPool2d(2),                   # H/2×W/2  ← Resolution loss!
    nn.Conv2d(64, 64, 3, padding=1),
    nn.MaxPool2d(2),                   # H/4×W/4
)

# Atrous approach: maintains resolution
atrous = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1, dilation=1),  # H×W
    nn.Conv2d(64, 64, 3, padding=2, dilation=2),  # H×W  ← Same resolution!
    nn.Conv2d(64, 64, 3, padding=4, dilation=4),  # H×W
)
```

**Trade-offs:**

| Advantage | Disadvantage |
|-----------|--------------|
| No resolution loss | Gridding artifacts at high rates |
| Same parameters | Memory for full resolution |
| Multi-scale context | Less effective for small objects |

**Gridding Problem Solution:**
Use hybrid rates (e.g., 6, 12, 18) rather than powers of 2 to avoid systematic gaps.

**Interview Tip:** Atrous convolutions are key to dense prediction tasks. They expand receptive field without downsampling, but beware of gridding artifacts. ASPP uses different rates to capture multi-scale context efficiently.

---

### Question 5
- [ ] Done
**Explain U-Net variants: U-Net++, Attention U-Net, and TransUNet. What problems does each solve?**

**Answer:**

U-Net variants address specific limitations: U-Net++ uses nested skip connections for better feature fusion, Attention U-Net adds attention gates to focus on relevant regions, and TransUNet incorporates transformers for global context. Each improves upon the original U-Net architecture.

**Comparison:**

| Variant | Key Innovation | Problem Solved |
|---------|----------------|----------------|
| U-Net | Skip connections | Basic encoder-decoder |
| U-Net++ | Nested/dense skip paths | Semantic gap in skips |
| Attention U-Net | Attention gates | Irrelevant feature suppression |
| TransUNet | Transformer encoder | Limited global context |

**U-Net++ (Nested U-Net):**

Problem: Direct skip connections have semantic gap between encoder and decoder.

Solution: Dense convolution blocks between encoder and decoder at each level.

```
Encoder                    Decoder
  X0,0 ─────────────────→ X0,4
   ↓      X0,1 ──→ X0,2 ──→ X0,3    ↑
  X1,0 ────→ X1,1 ──→ X1,2 ──→ X1,3 ↑
   ↓          X1,1 ──→ X1,2    ↑
  X2,0 ────────→ X2,1 ──→ X2,2 ↑
   ↓              ↑            ↑
  X3,0 ──────────→ X3,1 ───────↑
   ↓                           ↑
  X4,0 ────────────────────────↑
```

**Attention U-Net:**

Problem: Skip connections pass all features, including irrelevant ones.

Solution: Attention gates learn to suppress irrelevant regions.

Attention gate formula:
$$\alpha = \sigma(\psi^T(\sigma_1(W_x x + W_g g + b)))$$

Where:
- $x$ = skip features
- $g$ = gating signal from decoder
- $\alpha$ = attention coefficients

**TransUNet:**

Problem: CNNs have limited receptive field for global context.

Solution: Transformer encoder processes flattened CNN features.

```
Image → CNN Encoder → Flatten → Transformer → Reshape → CNN Decoder
                                   ↓
                            Global Context
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Net++ Block
class NestedBlock(nn.Module):
    """Nested skip connection block for U-Net++"""
    def __init__(self, in_ch, out_ch, num_concat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * num_concat, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.conv(x)

# Attention Gate
class AttentionGate(nn.Module):
    """Attention gate for Attention U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection features
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear')
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi  # Weighted skip features

class AttentionUNet(nn.Module):
    """U-Net with attention gates"""
    def __init__(self, in_ch=3, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        # Encoder
        for f in features:
            self.encoder.append(self._conv_block(in_ch, f))
            in_ch = f
        
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Decoder with attention
        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.attention.append(AttentionGate(f, f, f // 2))
            self.decoder.append(self._conv_block(f * 2, f))
        
        self.final = nn.Conv2d(features[0], out_ch, 1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skips = []
        
        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skips = skips[::-1]
        
        # Decoder with attention
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Upsample
            skip = self.attention[i // 2](x, skips[i // 2])  # Attention
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i + 1](x)  # Conv
        
        return self.final(x)

# TransUNet simplified
class TransUNet(nn.Module):
    """U-Net with Transformer encoder"""
    def __init__(self, img_size=224, patch_size=16, in_ch=3, num_classes=1):
        super().__init__()
        self.cnn_encoder = nn.Sequential(
            # ResNet-like encoder
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        # Transformer encoder
        self.patch_embed = nn.Conv2d(64, 768, patch_size, stride=patch_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=12
        )
        
        # Decoder
        self.decoder = UNetDecoder(768, num_classes)
    
    def forward(self, x):
        # CNN features
        cnn_feat = self.cnn_encoder(x)
        
        # Patch embedding
        patches = self.patch_embed(cnn_feat)
        B, C, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        
        # Transformer
        transformer_out = self.transformer(patches)
        
        # Reshape and decode
        transformer_out = transformer_out.transpose(1, 2).view(B, C, H, W)
        return self.decoder(transformer_out)
```

**When to Use Each:**

| Scenario | Best Choice |
|----------|-------------|
| Medical imaging (standard) | U-Net++ |
| Noisy backgrounds | Attention U-Net |
| Need global context | TransUNet |
| Resource constrained | Original U-Net |

**Interview Tip:** U-Net++ densely connects all levels (like DenseNet for skip connections). Attention U-Net adds soft attention to filter skip features. TransUNet brings transformer's global context to segmentation. Choose based on your specific needs.

---

### Question 6
- [ ] Done
**How do you handle class imbalance in segmentation when some classes occupy very few pixels?**

**Answer:**

Class imbalance in segmentation is addressed through: weighted loss functions (inverse frequency weighting), specialized losses (Dice, focal), oversampling of rare class images, class-balanced batching, and data augmentation targeting minority classes.

**Common Strategies:**

| Strategy | Level | Approach |
|----------|-------|----------|
| Loss reweighting | Loss | Higher weight for rare classes |
| Dice/Focal loss | Loss | Directly handles imbalance |
| Oversampling | Data | More samples with rare classes |
| Class-balanced batching | Training | Equal class representation |
| Copy-paste augmentation | Data | Add rare class instances |

**Weight Calculation Methods:**

| Method | Formula | When to Use |
|--------|---------|-------------|
| Inverse frequency | $w_c = \frac{1}{f_c}$ | Standard |
| Median frequency | $w_c = \frac{median(f)}{f_c}$ | Very imbalanced |
| Effective number | $w_c = \frac{1-\beta}{1-\beta^{n_c}}$ | Extreme imbalance |
| Log-smoothed | $w_c = \frac{1}{\log(1.1 + f_c)}$ | Prevent extreme weights |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_class_weights(dataset, num_classes, method='inverse'):
    """Compute class weights from dataset statistics"""
    pixel_counts = np.zeros(num_classes)
    
    for _, mask in dataset:
        for c in range(num_classes):
            pixel_counts[c] += (mask == c).sum()
    
    frequencies = pixel_counts / pixel_counts.sum()
    
    if method == 'inverse':
        weights = 1.0 / (frequencies + 1e-8)
    elif method == 'median':
        median_freq = np.median(frequencies[frequencies > 0])
        weights = median_freq / (frequencies + 1e-8)
    elif method == 'log':
        weights = 1.0 / np.log(1.1 + frequencies)
    
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

class WeightedCrossEntropy(nn.Module):
    """Cross-entropy with class weights"""
    def __init__(self, class_weights, ignore_index=255):
        super().__init__()
        self.weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        return F.cross_entropy(
            pred, target, 
            weight=self.weights.to(pred.device),
            ignore_index=self.ignore_index
        )

class BalancedDiceLoss(nn.Module):
    """Dice loss with class balancing"""
    def __init__(self, num_classes, class_weights=None, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.weights = class_weights
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2)
        
        dice_per_class = []
        for c in range(self.num_classes):
            p = pred[:, c].flatten()
            g = target_one_hot[:, c].float().flatten()
            
            intersection = (p * g).sum()
            dice = (2. * intersection + self.smooth) / \
                   (p.sum() + g.sum() + self.smooth)
            dice_per_class.append(dice)
        
        dice_tensor = torch.stack(dice_per_class)
        
        if self.weights is not None:
            weights = self.weights.to(pred.device)
            return 1 - (dice_tensor * weights).sum() / weights.sum()
        
        return 1 - dice_tensor.mean()

class FocalLossWithClassBalance(nn.Module):
    """Focal loss with class balancing"""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Per-class weight
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        
        # Get alpha for each pixel based on its class
        alpha = self.alpha[target]
        
        focal = alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()

# Class-balanced sampling
class ClassBalancedSampler(torch.utils.data.Sampler):
    """Sampler that balances class representation"""
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        self.indices_per_class = self._group_by_class()
    
    def _group_by_class(self):
        """Group image indices by dominant class"""
        indices = {c: [] for c in range(self.num_classes)}
        
        for idx, (_, mask) in enumerate(self.dataset):
            # Find dominant class in this image
            class_counts = [(mask == c).sum() for c in range(self.num_classes)]
            dominant = np.argmax(class_counts)
            indices[dominant].append(idx)
        
        return indices
    
    def __iter__(self):
        # Sample equally from each class
        samples = []
        max_samples = max(len(v) for v in self.indices_per_class.values())
        
        for c in range(self.num_classes):
            if len(self.indices_per_class[c]) > 0:
                samples.extend(np.random.choice(
                    self.indices_per_class[c], 
                    max_samples, 
                    replace=True
                ))
        
        np.random.shuffle(samples)
        return iter(samples)
    
    def __len__(self):
        return len(self.dataset)

# Copy-paste augmentation for rare classes
class CopyPasteAugmentation:
    """Copy instances of rare classes to other images"""
    def __init__(self, rare_classes, paste_prob=0.5):
        self.rare_classes = rare_classes
        self.paste_prob = paste_prob
        self.instance_bank = {c: [] for c in rare_classes}
    
    def add_to_bank(self, image, mask):
        for c in self.rare_classes:
            if (mask == c).any():
                # Extract instance
                binary_mask = (mask == c)
                instance = image * binary_mask
                self.instance_bank[c].append((instance, binary_mask))
    
    def __call__(self, image, mask):
        if np.random.random() > self.paste_prob:
            return image, mask
        
        # Pick a rare class to paste
        c = np.random.choice(self.rare_classes)
        if len(self.instance_bank[c]) == 0:
            return image, mask
        
        # Get random instance
        instance, inst_mask = random.choice(self.instance_bank[c])
        
        # Random position
        y = np.random.randint(0, image.shape[1] - inst_mask.shape[0])
        x = np.random.randint(0, image.shape[2] - inst_mask.shape[1])
        
        # Paste
        image[:, y:y+inst_mask.shape[0], x:x+inst_mask.shape[1]] = \
            torch.where(inst_mask, instance, 
                       image[:, y:y+inst_mask.shape[0], x:x+inst_mask.shape[1]])
        mask[y:y+inst_mask.shape[0], x:x+inst_mask.shape[1]] = \
            torch.where(inst_mask, c, 
                       mask[y:y+inst_mask.shape[0], x:x+inst_mask.shape[1]])
        
        return image, mask
```

**Best Practices:**

| Imbalance Ratio | Recommended Approach |
|-----------------|---------------------|
| <10:1 | Weighted CE |
| 10:1 - 100:1 | Dice + Weighted CE |
| >100:1 | Focal + Copy-paste + Oversampling |

**Interview Tip:** Combine multiple strategies for severe imbalance. Dice loss helps but fails when a class is absent. Use online hard example mining (OHEM) to focus on difficult pixels. Monitor per-class IoU, not just mIoU.

---

### Question 7
- [ ] Done
**What techniques improve segmentation performance on small or thin objects (boundary loss, deep supervision)?**

**Answer:**

Small and thin objects are challenging due to resolution loss during downsampling. Solutions include: boundary-aware losses, deep supervision at multiple scales, high-resolution feature branches, and dedicated boundary detection modules.

**Challenges:**

| Issue | Cause | Effect |
|-------|-------|--------|
| Resolution loss | Pooling/striding | Small objects disappear |
| Class imbalance | Few pixels | Model ignores them |
| Blurry boundaries | Upsampling artifacts | Imprecise edges |

**Techniques:**

| Technique | How It Helps |
|-----------|-------------|
| Boundary loss | Penalizes boundary errors directly |
| Deep supervision | Gradients at multiple resolutions |
| HRNet | Maintains high-resolution throughout |
| Edge detection branch | Explicit boundary learning |

**Boundary Loss:**

Distance-based loss using distance transform:
$$\mathcal{L}_{boundary} = \sum_p \phi(p) \cdot (y_p - \hat{y}_p)^2$$

Where $\phi(p)$ = distance to nearest boundary.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

class BoundaryLoss(nn.Module):
    """Loss that weights pixels based on distance to boundary"""
    def __init__(self, theta=10):
        super().__init__()
        self.theta = theta
    
    def compute_distance_map(self, mask):
        """Compute distance transform from boundary"""
        mask_np = mask.cpu().numpy()
        batch_dist = []
        
        for m in mask_np:
            # Find edges
            edges = ndimage.sobel(m.astype(float))
            edges = (np.abs(edges) > 0).astype(float)
            
            # Distance transform
            dist = ndimage.distance_transform_edt(1 - edges)
            batch_dist.append(dist)
        
        return torch.tensor(np.stack(batch_dist), device=mask.device)
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        # Compute boundary weights
        dist_map = self.compute_distance_map(target)
        boundary_weights = torch.exp(-dist_map / self.theta)
        
        # Weighted cross-entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * boundary_weights
        
        return weighted_loss.mean()

class HDLoss(nn.Module):
    """Hausdorff Distance Loss for boundary accuracy"""
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1]  # Binary
        
        # Distance transforms
        dist_pred = self._distance_transform(pred > 0.5)
        dist_target = self._distance_transform(target > 0)
        
        # Hausdorff-like loss
        hd_loss = (pred - target).abs() * (dist_pred + dist_target).pow(self.alpha)
        
        return hd_loss.mean()

class DeepSupervision(nn.Module):
    """Deep supervision with losses at multiple scales"""
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        
        # Auxiliary heads at different decoder stages
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(256, num_classes, 1),  # 1/4 scale
            nn.Conv2d(128, num_classes, 1),  # 1/2 scale
        ])
        
        self.main_head = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x, target=None):
        # Get intermediate features
        features = self.base.encode(x)
        
        outputs = []
        for i, (feat, head) in enumerate(zip(
            [features['1/4'], features['1/2'], features['1/1']],
            self.aux_heads + [self.main_head]
        )):
            out = head(feat)
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear')
            outputs.append(out)
        
        if target is not None:
            # Weighted loss from all outputs
            loss = 0
            weights = [0.4, 0.4, 1.0]  # Higher weight for final output
            for out, w in zip(outputs, weights):
                loss += w * F.cross_entropy(out, target)
            return outputs[-1], loss
        
        return outputs[-1]

class BoundaryAwareUNet(nn.Module):
    """U-Net with explicit boundary branch"""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.encoder = UNetEncoder(in_ch)
        
        # Main segmentation decoder
        self.seg_decoder = UNetDecoder(num_classes)
        
        # Boundary detection branch
        self.boundary_decoder = UNetDecoder(1)  # Binary boundary
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes + 1, num_classes, 3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        # Segmentation
        seg = self.seg_decoder(features)
        
        # Boundary
        boundary = self.boundary_decoder(features)
        
        # Fuse boundary with segmentation
        fused = self.fusion(torch.cat([seg, boundary], dim=1))
        
        return fused, boundary

class MultiScaleLoss(nn.Module):
    """Loss computed at multiple scales"""
    def __init__(self, scales=[1.0, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.weights = weights
    
    def forward(self, pred, target):
        total_loss = 0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1.0:
                # Downsample prediction and target
                h, w = int(pred.shape[2] * scale), int(pred.shape[3] * scale)
                pred_scaled = F.interpolate(pred, (h, w), mode='bilinear')
                target_scaled = F.interpolate(
                    target.unsqueeze(1).float(), (h, w), mode='nearest'
                ).squeeze(1).long()
            else:
                pred_scaled, target_scaled = pred, target
            
            loss = F.cross_entropy(pred_scaled, target_scaled)
            total_loss += weight * loss
        
        return total_loss
```

**HRNet Approach:**
```python
class HighResolutionBranch(nn.Module):
    """Maintain high resolution features throughout"""
    def __init__(self):
        super().__init__()
        # Multiple parallel branches at different resolutions
        self.branch_1x = nn.ModuleList()  # Full resolution
        self.branch_2x = nn.ModuleList()  # 1/2 resolution
        self.branch_4x = nn.ModuleList()  # 1/4 resolution
        
        # Cross-resolution fusion
        self.fuse = CrossResolutionFusion()
    
    def forward(self, x):
        x_1x = x
        x_2x = F.max_pool2d(x, 2)
        x_4x = F.max_pool2d(x_2x, 2)
        
        for b1, b2, b4 in zip(self.branch_1x, self.branch_2x, self.branch_4x):
            x_1x, x_2x, x_4x = self.fuse(b1(x_1x), b2(x_2x), b4(x_4x))
        
        return x_1x  # High-resolution output
```

**Results:**

| Technique | mIoU Gain | Boundary F1 Gain |
|-----------|-----------|------------------|
| Boundary loss | +1-2% | +5-10% |
| Deep supervision | +2-3% | +3-5% |
| HRNet | +3-5% | +8-12% |
| Combined | +5-8% | +15-20% |

**Interview Tip:** For small objects, avoid aggressive downsampling (use dilated convolutions instead). Deep supervision helps gradients flow to early layers. Boundary losses are essential for thin structures like vessels or roads.

---

### Question 8
- [ ] Done
**Explain the overlap-tile strategy for inference on large images that don't fit in memory.**

**Answer:**

Overlap-tile strategy divides large images into overlapping tiles, processes each independently, and stitches results using the overlap to avoid boundary artifacts. Essential for high-resolution medical, satellite, and pathology images.

**Why Overlap is Needed:**

| Without Overlap | With Overlap |
|-----------------|--------------|
| Edge artifacts | Smooth boundaries |
| Inconsistent predictions | Consistent results |
| Missing context at borders | Full context everywhere |

**Strategy:**

```
Original Large Image (8000×8000)
         ↓ Divide into tiles
[Tile 1][Tile 2][Tile 3]...
    ↓        ↓        ↓
  Process each tile
    ↓        ↓        ↓
[Result 1][Result 2][Result 3]
         ↓ Stitch (use center, discard edges)
Final Prediction (8000×8000)
```

**Key Parameters:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Tile size | Processing patch size | 256-1024 px |
| Overlap | Shared pixels between tiles | 32-128 px |
| Stride | Step between tiles | tile_size - overlap |

**Python Implementation:**
```python
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class TiledInference:
    """Sliding window inference for large images"""
    
    def __init__(self, model, tile_size=512, overlap=64, batch_size=4):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.batch_size = batch_size
    
    def predict(self, image):
        """
        Process large image using overlap-tile strategy
        
        image: numpy array (H, W, C) or (H, W)
        returns: prediction array (H, W) or (H, W, num_classes)
        """
        H, W = image.shape[:2]
        
        # Pad image to fit tiles exactly
        pad_h = (self.stride - (H - self.tile_size) % self.stride) % self.stride
        pad_w = (self.stride - (W - self.tile_size) % self.stride) % self.stride
        
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Initialize output and count arrays
        H_pad, W_pad = padded.shape[:2]
        output = np.zeros((H_pad, W_pad), dtype=np.float32)
        counts = np.zeros((H_pad, W_pad), dtype=np.float32)
        
        # Generate tile coordinates
        tiles = []
        coords = []
        for y in range(0, H_pad - self.tile_size + 1, self.stride):
            for x in range(0, W_pad - self.tile_size + 1, self.stride):
                tile = padded[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append(tile)
                coords.append((y, x))
        
        # Process in batches
        for i in range(0, len(tiles), self.batch_size):
            batch_tiles = tiles[i:i+self.batch_size]
            batch_coords = coords[i:i+self.batch_size]
            
            # Prepare batch
            batch = np.stack(batch_tiles)
            batch = torch.tensor(batch).permute(0, 3, 1, 2).float() / 255.0
            
            # Inference
            with torch.no_grad():
                preds = self.model(batch.cuda())
                preds = torch.argmax(preds, dim=1).cpu().numpy()
            
            # Create weight mask (higher weight in center)
            weight = self._create_weight_mask()
            
            # Accumulate predictions
            for (y, x), pred in zip(batch_coords, preds):
                output[y:y+self.tile_size, x:x+self.tile_size] += pred * weight
                counts[y:y+self.tile_size, x:x+self.tile_size] += weight
        
        # Average overlapping regions
        output = output / (counts + 1e-8)
        
        # Remove padding
        output = output[:H, :W]
        
        return output.astype(np.uint8)
    
    def _create_weight_mask(self):
        """Create weight mask: higher in center, lower at edges"""
        x = np.linspace(0, 1, self.tile_size)
        y = np.linspace(0, 1, self.tile_size)
        xx, yy = np.meshgrid(x, y)
        
        # Sigmoid-like weights
        weight_x = np.minimum(xx, 1 - xx) * 2
        weight_y = np.minimum(yy, 1 - yy) * 2
        weight = np.minimum(weight_x, weight_y)
        
        # Smooth edges
        weight = np.clip(weight * 4, 0, 1)
        
        return weight

class SmartTiledInference(TiledInference):
    """Improved tiling with multi-scale support"""
    
    def __init__(self, model, tile_size=512, overlap=64, scales=[1.0]):
        super().__init__(model, tile_size, overlap)
        self.scales = scales
    
    def predict_multiscale(self, image):
        """Average predictions across multiple scales"""
        H, W = image.shape[:2]
        all_preds = []
        
        for scale in self.scales:
            # Resize image
            h_scaled = int(H * scale)
            w_scaled = int(W * scale)
            
            if scale != 1.0:
                image_scaled = np.array(Image.fromarray(image).resize(
                    (w_scaled, h_scaled), Image.BILINEAR
                ))
            else:
                image_scaled = image
            
            # Process at this scale
            pred = self.predict(image_scaled)
            
            # Resize prediction back
            if scale != 1.0:
                pred = np.array(Image.fromarray(pred).resize((W, H), Image.NEAREST))
            
            all_preds.append(pred)
        
        # Vote or average
        stacked = np.stack(all_preds)
        final = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 
            axis=0, arr=stacked
        )
        
        return final

# Soft probability accumulation
def predict_with_soft_accumulation(model, image, tile_size, overlap, num_classes):
    """Accumulate soft probabilities instead of hard labels"""
    H, W = image.shape[:2]
    stride = tile_size - overlap
    
    # Initialize probability accumulator
    probs = np.zeros((num_classes, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile = image[y:y+tile_size, x:x+tile_size]
            tile_tensor = preprocess(tile).unsqueeze(0).cuda()
            
            with torch.no_grad():
                pred = F.softmax(model(tile_tensor), dim=1)
                pred = pred.squeeze().cpu().numpy()
            
            # Accumulate probabilities
            probs[:, y:y+tile_size, x:x+tile_size] += pred
            counts[y:y+tile_size, x:x+tile_size] += 1
    
    # Average
    probs = probs / (counts + 1e-8)
    
    # Argmax for final prediction
    return np.argmax(probs, axis=0)
```

**Memory-Efficient Tips:**

| Tip | Benefit |
|-----|---------|
| Process tiles in batches | GPU utilization |
| Use half precision (FP16) | 2× memory savings |
| Delete tensors after use | Free GPU memory |
| Use soft accumulation | Better boundaries |

**Interview Tip:** Overlap should be at least 2× the model's receptive field boundary effect. Weighted averaging (center > edges) gives smoother results than hard boundaries. Multi-scale can improve accuracy but increases computation.

---

### Question 9
- [ ] Done
**How do you implement data augmentation that preserves spatial relationships for segmentation?**

**Answer:**

Segmentation augmentation must apply identical geometric transforms to both image and mask. Photometric augmentations (brightness, contrast) apply only to images. Use augmentation libraries that handle paired transforms (Albumentations, torchvision.transforms.v2).

**Augmentation Categories:**

| Type | Apply to Image | Apply to Mask |
|------|----------------|---------------|
| Geometric (flip, rotate) | ✓ | ✓ |
| Photometric (color, brightness) | ✓ | ✗ |
| Noise/blur | ✓ | ✗ |
| Crop/resize | ✓ | ✓ (same coords) |

**Common Segmentation Augmentations:**

| Augmentation | Effect | Caution |
|--------------|--------|---------|
| Random flip | Invariance to orientation | May not suit directional data |
| Random crop | Scale invariance | Must keep enough context |
| Rotation | Orientation invariance | Use appropriate interpolation |
| Elastic deform | Shape variation | Can create artifacts |
| Scale jitter | Size invariance | Keep objects visible |

**Python Implementation:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision.transforms.v2 as T

# Using Albumentations (recommended)
class SegmentationTransforms:
    """Paired transforms for image and mask"""
    
    def __init__(self, train=True, img_size=256):
        if train:
            self.transform = A.Compose([
                # Geometric (applied to both)
                A.RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5
                ),
                
                # Elastic deformation (applied to both)
                A.ElasticTransform(alpha=120, sigma=6, p=0.3),
                
                # Photometric (image only)
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.GaussianBlur(blur_limit=7, p=0.3),
                
                # Normalize and convert
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __call__(self, image, mask):
        # Albumentations handles paired transforms automatically
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

# Using torchvision v2 transforms
class TorchvisionSegmentationTransforms:
    """Using torchvision.transforms.v2 for segmentation"""
    
    def __init__(self, train=True, img_size=256):
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(45),
            ])
            
            # Image-only transforms
            self.image_transform = T.Compose([
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.GaussianBlur(7, sigma=(0.1, 2.0)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.mask_transform = T.Compose([
                T.ToTensor()
            ])
        else:
            self.transform = T.Resize((img_size, img_size))
            self.image_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.mask_transform = T.ToTensor()
    
    def __call__(self, image, mask):
        # Apply paired geometric transforms
        # Stack for joint transform
        stacked = torch.cat([
            torch.tensor(image).permute(2, 0, 1),
            torch.tensor(mask).unsqueeze(0)
        ], dim=0)
        
        transformed = self.transform(stacked)
        
        image = transformed[:3]
        mask = transformed[3:]
        
        # Apply image-only transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask.long().squeeze(0)

# Custom paired transforms
class PairedTransform:
    """Manual paired transform implementation"""
    
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, image, mask):
        # Use same random seed for both
        seed = np.random.randint(2147483647)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        image = self._apply_transforms(image)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        mask = self._apply_transforms(mask, is_mask=True)
        
        return image, mask
    
    def _apply_transforms(self, x, is_mask=False):
        for t in self.transforms:
            if hasattr(t, 'is_geometric') and t.is_geometric:
                x = t(x)
            elif not is_mask:
                x = t(x)
        return x

# CutMix/MixUp for segmentation
class CutMixSegmentation:
    """CutMix augmentation for segmentation"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, image1, mask1, image2, mask2):
        lam = np.random.beta(self.alpha, self.alpha)
        
        H, W = image1.shape[1:3]
        
        # Random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        
        # Cut and paste
        image1[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        mask1[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
        
        return image1, mask1

# Copy-paste augmentation
class CopyPasteSegmentation:
    """Copy instances from one image to another"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image1, mask1, image2, mask2):
        if np.random.random() > self.p:
            return image1, mask1
        
        # Find instances in image2
        unique_ids = torch.unique(mask2)
        unique_ids = unique_ids[unique_ids != 0]  # Exclude background
        
        if len(unique_ids) == 0:
            return image1, mask1
        
        # Select random instance
        instance_id = unique_ids[np.random.randint(len(unique_ids))]
        instance_mask = (mask2 == instance_id)
        
        # Paste instance
        image1[:, instance_mask] = image2[:, instance_mask]
        mask1[instance_mask] = instance_id
        
        return image1, mask1
```

**Augmentation Pipeline Example:**
```python
# Complete training pipeline
train_transform = A.Compose([
    # Resize
    A.RandomResizedCrop(512, 512, scale=(0.5, 1.5)),
    
    # Flips
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    
    # Rotation
    A.RandomRotate90(p=0.5),
    
    # Color
    A.OneOf([
        A.ColorJitter(brightness=0.3, contrast=0.3),
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
    ], p=0.5),
    
    # Noise/blur
    A.OneOf([
        A.GaussNoise(),
        A.GaussianBlur(blur_limit=5),
        A.MotionBlur(blur_limit=5),
    ], p=0.3),
    
    # Normalize
    A.Normalize(),
    ToTensorV2()
])
```

**Interview Tip:** Always use the same random seed for paired transforms. Albumentations is the go-to library for segmentation augmentation. For medical imaging, elastic deformation is particularly effective. Avoid heavy augmentation that makes the task too hard.

---

### Question 10
- [ ] Done
**What approaches work for real-time semantic segmentation (BiSeNet, Fast-SCNN, EfficientPS)?**

**Answer:**

Real-time segmentation uses efficient architectures with two-branch designs (spatial + context), depthwise separable convolutions, early downsampling, and attention-based feature fusion. Target: >30 FPS on standard hardware while maintaining reasonable accuracy.

**Speed vs Accuracy Trade-off:**

| Model | mIoU (Cityscapes) | FPS (1024×2048) | Key Feature |
|-------|-------------------|-----------------|-------------|
| BiSeNet | 68.4% | 105 | Two-branch |
| Fast-SCNN | 68.0% | 123 | Learning to downsample |
| EfficientPS | 84.2% | 10 | Panoptic, accurate |
| DDRNet | 79.4% | 37 | Deep dual resolution |
| STDC | 76.8% | 97 | Short-term dense concat |

**BiSeNet Architecture:**

```
        Input Image
       /           \
Spatial Path    Context Path
(preserve       (semantic info)
 detail)        
 3 Conv3×3      Backbone (Xception)
    ↓                ↓
    ↓           Global Pooling + ARM
    ↓                ↓
      → Fusion (FFM) ←
             ↓
        Segmentation
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution (efficient)
class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))

# BiSeNet Components
class SpatialPath(nn.Module):
    """Preserve spatial information"""
    def __init__(self, in_ch=3, out_ch=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x  # 1/8 resolution

class ContextPath(nn.Module):
    """Extract semantic context"""
    def __init__(self, backbone='resnet18'):
        super().__init__()
        self.backbone = self._get_backbone(backbone)
        self.arm1 = AttentionRefinementModule(256, 128)
        self.arm2 = AttentionRefinementModule(512, 128)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        
        # Global context
        global_ctx = self.global_context(feat32)
        global_ctx = F.interpolate(global_ctx, feat32.shape[2:], mode='nearest')
        
        # ARM processing
        feat32 = self.arm2(feat32) + global_ctx
        feat16 = self.arm1(feat16)
        
        # Upsample and combine
        feat32_up = F.interpolate(feat32, feat16.shape[2:], mode='bilinear')
        return torch.cat([feat16, feat32_up], dim=1)

class AttentionRefinementModule(nn.Module):
    """Channel attention for feature refinement"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        att = self.attention(x)
        return x * att

class FeatureFusionModule(nn.Module):
    """Fuse spatial and context features"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spatial, context):
        x = torch.cat([spatial, context], dim=1)
        x = self.conv(x)
        att = self.attention(x)
        return x + x * att

class BiSeNet(nn.Module):
    """Complete BiSeNet for real-time segmentation"""
    def __init__(self, num_classes, backbone='resnet18'):
        super().__init__()
        self.spatial_path = SpatialPath(3, 128)
        self.context_path = ContextPath(backbone)
        self.ffm = FeatureFusionModule(128 + 256, 256)
        self.head = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        size = x.shape[2:]
        
        spatial = self.spatial_path(x)
        context = self.context_path(x)
        
        # Resize context to match spatial
        context = F.interpolate(context, spatial.shape[2:], mode='bilinear')
        
        fused = self.ffm(spatial, context)
        out = self.head(fused)
        
        return F.interpolate(out, size, mode='bilinear')

# Fast-SCNN: Learning to Downsample
class LearningToDownsample(nn.Module):
    """Efficient downsampling module"""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 32, 3, stride=2, padding=1)
        self.dsconv1 = DSConv(32, 48, stride=2)
        self.dsconv2 = DSConv(48, out_ch, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x  # 1/8 resolution

class FastSCNN(nn.Module):
    """Fast Semantic Segmentation"""
    def __init__(self, num_classes):
        super().__init__()
        self.learning_to_downsample = LearningToDownsample(3, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, 128)
        self.feature_fusion = FeatureFusionModule(64 + 128, 128)
        self.classifier = nn.Sequential(
            DSConv(128, 128),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Learning to downsample
        lower = self.learning_to_downsample(x)
        
        # Global features
        upper = self.global_feature_extractor(lower)
        
        # Fusion
        fused = self.feature_fusion(lower, upper)
        
        # Classify
        out = self.classifier(fused)
        return F.interpolate(out, size, mode='bilinear')
```

**Optimization Techniques:**

| Technique | Speedup | Accuracy Loss |
|-----------|---------|---------------|
| Depthwise separable conv | 2-3× | <1% |
| Early downsampling | 2× | 2-3% |
| Lower resolution input | 4× | 3-5% |
| TensorRT/ONNX | 2× | 0% |

**Interview Tip:** BiSeNet's key insight is separating spatial (detail) and semantic (context) processing into two branches. For production, always optimize with TensorRT. Trade-off: use lower resolution or fewer channels for speed, but test on your specific use case.

---

### Question 11
- [ ] Done
**How do you handle segmentation of objects with fuzzy or ambiguous boundaries?**

**Answer:**

Fuzzy boundaries (smoke, hair, glass) require soft predictions instead of hard labels. Use soft labels during training, probabilistic outputs, boundary-aware losses, and uncertainty estimation. Matting techniques help for semi-transparent objects.

**Challenges:**

| Object Type | Issue | Example |
|-------------|-------|---------|
| Semi-transparent | No clear boundary | Glass, smoke |
| Fine details | Sub-pixel accuracy needed | Hair, fur |
| Motion blur | Boundaries spread | Fast-moving objects |
| Gradual transitions | No sharp edge | Clouds, fog |

**Approaches:**

| Approach | How It Helps |
|----------|-------------|
| Soft labels | Allow fractional class membership |
| Alpha matting | Predict opacity (0-1) instead of binary |
| Uncertainty output | Model predicts confidence |
| Multi-label | Pixel can belong to multiple classes |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSegmentationModel(nn.Module):
    """Model that outputs soft probabilities"""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        
        # Main segmentation head
        self.seg_head = nn.Conv2d(256, num_classes, 1)
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        logits = self.seg_head(features)
        probs = F.softmax(logits, dim=1)
        
        uncertainty = self.uncertainty_head(features)
        
        return probs, uncertainty

class SoftLabelLoss(nn.Module):
    """Loss for soft/fuzzy labels"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target, soft_target=None):
        if soft_target is not None:
            # Use provided soft labels
            log_probs = F.log_softmax(pred, dim=1)
            loss = -(soft_target * log_probs).sum(dim=1).mean()
        else:
            # Label smoothing
            num_classes = pred.shape[1]
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
            
            log_probs = F.log_softmax(pred, dim=1)
            loss = -(smooth_target * log_probs).sum(dim=1).mean()
        
        return loss

class AlphaMattingHead(nn.Module):
    """Predict alpha matte for transparency"""
    def __init__(self, in_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, coarse_mask):
        # Combine features with coarse mask
        combined = torch.cat([features, coarse_mask], dim=1)
        alpha = self.refine(combined)
        return alpha  # Values in [0, 1]

class MattingLoss(nn.Module):
    """Loss for alpha matting"""
    def __init__(self, alpha_weight=1.0, comp_weight=1.0, grad_weight=0.5):
        super().__init__()
        self.alpha_weight = alpha_weight
        self.comp_weight = comp_weight
        self.grad_weight = grad_weight
    
    def forward(self, pred_alpha, gt_alpha, image, fg, bg):
        # Alpha loss
        alpha_loss = F.l1_loss(pred_alpha, gt_alpha)
        
        # Composition loss
        pred_comp = pred_alpha * fg + (1 - pred_alpha) * bg
        gt_comp = gt_alpha * fg + (1 - gt_alpha) * bg
        comp_loss = F.l1_loss(pred_comp, gt_comp)
        
        # Gradient loss (preserve edges)
        pred_grad = self._gradient(pred_alpha)
        gt_grad = self._gradient(gt_alpha)
        grad_loss = F.l1_loss(pred_grad, gt_grad)
        
        return (self.alpha_weight * alpha_loss + 
                self.comp_weight * comp_loss + 
                self.grad_weight * grad_loss)
    
    def _gradient(self, x):
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        return torch.abs(grad_x).mean() + torch.abs(grad_y).mean()

class BoundaryUncertaintyModel(nn.Module):
    """Model with boundary-aware uncertainty"""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.seg_head = nn.Conv2d(256, num_classes, 1)
        
        # Boundary detection
        self.boundary_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        seg = self.seg_head(features)
        boundary = self.boundary_head(features)
        
        return seg, boundary
    
    def predict_soft(self, x):
        seg, boundary = self.forward(x)
        probs = F.softmax(seg, dim=1)
        
        # Soften predictions near boundaries
        boundary_expanded = boundary.expand_as(probs)
        soft_probs = probs * (1 - boundary_expanded * 0.3)  # Reduce confidence at boundaries
        
        return soft_probs

# Training with soft labels
def create_soft_labels(hard_mask, sigma=2):
    """Create soft labels by blurring boundaries"""
    from scipy.ndimage import gaussian_filter
    
    num_classes = hard_mask.max() + 1
    soft_labels = np.zeros((*hard_mask.shape, num_classes))
    
    for c in range(num_classes):
        binary = (hard_mask == c).astype(float)
        soft_labels[..., c] = gaussian_filter(binary, sigma)
    
    # Normalize
    soft_labels = soft_labels / soft_labels.sum(axis=-1, keepdims=True)
    
    return soft_labels

# Trimap-based matting
def generate_trimap(mask, erosion=10, dilation=10):
    """Create trimap: definite foreground, definite background, unknown"""
    import cv2
    
    kernel = np.ones((3, 3), np.uint8)
    
    fg = cv2.erode(mask.astype(np.uint8), kernel, iterations=erosion)
    bg = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation)
    
    trimap = np.zeros_like(mask)
    trimap[bg == 0] = 0  # Definite background
    trimap[fg == 1] = 255  # Definite foreground
    trimap[(bg == 1) & (fg == 0)] = 128  # Unknown
    
    return trimap
```

**Evaluation Metrics:**

| Metric | For | Formula |
|--------|-----|---------|
| Soft IoU | Soft predictions | $\frac{\sum \min(p, g)}{\sum \max(p, g)}$ |
| MSE | Alpha matting | $\frac{1}{N}\sum(p - g)^2$ |
| Gradient error | Edge quality | $\|\nabla p - \nabla g\|$ |
| SAD | Overall | $\sum|p - g|$ |

**Interview Tip:** For fuzzy boundaries, avoid hard argmax predictions. Output soft probabilities and let downstream tasks decide threshold. Alpha matting is essential for compositing applications (green screen, portrait mode).

---

### Question 12
- [ ] Done
**Explain weakly supervised segmentation using image-level labels or bounding boxes instead of pixel masks.**

**Answer:**

Weakly supervised segmentation trains with cheap annotations (image labels, bounding boxes, scribbles) instead of expensive pixel-level masks. Uses CAM (Class Activation Maps) for localization, pseudo-label generation, and iterative refinement to achieve near-fully-supervised performance.

**Supervision Levels:**

| Level | Annotation | Cost | Quality |
|-------|------------|------|---------|
| Fully supervised | Pixel masks | Very High | Best |
| Bounding boxes | Boxes | Medium | Good |
| Image-level | Class labels | Low | Moderate |
| Scribbles | Few clicks | Low | Moderate |

**Approaches:**

| Method | Input | Technique |
|--------|-------|-----------|
| CAM-based | Image labels | Grad-CAM → pseudo-mask |
| BoxSup | Bounding boxes | Box as supervision |
| ScribbleSup | Scribbles | Propagate labels |
| Self-training | Any weak | Iterative pseudo-labeling |

**CAM-Based Pipeline:**
```
Image → Classifier → Grad-CAM → Threshold → Pseudo-mask → Train Segmentation
                                    ↓
                              CRF Refinement
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassActivationMap:
    """Generate CAMs from classifier"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image, class_idx):
        # Forward pass
        output = self.model(image)
        
        # Backward for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)
        
        # Weight by gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam / cam.max()
        
        # Upsample to image size
        cam = F.interpolate(cam, image.shape[2:], mode='bilinear')
        
        return cam.squeeze().cpu().numpy()

class WeaklySupervised Segmentation:
    """Weakly supervised segmentation with image-level labels"""
    
    def __init__(self, classifier, segmenter, num_classes):
        self.classifier = classifier
        self.segmenter = segmenter
        self.num_classes = num_classes
        self.cam_generator = ClassActivationMap(classifier, classifier.layer4)
    
    def generate_pseudo_masks(self, images, labels, threshold=0.4):
        """Generate pseudo-masks from CAMs"""
        pseudo_masks = []
        
        for image, label in zip(images, labels):
            mask = np.zeros(image.shape[1:], dtype=np.int64)
            
            # Generate CAM for each present class
            for class_idx in label.nonzero()[0]:
                cam = self.cam_generator.generate_cam(
                    image.unsqueeze(0), class_idx
                )
                
                # Threshold
                class_mask = cam > threshold
                mask[class_mask] = class_idx + 1  # +1 for background=0
            
            # CRF refinement
            mask = self.crf_refine(image, mask)
            pseudo_masks.append(mask)
        
        return pseudo_masks
    
    def crf_refine(self, image, mask):
        """Refine pseudo-mask with CRF"""
        import pydensecrf.densecrf as dcrf
        
        h, w = mask.shape
        d = dcrf.DenseCRF2D(w, h, self.num_classes + 1)
        
        # Unary potentials from mask
        U = np.zeros((self.num_classes + 1, h * w), dtype=np.float32)
        for c in range(self.num_classes + 1):
            U[c] = (mask.flatten() == c).astype(np.float32) * (-10) + \
                   (mask.flatten() != c).astype(np.float32) * 10
        d.setUnaryEnergy(U)
        
        # Pairwise potentials
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img_np, compat=10)
        
        Q = d.inference(5)
        return np.argmax(Q, axis=0).reshape(h, w)
    
    def train_step(self, images, image_labels):
        """One training step with pseudo-labels"""
        # Generate pseudo-masks
        pseudo_masks = self.generate_pseudo_masks(images, image_labels)
        pseudo_masks = torch.tensor(pseudo_masks)
        
        # Train segmenter
        pred = self.segmenter(images)
        loss = F.cross_entropy(pred, pseudo_masks, ignore_index=255)
        
        return loss

class BoundingBoxSupervision(nn.Module):
    """Segmentation with bounding box supervision"""
    
    def __init__(self, segmenter):
        super().__init__()
        self.segmenter = segmenter
    
    def forward(self, images, boxes, box_labels):
        pred = self.segmenter(images)
        
        loss = 0
        for i, (box_list, label_list) in enumerate(zip(boxes, box_labels)):
            for box, label in zip(box_list, label_list):
                x1, y1, x2, y2 = box.int()
                
                # Loss inside box: should be the class
                inside_pred = pred[i, :, y1:y2, x1:x2]
                inside_loss = self.box_inside_loss(inside_pred, label)
                
                # Loss outside all boxes: should be background
                outside_mask = self.get_outside_mask(pred.shape[2:], box_list)
                outside_pred = pred[i, :, outside_mask]
                outside_loss = F.cross_entropy(
                    outside_pred.T.unsqueeze(0), 
                    torch.zeros(1, dtype=torch.long)
                )
                
                loss += inside_loss + outside_loss
        
        return loss
    
    def box_inside_loss(self, pred, label):
        """GrabCut-style loss: fill box with class"""
        # At least some pixels should be the class
        probs = F.softmax(pred, dim=0)
        class_prob = probs[label].mean()
        return -torch.log(class_prob + 1e-8)

class IterativePseudoLabeling:
    """Self-training with iterative pseudo-label refinement"""
    
    def __init__(self, model, confidence_threshold=0.9):
        self.model = model
        self.threshold = confidence_threshold
    
    def train(self, labeled_data, unlabeled_data, epochs=10, rounds=5):
        for round in range(rounds):
            print(f"Round {round + 1}/{rounds}")
            
            # Generate pseudo-labels for unlabeled data
            pseudo_labels = self.generate_pseudo_labels(unlabeled_data)
            
            # Combine with labeled data
            combined = self.combine_data(labeled_data, unlabeled_data, pseudo_labels)
            
            # Train
            for epoch in range(epochs):
                for images, masks in combined:
                    pred = self.model(images)
                    loss = F.cross_entropy(pred, masks, ignore_index=255)
                    loss.backward()
    
    def generate_pseudo_labels(self, data):
        pseudo = []
        for images in data:
            with torch.no_grad():
                pred = self.model(images)
                probs = F.softmax(pred, dim=1)
                
                max_probs, labels = probs.max(dim=1)
                
                # Only keep high-confidence predictions
                labels[max_probs < self.threshold] = 255  # Ignore
                pseudo.append(labels)
        
        return pseudo
```

**Results Comparison:**

| Supervision | VOC mIoU | Annotation Time |
|-------------|----------|-----------------|
| Fully supervised | 77% | 100× |
| Bounding boxes | 70% | 5× |
| Image labels | 62% | 1× |

**Interview Tip:** CAM-based methods are simple but coarse. Combine with CRF for better boundaries. Self-training with confidence thresholding iteratively improves pseudo-labels. Bounding box supervision with GrabCut achieves near-full supervision quality.

---

## Instance Segmentation

### Question 13
- [ ] Done
**How does Mask R-CNN's architecture balance object detection and pixel-level segmentation?**

**Answer:**

Mask R-CNN extends Faster R-CNN by adding a parallel mask prediction branch. Detection and mask heads share RoI features but have separate loss functions. The key is decoupling mask and classification—masks are predicted for each class independently.

**Architecture:**

```
Image → Backbone (ResNet+FPN) → RPN → RoI Proposals
                                         ↓
                                    RoIAlign
                                    /   |   \
                              Class  Box  Mask
                              Head   Head Head
                                ↓      ↓     ↓
                              Labels Boxes Masks
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| Backbone (FPN) | Multi-scale feature extraction |
| RPN | Generate object proposals |
| RoIAlign | Extract aligned features per proposal |
| Box head | Predict bounding box + class |
| Mask head | Predict binary mask per class |

**Key Insight: Decoupled Mask and Class**

- Mask head predicts K binary masks (one per class)
- Classification head selects which mask to use
- Avoids competition between classes in mask prediction

**Loss Function:**
$$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{box} + \mathcal{L}_{mask}$$

Where mask loss is binary cross-entropy on the GT class mask only.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import RoIAlign

class MaskRCNN(nn.Module):
    """Simplified Mask R-CNN architecture"""
    
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        # Backbone with FPN
        self.backbone = self._build_backbone(backbone)
        
        # Region Proposal Network
        self.rpn = RegionProposalNetwork(256, 256)
        
        # RoI feature extraction
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16, 
                                   sampling_ratio=2)
        self.roi_align_mask = RoIAlign(output_size=(14, 14), spatial_scale=1/16,
                                        sampling_ratio=2)
        
        # Detection heads
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.cls_predictor = nn.Linear(1024, num_classes)
        self.box_predictor = nn.Linear(1024, num_classes * 4)
        
        # Mask head (FCN)
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)  # K masks
        )
    
    def forward(self, images, targets=None):
        # Feature extraction
        features = self.backbone(images)
        
        # RPN proposals
        proposals, rpn_losses = self.rpn(features, targets)
        
        if self.training:
            proposals, matched_targets = self.match_proposals(proposals, targets)
        
        # RoI features for detection
        roi_features = self.roi_align(features['p3'], proposals)
        box_features = self.box_head(roi_features)
        
        class_logits = self.cls_predictor(box_features)
        box_regression = self.box_predictor(box_features)
        
        # RoI features for mask (higher resolution)
        mask_features = self.roi_align_mask(features['p3'], proposals)
        mask_logits = self.mask_head(mask_features)  # [N, K, 28, 28]
        
        if self.training:
            losses = self.compute_losses(
                class_logits, box_regression, mask_logits,
                matched_targets
            )
            return losses
        else:
            return self.post_process(class_logits, box_regression, mask_logits, proposals)
    
    def compute_losses(self, class_logits, box_regression, mask_logits, targets):
        # Classification loss
        cls_loss = F.cross_entropy(class_logits, targets['labels'])
        
        # Box regression loss (only for positive samples)
        box_loss = F.smooth_l1_loss(
            box_regression[targets['labels'] > 0],
            targets['boxes'][targets['labels'] > 0]
        )
        
        # Mask loss (only for positive samples, only GT class mask)
        mask_loss = 0
        for i, label in enumerate(targets['labels']):
            if label > 0:  # Positive sample
                # Get predicted mask for GT class
                pred_mask = mask_logits[i, label]  # [28, 28]
                gt_mask = targets['masks'][i]  # [28, 28]
                
                mask_loss += F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
        
        mask_loss = mask_loss / max(1, (targets['labels'] > 0).sum())
        
        return {
            'loss_cls': cls_loss,
            'loss_box': box_loss,
            'loss_mask': mask_loss
        }
    
    def post_process(self, class_logits, box_regression, mask_logits, proposals):
        # Get predicted classes
        scores = F.softmax(class_logits, dim=1)
        max_scores, labels = scores.max(dim=1)
        
        # Select mask for predicted class
        masks = []
        for i, label in enumerate(labels):
            mask = torch.sigmoid(mask_logits[i, label])
            masks.append(mask)
        
        return {
            'boxes': proposals,
            'labels': labels,
            'scores': max_scores,
            'masks': torch.stack(masks)
        }

# Using torchvision (recommended)
def get_maskrcnn_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace heads for custom classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    
    return model
```

**Balancing Detection and Segmentation:**

| Aspect | Detection | Segmentation |
|--------|-----------|--------------|
| Features | 7×7 RoI | 14×14 RoI (finer) |
| Loss weight | 1.0 | 1.0 (equal) |
| Output | Class + box | Per-class binary mask |
| Training | Multi-task | Shared backbone |

**Interview Tip:** Mask R-CNN's key insight is predicting class-agnostic masks (one per class) and selecting based on classification. This decoupling prevents mask quality from being affected by classification uncertainty. RoIAlign (vs RoIPool) is crucial for mask quality.

---

### Question 14
- [ ] Done
**Explain the difference between ROIPool and ROIAlign. Why is ROIAlign crucial for mask quality?**

**Answer:**

ROIPool uses coarse quantization that misaligns features with input pixels, causing spatial distortions. ROIAlign uses bilinear interpolation for exact alignment, eliminating quantization errors. This precision is crucial for pixel-level mask predictions where small misalignments cause visible artifacts.

**The Problem with ROIPool:**

```
Input feature map: 7.3 → 7 (floor)
                          ↓ Quantization error
                   Misaligned features
```

**Quantization Errors in ROIPool:**
1. Rounding proposal coordinates to integers
2. Dividing into bins with floor/ceil
3. Each quantization loses sub-pixel precision

**ROIAlign Solution:**
- Uses bilinear interpolation
- Samples at exact floating-point locations
- No rounding of coordinates

**Visual Comparison:**

```
ROIPool:                          ROIAlign:
[Grid snaps to pixels]            [Grid aligns exactly]
┌───┬───┬───┐                     ┌───┬───┬───┐
│ ● │   │   │  ← Sampled at      │ ⊕ │ ⊕ │ ⊕ │  ← Sampled at
├───┼───┼───┤    integer pos     ├───┼───┼───┤    exact positions
│   │ ● │   │                     │ ⊕ │ ⊕ │ ⊕ │
└───┴───┴───┘                     └───┴───┴───┘
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
from torchvision.ops import roi_align, roi_pool

class RoIPoolVsAlign:
    """Compare ROIPool and ROIAlign"""
    
    @staticmethod
    def roi_pool_manual(features, boxes, output_size):
        """
        ROIPool with quantization
        """
        B, C, H, W = features.shape
        pooled = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Quantization step 1: Round box coordinates
            x1, y1 = int(x1), int(y1)  # ← Precision loss!
            x2, y2 = int(x2), int(y2)
            
            # Extract region
            roi = features[:, :, y1:y2, x1:x2]
            
            # Quantization step 2: Divide into bins
            bin_h = (y2 - y1) // output_size[0]  # ← More precision loss!
            bin_w = (x2 - x1) // output_size[1]
            
            # Max pool each bin
            output = torch.zeros(C, *output_size)
            for i in range(output_size[0]):
                for j in range(output_size[1]):
                    bin_region = roi[:, :, 
                                     i*bin_h:(i+1)*bin_h, 
                                     j*bin_w:(j+1)*bin_w]
                    output[:, i, j] = bin_region.max()
            
            pooled.append(output)
        
        return torch.stack(pooled)
    
    @staticmethod
    def roi_align_manual(features, boxes, output_size, sampling_ratio=2):
        """
        ROIAlign with bilinear interpolation
        """
        B, C, H, W = features.shape
        aligned = []
        
        for box in boxes:
            x1, y1, x2, y2 = box  # Keep as float!
            
            roi_h = y2 - y1
            roi_w = x2 - x1
            
            # Bin size (floating point)
            bin_h = roi_h / output_size[0]
            bin_w = roi_w / output_size[1]
            
            output = torch.zeros(C, *output_size)
            
            for i in range(output_size[0]):
                for j in range(output_size[1]):
                    # Sample points within bin
                    samples = []
                    for si in range(sampling_ratio):
                        for sj in range(sampling_ratio):
                            # Exact sample location (no quantization)
                            y = y1 + bin_h * (i + (si + 0.5) / sampling_ratio)
                            x = x1 + bin_w * (j + (sj + 0.5) / sampling_ratio)
                            
                            # Bilinear interpolation
                            val = bilinear_interpolate(features[0], x, y)
                            samples.append(val)
                    
                    # Average samples
                    output[:, i, j] = torch.stack(samples).mean(dim=0)
            
            aligned.append(output)
        
        return torch.stack(aligned)

def bilinear_interpolate(features, x, y):
    """Bilinear interpolation at (x, y)"""
    C, H, W = features.shape
    
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    
    # Weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    return (wa * features[:, y0, x0] + 
            wb * features[:, y0, x1] + 
            wc * features[:, y1, x0] + 
            wd * features[:, y1, x1])

# Using torchvision (recommended)
def compare_pool_vs_align():
    features = torch.randn(1, 256, 50, 50)
    boxes = torch.tensor([[10.3, 15.7, 30.2, 40.8]])  # Non-integer coords
    
    # ROIPool (with quantization)
    pooled = roi_pool(features, [boxes], output_size=(7, 7), spatial_scale=1.0)
    
    # ROIAlign (exact)
    aligned = roi_align(features, [boxes], output_size=(7, 7), 
                        spatial_scale=1.0, sampling_ratio=2)
    
    print(f"Max difference: {(pooled - aligned).abs().max()}")
    # Significant difference due to quantization
```

**Impact on Mask Quality:**

| Metric | ROIPool | ROIAlign | Improvement |
|--------|---------|----------|-------------|
| Mask AP | 23.6 | 37.1 | +13.5 |
| Mask AP (small) | 8.2 | 17.2 | +9.0 |
| Box AP | 26.9 | 38.2 | +11.3 |

**Why Masks Need Alignment:**

| Task | Tolerance to Misalignment |
|------|---------------------------|
| Classification | High (global pooling) |
| Box regression | Medium |
| Mask prediction | Very Low (pixel-level) |

**Key Parameters:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| output_size | RoI feature resolution | 7 (box), 14 (mask) |
| spatial_scale | Feature map / input ratio | 1/16 or 1/32 |
| sampling_ratio | Points per bin | 2 |

**Interview Tip:** ROIAlign is essential for any pixel-level prediction from region proposals. The 13.5 AP improvement shows how important sub-pixel precision is. Small objects benefit most because quantization error is proportionally larger.

---

---

### Question 15
- [ ] Done
**What are the trade-offs between two-stage (Mask R-CNN) and single-stage (YOLACT, SOLOv2) instance segmentation?**

**Answer:**

Two-stage methods (Mask R-CNN) provide higher accuracy with separate proposal and segmentation steps. Single-stage methods (YOLACT, SOLOv2) are faster by predicting masks directly without region proposals. Trade-off is speed vs. accuracy.

**Comparison:**

| Aspect | Two-Stage (Mask R-CNN) | Single-Stage (YOLACT) |
|--------|----------------------|----------------------|
| Speed | ~5 FPS | ~30+ FPS |
| Accuracy | Higher (mAP ~37) | Lower (mAP ~30) |
| Architecture | RPN + RoIAlign + Mask head | Direct mask prediction |
| Memory | Higher | Lower |
| Small objects | Better | Challenging |

**Architectures:**

**Mask R-CNN (Two-Stage):**
```
Image → Backbone → RPN → Proposals → RoIAlign → Box/Class → Mask
                    ↓
               Stage 1: Where    Stage 2: What + Segment
```

**YOLACT (Single-Stage):**
```
Image → Backbone → Prototype masks (K global masks)
                 → Coefficients per detection
                 → Mask = Σ(coef × prototype)
```

**SOLOv2:**
```
Image → Backbone → FPN → Category branch
                       → Kernel branch → Dynamic conv → Masks
```

**Key Differences:**

| Feature | Mask R-CNN | YOLACT | SOLOv2 |
|---------|-----------|--------|--------|
| Mask computation | Per-instance RoI | Linear combo of prototypes | Dynamic convolutions |
| NMS | Required | Required | Matrix NMS |
| Instance representation | Box-based | Coefficients | Location-based |

**Python Comparison:**
```python
# Mask R-CNN mask generation (per-instance)
def maskrcnn_mask(roi_features, mask_head):
    # Extract features for each RoI
    roi_aligned = roi_align(features, proposals)  # [N, C, 14, 14]
    masks = mask_head(roi_aligned)  # [N, K, 28, 28]
    return masks

# YOLACT mask generation (global + coefficients)
def yolact_mask(features, prototypes, coefficients):
    # prototypes: [K, H, W] global masks
    # coefficients: [N, K] per detection
    masks = torch.einsum('nk,khw->nhw', coefficients, prototypes)
    masks = torch.sigmoid(masks)
    return masks

# SOLOv2 mask generation (dynamic convolution)
def solov2_mask(features, kernels):
    # kernels: [N, C] predicted per instance
    # features: [1, C, H, W] mask features
    masks = []
    for kernel in kernels:
        mask = F.conv2d(features, kernel.view(1, -1, 1, 1))
        masks.append(mask)
    return torch.cat(masks)
```

**Speed vs Accuracy Trade-off:**

| Model | mAP | FPS | Use Case |
|-------|-----|-----|----------|
| Mask R-CNN | 37.1 | 5 | Accuracy-critical |
| Cascade Mask R-CNN | 40.0 | 3 | Best accuracy |
| YOLACT | 29.8 | 33 | Real-time |
| YOLACT++ | 34.1 | 27 | Balanced |
| SOLOv2 | 37.5 | 18 | Good balance |

**When to Use:**

| Scenario | Recommendation |
|----------|----------------|
| Real-time video | YOLACT, SOLOv2 |
| High accuracy needed | Mask R-CNN |
| Edge deployment | YOLACT with TensorRT |
| Dense objects | SOLOv2 (better separation) |

**Interview Tip:** YOLACT's innovation is decomposing masks into global prototypes + instance coefficients—very efficient. SOLOv2 uses location-based instance assignment without boxes. Both are faster but struggle with small/overlapping objects compared to Mask R-CNN.

---

### Question 16
- [ ] Done
**How do you handle overlapping instances in dense object arrangements?**

**Answer:**

Overlapping instances challenge segmentation because pixels may belong to multiple objects. Solutions include: learning to predict ordering/depth, using center-based representations, employing occlusion-aware losses, and applying instance-specific feature channels.

**Challenges:**

| Issue | Effect |
|-------|--------|
| Pixel ambiguity | Same pixel assigned to multiple instances |
| Boundary confusion | Edges blend between instances |
| NMS failures | Overlapping boxes suppressed incorrectly |
| Feature mixing | RoI features contaminated by neighbors |

**Approaches:**

| Method | How It Handles Overlap |
|--------|----------------------|
| Instance ordering | Predict depth/layer order |
| Center-based | Use instance centers for separation |
| Amodal segmentation | Predict complete object (including occluded) |
| Embedding-based | Different embeddings for each instance |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapAwareSegmentation(nn.Module):
    """Handle overlapping instances with ordering"""
    
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        
        # Main segmentation head
        self.seg_head = nn.Conv2d(256, num_classes, 1)
        
        # Instance depth/ordering head
        self.depth_head = nn.Conv2d(256, 1, 1)
        
        # Instance center head
        self.center_head = nn.Conv2d(256, 1, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        
        seg = self.seg_head(features)
        depth = self.depth_head(features)
        center = torch.sigmoid(self.center_head(features))
        
        return seg, depth, center
    
    def resolve_overlaps(self, masks, depths):
        """Resolve overlapping masks using depth ordering"""
        H, W = masks.shape[1:]
        final_mask = torch.zeros(H, W, dtype=torch.long)
        
        # Sort instances by depth (front to back)
        sorted_idx = torch.argsort(depths, descending=True)
        
        # Assign pixels to frontmost instance
        for idx in sorted_idx:
            mask = masks[idx] > 0.5
            final_mask[mask] = idx + 1
        
        return final_mask

class CenterBasedInstanceSeg(nn.Module):
    """CenterNet-style instance segmentation"""
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # Center heatmap
        self.center_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Instance embedding
        self.embedding_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1)  # 32-dim embedding
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        centers = self.center_head(features)
        embeddings = self.embedding_head(features)
        
        return centers, embeddings
    
    def group_instances(self, centers, embeddings, threshold=0.5):
        """Group pixels into instances using centers and embeddings"""
        # Find instance centers
        center_mask = centers.squeeze() > threshold
        center_coords = center_mask.nonzero()
        
        if len(center_coords) == 0:
            return torch.zeros_like(centers.squeeze(), dtype=torch.long)
        
        # Get center embeddings
        center_embeds = embeddings[:, center_coords[:, 0], center_coords[:, 1]]
        
        # Assign each pixel to nearest center in embedding space
        H, W = centers.shape[2:]
        pixel_embeds = embeddings.view(embeddings.shape[1], -1).T  # [H*W, D]
        
        distances = torch.cdist(pixel_embeds, center_embeds.T)
        assignments = distances.argmin(dim=1)
        
        return assignments.view(H, W)

class EmbeddingLoss(nn.Module):
    """Loss for instance embeddings (pull together, push apart)"""
    
    def __init__(self, margin=0.5, pull_weight=1.0, push_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pull_weight = pull_weight
        self.push_weight = push_weight
    
    def forward(self, embeddings, instance_masks):
        """
        embeddings: [B, D, H, W]
        instance_masks: [B, H, W] with instance IDs
        """
        B, D, H, W = embeddings.shape
        
        total_loss = 0
        for b in range(B):
            instance_ids = instance_masks[b].unique()
            instance_ids = instance_ids[instance_ids > 0]  # Skip background
            
            centers = []
            for inst_id in instance_ids:
                mask = instance_masks[b] == inst_id
                inst_embeds = embeddings[b, :, mask]  # [D, N]
                center = inst_embeds.mean(dim=1)
                centers.append(center)
                
                # Pull loss: embeddings close to their center
                pull = (inst_embeds - center.unsqueeze(1)).norm(dim=0).mean()
                total_loss += self.pull_weight * pull
            
            # Push loss: centers far from each other
            if len(centers) > 1:
                centers = torch.stack(centers)
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = (centers[i] - centers[j]).norm()
                        push = F.relu(self.margin - dist)
                        total_loss += self.push_weight * push
        
        return total_loss / B

class OcclusionAwareMaskHead(nn.Module):
    """Mask head that reasons about occlusion"""
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Visible mask
        self.visible_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Amodal mask (complete object including occluded parts)
        self.amodal_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Occlusion mask
        self.occlusion_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        visible = self.visible_head(features)
        amodal = self.amodal_head(features)
        occlusion = self.occlusion_head(features)
        
        return visible, amodal, occlusion
```

**Best Practices:**

| Technique | When to Use |
|-----------|-------------|
| Center-based | Circular/compact objects |
| Embedding | Dense, touching objects |
| Depth ordering | Known occlusion patterns |
| Matrix NMS | Replace standard NMS |

**Interview Tip:** For dense objects, standard NMS fails. Use Soft-NMS or Matrix NMS. Embedding-based methods work well but require careful loss balancing. Amodal segmentation helps downstream tasks like tracking.

---

### Question 17
- [ ] Done
**Explain panoptic segmentation and how it combines semantic and instance segmentation.**

**Answer:**

Panoptic segmentation unifies semantic segmentation (stuff: sky, road) with instance segmentation (things: cars, people). Every pixel gets a class label and instance ID. Requires resolving conflicts between overlapping predictions and handling both countable and uncountable objects.

**Task Comparison:**

| Task | Classes | Instance IDs | Output |
|------|---------|--------------|--------|
| Semantic | All pixels | No | Class per pixel |
| Instance | Things only | Yes | Mask per instance |
| Panoptic | All pixels | Things only | Class + instance per pixel |

**Key Concepts:**

- **Things**: Countable objects (person, car) → need instance IDs
- **Stuff**: Amorphous regions (sky, grass) → no instance IDs

**Panoptic Quality (PQ) Metric:**
$$PQ = \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{Recognition\ Quality\ (RQ)} \times \underbrace{\frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP|}}_{Segmentation\ Quality\ (SQ)}$$

**Architecture Approaches:**

```
Approach 1: Separate branches (Panoptic FPN)
Image → Backbone → FPN → Semantic branch → Stuff predictions
                       → Instance branch → Thing predictions
                              ↓
                       Panoptic fusion

Approach 2: Unified (MaskFormer)
Image → Backbone → Transformer → Unified mask predictions
                                       ↓
                               Thing + Stuff together
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PanopticFPN(nn.Module):
    """Panoptic FPN: Separate semantic and instance branches"""
    
    def __init__(self, backbone, num_stuff, num_things):
        super().__init__()
        self.backbone = backbone
        self.num_stuff = num_stuff
        self.num_things = num_things
        
        # FPN
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)
        
        # Semantic segmentation branch (for stuff + things semantics)
        self.semantic_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_stuff + num_things, 1)
        )
        
        # Instance segmentation branch (Mask R-CNN style)
        self.instance_branch = MaskRCNNHead(256, num_things)
    
    def forward(self, x, targets=None):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        
        # Semantic predictions
        semantic_logits = self.semantic_head(fpn_features['p2'])
        semantic_logits = F.interpolate(semantic_logits, x.shape[2:], mode='bilinear')
        
        # Instance predictions
        instance_outputs = self.instance_branch(fpn_features, targets)
        
        return semantic_logits, instance_outputs
    
    def panoptic_fusion(self, semantic_logits, instance_outputs, stuff_threshold=0.5):
        """Merge semantic and instance predictions"""
        H, W = semantic_logits.shape[2:]
        
        # Initialize panoptic output
        panoptic = torch.zeros(H, W, dtype=torch.long)
        segments_info = []
        
        # 1. Add stuff classes (from semantic)
        semantic_pred = semantic_logits.argmax(dim=1).squeeze()
        instance_id = 1
        
        for stuff_id in range(self.num_stuff):
            stuff_mask = semantic_pred == stuff_id
            if stuff_mask.sum() > 0:
                panoptic[stuff_mask] = stuff_id * 1000  # Encode class
                segments_info.append({
                    'id': stuff_id * 1000,
                    'category_id': stuff_id,
                    'isthing': False
                })
        
        # 2. Add things instances (from instance branch)
        for det in instance_outputs:
            mask = det['mask'] > 0.5
            score = det['score']
            class_id = det['label'] + self.num_stuff  # Offset by stuff classes
            
            if score > stuff_threshold:
                # Resolve conflicts (things override stuff)
                panoptic[mask] = class_id * 1000 + instance_id
                segments_info.append({
                    'id': class_id * 1000 + instance_id,
                    'category_id': class_id,
                    'isthing': True
                })
                instance_id += 1
        
        return panoptic, segments_info

class MaskFormer(nn.Module):
    """Unified panoptic segmentation with transformers"""
    
    def __init__(self, backbone, num_classes, num_queries=100):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Pixel decoder
        self.pixel_decoder = PixelDecoder(256)
        
        # Transformer decoder
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(256, 8),
            num_layers=6
        )
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, 256)
        
        # Prediction heads
        self.class_head = nn.Linear(256, num_classes + 1)  # +1 for no-object
        self.mask_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    
    def forward(self, x):
        # Backbone features
        features = self.backbone(x)
        
        # Pixel decoder for high-res features
        pixel_features = self.pixel_decoder(features)
        
        # Flatten for transformer
        B, C, H, W = features[-1].shape
        memory = features[-1].flatten(2).permute(2, 0, 1)
        
        # Transformer decoder
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        output = self.transformer(queries, memory)
        output = output.permute(1, 0, 2)  # [B, N, C]
        
        # Predictions
        class_logits = self.class_head(output)  # [B, N, K+1]
        mask_embeddings = self.mask_head(output)  # [B, N, C]
        
        # Generate masks via dot product
        masks = torch.einsum('bnc,bchw->bnhw', mask_embeddings, pixel_features)
        
        return class_logits, masks
    
    def inference(self, class_logits, masks):
        """Generate panoptic prediction"""
        # Get class predictions
        probs = F.softmax(class_logits, dim=-1)
        scores, labels = probs[..., :-1].max(dim=-1)  # Exclude no-object
        
        # Filter by score
        keep = scores > 0.5
        
        # Combine masks
        H, W = masks.shape[2:]
        panoptic = torch.zeros(H, W, dtype=torch.long)
        
        for idx, (label, mask, score) in enumerate(zip(labels[keep], masks[keep], scores[keep])):
            mask_binary = mask.sigmoid() > 0.5
            panoptic[mask_binary] = label * 1000 + idx + 1
        
        return panoptic

def compute_pq(pred_segments, gt_segments, num_classes):
    """Compute Panoptic Quality"""
    pq_per_class = []
    
    for c in range(num_classes):
        pred_c = [(s, m) for s, m in pred_segments if s['category_id'] == c]
        gt_c = [(s, m) for s, m in gt_segments if s['category_id'] == c]
        
        matched = []
        iou_sum = 0
        
        for ps, pm in pred_c:
            for gs, gm in gt_c:
                iou = compute_iou(pm, gm)
                if iou > 0.5:
                    matched.append((ps, gs, iou))
                    iou_sum += iou
        
        tp = len(matched)
        fp = len(pred_c) - tp
        fn = len(gt_c) - tp
        
        if tp > 0:
            sq = iou_sum / tp
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
            pq = sq * rq
        else:
            pq = 0
        
        pq_per_class.append(pq)
    
    return np.mean(pq_per_class)
```

**Model Comparison:**

| Model | PQ | Approach |
|-------|-----|----------|
| Panoptic FPN | 40.3 | Separate branches |
| MaskFormer | 52.7 | Unified queries |
| Mask2Former | 57.8 | Masked attention |

**Interview Tip:** Panoptic segmentation is now dominated by transformer-based unified approaches (MaskFormer, Mask2Former) that treat all segments equally. The key insight is that things vs stuff distinction can be learned rather than architected.

---

### Question 18
- [ ] Done
**What techniques help with segmenting objects with complex or irregular shapes?**

**Answer:**

Complex shapes (vehicles, furniture, animals) require flexible representations. Techniques include: contour-based methods, polygon regression, deformable convolutions, multi-scale processing, and transformer-based architectures that capture global context.

**Challenges:**

| Shape Type | Challenge | Example |
|------------|-----------|---------|
| Articulated | Non-rigid parts | Person, animal |
| Thin structures | Limited pixels | Poles, wires |
| Concave | Interior gaps | U-shaped objects |
| Complex boundary | Many vertices | Trees, buildings |

**Techniques:**

| Technique | How It Helps |
|-----------|-------------|
| Deformable convolution | Adaptive receptive field |
| Contour regression | Explicit boundary modeling |
| Point-based | Flexible polygon representation |
| GCN on contours | Graph-based boundary refinement |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class DeformableConv2d(nn.Module):
    """Deformable convolution for adaptive receptive field"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Offset prediction
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size,  # x,y offset per position
            kernel_size, stride, padding
        )
        
        # Main convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(x, offset, self.weight, self.bias, 
                            stride=self.stride, padding=self.padding)

class ContourHead(nn.Module):
    """Predict contour points for complex shapes"""
    
    def __init__(self, in_channels, num_points=128):
        super().__init__()
        self.num_points = num_points
        
        # Initial contour prediction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_points * 2)  # x,y per point
        )
        
        # Iterative refinement
        self.refine = nn.Sequential(
            nn.Linear(num_points * 2 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 2)
        )
    
    def forward(self, features, num_iterations=3):
        # Initial contour
        contour = self.initial(features)
        contour = contour.view(-1, self.num_points, 2)
        
        # Iterative refinement
        global_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        for _ in range(num_iterations):
            contour_flat = contour.flatten(1)
            delta = self.refine(torch.cat([contour_flat, global_features], dim=1))
            contour = contour + delta.view(-1, self.num_points, 2)
        
        return contour

class PolygonRNN(nn.Module):
    """Sequential polygon vertex prediction"""
    
    def __init__(self, feature_dim, hidden_dim=256, max_vertices=60):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vertices = max_vertices
        
        # RNN for sequential prediction
        self.rnn = nn.GRU(feature_dim + 2, hidden_dim, batch_first=True)
        
        # Vertex predictor
        self.vertex_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # x, y
        )
        
        # Stop token predictor
        self.stop_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, features, start_vertex):
        B = features.shape[0]
        hidden = torch.zeros(1, B, self.hidden_dim, device=features.device)
        
        vertices = [start_vertex]
        current = start_vertex
        
        for _ in range(self.max_vertices):
            # Combine features with current vertex
            input = torch.cat([features, current], dim=-1).unsqueeze(1)
            
            # RNN step
            output, hidden = self.rnn(input, hidden)
            
            # Predict next vertex
            next_vertex = self.vertex_head(output.squeeze(1))
            stop_prob = torch.sigmoid(self.stop_head(output.squeeze(1)))
            
            if stop_prob.mean() > 0.5:
                break
            
            vertices.append(next_vertex)
            current = next_vertex
        
        return torch.stack(vertices, dim=1)

class MultiScaleShapeHead(nn.Module):
    """Multi-scale processing for complex shapes"""
    
    def __init__(self, num_classes, fpn_channels=256):
        super().__init__()
        
        # Process each FPN level
        self.heads = nn.ModuleList([
            nn.Sequential(
                DeformableConv2d(fpn_channels, 256),
                nn.ReLU(),
                DeformableConv2d(256, 256),
                nn.ReLU(),
                nn.Conv2d(256, num_classes, 1)
            ) for _ in range(4)  # P2-P5
        ])
        
        # Feature fusion
        self.fusion = nn.Conv2d(num_classes * 4, num_classes, 1)
    
    def forward(self, fpn_features):
        outputs = []
        target_size = fpn_features[0].shape[2:]
        
        for feat, head in zip(fpn_features, self.heads):
            out = head(feat)
            out = F.interpolate(out, target_size, mode='bilinear')
            outputs.append(out)
        
        fused = self.fusion(torch.cat(outputs, dim=1))
        return fused

class BoundaryRefinementModule(nn.Module):
    """Refine boundaries for complex shapes"""
    
    def __init__(self, in_channels):
        super().__init__()
        
        # Boundary detection
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # GCN for boundary refinement
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(in_channels, 128),
            GraphConvLayer(128, 64),
            GraphConvLayer(64, 2)  # x,y offset
        ])
    
    def forward(self, features, initial_mask):
        # Detect boundaries
        boundary = self.boundary_conv(features)
        
        # Extract boundary points
        boundary_points = (boundary > 0.5).nonzero()
        
        # Build adjacency (connect nearby points)
        adj = self.build_adjacency(boundary_points)
        
        # GCN refinement
        x = self.sample_features(features, boundary_points)
        for gcn in self.gcn_layers:
            x = gcn(x, adj)
        
        # Apply offsets
        refined_points = boundary_points.float() + x
        
        return refined_points, boundary
```

**Mask Representation Comparison:**

| Representation | Pros | Cons |
|----------------|------|------|
| Dense mask | Simple, standard | Fixed resolution |
| Polygon | Compact, scalable | Fixed num vertices |
| Contour | Flexible | Complex training |
| Implicit (NeRF-style) | Continuous | Slow inference |

**Interview Tip:** Deformable convolutions let the network learn where to sample features, adapting to object shape. For very complex shapes, consider contour-based approaches or iterative refinement. Multi-scale processing is essential for objects with thin parts.

---

### Question 19
- [ ] Done
**How do you optimize Mask R-CNN for real-time applications without significant accuracy loss?**

**Answer:**

Optimize Mask R-CNN via: lighter backbone (MobileNet, EfficientNet), fewer proposals, smaller RoI size, TensorRT optimization, reduced input resolution, and cascade removal. Can achieve 2-5× speedup with <2% accuracy loss.

**Optimization Strategies:**

| Strategy | Speedup | Accuracy Loss | Difficulty |
|----------|---------|---------------|------------|
| Lighter backbone | 2-3× | 2-4% | Low |
| Reduce proposals | 1.5× | 1% | Low |
| Smaller input | 2× | 3-5% | Low |
| TensorRT | 2-3× | 0% | Medium |
| Quantization (INT8) | 2-4× | 1-2% | Medium |
| Knowledge distillation | - | Reduces loss | High |

**Python Implementation:**
```python
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def create_fast_maskrcnn(num_classes, backbone='mobilenet_v3_large'):
    """Create optimized Mask R-CNN"""
    
    if backbone == 'mobilenet_v3_large':
        # MobileNet backbone (faster)
        backbone = torchvision.models.mobilenet_v3_large(pretrained=True).features
        backbone.out_channels = 960
        
        model = MaskRCNN(
            backbone,
            num_classes=num_classes,
            # Reduce proposals
            rpn_pre_nms_top_n_train=1000,  # Default: 2000
            rpn_pre_nms_top_n_test=500,    # Default: 1000
            rpn_post_nms_top_n_train=500,  # Default: 2000
            rpn_post_nms_top_n_test=100,   # Default: 1000
            # Smaller RoI
            box_roi_pool=torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'], output_size=5, sampling_ratio=2  # Default: 7
            ),
            mask_roi_pool=torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'], output_size=10, sampling_ratio=2  # Default: 14
            )
        )
    else:
        # Lighter ResNet
        backbone = resnet_fpn_backbone('resnet18', pretrained=True)
        model = MaskRCNN(backbone, num_classes=num_classes)
    
    return model

class OptimizedMaskRCNN:
    """Wrapper with inference optimizations"""
    
    def __init__(self, model, input_size=512):
        self.model = model
        self.input_size = input_size
        
    def preprocess(self, images):
        """Resize to fixed size for faster inference"""
        resized = []
        for img in images:
            h, w = img.shape[1:]
            scale = self.input_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized.append(F.interpolate(
                img.unsqueeze(0), (new_h, new_w), mode='bilinear'
            ).squeeze(0))
        
        return resized
    
    @torch.no_grad()
    def predict(self, images, score_threshold=0.7):
        """Optimized inference"""
        self.model.eval()
        
        # Preprocess
        images = self.preprocess(images)
        
        # Inference
        outputs = self.model(images)
        
        # Filter by score
        filtered = []
        for out in outputs:
            keep = out['scores'] > score_threshold
            filtered.append({k: v[keep] for k, v in out.items()})
        
        return filtered

def export_to_tensorrt(model, input_shape, output_path):
    """Export to TensorRT for maximum speed"""
    import torch_tensorrt
    
    model.eval()
    
    # Trace model
    example_input = torch.randn(input_shape).cuda()
    
    # Compile with TensorRT
    trt_model = torch_tensorrt.compile(
        model.cuda(),
        inputs=[torch_tensorrt.Input(input_shape)],
        enabled_precisions={torch.half},  # FP16
        workspace_size=1 << 30  # 1GB
    )
    
    torch.jit.save(trt_model, output_path)
    return trt_model

def quantize_maskrcnn(model, calibration_data):
    """INT8 quantization"""
    from torch.quantization import quantize_dynamic, prepare_qat
    
    # Dynamic quantization (simple)
    quantized = quantize_dynamic(
        model.backbone,
        {torch.nn.Conv2d, torch.nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized

# Knowledge distillation for accuracy recovery
class DistillationLoss(nn.Module):
    """Distill from large to small Mask R-CNN"""
    
    def __init__(self, teacher, student, temperature=4.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
    
    def forward(self, images, targets):
        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(images)
        
        # Get student predictions
        student_outputs = self.student(images, targets)
        
        # Task loss
        task_loss = sum(student_outputs.values())
        
        # Distillation losses
        distill_loss = 0
        
        # Feature distillation
        teacher_feats = self.teacher.backbone(images)
        student_feats = self.student.backbone(images)
        
        for t_feat, s_feat in zip(teacher_feats.values(), student_feats.values()):
            distill_loss += F.mse_loss(s_feat, t_feat)
        
        return task_loss + 0.5 * distill_loss

# Benchmark different configurations
def benchmark_configs():
    configs = [
        {'backbone': 'resnet50', 'proposals': 1000, 'input': 800},  # Baseline
        {'backbone': 'resnet18', 'proposals': 1000, 'input': 800},  # Lighter
        {'backbone': 'resnet50', 'proposals': 300, 'input': 512},   # Faster
        {'backbone': 'mobilenet', 'proposals': 300, 'input': 512},  # Fastest
    ]
    
    results = []
    for cfg in configs:
        model = create_model(cfg)
        fps = measure_fps(model)
        ap = evaluate_ap(model)
        results.append({'config': cfg, 'fps': fps, 'ap': ap})
    
    return results
```

**Performance Comparison:**

| Configuration | FPS | Mask AP |
|---------------|-----|---------|
| ResNet-50 FPN, 1000 props | 5 | 37.1 |
| ResNet-18 FPN, 500 props | 12 | 33.5 |
| MobileNet, 300 props | 22 | 29.8 |
| MobileNet + TensorRT | 35 | 29.8 |

**Optimization Checklist:**
1. ✓ Use lighter backbone (MobileNet/EfficientNet)
2. ✓ Reduce proposal count (300-500)
3. ✓ Lower input resolution (512-640)
4. ✓ Apply TensorRT optimization
5. ✓ Use FP16 inference
6. ✓ Batch processing when possible

**Interview Tip:** For production, always export to TensorRT or ONNX Runtime. Backbone choice has the biggest impact—MobileNet is 4× faster than ResNet-50. Reduce proposals carefully; too few hurts recall on small objects.

---

## Vision Transformers (ViT)

### Question 20
- [ ] Done
**Explain the core innovation of Vision Transformers compared to CNNs. What inductive biases does ViT lack?**

**Answer:**

ViT treats images as sequences of patches and applies transformer attention, enabling global context from the first layer. Unlike CNNs, ViT lacks built-in spatial locality, translation equivariance, and hierarchical structure—requiring more data to learn these properties from scratch.

**CNN vs ViT Comparison:**

| Aspect | CNN | ViT |
|--------|-----|-----|
| Receptive field | Local → Global (gradual) | Global from layer 1 |
| Inductive bias | Locality, equivariance | None (learned) |
| Data efficiency | Good with small data | Needs large data |
| Scalability | Limited by depth | Scales well |
| Long-range | Requires deep networks | Native |

**CNN Inductive Biases ViT Lacks:**

| Bias | What It Means | Effect Without It |
|------|---------------|-------------------|
| Locality | Nearby pixels are related | Must learn local patterns |
| Translation equivariance | Same filter everywhere | Must learn at each position |
| Hierarchy | Low→high level features | Flat representation |
| Scale invariance | Multi-resolution | Fixed patch size |

**ViT Architecture:**
```
Image (224×224×3)
      ↓ Split into patches (14×14 patches of 16×16)
[P1][P2][P3]...[P196]
      ↓ Linear projection + position embedding
[E1][E2][E3]...[E196] + [CLS]
      ↓ Transformer Encoder (12 layers)
[O1][O2][O3]...[O196] + [CLS']
      ↓ Take [CLS'] output
Classification
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """Convert image to sequence of patch embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.projection(x)  # [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, E]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, E]
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.encoder(x)
        
        # Classification from CLS token
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x

# Comparison: CNN vs ViT attention patterns
def compare_receptive_fields():
    """
    CNN: Layer 1 sees 3×3, Layer 12 sees ~50×50 (gradual)
    ViT: Layer 1 sees entire image (global attention)
    """
    print("CNN receptive field growth: 3→7→15→31→63→127...")
    print("ViT receptive field: 224×224 from layer 1")

# Why ViT needs more data
def data_requirement_intuition():
    """
    CNN: Prior knowledge (locality) reduces search space
    ViT: Must learn everything from data
    
    Like teaching someone to read:
    - CNN: Given alphabet + rules (inductive bias)
    - ViT: Given nothing, must discover patterns
    """
    pass
```

**When CNN Biases Help/Hurt:**

| Scenario | CNN Advantage | ViT Advantage |
|----------|---------------|---------------|
| Small dataset | Better generalization | Overfits |
| Large dataset | Plateaus | Keeps improving |
| Translation variance | Struggles | Learns it |
| Global context | Needs depth | Native |

**Data Requirements:**

| Model | Dataset Size | Top-1 Accuracy |
|-------|--------------|----------------|
| ViT-B/16 | ImageNet-1K | 77.9% |
| ViT-B/16 | ImageNet-21K | 84.0% |
| ViT-L/16 | JFT-300M | 87.8% |
| ResNet-50 | ImageNet-1K | 76.2% |

**Interview Tip:** ViT's lack of inductive bias is both weakness (needs data) and strength (no limiting assumptions). Pre-training on large data, then fine-tuning transfers the learned biases. DeiT shows how to train ViT efficiently on ImageNet-1K using distillation.

---

### Question 21
- [ ] Done
**How are images converted into patch embeddings in ViT? Explain the linear projection layer.**

**Answer:**

ViT divides images into fixed-size patches (e.g., 16×16), flattens each patch into a vector, and applies a linear projection to create embeddings. This is equivalent to a convolution with kernel_size=stride=patch_size, transforming spatial data into a sequence.

**Patch Embedding Process:**

```
Input: 224×224×3 image
    ↓ Divide into 14×14 grid of 16×16 patches
196 patches of size 16×16×3
    ↓ Flatten each patch
196 vectors of size 768 (16×16×3)
    ↓ Linear projection
196 embeddings of size D (e.g., 768)
```

**Mathematical Formulation:**

For patch $p_i$ of size $P \times P \times C$:
$$z_i = p_i \cdot W_E + b_E$$

Where:
- $p_i \in \mathbb{R}^{P^2 \cdot C}$ (flattened patch)
- $W_E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ (projection matrix)
- $z_i \in \mathbb{R}^{D}$ (patch embedding)

**Equivalent Conv2d:**
```python
# These are equivalent:
# Method 1: Manual flatten + linear
patch = image[:, y:y+P, x:x+P].flatten()
embedding = linear(patch)

# Method 2: Conv2d (more efficient)
embedding = conv2d(image, kernel_size=P, stride=P)
```

**Python Implementation:**
```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Method 1: Explicit linear projection
        self.flatten_dim = patch_size * patch_size * in_channels
        self.linear_proj = nn.Linear(self.flatten_dim, embed_dim)
        
        # Method 2: Equivalent Conv2d (used in practice)
        self.conv_proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward_manual(self, x):
        """Manual implementation for understanding"""
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Extract patches
        patches = []
        for i in range(0, H, P):
            for j in range(0, W, P):
                patch = x[:, :, i:i+P, j:j+P]  # [B, C, P, P]
                patch = patch.flatten(1)  # [B, C*P*P]
                patches.append(patch)
        
        # Stack and project
        patches = torch.stack(patches, dim=1)  # [B, N, C*P*P]
        embeddings = self.linear_proj(patches)  # [B, N, D]
        
        return embeddings
    
    def forward(self, x):
        """Efficient implementation using Conv2d"""
        # x: [B, C, H, W]
        x = self.conv_proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        return x

class HybridPatchEmbedding(nn.Module):
    """Use CNN features instead of raw patches"""
    
    def __init__(self, cnn_backbone, embed_dim=768):
        super().__init__()
        self.backbone = cnn_backbone
        self.proj = nn.Conv2d(cnn_backbone.out_channels, embed_dim, 1)
    
    def forward(self, x):
        # CNN feature extraction
        features = self.backbone(x)  # [B, C, H', W']
        
        # Project to embedding dimension
        embeddings = self.proj(features)  # [B, D, H', W']
        embeddings = embeddings.flatten(2).transpose(1, 2)  # [B, N, D]
        
        return embeddings

# Verify equivalence
def verify_equivalence():
    img = torch.randn(1, 3, 224, 224)
    
    # Method 1: Manual
    patches = []
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            patches.append(img[:, :, i:i+16, j:j+16].flatten(1))
    patches = torch.stack(patches, dim=1)  # [1, 196, 768]
    
    # Method 2: Conv2d
    conv = nn.Conv2d(3, 768, 16, 16, bias=False)
    conv_out = conv(img).flatten(2).transpose(1, 2)  # [1, 196, 768]
    
    # Shapes match
    assert patches.shape == conv_out.shape == (1, 196, 768)
```

**Key Design Choices:**

| Choice | Impact |
|--------|--------|
| Patch size 16 | 14×14=196 tokens for 224 input |
| Patch size 32 | 7×7=49 tokens (faster, less detail) |
| Patch size 8 | 28×28=784 tokens (slower, more detail) |
| Overlap | Smoother but more tokens |

**Patch Size Trade-offs:**

| Patch Size | Tokens (224 input) | Memory | Detail |
|------------|-------------------|--------|--------|
| 8 | 784 | High | High |
| 16 | 196 | Medium | Medium |
| 32 | 49 | Low | Low |

**Interview Tip:** The linear projection is conceptually a learned "bag of pixels" representation. Using Conv2d is more efficient and functionally identical. Smaller patches capture more detail but increase sequence length quadratically, affecting transformer complexity.

---

### Question 22
- [ ] Done
**What is the role of the [CLS] token and positional encodings in Vision Transformers?**

**Answer:**

The [CLS] token aggregates global image information for classification—it attends to all patches and its final representation is used for prediction. Positional encodings provide spatial information since transformers are permutation-invariant and don't inherently know patch positions.

**[CLS] Token:**

```
Input:  [CLS] [P1] [P2] ... [P196]
           ↓    ↓    ↓        ↓
Transformer layers (attention between all)
           ↓    ↓    ↓        ↓
Output: [CLS'] [P1'] [P2'] ... [P196']
           ↓
    Classification head
```

**Purpose:**
- Attends to all patches during forward pass
- Aggregates global representation
- Single output for classification

**Positional Encodings:**

Without positions, transformer sees:
```
[P1] [P2] [P3] same as [P3] [P1] [P2]
```

With positions:
```
[P1 + pos1] [P2 + pos2] [P3 + pos3] ≠ [P3 + pos3] [P1 + pos1] [P2 + pos2]
```

**Types of Position Encoding:**

| Type | Method | Properties |
|------|--------|------------|
| Learned | Trainable parameters | Flexible, data-driven |
| Sinusoidal | Fixed sin/cos functions | Generalizes to longer sequences |
| Relative | Attention bias | Position-aware attention |
| 2D | Separate row/col | Better for images |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import math

class ViTWithPositions(nn.Module):
    """ViT with CLS token and positional encoding"""
    
    def __init__(self, num_patches=196, embed_dim=768, num_classes=1000):
        super().__init__()
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding (learnable) - includes CLS position
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, patch_embeddings):
        B = patch_embeddings.shape[0]
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        
        # Prepend CLS token
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)  # [B, N+1, D]
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # ... transformer layers ...
        
        # Use CLS token for classification
        cls_output = x[:, 0]  # [B, D]
        return self.head(cls_output)

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding"""
    
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        
        # Create position encoding matrix
        position = torch.arange(num_patches + 1).unsqueeze(1)  # +1 for CLS
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                            (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(1, num_patches + 1, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Learned2DPositionalEncoding(nn.Module):
    """Separate learned encodings for row and column"""
    
    def __init__(self, grid_size=14, embed_dim=768):
        super().__init__()
        self.grid_size = grid_size
        
        # Separate row and column embeddings
        self.row_embed = nn.Parameter(torch.zeros(1, grid_size, embed_dim // 2))
        self.col_embed = nn.Parameter(torch.zeros(1, grid_size, embed_dim // 2))
        
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)
    
    def forward(self, x):
        B, N, D = x.shape
        
        # Create 2D position encoding
        pos = torch.zeros(1, N, D, device=x.device)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if idx < N:  # Skip if CLS token included
                    pos[0, idx, :D//2] = self.row_embed[0, i]
                    pos[0, idx, D//2:] = self.col_embed[0, j]
        
        return x + pos

# Visualizing positional encoding similarity
def visualize_position_similarity(pos_embed):
    """Show which positions attend to each other"""
    # Cosine similarity between position embeddings
    pos = pos_embed[0, 1:]  # Exclude CLS
    similarity = torch.cosine_similarity(
        pos.unsqueeze(0), pos.unsqueeze(1), dim=2
    )
    
    # Reshape to 2D grid
    grid_size = int(math.sqrt(len(pos)))
    similarity = similarity.view(grid_size, grid_size, grid_size, grid_size)
    
    return similarity

# CLS token alternatives
class GlobalAveragePooling(nn.Module):
    """Alternative to CLS: average all patch tokens"""
    
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x: [B, N, D] (no CLS token)
        pooled = x.mean(dim=1)  # [B, D]
        return self.head(pooled)
```

**CLS Token vs Global Average Pooling:**

| Approach | Pros | Cons |
|----------|------|------|
| CLS token | Learnable aggregation | Extra token |
| GAP | Simpler, no extra token | Less flexible |

**Position Encoding Analysis:**

Learned position embeddings often show:
- Similar embeddings for nearby patches
- 2D structure emerges during training
- CLS position is distinct from patches

**Interview Tip:** CLS token is borrowed from BERT—it's an aggregation point that learns to combine patch information. Position encodings are crucial; without them, ViT would be invariant to patch order (shuffled patches would give same output). Learned positions work well for fixed-size inputs.

---

### Question 23
- [ ] Done
**Explain the computational complexity of ViT (O(n²) attention) and how it limits resolution scalability.**

**Answer:**

Self-attention computes pairwise interactions between all tokens, giving O(n²) complexity where n = number of patches. For images, n grows quadratically with resolution, making high-resolution inputs prohibitively expensive. This limits ViT's use for dense prediction tasks.

**Complexity Analysis:**

| Component | Complexity | Memory |
|-----------|-----------|--------|
| Attention | O(n²·d) | O(n²) |
| MLP | O(n·d²) | O(n·d) |
| Total | O(n²·d + n·d²) | O(n²) |

Where:
- n = number of tokens (patches)
- d = embedding dimension

**Resolution Scaling Problem:**

| Resolution | Patch Size | Tokens (n) | Attention Cost (n²) |
|------------|------------|------------|---------------------|
| 224×224 | 16 | 196 | 38K |
| 384×384 | 16 | 576 | 332K |
| 512×512 | 16 | 1024 | 1M |
| 1024×1024 | 16 | 4096 | 16.7M |

**Memory grows 43× from 224 to 1024!**

**Python Demonstration:**
```python
import torch
import torch.nn as nn
import time

def attention_complexity_demo():
    """Demonstrate O(n²) complexity"""
    
    embed_dim = 768
    num_heads = 12
    attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    results = []
    for n in [196, 576, 1024, 4096]:
        x = torch.randn(1, n, embed_dim)
        
        # Measure time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        for _ in range(10):
            _ = attention(x, x, x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        results.append({'tokens': n, 'time': elapsed, 'n_squared': n*n})
        print(f"n={n}: time={elapsed:.4f}s, n²={n*n}")
    
    return results

class StandardAttention(nn.Module):
    """Standard O(n²) self-attention"""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        
        # Attention: O(n²) operation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        
        # Memory: storing N×N attention matrix per head
        # This is the bottleneck!
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

# Solutions to O(n²) problem
class LinearAttention(nn.Module):
    """O(n) linear attention (approximation)"""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Kernel trick: φ(Q)φ(K)ᵀV instead of softmax(QKᵀ)V
        # Using ELU+1 as feature map
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # O(n) computation: (K'V) then Q @ (K'V)
        kv = k.transpose(-2, -1) @ v  # [B, H, D, D]
        x = (q @ kv) / (q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1))
        
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class WindowedAttention(nn.Module):
    """O(n·w²) windowed attention (Swin-style)"""
    
    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.attention = StandardAttention(embed_dim, num_heads)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        
        # Partition into windows
        ws = self.window_size
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        
        # Attention within windows
        x = self.attention(x)
        
        # Merge windows back
        x = x.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, N, C)
        
        return x
```

**Solutions to Scalability:**

| Method | Complexity | Approach |
|--------|------------|----------|
| Swin Transformer | O(n·w²) | Windowed attention |
| Linear Attention | O(n·d) | Kernel approximation |
| Performer | O(n·d) | Random features |
| Pooling ViT | O((n/k)²) | Downsample tokens |
| FlashAttention | O(n²) but fast | Memory-efficient |

**Practical Implications:**

| Task | Challenge | Solution |
|------|-----------|----------|
| Classification | 224² ok | Standard ViT |
| Segmentation | Need high-res | Swin, windowed |
| Detection | Multi-scale | Hierarchical ViT |

**Interview Tip:** O(n²) is the key limitation of ViT for dense tasks. Swin Transformer solves this with windowed attention (O(n·w²)). For classification with fixed 224² input, it's not a problem. FlashAttention doesn't change complexity but dramatically reduces memory through clever tiling.

---

### Question 24
- [ ] Done
**What are the data requirements for training ViT from scratch vs. using pre-trained models?**

**Answer:**

ViT requires massive data (100M+ images) to train from scratch due to lack of inductive biases. Pre-trained models transfer well to smaller datasets. ImageNet-1K alone is insufficient for scratch training, but works well for fine-tuning pre-trained ViT.

**Data Requirements:**

| Training Approach | Minimum Data | Performance |
|------------------|--------------|-------------|
| ViT from scratch | 100M+ images | Competitive |
| ViT on ImageNet-1K only | 1.2M | Poor (77%) |
| Pre-trained + fine-tune | 1K+ images | Excellent |
| DeiT (with tricks) | 1.2M | Good (81%) |

**Why ViT Needs More Data:**

| Factor | CNN | ViT |
|--------|-----|-----|
| Locality bias | Built-in | Must learn |
| Translation equiv. | Built-in | Must learn |
| Hierarchical | Built-in | Must learn |
| Effective parameters | Shared filters | Position-specific |

**Pre-training Datasets:**

| Dataset | Size | ViT-B Accuracy |
|---------|------|----------------|
| ImageNet-1K | 1.2M | 77.9% |
| ImageNet-21K | 14M | 84.0% |
| JFT-300M | 300M | 87.8% |

**Python Implementation:**
```python
import torch
import timm
from torch.utils.data import DataLoader
import torch.optim as optim

def load_pretrained_vit(num_classes, pretrained_dataset='imagenet21k'):
    """Load pre-trained ViT for fine-tuning"""
    
    if pretrained_dataset == 'imagenet21k':
        model = timm.create_model(
            'vit_base_patch16_224_in21k',
            pretrained=True,
            num_classes=num_classes
        )
    elif pretrained_dataset == 'imagenet1k':
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=num_classes
        )
    
    return model

def finetune_vit(model, train_loader, val_loader, num_epochs=30, lr=1e-4):
    """Fine-tune pre-trained ViT on downstream task"""
    
    # Freeze backbone initially (optional)
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
    
    # Train only head for warmup
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 10  # Higher LR for head
    )
    
    for epoch in range(5):
        train_epoch(model, train_loader, optimizer)
    
    # Unfreeze all and fine-tune
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - 5)
    
    for epoch in range(num_epochs - 5):
        train_epoch(model, train_loader, optimizer)
        scheduler.step()
        validate(model, val_loader)

def data_augmentation_for_small_datasets():
    """Strong augmentation when data is limited"""
    from torchvision import transforms
    from timm.data.auto_augment import rand_augment_transform
    
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        rand_augment_transform('rand-m9-mstd0.5', {}),  # RandAugment
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),  # CutOut
    ])

# Training from scratch (requires massive data)
def train_vit_from_scratch(model, dataset):
    """
    Requirements for training from scratch:
    - 100M+ images (JFT-300M, LAION)
    - Strong augmentation
    - Long training (300+ epochs)
    - Large batch sizes (4096+)
    """
    
    if len(dataset) < 10_000_000:
        print("Warning: ViT may not train well with < 10M images")
        print("Consider using pre-trained model or DeiT approach")
    
    # Heavy regularization
    model = add_dropout(model, rate=0.1)
    
    # MixUp + CutMix
    mixup = MixUp(alpha=0.8)
    cutmix = CutMix(alpha=1.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.3)
    
    for epoch in range(300):  # Long training
        for images, labels in train_loader:
            # Apply MixUp or CutMix
            if random.random() > 0.5:
                images, labels = mixup(images, labels)
            else:
                images, labels = cutmix(images, labels)
            
            loss = train_step(model, images, labels)

# Small dataset strategies
class SmallDatasetStrategies:
    """Techniques for training ViT on small datasets"""
    
    @staticmethod
    def use_pretrained():
        """Best approach: use pre-trained model"""
        return timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
    
    @staticmethod
    def use_deit_approach():
        """DeiT: Knowledge distillation from CNN"""
        teacher = timm.create_model('regnet_y_160', pretrained=True)
        student = timm.create_model('deit_small_patch16_224', pretrained=False)
        return teacher, student
    
    @staticmethod
    def use_hybrid():
        """Hybrid: CNN backbone + Transformer head"""
        return timm.create_model('vit_base_resnet50_384', pretrained=True)
    
    @staticmethod
    def heavy_augmentation():
        """RandAugment + MixUp + CutMix + CutOut"""
        pass
```

**Transfer Learning Guidelines:**

| Source Dataset | Target Size | Strategy |
|---------------|-------------|----------|
| ImageNet-21K | Any | Fine-tune all layers |
| ImageNet-21K | < 1K images | Freeze backbone, train head |
| ImageNet-1K | Medium | Fine-tune with lower LR |
| None | < 1M | Don't train ViT from scratch |

**Small Dataset Alternatives:**

| Method | Data Needed | Accuracy |
|--------|-------------|----------|
| Pre-trained ViT | 1K+ | High |
| DeiT + distillation | 10K+ | Good |
| Hybrid CNN-ViT | 50K+ | Good |
| Pure ViT from scratch | 10M+ | Variable |

**Interview Tip:** Never train ViT from scratch on small datasets. Pre-training on ImageNet-21K or larger, then fine-tuning is the standard approach. DeiT showed that distillation from CNN teachers can bridge the gap on ImageNet-1K. Strong augmentation (RandAugment, MixUp, CutMix) is essential for smaller datasets.

---

### Question 25
- [ ] Done
**Explain DeiT (Data-efficient Image Transformers) and how knowledge distillation improves ViT training on smaller datasets.**

**Answer:**

DeiT enables training ViT on ImageNet-1K (1.2M images) by using knowledge distillation from a CNN teacher, strong augmentation, and a distillation token. The teacher's soft labels provide richer supervision than hard labels alone, transferring CNN's inductive biases to the transformer.

**Key DeiT Innovations:**

| Innovation | Purpose |
|------------|---------|
| Distillation token | Separate token for learning from teacher |
| Hard distillation | Match teacher's predictions |
| CNN teacher | Transfer locality/hierarchy biases |
| Strong augmentation | RandAugment, MixUp, CutMix |

**Architecture:**
```
Input: [CLS] [DIST] [P1] [P2] ... [P196]
                ↓
        Transformer Encoder
                ↓
Output: [CLS'] [DIST'] [P1'] [P2'] ...
           ↓       ↓
      Class Head  Distill Head
           ↓       ↓
      Class Loss  Distill Loss
```

**Distillation Types:**

| Type | Teacher Output | Student Loss |
|------|---------------|--------------|
| Soft distillation | Soft probabilities | KL divergence |
| Hard distillation | Hard labels (argmax) | Cross-entropy |

DeiT uses **hard distillation** (surprisingly works better).

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DeiT(nn.Module):
    """Data-efficient Image Transformer with distillation"""
    
    def __init__(self, num_patches=196, embed_dim=768, num_classes=1000, depth=12):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        
        # CLS and distillation tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # NEW!
        
        # Position embedding (includes CLS and DIST)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 12, embed_dim * 4, 
                                       activation='gelu', batch_first=True),
            num_layers=depth
        )
        
        # Two heads: classification and distillation
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)  # NEW!
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS and distillation tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Get outputs from both tokens
        cls_output = x[:, 0]
        dist_output = x[:, 1]
        
        # Classification outputs
        x_cls = self.head(cls_output)
        x_dist = self.head_dist(dist_output)
        
        if self.training:
            return x_cls, x_dist
        else:
            # Average at inference
            return (x_cls + x_dist) / 2

class DistillationLoss(nn.Module):
    """Loss for DeiT training"""
    
    def __init__(self, teacher, alpha=0.5, temperature=3.0, hard_distill=True):
        super().__init__()
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.hard_distill = hard_distill
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, student_cls, student_dist, images, targets):
        # Get teacher predictions
        with torch.no_grad():
            teacher_output = self.teacher(images)
        
        # Classification loss (from CLS token)
        cls_loss = F.cross_entropy(student_cls, targets)
        
        # Distillation loss (from DIST token)
        if self.hard_distill:
            # Hard labels from teacher
            teacher_labels = teacher_output.argmax(dim=1)
            dist_loss = F.cross_entropy(student_dist, teacher_labels)
        else:
            # Soft labels with temperature
            soft_targets = F.softmax(teacher_output / self.temperature, dim=1)
            soft_student = F.log_softmax(student_dist / self.temperature, dim=1)
            dist_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
            dist_loss = dist_loss * (self.temperature ** 2)
        
        # Combined loss
        return (1 - self.alpha) * cls_loss + self.alpha * dist_loss

def train_deit():
    """Train DeiT with knowledge distillation"""
    
    # Teacher: pre-trained CNN (RegNet works well)
    teacher = timm.create_model('regnetx_160', pretrained=True).eval()
    
    # Student: DeiT
    student = DeiT(num_classes=1000)
    
    # Loss
    criterion = DistillationLoss(teacher, alpha=0.5, hard_distill=True)
    
    # Strong augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugment(2, 9),  # RandAugment
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # MixUp and CutMix
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.05)
    
    for epoch in range(300):
        for images, targets in train_loader:
            images, targets = mixup(images, targets)
            
            cls_out, dist_out = student(images)
            loss = criterion(cls_out, dist_out, images, targets)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

**DeiT Results:**

| Model | Teacher | Top-1 Acc | Training |
|-------|---------|-----------|----------|
| ViT-B (no distill) | - | 77.9% | JFT-300M |
| DeiT-B (no distill) | - | 79.8% | ImageNet-1K |
| DeiT-B (distilled) | RegNetY-16G | 83.4% | ImageNet-1K |

**Why Hard Distillation Works:**
- CNN teacher has high confidence on correct class
- Hard labels are "cleaned" by teacher's inductive bias
- Prevents student from overfitting to data noise

**Interview Tip:** DeiT's key insight is that CNN teachers effectively inject inductive biases into ViT training. The distillation token learns differently from CLS—it focuses on what the teacher knows. Hard distillation outperforms soft, likely because confident teacher predictions provide cleaner supervision.

---

### Question 26
- [ ] Done
**Describe Masked Autoencoder (MAE) pre-training for Vision Transformers. How does masking 75% of patches work?**

**Answer:**

MAE pre-trains ViT by masking 75% of image patches and reconstructing them. The encoder processes only visible patches (efficient), while the decoder reconstructs missing patches. This forces the model to learn meaningful representations by understanding image structure.

**MAE Pipeline:**
```
Image → Split patches → Random mask 75% → Encode visible → Decode all → Reconstruct
         196 patches     ~50 visible      ViT encoder      Light decoder   Loss on masked
```

**Key Design Choices:**

| Choice | Value | Reason |
|--------|-------|--------|
| Mask ratio | 75% | High redundancy in images |
| Encoder | ViT | Processes visible only |
| Decoder | Shallow ViT | Reconstruct + lightweight |
| Target | Raw pixels | Simple, effective |

**Why 75% Masking Works:**

Images have high spatial redundancy—neighboring patches are similar. Masking 75% forces the model to truly understand structure rather than interpolate from neighbors.

| Mask Ratio | Task Difficulty | Quality |
|------------|-----------------|---------|
| 25% | Too easy | Poor representations |
| 50% | Medium | Okay |
| 75% | Hard | Best representations |
| 90% | Too hard | Underfitting |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAE(nn.Module):
    """Masked Autoencoder for Vision Transformers"""
    
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 encoder_depth=12, decoder_depth=4, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Encoder (standard ViT)
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 12, embed_dim * 4, 
                                       activation='gelu', batch_first=True),
            num_layers=encoder_depth
        )
        
        # Decoder (lightweight)
        decoder_dim = 512
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(decoder_dim, 8, decoder_dim * 4,
                                       activation='gelu', batch_first=True),
            num_layers=decoder_depth
        )
        
        # Reconstruction head
        self.pred_head = nn.Linear(decoder_dim, patch_size ** 2 * 3)
    
    def random_masking(self, x):
        """Random mask patches"""
        B, N, D = x.shape
        num_keep = int(N * (1 - self.mask_ratio))
        
        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, 
                                index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Binary mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Masking
        x, mask, ids_restore = self.random_masking(x)
        
        # Encode only visible patches
        x = self.encoder(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        B, N_visible, _ = x.shape
        
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(B, self.num_patches - N_visible, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to original order
        x_full = torch.gather(x_full, dim=1,
                             index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))
        
        # Add position embedding
        x_full = x_full + self.decoder_pos_embed
        
        # Decode
        x_full = self.decoder(x_full)
        
        # Predict pixels
        pred = self.pred_head(x_full)
        
        return pred
    
    def forward(self, x):
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask
    
    def compute_loss(self, x, pred, mask):
        """MSE loss on masked patches only"""
        # Patchify target
        target = self.patchify(x)
        
        # Normalize target
        mean = target.mean(dim=-1, keepdim=True)
        std = target.std(dim=-1, keepdim=True)
        target = (target - mean) / (std + 1e-6)
        
        # Loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Per patch
        loss = (loss * mask).sum() / mask.sum()  # Average over masked
        
        return loss
    
    def patchify(self, x):
        """Convert image to patches"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, (H // p) * (W // p), p * p * C)
        return x

def train_mae(model, train_loader, epochs=400):
    """Pre-train MAE"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, 
                                   betas=(0.9, 0.95), weight_decay=0.05)
    
    for epoch in range(epochs):
        for images, _ in train_loader:  # Labels not used!
            pred, mask = model(images)
            loss = model.compute_loss(images, pred, mask)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def finetune_from_mae(mae_model, num_classes):
    """Fine-tune encoder for classification"""
    # Take only encoder
    encoder = mae_model.encoder
    
    # Add classification head
    model = nn.Sequential(
        mae_model.patch_embed,
        # Add CLS token handling
        encoder,
        nn.LayerNorm(768),
        nn.Linear(768, num_classes)
    )
    
    return model
```

**MAE vs Other Pre-training:**

| Method | Pretext Task | Mask Ratio | Efficiency |
|--------|-------------|------------|------------|
| MAE | Reconstruction | 75% | Very high |
| BEiT | Token prediction | 40% | Medium |
| SimMIM | Reconstruction | 60% | High |
| Contrastive | Instance discrimination | 0% | Low |

**Why MAE is Efficient:**
- Encoder sees only 25% of patches (4× faster)
- Decoder is shallow
- No negative samples needed

**Interview Tip:** MAE's key insight is that images have high redundancy, so 75% masking is needed to make the task hard. The asymmetric encoder-decoder (heavy encoder, light decoder) is efficient. Pre-trained representations transfer well to downstream tasks.

---

### Question 27
- [ ] Done
**How do hybrid architectures combine CNN feature extraction with transformer attention?**

**Answer:**

Hybrid architectures combine CNN's efficient local feature extraction with Transformer's global attention. CNNs reduce spatial resolution while capturing local patterns; Transformers then model long-range dependencies on the reduced feature map.

**Design Strategies:**

| Strategy | Approach | Example |
|----------|----------|---------|
| CNN backbone + Transformer head | CNN extracts features, Transformer refines | ViT with ResNet stem |
| Interleaved | Alternate CNN and attention layers | CoAtNet, LeViT |
| Parallel | Both branches, fuse outputs | Conformer |
| CNN tokenizer | Replace patch embedding with CNN | Early Convolutions ViT |

**Why Hybrid Works:**

| Component | Strength | Limitation |
|-----------|----------|------------|
| CNN | Local features, translation equivariance | Limited receptive field |
| Transformer | Global context, flexible | Quadratic complexity, no inductive bias |
| **Hybrid** | **Best of both** | **Slightly more complex** |

**Python Implementation:**
```python
import torch
import torch.nn as nn

class HybridViT(nn.Module):
    """CNN backbone + Transformer"""
    
    def __init__(self, num_classes=1000, embed_dim=768):
        super().__init__()
        
        # CNN backbone (ResNet stages)
        self.cnn_backbone = nn.Sequential(
            # Stem
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Stage 1-3: 224 -> 14x14
            self._make_stage(64, 256, 3, stride=1),
            self._make_stage(256, 512, 4, stride=2),
            self._make_stage(512, 1024, 6, stride=2),
        )
        
        # Project to transformer dimension
        self.proj = nn.Conv2d(1024, embed_dim, 1)
        
        # Transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 14*14 + 1, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 12, embed_dim * 4,
                                       activation='gelu', batch_first=True),
            num_layers=12
        )
        
        self.head = nn.Linear(embed_dim, num_classes)
    
    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        layers = [Bottleneck(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # CNN: 224x224 -> 14x14
        x = self.cnn_backbone(x)  # [B, 1024, 14, 14]
        
        # Project and flatten
        x = self.proj(x)  # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]
        
        # Add CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 197, 768]
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        return self.head(x[:, 0])


class EarlyConvViT(nn.Module):
    """Replace patch embedding with convolutional stem"""
    
    def __init__(self, num_classes=1000, embed_dim=768):
        super().__init__()
        
        # Convolutional tokenizer
        self.tokenizer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, 3, stride=2, padding=1),
        )
        
        # Standard ViT components
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 12, embed_dim * 4,
                                       activation='gelu', batch_first=True),
            num_layers=12
        )
        
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.tokenizer(x)  # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]
        
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        return self.head(x[:, 0])
```

**Comparison:**

| Model | ImageNet Acc | Design |
|-------|-------------|--------|
| ViT-B | 77.9% | Pure Transformer |
| ResNet-50 | 76.1% | Pure CNN |
| ViT-B + ResNet stem | 79.0% | Hybrid |
| CoAtNet | 84.5% | Interleaved |

**When to Use:**

| Scenario | Recommendation |
|----------|---------------|
| Limited data | Hybrid (CNN provides inductive bias) |
| Large data | Pure Transformer |
| Efficiency | CNN-heavy stages, light attention |

**Interview Tip:** Hybrids leverage CNNs for efficient local feature extraction (edges, textures) while Transformers handle global semantic relationships. Key design choice is where to transition from convolutions to attention.

---

### Question 28
- [ ] Done
**Explain attention visualization in ViT. How do you interpret attention patterns across layers?**

**Answer:**

ViT attention maps show which image patches each token attends to. By visualizing attention weights from the CLS token to all patches, we can see what regions the model focuses on for classification. Deeper layers show more semantic attention patterns.

**Attention Visualization Types:**

| Type | What It Shows | Use Case |
|------|--------------|----------|
| CLS attention | What CLS token attends to | Classification focus |
| Head attention | Individual head patterns | Specialized features |
| Layer attention | How attention evolves | Understanding processing |
| Attention rollout | Accumulated attention | End-to-end flow |

**Attention Patterns by Layer:**

| Layer | Typical Pattern | Interpretation |
|-------|----------------|----------------|
| Early (1-3) | Local, grid-like | Edge, texture detection |
| Middle (4-8) | Mixed local/global | Part detection |
| Late (9-12) | Semantic, object-focused | Class-relevant regions |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def get_attention_maps(model, image):
    """Extract attention maps from ViT"""
    attention_maps = []
    
    # Register hooks to capture attention
    def hook_fn(module, input, output):
        # For standard transformer: output is (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attention_maps.append(output[1].detach())
    
    hooks = []
    for layer in model.transformer.layers:
        hooks.append(layer.self_attn.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps  # List of [B, heads, tokens, tokens]


def visualize_cls_attention(attention_maps, img_size=224, patch_size=16):
    """Visualize CLS token attention across layers"""
    num_patches = (img_size // patch_size) ** 2
    grid_size = img_size // patch_size
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for layer_idx, attn in enumerate(attention_maps):
        if layer_idx >= 12:
            break
            
        # Get CLS attention (first row, skip CLS self-attention)
        cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # Average heads
        cls_attn = cls_attn.reshape(grid_size, grid_size).numpy()
        
        ax = axes[layer_idx // 4, layer_idx % 4]
        ax.imshow(cls_attn, cmap='viridis')
        ax.set_title(f'Layer {layer_idx + 1}')
        ax.axis('off')
    
    plt.suptitle('CLS Token Attention Across Layers')
    plt.tight_layout()
    return fig


def attention_rollout(attention_maps, discard_ratio=0.1):
    """Compute attention rollout for end-to-end visualization"""
    result = torch.eye(attention_maps[0].shape[-1])
    
    for attn in attention_maps:
        # Average across heads
        attn_avg = attn.mean(dim=1)[0]  # [tokens, tokens]
        
        # Add residual connection (identity)
        attn_with_residual = 0.5 * attn_avg + 0.5 * torch.eye(attn_avg.shape[0])
        
        # Normalize rows
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        
        # Multiply with previous
        result = torch.matmul(attn_with_residual, result)
    
    # Get CLS attention to patches
    cls_attention = result[0, 1:]  # Skip CLS self
    return cls_attention


def visualize_head_diversity(attention_maps, layer_idx=11):
    """Show attention patterns of different heads"""
    attn = attention_maps[layer_idx][0]  # [heads, tokens, tokens]
    num_heads = attn.shape[0]
    grid_size = int(np.sqrt(attn.shape[-1] - 1))
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for head_idx in range(min(12, num_heads)):
        cls_attn = attn[head_idx, 0, 1:].reshape(grid_size, grid_size).numpy()
        
        ax = axes[head_idx // 4, head_idx % 4]
        ax.imshow(cls_attn, cmap='hot')
        ax.set_title(f'Head {head_idx + 1}')
        ax.axis('off')
    
    plt.suptitle(f'Attention Heads in Layer {layer_idx + 1}')
    return fig


class AttentionVisualizerViT(nn.Module):
    """ViT with built-in attention extraction"""
    
    def __init__(self, base_vit):
        super().__init__()
        self.vit = base_vit
        self.attention_maps = []
    
    def forward(self, x):
        self.attention_maps = []
        
        # Patch embedding
        x = self.vit.patch_embed(x)
        cls = self.vit.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.vit.pos_embed
        
        # Transformer with attention capture
        for layer in self.vit.transformer.layers:
            # Multi-head attention with weights
            attn_output, attn_weights = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            self.attention_maps.append(attn_weights)
            
            # Rest of layer
            x = x + attn_output
            x = x + layer.mlp(layer.norm2(x))
        
        return self.vit.head(x[:, 0])
```

**Interpreting Patterns:**

| Pattern | Meaning |
|---------|---------|
| Uniform attention | Model uncertain, averaging |
| Focused on object | Good classification signal |
| Background attention | Potential overfitting |
| Grid patterns (early) | Local feature detection |
| Diagonal patterns | Self-attention, identity |

**Head Specialization:**

Different attention heads often specialize:
- Some heads: horizontal/vertical edges
- Some heads: global context
- Some heads: semantic parts (eyes, wheels)

**Interview Tip:** Attention visualization helps interpretability but doesn't always align with human intuition. Attention rollout provides a better end-to-end view by accumulating attention through layers with residual connections.

---

### Question 29
- [ ] Done
**What are the architectural variants of ViT (ViT-B, ViT-L, ViT-H) and their trade-offs?**

**Answer:**

ViT comes in three main sizes: Base (B), Large (L), and Huge (H). Each scales up embedding dimension, number of layers, and attention heads. Larger models achieve higher accuracy but require more compute and data.

**ViT Variants:**

| Variant | Layers | Embed Dim | Heads | Params | GFLOPs |
|---------|--------|-----------|-------|--------|--------|
| ViT-B/16 | 12 | 768 | 12 | 86M | 17.6 |
| ViT-L/16 | 24 | 1024 | 16 | 307M | 61.6 |
| ViT-H/14 | 32 | 1280 | 16 | 632M | 167 |

**Naming Convention:**
- ViT-B/16: Base model, 16×16 patch size
- ViT-L/14: Large model, 14×14 patch size

**Trade-offs:**

| Aspect | ViT-B | ViT-L | ViT-H |
|--------|-------|-------|-------|
| ImageNet Acc | 77.9% | 76.5%* | 88.6%** |
| Training data needed | Large | Very large | Huge |
| Inference speed | Fast | Medium | Slow |
| Fine-tune data | 100K+ | 1M+ | 10M+ |
| Memory | 4GB | 12GB | 32GB+ |

*ViT-L underperforms ViT-B without sufficient pre-training data
**With JFT-300M pre-training

**Python Implementation:**
```python
import torch
import torch.nn as nn

def create_vit(variant='base', patch_size=16, num_classes=1000, img_size=224):
    """Create ViT variant"""
    configs = {
        'tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
        'huge': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
    }
    
    config = configs[variant]
    return ViT(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        **config
    )


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Compare variants
for variant in ['tiny', 'small', 'base', 'large', 'huge']:
    model = create_vit(variant)
    params = count_parameters(model) / 1e6
    print(f"ViT-{variant[0].upper()}: {params:.1f}M parameters")
```

**Choosing the Right Variant:**

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Limited compute | ViT-B/16 | Balanced |
| Large dataset (1M+) | ViT-L/16 | Better capacity |
| Massive dataset | ViT-H/14 | Maximum accuracy |
| Real-time inference | ViT-S or ViT-Ti | Speed priority |
| Transfer learning | ViT-B/16 | Good pre-trained checkpoints |

**Patch Size Impact:**

| Patch Size | Tokens (224×224) | Speed | Accuracy |
|------------|-----------------|-------|----------|
| 32 | 49 | Fastest | Lower |
| 16 | 196 | Balanced | Good |
| 14 | 256 | Slower | Better |
| 8 | 784 | Slowest | Highest |

**Interview Tip:** Larger ViT models need more pre-training data to outperform smaller ones. ViT-L actually underperforms ViT-B on ImageNet-1K only training. The patch size affects the accuracy-speed trade-off more than model depth in many cases.

---

### Question 30
- [ ] Done
**How does ViT handle different input image resolutions during fine-tuning vs. pre-training?**

**Answer:**

ViT uses fixed-size position embeddings during pre-training. When fine-tuning at different resolutions, the position embeddings are interpolated to match the new number of patches. Higher resolution = more patches = need to interpolate position embeddings.

**Resolution Change Problem:**

```
Pre-training: 224×224, patch 16 → 14×14 = 196 patches + 1 CLS = 197 positions
Fine-tuning:  384×384, patch 16 → 24×24 = 576 patches + 1 CLS = 577 positions

Position embeddings: 197 → 577 (need interpolation!)
```

**Solution: Position Embedding Interpolation:**

```python
import torch
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed, target_size, patch_size=16):
    """Interpolate position embeddings for different resolutions"""
    
    # Original dimensions
    N = pos_embed.shape[1] - 1  # Exclude CLS token
    orig_size = int(N ** 0.5)
    
    # Target dimensions
    new_N = (target_size // patch_size) ** 2
    new_size = int(new_N ** 0.5)
    
    if N == new_N:
        return pos_embed
    
    # Separate CLS token
    cls_token = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]  # [1, N, D]
    
    # Reshape to 2D grid
    D = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, orig_size, orig_size, D)
    patch_pos = patch_pos.permute(0, 3, 1, 2)  # [1, D, H, W]
    
    # Bicubic interpolation
    patch_pos = F.interpolate(
        patch_pos,
        size=(new_size, new_size),
        mode='bicubic',
        align_corners=False
    )
    
    # Reshape back
    patch_pos = patch_pos.permute(0, 2, 3, 1)  # [1, H, W, D]
    patch_pos = patch_pos.reshape(1, new_N, D)
    
    # Concatenate CLS token
    return torch.cat([cls_token, patch_pos], dim=1)


class ResolutionAdaptiveViT(nn.Module):
    """ViT that handles variable input resolutions"""
    
    def __init__(self, pretrained_pos_embed, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Store original position embedding
        self.register_buffer('pretrained_pos_embed', pretrained_pos_embed)
        
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 12, embed_dim * 4,
                                       activation='gelu', batch_first=True),
            num_layers=12
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute patches
        x = self.patch_embed(x)  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Interpolate position embedding for current resolution
        pos_embed = interpolate_pos_embed(
            self.pretrained_pos_embed, H, self.patch_size
        )
        x = x + pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        return x[:, 0]
```

**Resolution Scaling Effects:**

| Resolution | Patches | Accuracy | Inference Time |
|------------|---------|----------|----------------|
| 224×224 | 196 | 77.9% | 1× |
| 384×384 | 576 | 79.8% | 2.9× |
| 512×512 | 1024 | 80.5% | 5.2× |

**Best Practices:**

| Practice | Reason |
|----------|--------|
| Pre-train at lower resolution | Faster, more data |
| Fine-tune at higher resolution | Better accuracy |
| Use bicubic interpolation | Smooth transitions |
| Fine-tune position embeddings | Adapt to new resolution |

**Alternative: Learnable Interpolation:**
```python
class LearnablePositionInterpolation(nn.Module):
    """Learn position embedding interpolation"""
    
    def __init__(self, pretrained_pos_embed, target_patches):
        super().__init__()
        orig_patches = pretrained_pos_embed.shape[1] - 1
        
        # Initialize with interpolated values
        interpolated = interpolate_pos_embed(
            pretrained_pos_embed, 
            int(target_patches ** 0.5) * 16
        )
        
        # Make learnable
        self.pos_embed = nn.Parameter(interpolated)
```

**Interview Tip:** Position embedding interpolation is necessary because ViT position embeddings are learned, not computed. Bicubic interpolation works well but fine-tuning the interpolated embeddings for a few epochs improves results. Higher resolution generally improves accuracy but with quadratic compute cost.

---

## Swin Transformer

### Question 31
- [ ] Done
**Explain shifted window partitioning in Swin Transformer. How does it enable cross-window connections?**

**Answer:**

Swin Transformer partitions the image into non-overlapping windows and computes self-attention within each window (efficient). Shifted window partitioning offsets the windows in alternate layers, allowing tokens at window boundaries to interact with tokens from adjacent windows.

**The Problem with Fixed Windows:**

Regular windows: tokens can only attend within their window
```
Layer 1:          Layer 2 (same):
┌───┬───┐         ┌───┬───┐
│ A │ B │         │ A │ B │  ← No cross-window connection!
├───┼───┤         ├───┼───┤
│ C │ D │         │ C │ D │
└───┴───┘         └───┴───┘
```

**Shifted Window Solution:**

```
Layer 1 (Regular):    Layer 2 (Shifted by M/2):
┌───┬───┐             ╔═══════════╗
│ A │ B │             ║   Shift   ║
├───┼───┤      →      ╠═══╦═══╦═══╣
│ C │ D │             ║   ║ X ║   ║  ← Tokens at boundaries mix!
└───┴───┘             ╚═══╩═══╩═══╝
```

After shifting, elements that were at window boundaries now share a window, enabling cross-window information flow.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """Partition image into non-overlapping windows"""
    B, H, W, C = x.shape
    
    x = x.view(B, H // window_size, window_size, 
               W // window_size, window_size, C)
    
    # [B, num_h, num_w, M, M, C] -> [B*num_windows, M, M, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention"""
    
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        
        # Compute relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size)
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape  # N = window_size^2
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask for shifted windows
        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with regular and shifted windows"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
```

**Why It Works:**

| Layer | Window Type | Effect |
|-------|-------------|--------|
| Odd layers | Regular | Local attention |
| Even layers | Shifted | Cross-window connection |
| Alternating | Both | Global receptive field |

**Interview Tip:** Shifted windows cleverly achieve cross-window connections without overlapping windows or global attention. The cyclic shift is efficient (just memory reordering), and masking handles boundary cases. This gives linear complexity while maintaining global connectivity.

---

### Question 32
- [ ] Done
**How does Swin Transformer's hierarchical representation differ from ViT's flat structure?**

**Answer:**

ViT maintains the same resolution throughout (flat structure), while Swin Transformer progressively reduces spatial resolution and increases channels (hierarchical), similar to CNNs. This creates multi-scale feature maps essential for dense prediction tasks.

**Architecture Comparison:**

```
ViT (Flat):
Input → Patch Embed → [Block]×12 → CLS → Output
        14×14×768    14×14×768    768

Swin (Hierarchical):
Input → Patch Embed → Stage1 → Stage2 → Stage3 → Stage4 → Output
        56×56×96     56×56    28×28    14×14     7×7
                     ×96      ×192     ×384      ×768
```

**Key Differences:**

| Aspect | ViT | Swin |
|--------|-----|------|
| Resolution | Fixed (e.g., 14×14) | Decreasing (56→28→14→7) |
| Channels | Fixed (e.g., 768) | Increasing (96→192→384→768) |
| Output | Single scale | Multi-scale pyramid |
| Dense prediction | Needs decoder | Native FPN compatibility |

**Why Hierarchy Matters:**

| Task | Need | ViT Problem | Swin Solution |
|------|------|-------------|---------------|
| Detection | Multi-scale features | Single scale | Built-in pyramid |
| Segmentation | High-res features | Low resolution | Hierarchical features |
| Small objects | Fine details | 16× downsampling | 4× initial downsampling |

**Python Implementation:**
```python
import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    """Downsample by merging 2×2 patches"""
    
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # Merge 2×2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2 * W/2, 2C]
        
        return x


class SwinTransformer(nn.Module):
    """Hierarchical Swin Transformer"""
    
    def __init__(self, img_size=224, patch_size=4, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        # Patch embedding (4× downsampling)
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        # Build stages
        self.stages = nn.ModuleList()
        self.merging = nn.ModuleList()
        
        dim = embed_dim
        resolution = img_size // patch_size
        
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            # Swin blocks for this stage
            stage = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=7,
                    shift_size=0 if j % 2 == 0 else 3
                )
                for j in range(depth)
            ])
            self.stages.append(stage)
            
            # Patch merging (except last stage)
            if i < len(depths) - 1:
                self.merging.append(PatchMerging(dim))
                dim *= 2
                resolution //= 2
            else:
                self.merging.append(None)
        
        # Head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, C, H, W]
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        features = []  # Multi-scale features
        
        for stage, merge in zip(self.stages, self.merging):
            # Apply blocks
            for block in stage:
                x = block(x, H, W)
            
            features.append(x.view(B, H, W, -1).permute(0, 3, 1, 2))
            
            # Downsample
            if merge is not None:
                x = merge(x, H, W)
                H, W = H // 2, W // 2
        
        return features  # [C1×H1×W1, C2×H2×W2, ...]
    
    def forward(self, x):
        features = self.forward_features(x)
        x = features[-1]  # Last stage
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.norm(x)
        return self.head(x)


# Comparison: Using for detection/segmentation
class SwinFPN(nn.Module):
    """Swin + FPN for detection"""
    
    def __init__(self, swin_backbone, out_channels=256):
        super().__init__()
        self.backbone = swin_backbone
        
        # FPN lateral connections
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1)
            for c in [96, 192, 384, 768]  # Swin-T channels
        ])
        
        # FPN output convs
        self.outputs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(4)
        ])
    
    def forward(self, x):
        # Get multi-scale features
        features = self.backbone.forward_features(x)  # [P1, P2, P3, P4]
        
        # FPN top-down pathway
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], scale_factor=2, mode='nearest'
            )
        
        outputs = [out(lat) for out, lat in zip(self.outputs, laterals)]
        
        return outputs  # Multi-scale feature pyramid
```

**Feature Map Comparison:**

| Stage | Swin Resolution | Swin Channels | ViT |
|-------|----------------|---------------|-----|
| 1 | 56×56 | 96 | 14×14×768 |
| 2 | 28×28 | 192 | 14×14×768 |
| 3 | 14×14 | 384 | 14×14×768 |
| 4 | 7×7 | 768 | 14×14×768 |

**Interview Tip:** Swin's hierarchical design makes it a drop-in replacement for CNN backbones in detection/segmentation frameworks (Faster R-CNN, Mask R-CNN, UPerNet). ViT requires additional adapters or decoders to produce multi-scale features.

---

### Question 33
- [ ] Done
**Explain the linear complexity (O(n)) of Swin vs. quadratic complexity of ViT. How is this achieved?**

**Answer:**

ViT computes global self-attention over all patches, resulting in O(n²) complexity where n = number of patches. Swin computes attention only within fixed-size local windows, giving O(n) complexity. Window size is constant regardless of image size.

**Complexity Analysis:**

```
ViT: Every patch attends to every patch
     224×224 image, patch 16 → 196 patches
     Attention: 196 × 196 = 38,416 operations per head

Swin: Attention only within 7×7 windows
      196 patches = 4 windows of 49 patches
      Attention: 4 × (49 × 49) = 9,604 operations per head
```

**Mathematical Breakdown:**

| Model | Attention Complexity | For 224×224 | For 448×448 |
|-------|---------------------|-------------|-------------|
| ViT | O(n²) = O((HW/P²)²) | 38,416 | 614,656 (16×) |
| Swin | O(n) = O((HW/P²) × M²) | 9,604 | 38,416 (4×) |

Where:
- n = number of patches = HW/P²
- M = window size (fixed at 7)
- P = patch size

**Why Swin is O(n):**

```python
# Complexity calculation
def complexity_analysis(img_size, patch_size=4, window_size=7):
    num_patches = (img_size // patch_size) ** 2
    num_windows = num_patches // (window_size ** 2)
    tokens_per_window = window_size ** 2
    
    # ViT: global attention
    vit_complexity = num_patches ** 2
    
    # Swin: local window attention
    swin_complexity = num_windows * (tokens_per_window ** 2)
    # Simplifies to: num_patches * window_size^2
    # = O(n) since window_size is constant
    
    return {
        'ViT': vit_complexity,
        'Swin': swin_complexity,
        'Ratio': vit_complexity / swin_complexity
    }

# Example
print(complexity_analysis(224))   # ViT: 3.1M, Swin: 0.8M, Ratio: 4×
print(complexity_analysis(448))   # ViT: 50M,  Swin: 3.1M, Ratio: 16×
print(complexity_analysis(896))   # ViT: 800M, Swin: 12.5M, Ratio: 64×
```

**Python Implementation:**
```python
import torch
import torch.nn as nn

class GlobalAttention(nn.Module):
    """ViT-style global attention - O(n²)"""
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # O(N²) attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class WindowAttention(nn.Module):
    """Swin-style window attention - O(n)"""
    
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        M = self.window_size
        
        # Reshape to spatial
        x = x.view(B, H, W, C)
        
        # Partition into windows: [B, H, W, C] -> [B*num_windows, M², C]
        x = x.view(B, H // M, M, W // M, M, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, M * M, C)  # [B*num_windows, M², C]
        
        # Attention within each window - O(M²) per window
        qkv = self.qkv(x).reshape(-1, M * M, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [windows, heads, M², M²]
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, M * M, C)
        x = self.proj(x)
        
        # Reverse partition
        num_windows = (H // M) * (W // M)
        x = x.view(B, H // M, W // M, M, M, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, L, C)
        
        return x


def benchmark_complexity(img_sizes=[224, 448, 672, 896]):
    """Benchmark ViT vs Swin complexity"""
    import time
    
    results = []
    for size in img_sizes:
        patches = (size // 4) ** 2  # Swin uses patch_size=4
        
        # Simulate attention computation time
        x = torch.randn(1, patches, 96)
        
        # ViT global
        global_attn = GlobalAttention(96, 4)
        start = time.time()
        for _ in range(10):
            _ = global_attn(x)
        vit_time = time.time() - start
        
        # Swin window
        H = W = size // 4
        window_attn = WindowAttention(96, 7, 4)
        start = time.time()
        for _ in range(10):
            _ = window_attn(x, H, W)
        swin_time = time.time() - start
        
        results.append({
            'size': size,
            'patches': patches,
            'ViT_time': vit_time,
            'Swin_time': swin_time,
            'speedup': vit_time / swin_time
        })
    
    return results
```

**Scaling Comparison:**

| Image Size | Patches | ViT (O(n²)) | Swin (O(n)) | Speedup |
|------------|---------|-------------|-------------|---------|
| 224×224 | 3,136 | 9.8M ops | 0.15M ops | 64× |
| 448×448 | 12,544 | 157M ops | 0.61M ops | 256× |
| 896×896 | 50,176 | 2.5B ops | 2.4M ops | 1024× |

**Interview Tip:** Swin achieves O(n) by fixing the attention window size. Total operations = (num_patches / window_size²) × window_size⁴ = num_patches × window_size². Since window_size is constant (7), this is O(n). This enables Swin to scale to high-resolution images where ViT becomes impractical.

---

### Question 34
- [ ] Done
**Describe patch merging layers and how they create multi-scale feature maps.**

**Answer:**

Patch merging downsamples spatial resolution by 2× while doubling channels, similar to strided convolutions in CNNs. It concatenates 2×2 neighboring patches and projects them to half the concatenated dimension, creating hierarchical multi-scale features.

**Patch Merging Process:**

```
Input: H×W×C
  ↓
Split into 2×2 groups → 4 patches of (H/2)×(W/2)×C each
  ↓
Concatenate → (H/2)×(W/2)×4C
  ↓
Linear projection → (H/2)×(W/2)×2C

Result: 2× smaller spatial, 2× more channels
```

**Visual Example:**
```
Before (8×8×C):                After (4×4×2C):
┌─┬─┬─┬─┬─┬─┬─┬─┐              ┌───┬───┬───┬───┐
│A│B│A│B│A│B│A│B│              │   │   │   │   │
├─┼─┼─┼─┼─┼─┼─┼─┤   Merge      ├───┼───┼───┼───┤
│C│D│C│D│C│D│C│D│   ────→      │   │   │   │   │
├─┼─┼─┼─┼─┼─┼─┼─┤              ├───┼───┼───┼───┤
│A│B│...│...│...│              │ concat(A,B,C,D)│
└─┴─┴─┴─┴─┴─┴─┴─┘              └───┴───┴───┴───┘

Each new patch = [A; B; C; D] → Linear → 2C channels
```

**Python Implementation:**
```python
import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
    
    def forward(self, x, H, W):
        """
        Args:
            x: [B, H*W, C]
            H, W: spatial dimensions
        Returns:
            x: [B, H/2 * W/2, 2C]
        """
        B, L, C = x.shape
        assert L == H * W, "Input size mismatch"
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
        
        x = x.view(B, H, W, C)
        
        # Sample 2×2 patches
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C] - top-left
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C] - bottom-left
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C] - top-right
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C] - bottom-right
        
        # Concatenate: [B, H/2, W/2, 4C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)  # [B, H/2 * W/2, 4C]
        
        # Normalize and reduce
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2 * W/2, 2C]
        
        return x


class SwinStage(nn.Module):
    """Swin Transformer Stage with optional downsampling"""
    
    def __init__(self, dim, depth, num_heads, window_size=7, downsample=True):
        super().__init__()
        
        # Swin blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2
            )
            for i in range(depth)
        ])
        
        # Patch merging for downsampling
        self.downsample = PatchMerging(dim) if downsample else None
    
    def forward(self, x, H, W):
        # Apply blocks
        for block in self.blocks:
            x = block(x, H, W)
        
        # Feature before downsampling (for FPN)
        feature = x.view(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2)
        
        # Downsample
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = H // 2, W // 2
        
        return x, H, W, feature


class HierarchicalSwin(nn.Module):
    """Full Swin with hierarchical features"""
    
    def __init__(self, img_size=224, embed_dim=96, 
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        # Initial patch embedding (4× downsample)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=4, stride=4),
            nn.LayerNorm([embed_dim, img_size // 4, img_size // 4])
        )
        
        # Stages
        self.stages = nn.ModuleList()
        dim = embed_dim
        
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            downsample = (i < len(depths) - 1)  # No downsample at last stage
            stage = SwinStage(dim, depth, heads, downsample=downsample)
            self.stages.append(stage)
            if downsample:
                dim *= 2
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed[0](x)  # [B, 96, 56, 56]
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, 3136, 96]
        
        # Collect multi-scale features
        features = []
        
        for stage in self.stages:
            x, H, W, feature = stage(x, H, W)
            features.append(feature)
        
        return features  # [C1×56×56, C2×28×28, C3×14×14, C4×7×7]
```

**Multi-Scale Feature Map:**

| Stage | Resolution | Channels | Stride |
|-------|------------|----------|--------|
| Stage 1 | 56×56 | 96 | 4 |
| Stage 2 | 28×28 | 192 | 8 |
| Stage 3 | 14×14 | 384 | 16 |
| Stage 4 | 7×7 | 768 | 32 |

**Comparison with CNN Pooling:**

| Aspect | Patch Merging | Max/Avg Pooling |
|--------|---------------|-----------------|
| Information | Preserves all (concatenate) | Discards (max/avg) |
| Learnable | Yes (linear projection) | No |
| Channel change | 2× increase | Same |

**Use in Downstream Tasks:**
```python
# Object Detection with FPN
features = swin_backbone(image)  # [P1, P2, P3, P4]
fpn_features = fpn(features)  # Multi-scale pyramid
boxes = detection_head(fpn_features)

# Semantic Segmentation
features = swin_backbone(image)
mask = upernet_decoder(features)
```

**Interview Tip:** Patch merging is Swin's equivalent to strided convolution for downsampling. By concatenating 2×2 patches before projection, it preserves all information (unlike pooling). The hierarchical features enable direct compatibility with FPN-based detection and segmentation frameworks.

---

### Question 35
- [ ] Done
**Explain relative positional bias in Swin vs. absolute positional encoding in ViT.**

**Answer:**

ViT uses absolute positional encodings added to patch embeddings, encoding each patch's fixed position. Swin uses relative positional bias added to attention scores, encoding the relative distance between patches. Relative bias generalizes better to different resolutions.

**Key Difference:**

| Type | What It Encodes | Added To | Resolution Change |
|------|-----------------|----------|-------------------|
| Absolute (ViT) | Patch (3,5) is at position 3,5 | Embeddings | Needs interpolation |
| Relative (Swin) | Patch A is 2 right, 1 down from B | Attention scores | Generalizes naturally |

**Visual Example:**
```
Absolute: "I am patch at position (2,3)"
Relative: "Neighbor is 1 step right, 0 steps down"

Absolute for 14×14:              Relative (same for any size):
pos[0,0], pos[0,1], ...          [-6,-6] [-6,-5] ... [-6,+6]
pos[1,0], pos[1,1], ...          [-5,-6] [-5,-5] ... [-5,+6]
...                               ...
pos[13,13]                       [+6,-6] [+6,-5] ... [+6,+6]
                                 (for 7×7 window)
```

**Python Implementation:**
```python
import torch
import torch.nn as nn

# ViT: Absolute Positional Encoding
class AbsolutePositionalEncoding(nn.Module):
    """ViT-style absolute position embedding"""
    
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # Learnable embedding for each position
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        # Add to embeddings
        return x + self.pos_embed


# Swin: Relative Positional Bias
class RelativePositionalBias(nn.Module):
    """Swin-style relative position bias"""
    
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        
        # Bias table: (2M-1) × (2M-1) possible relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = coords.flatten(1)  # [2, M*M]
        
        # Relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [M², M², 2]
        
        # Shift to start from 0
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        
        # Convert 2D to 1D index
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [M², M²]
        
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self):
        """Return bias to add to attention scores"""
        # Look up bias from table
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        bias = bias.view(
            self.window_size ** 2, 
            self.window_size ** 2, 
            -1
        )
        # [M², M², num_heads] -> [num_heads, M², M²]
        return bias.permute(2, 0, 1).contiguous()


class WindowAttentionWithBias(nn.Module):
    """Window attention with relative positional bias"""
    
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.rel_pos_bias = RelativePositionalBias(window_size, num_heads)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention with relative bias
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias()  # Add bias to scores
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj(x)


def compare_position_encodings():
    """Compare absolute vs relative approaches"""
    
    # Absolute: needs different embedding for each resolution
    pos_224 = AbsolutePositionalEncoding(196, 768)  # 14×14
    pos_384 = AbsolutePositionalEncoding(576, 768)  # 24×24 - different!
    
    # Relative: same bias works for any resolution (within window)
    rel_bias = RelativePositionalBias(7, 12)  # 7×7 window, 12 heads
    # Works for any image size using this window
    
    print(f"Absolute 224: {pos_224.pos_embed.shape}")
    print(f"Absolute 384: {pos_384.pos_embed.shape}")
    print(f"Relative bias table: {rel_bias.relative_position_bias_table.shape}")
```

**Resolution Generalization:**

| Scenario | Absolute (ViT) | Relative (Swin) |
|----------|---------------|-----------------|
| Pre-train 224, fine-tune 224 | ✅ Works | ✅ Works |
| Pre-train 224, fine-tune 384 | ⚠️ Interpolate | ✅ Works directly |
| Pre-train 224, fine-tune 512 | ⚠️ Significant degradation | ✅ Minimal impact |

**Why Relative Bias Works Better:**

1. **Translation equivariance**: Same relative bias regardless of absolute position
2. **Resolution invariance**: Bias depends only on window size, not image size
3. **Learned preferences**: Model learns that "2 patches away" has specific relationship

**Bias Table Size:**

For window size M=7:
- Relative positions range: [-(M-1), M-1] in each dimension
- Table size: (2×7-1)² = 13² = 169 entries per head

**Interview Tip:** Relative positional bias is more efficient (smaller table) and generalizes better. It encodes spatial relationships between tokens rather than absolute locations, making it translation equivariant and resolution-agnostic within each window.

---

### Question 36
- [ ] Done
**How is Swin Transformer used as a backbone for object detection (Swin + FPN) and segmentation (Swin + UPerNet)?**

**Answer:**

Swin's hierarchical features make it a drop-in replacement for CNN backbones. For detection, Swin extracts multi-scale features fed to FPN and detection heads. For segmentation, Swin features are processed by UPerNet decoder to produce dense predictions.

**Swin + FPN for Detection:**

```
                     ┌─────────────┐
                     │ Detection   │
                     │ Head (RCNN) │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           ┌──────┐     ┌──────┐     ┌──────┐
           │ P3   │     │ P4   │     │ P5   │  ← FPN outputs
           └──┬───┘     └──┬───┘     └──┬───┘
              │            │            │
              ▼            ▼            ▼
           ┌──────┐     ┌──────┐     ┌──────┐
           │Stage2│     │Stage3│     │Stage4│  ← Swin features
           │28×28 │     │14×14 │     │ 7×7  │
           └──────┘     └──────┘     └──────┘
```

**Swin + UPerNet for Segmentation:**

```
Swin Backbone          UPerNet Decoder           Output
┌────────┐             ┌─────────────┐          ┌──────┐
│Stage 1 │─────────────┤   Lateral   │          │      │
│ 56×56  │             │   + Fuse    │          │ Mask │
├────────┤             │             │   ────►  │H × W │
│Stage 2 │─────────────┤   PPM       │          │      │
│ 28×28  │             │   Module    │          └──────┘
├────────┤             │             │
│Stage 3 │─────────────┤   Upsample  │
│ 14×14  │             │   & Merge   │
├────────┤             └─────────────┘
│Stage 4 │
│  7×7   │
└────────┘
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinFPN(nn.Module):
    """Swin Transformer + Feature Pyramid Network"""
    
    def __init__(self, swin_backbone, out_channels=256):
        super().__init__()
        self.backbone = swin_backbone
        
        # Swin-T channel dimensions
        in_channels = [96, 192, 384, 768]
        
        # Lateral connections (1×1 conv to match channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        
        # Output convolutions (3×3 for smoothing)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        ])
        
        # Extra level for detection (P6, P7)
        self.extra_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        ])
    
    def forward(self, x):
        # Extract hierarchical features from Swin
        features = self.backbone.forward_features(x)  # [C1, C2, C3, C4]
        
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway with upsampling
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[2:],
                mode='nearest'
            )
        
        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        
        # Extra levels (P6, P7)
        outputs.append(self.extra_convs[0](outputs[-1]))
        outputs.append(self.extra_convs[1](outputs[-1]))
        
        return outputs  # P2, P3, P4, P5, P6, P7


class SwinUPerNet(nn.Module):
    """Swin Transformer + UPerNet for Semantic Segmentation"""
    
    def __init__(self, swin_backbone, num_classes=150, feature_channels=512):
        super().__init__()
        self.backbone = swin_backbone
        in_channels = [96, 192, 384, 768]  # Swin-T
        
        # Pyramid Pooling Module (on deepest feature)
        self.ppm = PyramidPoolingModule(in_channels[-1], feature_channels // 4)
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, feature_channels, 1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU()
            ) for c in in_channels[:-1]
        ])
        
        # FPN-style top-down
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU()
            ) for _ in in_channels[:-1]
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_channels * 4, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU()
        )
        
        # Classification head
        self.head = nn.Conv2d(feature_channels, num_classes, 1)
    
    def forward(self, x):
        H, W = x.shape[2:]
        
        # Get hierarchical features
        features = self.backbone.forward_features(x)  # [C1, C2, C3, C4]
        
        # PPM on deepest feature
        ppm_out = self.ppm(features[-1])  # [B, 512, H/32, W/32]
        
        # Top-down with lateral connections
        fpn_features = [ppm_out]
        
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            
            # Upsample and add
            upsampled = F.interpolate(
                fpn_features[-1], 
                size=lateral.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            fpn_features.append(self.fpn_convs[i](lateral + upsampled))
        
        # Upsample all to same size and concatenate
        target_size = features[0].shape[2:]
        aligned = []
        for f in fpn_features:
            aligned.append(F.interpolate(f, size=target_size, 
                                         mode='bilinear', align_corners=False))
        
        fused = self.fusion(torch.cat(aligned, dim=1))
        
        # Upsample to original resolution
        output = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
        output = self.head(output)
        
        return output


class PyramidPoolingModule(nn.Module):
    """PPM from PSPNet"""
    
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for s in pool_sizes
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_sizes), 
                      out_channels * len(pool_sizes), 3, padding=1),
            nn.BatchNorm2d(out_channels * len(pool_sizes)),
            nn.ReLU()
        )
    
    def forward(self, x):
        H, W = x.shape[2:]
        
        pyramids = [x]
        for stage in self.stages:
            pooled = stage(x)
            pyramids.append(F.interpolate(pooled, size=(H, W), 
                                          mode='bilinear', align_corners=False))
        
        return self.bottleneck(torch.cat(pyramids, dim=1))


# Usage examples
def build_detection_model():
    """Build Swin + Faster R-CNN"""
    swin = SwinTransformer()
    fpn = SwinFPN(swin, out_channels=256)
    # Add RPN and ROI heads...
    return fpn

def build_segmentation_model():
    """Build Swin + UPerNet"""
    swin = SwinTransformer()
    model = SwinUPerNet(swin, num_classes=150)
    return model
```

**Performance Comparison:**

| Backbone | COCO mAP | ADE20K mIoU |
|----------|----------|-------------|
| ResNet-50 | 41.0 | 42.1 |
| Swin-T | 46.0 | 44.5 |
| Swin-S | 48.5 | 47.6 |
| Swin-B | 49.7 | 48.1 |

**Interview Tip:** Swin's hierarchical design with 4 stages producing features at 4×, 8×, 16×, 32× strides mirrors CNN backbones, making it compatible with existing detection/segmentation frameworks. Just replace ResNet with Swin and adjust channel dimensions.

---

### Question 37
- [ ] Done
**Compare Swin-T, Swin-S, Swin-B, and Swin-L configurations. How do you choose for your task?**

**Answer:**

Swin comes in four sizes: Tiny (T), Small (S), Base (B), and Large (L). They differ in embedding dimension, number of layers per stage, and attention heads. Larger models achieve higher accuracy but require more compute and memory.

**Configuration Comparison:**

| Config | Embed Dim | Depths | Heads | Params | FLOPs |
|--------|-----------|--------|-------|--------|-------|
| Swin-T | 96 | [2,2,6,2] | [3,6,12,24] | 28M | 4.5G |
| Swin-S | 96 | [2,2,18,2] | [3,6,12,24] | 50M | 8.7G |
| Swin-B | 128 | [2,2,18,2] | [4,8,16,32] | 88M | 15.4G |
| Swin-L | 192 | [2,2,18,2] | [6,12,24,48] | 197M | 34.5G |

**Key Observations:**
- Swin-S is Swin-T with deeper Stage 3 (18 vs 6 blocks)
- Swin-B increases embed_dim from 96 to 128
- Swin-L further increases embed_dim to 192

**Performance Comparison:**

| Model | ImageNet Top-1 | COCO mAP | ADE20K mIoU |
|-------|---------------|----------|-------------|
| Swin-T | 81.3% | 46.0 | 44.5 |
| Swin-S | 83.0% | 48.5 | 47.6 |
| Swin-B | 83.5% | 49.7 | 48.1 |
| Swin-L | 86.3%* | 51.9 | 52.1 |

*With ImageNet-22K pre-training

**Python Configuration:**
```python
import torch
import torch.nn as nn

def get_swin_config(variant='tiny'):
    """Get Swin configuration by variant"""
    configs = {
        'tiny': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7
        },
        'small': {
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],  # Deeper stage 3
            'num_heads': [3, 6, 12, 24],
            'window_size': 7
        },
        'base': {
            'embed_dim': 128,  # Wider
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': 7
        },
        'large': {
            'embed_dim': 192,  # Even wider
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
            'window_size': 7
        }
    }
    return configs[variant]


def create_swin(variant='tiny', num_classes=1000, img_size=224):
    """Create Swin Transformer"""
    config = get_swin_config(variant)
    return SwinTransformer(
        img_size=img_size,
        num_classes=num_classes,
        **config
    )


# Channel dimensions at each stage
def get_stage_channels(variant):
    """Get output channels for each stage"""
    base_dim = get_swin_config(variant)['embed_dim']
    return [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

# Example
print(get_stage_channels('tiny'))   # [96, 192, 384, 768]
print(get_stage_channels('base'))   # [128, 256, 512, 1024]
print(get_stage_channels('large'))  # [192, 384, 768, 1536]
```

**Selection Guidelines:**

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Edge deployment | Swin-T | Smallest, 28M params |
| Standard training | Swin-S | Good accuracy/cost balance |
| High accuracy needed | Swin-B | Best single-scale accuracy |
| Competition/SOTA | Swin-L | Maximum capacity |
| Limited GPU memory | Swin-T/S | Lower memory footprint |
| Pre-training available | Swin-B/L | Better transfer |

**Memory and Speed Comparison:**

| Model | Training Batch (V100) | Inference Speed |
|-------|----------------------|-----------------|
| Swin-T | 128 | 1.0× |
| Swin-S | 64 | 0.6× |
| Swin-B | 32 | 0.4× |
| Swin-L | 16 | 0.25× |

**Task-Specific Recommendations:**

| Task | Best Choice | Why |
|------|-------------|-----|
| ImageNet classification | Swin-B (22K pretrain) | Accuracy priority |
| Real-time detection | Swin-T | Speed priority |
| High-res segmentation | Swin-S/B | Memory for large images |
| Transfer to small dataset | Swin-B/L (pretrained) | Better features |
| Resource-constrained | Swin-T | 28M params, 4.5 GFLOPs |

**Interview Tip:** Choose based on your constraints: Swin-T for speed/memory, Swin-S for balanced performance, Swin-B for high accuracy, Swin-L with ImageNet-22K pretraining for SOTA. The main difference between T and S is Stage 3 depth; between S and B/L is embedding dimension.

---

### Question 38
- [ ] Done
**Explain Swin-V2 improvements: log-scaled continuous position bias, residual post-norm, and scaled cosine attention.**

**Answer:**

Swin-V2 introduces three key improvements to enable training at much larger scales (up to 3 billion parameters) and higher resolutions (up to 1536×1536). These changes address training instability and resolution transfer issues in the original Swin.

**Key Improvements:**

| Issue in Swin-V1 | Swin-V2 Solution |
|------------------|------------------|
| Training instability at scale | Residual post-norm |
| Attention value explosion | Scaled cosine attention |
| Poor resolution transfer | Log-spaced continuous position bias |

**1. Residual Post-Norm:**

V1 uses pre-norm (LayerNorm before attention), V2 uses post-norm (after attention) with residual scaling.

```python
# Swin-V1: Pre-normalization
def forward_v1(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x

# Swin-V2: Post-normalization with residual
def forward_v2(self, x):
    x = x + self.norm1(self.attn(x))  # Norm after attention
    x = x + self.norm2(self.mlp(x))   # Norm after MLP
    return x
```

**Why:** Pre-norm accumulates values across layers, causing activation explosion in very deep models. Post-norm keeps activations bounded.

**2. Scaled Cosine Attention:**

Replaces dot-product attention with cosine similarity scaled by a learnable parameter.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledCosineAttention(nn.Module):
    """Swin-V2 scaled cosine attention"""
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Learnable temperature (log scale for stability)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads, 1, 1)))
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # L2 normalize q and k (cosine similarity)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Scaled cosine attention
        attn = (q @ k.transpose(-2, -1))
        
        # Apply learnable scale (clamp for stability)
        logit_scale = torch.clamp(self.logit_scale, max=4.6052)  # max = ln(100)
        attn = attn * logit_scale.exp()
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj(x)
```

**Why:** Dot-product attention can have unbounded magnitudes. Cosine similarity is always in [-1, 1], preventing attention weight explosion.

**3. Log-Spaced Continuous Position Bias:**

Uses log-scale coordinates for position bias, enabling smooth interpolation to unseen positions at higher resolutions.

```python
class LogSpacedContinuousPositionBias(nn.Module):
    """Swin-V2 continuous relative position bias"""
    
    def __init__(self, window_size, num_heads, pretrained_window_size=0):
        super().__init__()
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        
        # MLP to generate bias from coordinates
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, num_heads)
        )
        
        # Compute log-spaced relative coordinates
        relative_coords = self._get_relative_coords(window_size)
        self.register_buffer("relative_coords_table", relative_coords)
    
    def _get_relative_coords(self, window_size):
        """Compute log-scaled relative coordinates"""
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = coords.flatten(1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).float()
        
        # Normalize to [-1, 1]
        relative_coords[:, :, 0] /= (window_size - 1)
        relative_coords[:, :, 1] /= (window_size - 1)
        
        # Log-scale transformation
        # sign * log(1 + |x|) / log(1 + 8)
        relative_coords = torch.sign(relative_coords) * torch.log2(
            1 + torch.abs(relative_coords) * 8
        ) / 3.0
        
        return relative_coords
    
    def forward(self):
        # Generate bias through MLP
        bias = self.cpb_mlp(self.relative_coords_table)
        bias = bias.permute(2, 0, 1)  # [num_heads, M², M²]
        
        # Apply sigmoid and scale
        bias = 16 * torch.sigmoid(bias)
        
        return bias
```

**Why Log-Scale:**

| Coordinate | Linear | Log-scale |
|------------|--------|-----------|
| Position 0 | 0.0 | 0.0 |
| Position 1 | 0.14 | 0.35 |
| Position 7 | 1.0 | 1.0 |
| Position 14 (new) | 2.0 (extrapolate) | 1.26 (within range) |

Log-scale compresses large distances, making extrapolation to higher resolutions smoother.

**Full Swin-V2 Block:**
```python
class SwinV2Block(nn.Module):
    """Swin Transformer V2 Block"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.attn = ScaledCosineAttention(dim, num_heads)
        self.pos_bias = LogSpacedContinuousPositionBias(window_size, num_heads)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
        self.window_size = window_size
        self.shift_size = shift_size
    
    def forward(self, x, H, W):
        shortcut = x
        
        # Window partition, attention, reverse
        x = self._window_attention(x, H, W)
        
        # Post-norm (V2 change)
        x = shortcut + self.norm1(x)
        
        # MLP with post-norm
        x = x + self.norm2(self.mlp(x))
        
        return x
```

**Scale Comparison:**

| Model | Params | Resolution | Training Stability |
|-------|--------|------------|-------------------|
| Swin-V1-L | 197M | 384 | Good |
| Swin-V1 scaled | 1B+ | 640+ | Unstable |
| Swin-V2-G | 3B | 1536 | Stable |

**Interview Tip:** Swin-V2 changes are about scaling: post-norm prevents activation explosion, cosine attention bounds attention weights, and log-spaced position bias enables smooth resolution transfer. These enable 3B parameter models at 1536×1536 resolution.

---

### Question 39
- [ ] Done
**How does Video Swin Transformer extend the architecture for temporal modeling?**

**Answer:**

Video Swin extends Swin from 2D images to 3D video by using 3D shifted windows (spatial + temporal). It partitions video into 3D patches and computes attention within local spatiotemporal windows, with shifting for cross-window connections.

**Key Extensions:**

| Aspect | Image Swin | Video Swin |
|--------|------------|------------|
| Input | H×W×3 | T×H×W×3 |
| Patch | 2D (4×4) | 3D (2×4×4) |
| Window | 2D (M×M) | 3D (P×M×M) |
| Shift | (M/2, M/2) | (P/2, M/2, M/2) |
| Attention | Spatial only | Spatiotemporal |

**Architecture:**

```
Video Input [T×H×W×3]
      ↓
3D Patch Partition [T/2 × H/4 × W/4 × C]
      ↓
Stage 1: 3D Swin Blocks (spatiotemporal attention)
      ↓
3D Patch Merging [T/2 × H/8 × W/8 × 2C]
      ↓
Stage 2-4: More blocks + merging
      ↓
Global Average Pool → Classification
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Video3DPatchEmbed(nn.Module):
    """3D Patch Embedding for Video"""
    
    def __init__(self, patch_size=(2, 4, 4), in_channels=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, D, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', D]
        x = self.norm(x)
        return x


def window_partition_3d(x, window_size):
    """Partition video into 3D windows"""
    B, T, H, W, C = x.shape
    Pt, Ph, Pw = window_size
    
    x = x.view(B, T // Pt, Pt, H // Ph, Ph, W // Pw, Pw, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, Pt * Ph * Pw, C)
    
    return windows


def window_reverse_3d(windows, window_size, T, H, W):
    """Reverse 3D window partition"""
    Pt, Ph, Pw = window_size
    B = windows.shape[0] // ((T // Pt) * (H // Ph) * (W // Pw))
    
    x = windows.view(B, T // Pt, H // Ph, W // Pw, Pt, Ph, Pw, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, T, H, W, -1)
    
    return x


class Video3DWindowAttention(nn.Module):
    """3D Window Attention for Video"""
    
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (P, M, M)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # 3D relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape  # N = P * M * M
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add 3D relative position bias
        # (implementation similar to 2D but with 3 dimensions)
        
        if mask is not None:
            attn = attn + mask
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        return self.proj(x)


class VideoSwinBlock(nn.Module):
    """Video Swin Transformer Block with 3D shifted windows"""
    
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0)):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Video3DWindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    
    def forward(self, x, T, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)
        
        # 3D cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            shifted_x = x
        
        # Partition into 3D windows
        x_windows = window_partition_3d(shifted_x, self.window_size)
        
        # 3D window attention
        attn_windows = self.attn(x_windows)
        
        # Reverse partition
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, T, H, W)
        
        # Reverse shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x
        
        x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class VideoSwinTransformer(nn.Module):
    """Video Swin Transformer for Action Recognition"""
    
    def __init__(self, num_frames=32, img_size=224, num_classes=400,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        self.patch_embed = Video3DPatchEmbed(
            patch_size=(2, 4, 4),
            embed_dim=embed_dim
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        dim = embed_dim
        
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage = nn.ModuleList([
                VideoSwinBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=(2, 7, 7),
                    shift_size=(0, 0, 0) if j % 2 == 0 else (1, 3, 3)
                )
                for j in range(depth)
            ])
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                dim *= 2  # After patch merging
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.patch_embed(x)  # [B, L, D]
        
        # Track dimensions
        T, H, W = 16, 56, 56  # After initial patch embed
        
        for stage in self.stages:
            for block in stage:
                x = block(x, T, H, W)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        
        return self.head(x)
```

**Performance on Kinetics-400:**

| Model | Top-1 | Params | GFLOPs |
|-------|-------|--------|--------|
| TimeSformer | 78.0 | 121M | 590 |
| ViViT | 79.3 | 89M | 1446 |
| Video Swin-T | 78.8 | 28M | 88 |
| Video Swin-B | 80.6 | 88M | 282 |

**Interview Tip:** Video Swin maintains linear complexity by using 3D local windows. The temporal window size P is typically 2 (short-term) or 8 (long-term). Shifted windows enable temporal information flow across clips while keeping computation efficient.

---

### Question 40
- [ ] Done
**Compare Swin Transformer to ConvNeXt. What design principles from Swin were adopted back into CNNs?**

**Answer:**

ConvNeXt modernizes ResNet by adopting design choices from Swin Transformer, achieving competitive performance with pure convolutions. It shows that many ViT/Swin advantages come from training recipes and architectural tweaks, not just attention.

**Design Principles Borrowed from Swin:**

| Swin Design | ConvNeXt Adaptation |
|-------------|---------------------|
| Patchify stem (4×4 stride) | 4×4 non-overlapping conv |
| Swin stage ratio [2,2,6,2] | Stage ratio [3,3,9,3] |
| Inverted bottleneck | Depthwise conv (wide→narrow) |
| GELU activation | GELU instead of ReLU |
| LayerNorm | LayerNorm instead of BatchNorm |
| Fewer normalization layers | Single LN per block |
| Large kernel | 7×7 depthwise conv |

**Architecture Comparison:**

```
ResNet Block:               Swin Block:                ConvNeXt Block:
┌─────────────┐            ┌──────────────┐            ┌──────────────┐
│ 1×1 Conv    │            │ LayerNorm    │            │ 7×7 DWConv   │
│ 3×3 Conv    │            │ Window Attn  │            │ LayerNorm    │
│ 1×1 Conv    │            │ LayerNorm    │            │ 1×1 Conv     │
│ BatchNorm   │            │ MLP          │            │ GELU         │
│ ReLU        │            │ GELU         │            │ 1×1 Conv     │
└─────────────┘            └──────────────┘            └──────────────┘
```

**Python Implementation:**
```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block - CNN with Transformer-inspired design"""
    
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        
        # Depthwise conv (like Swin's local attention)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm (from Transformers, not BatchNorm)
        self.norm = nn.LayerNorm(dim)
        
        # Inverted bottleneck MLP (like Swin/ViT)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Expand
        self.act = nn.GELU()  # GELU not ReLU
        self.pwconv2 = nn.Linear(4 * dim, dim)  # Contract
        
        # Layer scale (from Transformer training)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        
        # Stochastic depth (from DeiT)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        shortcut = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Channel-last for LayerNorm (like Transformers)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        
        # MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        x = self.gamma * x
        
        # Back to channel-first
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Residual with drop path
        x = shortcut + self.drop_path(x)
        
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt - A ConvNet for the 2020s"""
    
    def __init__(self, num_classes=1000, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.0):
        super().__init__()
        
        # Patchify stem (4×4 non-overlapping, like Swin)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm([dims[0], 56, 56])
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        
        cur = 0
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            # Downsampling (except first stage)
            if i > 0:
                downsample = nn.Sequential(
                    nn.LayerNorm([dims[i-1], 56 // (2 ** i), 56 // (2 ** i)]),
                    nn.Conv2d(dims[i-1], dim, kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()
            
            # Blocks
            blocks = nn.Sequential(*[
                ConvNeXtBlock(dim, drop_path=dp_rates[cur + j])
                for j in range(depth)
            ])
            
            self.stages.append(nn.Sequential(downsample, blocks))
            cur += depth
        
        # Head
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.norm(x)
        
        return self.head(x)


def convnext_tiny():
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])

def convnext_base():
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
```

**Key Modernizations:**

| Aspect | ResNet-50 | ConvNeXt-T |
|--------|-----------|------------|
| Stem | 7×7 conv, 3×3 pool | 4×4 conv stride 4 |
| Activation | ReLU | GELU |
| Normalization | BatchNorm | LayerNorm |
| Stage ratio | [3,4,6,3] | [3,3,9,3] |
| Block design | Bottleneck | Inverted bottleneck |
| Kernel size | 3×3 | 7×7 depthwise |

**Performance Comparison:**

| Model | Params | ImageNet Top-1 |
|-------|--------|---------------|
| Swin-T | 28M | 81.3% |
| ConvNeXt-T | 28M | 82.1% |
| Swin-B | 88M | 83.5% |
| ConvNeXt-B | 89M | 83.8% |

**What This Teaches Us:**

1. **Attention isn't everything**: Many Swin advantages come from training and architecture design
2. **Large kernels matter**: 7×7 depthwise ≈ local attention receptive field
3. **Modern training helps all**: DeiT training recipes boost CNNs too
4. **Inductive bias is useful**: Convolutions still have advantages for some tasks

**Interview Tip:** ConvNeXt shows that transformers don't have inherent superiority—careful design choices matter more. It adopts Swin's macro design (stage ratios, patchify stem) and micro design (LayerNorm, GELU, inverted bottleneck) while keeping pure convolutions. This is important for understanding what actually drives vision model performance.

---

## Segmentation Transformers

### Question 41
- [ ] Done
**Explain SETR (Segmentation Transformer) and how pure transformers handle dense prediction.**

**Answer:**

SETR (Segmentation Transformer) uses a pure ViT encoder to extract features, then uses CNN decoders to upsample to full resolution. It shows transformers can handle dense prediction by treating each patch as a token and decoding the sequence back to spatial predictions.

**Architecture:**

```
Image 480×480 → ViT Encoder (24 layers) → Sequence 900×1024 → Decoder → Mask 480×480
        ↓                                        ↓
    16×16 patches                        Reshape to 30×30
    30×30 = 900 tokens                   Upsample to full res
```

**Three Decoder Variants:**

| Decoder | Method | Complexity |
|---------|--------|------------|
| SETR-Naive | Simple 1×1 conv + bilinear upsample | Lowest |
| SETR-PUP | Progressive Upsampling (4 stages) | Medium |
| SETR-MLA | Multi-Level Aggregation | Highest |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SETR(nn.Module):
    """Segmentation Transformer"""
    
    def __init__(self, img_size=480, patch_size=16, num_classes=150,
                 embed_dim=1024, depth=24, num_heads=16):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # ViT Encoder
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4,
                                       activation='gelu', batch_first=True),
            num_layers=depth
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward_encoder(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class SETR_Naive(SETR):
    """SETR with Naive Upsampling decoder"""
    
    def __init__(self, num_classes=150, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Simple 1×1 conv
        self.head = nn.Conv2d(1024, num_classes, 1)
    
    def forward(self, x):
        H, W = x.shape[2:]
        
        x = self.forward_encoder(x)  # [B, N, D]
        
        # Reshape to spatial
        x = x.transpose(1, 2).reshape(-1, 1024, self.grid_size, self.grid_size)
        
        # Simple head + upsample
        x = self.head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


class SETR_PUP(SETR):
    """SETR with Progressive Upsampling"""
    
    def __init__(self, num_classes=150, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Progressive upsampling: 30 → 60 → 120 → 240 → 480
        self.up1 = self._upsample_block(1024, 256)
        self.up2 = self._upsample_block(256, 256)
        self.up3 = self._upsample_block(256, 256)
        self.up4 = self._upsample_block(256, 256)
        
        self.head = nn.Conv2d(256, num_classes, 1)
    
    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
    
    def forward(self, x):
        H, W = x.shape[2:]
        
        x = self.forward_encoder(x)
        x = x.transpose(1, 2).reshape(-1, 1024, self.grid_size, self.grid_size)
        
        # Progressive upsampling
        x = self.up1(x)  # 60×60
        x = self.up2(x)  # 120×120
        x = self.up3(x)  # 240×240
        x = self.up4(x)  # 480×480
        
        return self.head(x)


class SETR_MLA(SETR):
    """SETR with Multi-Level Aggregation"""
    
    def __init__(self, num_classes=150, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Aggregate from multiple transformer layers
        self.mla_layers = [5, 11, 17, 23]  # Layers to tap
        
        # Lateral connections
        self.laterals = nn.ModuleList([
            nn.Conv2d(1024, 256, 1) for _ in self.mla_layers
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.head = nn.Conv2d(256, num_classes, 1)
    
    def forward_encoder_multilevel(self, x):
        """Extract features from multiple layers"""
        B = x.shape[0]
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        features = []
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            if i in self.mla_layers:
                features.append(x)
        
        return features
    
    def forward(self, x):
        H, W = x.shape[2:]
        
        features = self.forward_encoder_multilevel(x)
        
        # Process each level
        processed = []
        for feat, lateral in zip(features, self.laterals):
            f = feat.transpose(1, 2).reshape(-1, 1024, self.grid_size, self.grid_size)
            f = lateral(f)
            f = F.interpolate(f, size=(H // 4, W // 4), mode='bilinear')
            processed.append(f)
        
        # Concatenate and fuse
        x = torch.cat(processed, dim=1)
        x = self.fusion(x)
        
        # Final upsample
        x = F.interpolate(x, size=(H, W), mode='bilinear')
        
        return self.head(x)
```

**Performance (ADE20K):**

| Model | mIoU | Params |
|-------|------|--------|
| SETR-Naive | 48.2 | 306M |
| SETR-PUP | 49.1 | 308M |
| SETR-MLA | 50.2 | 310M |

**Key Insights:**
1. **Pure transformer works**: No CNN backbone needed
2. **Multi-level helps**: Aggregating from multiple transformer layers improves results
3. **Large model needed**: ViT-L (24 layers) for good performance
4. **Decoder matters less**: Simple upsampling works reasonably

**Interview Tip:** SETR proved pure transformers can handle dense prediction. The key insight is that transformer output tokens preserve spatial relationships (each token = one patch location), so reshaping the sequence to 2D and upsampling works. Multi-level aggregation from different transformer depths helps capture both low-level and high-level features.

---

### Question 42
- [ ] Done
**Describe SegFormer architecture and its efficient self-attention mechanism for segmentation.**

**Answer:**

SegFormer combines a hierarchical transformer encoder with a lightweight MLP decoder. Its efficient self-attention uses reduced spatial resolution for keys/values, achieving O(n) complexity instead of O(n²). It's both accurate and efficient for semantic segmentation.

**Architecture Overview:**

```
Image 512×512
      ↓
Hierarchical Encoder (Mix Transformer)
├── Stage 1: 128×128, C1=32
├── Stage 2: 64×64, C2=64
├── Stage 3: 32×32, C3=160
└── Stage 4: 16×16, C4=256
      ↓
All-MLP Decoder (lightweight)
      ↓
Segmentation Mask 512×512
```

**Key Innovations:**

| Innovation | Description | Benefit |
|------------|-------------|---------|
| Efficient Self-Attention | Reduce K,V spatial resolution | O(n) complexity |
| Overlapping Patch Merge | 3×3 conv for downsampling | Local continuity |
| Mix-FFN | 3×3 depthwise conv in FFN | Positional info |
| All-MLP Decoder | Only MLPs, no convs | Simple, unified |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientSelfAttention(nn.Module):
    """SegFormer's efficient attention with spatial reduction"""
    
    def __init__(self, dim, num_heads, sr_ratio=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
        # Spatial reduction for K, V
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Query: full resolution
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # [B, heads, N, dim]
        
        # Key, Value: reduced resolution
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)  # [B, N', C]
            x_ = self.norm(x_)
        else:
            x_ = x
        
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, heads, N', dim]
        
        # Attention: Q @ K' where K' is reduced
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MixFFN(nn.Module):
    """Mix-FFN with 3×3 depth-wise convolution"""
    
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = int(dim * expansion)
        
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        x = self.fc1(x)
        
        # Apply depthwise conv (provides position info)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = self.act(x)
        x = self.fc2(x)
        
        return x


class MixTransformerBlock(nn.Module):
    """SegFormer encoder block"""
    
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)
    
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding for smooth downsampling"""
    
    def __init__(self, patch_size=7, stride=4, in_ch=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        return x, H, W


class MixTransformerEncoder(nn.Module):
    """Hierarchical Mix Transformer Encoder"""
    
    def __init__(self, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8],
                 depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        
        self.stages = nn.ModuleList()
        
        for i in range(4):
            # Patch embedding
            if i == 0:
                patch_embed = OverlapPatchEmbed(7, 4, 3, embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(3, 2, embed_dims[i-1], embed_dims[i])
            
            # Transformer blocks
            blocks = nn.ModuleList([
                MixTransformerBlock(embed_dims[i], num_heads[i], sr_ratios[i])
                for _ in range(depths[i])
            ])
            
            self.stages.append(nn.ModuleDict({
                'patch_embed': patch_embed,
                'blocks': blocks,
                'norm': nn.LayerNorm(embed_dims[i])
            }))
    
    def forward(self, x):
        features = []
        
        for stage in self.stages:
            x, H, W = stage['patch_embed'](x)
            
            for block in stage['blocks']:
                x = block(x, H, W)
            
            x = stage['norm'](x)
            x = x.transpose(1, 2).reshape(-1, x.shape[-1], H, W)
            features.append(x)
        
        return features


class SegFormerHead(nn.Module):
    """All-MLP Decoder"""
    
    def __init__(self, in_channels=[32, 64, 160, 256], 
                 embed_dim=256, num_classes=150):
        super().__init__()
        
        # Linear projections for each scale
        self.linear_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1),
                nn.BatchNorm2d(embed_dim)
            ) for c in in_channels
        ])
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        self.head = nn.Conv2d(embed_dim, num_classes, 1)
    
    def forward(self, features):
        # Upsample all to largest feature size
        target_size = features[0].shape[2:]
        
        fused = []
        for f, proj in zip(features, self.linear_fuse):
            f = proj(f)
            f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            fused.append(f)
        
        x = torch.cat(fused, dim=1)
        x = self.fusion(x)
        
        return self.head(x)


class SegFormer(nn.Module):
    """Complete SegFormer"""
    
    def __init__(self, num_classes=150):
        super().__init__()
        self.encoder = MixTransformerEncoder()
        self.decoder = SegFormerHead(num_classes=num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear')
        return out
```

**Complexity Comparison:**

| Model | Attention Complexity | For 512×512 |
|-------|---------------------|-------------|
| ViT | O(N²) | Very slow |
| Swin | O(N × M²) | Fast |
| SegFormer | O(N × N/R²) | Very fast |

Where R = spatial reduction ratio (8,4,2,1 per stage)

**Performance:**

| Model | ADE20K mIoU | Params | GFLOPs |
|-------|-------------|--------|--------|
| SegFormer-B0 | 37.4 | 3.8M | 8.4 |
| SegFormer-B2 | 47.3 | 27.4M | 62.4 |
| SegFormer-B5 | 51.0 | 84.6M | 183.3 |

**Interview Tip:** SegFormer's key innovations are: (1) efficient attention via K,V spatial reduction, (2) Mix-FFN with depthwise conv eliminating need for position embeddings, (3) lightweight MLP decoder. It's a great example of designing transformers specifically for dense prediction.

---

### Question 43
- [ ] Done
**How does Mask2Former unify semantic, instance, and panoptic segmentation with a single architecture?**

**Answer:**

Mask2Former treats all segmentation tasks as mask classification: predict N masks with corresponding class labels. For semantic segmentation, masks correspond to stuff classes. For instance segmentation, masks correspond to individual objects. For panoptic, both are combined.

**Unified Formulation:**

| Task | What Masks Represent | Example Output |
|------|---------------------|----------------|
| Semantic | Class regions | "road", "sky", "building" |
| Instance | Individual objects | "car_1", "car_2", "person_1" |
| Panoptic | Both stuff + things | "road", "car_1", "person_1" |

**Architecture:**

```
                    Image
                      ↓
              Backbone (ResNet/Swin)
                      ↓
              Pixel Decoder (FPN)
                      ↓
              Multi-Scale Features
                      ↓
         ┌────────────────────────┐
         │   Transformer Decoder  │
         │   (Masked Attention)   │
         │                        │
         │   N learnable queries  │
         └────────────────────────┘
                      ↓
              ┌───────┴───────┐
              ↓               ↓
         N Masks         N Classes
         [H×W×N]          [N×K]
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mask2Former(nn.Module):
    """Unified architecture for all segmentation tasks"""
    
    def __init__(self, num_classes, num_queries=100, hidden_dim=256):
        super().__init__()
        
        # Backbone
        self.backbone = ResNet50()
        
        # Pixel decoder (multi-scale feature fusion)
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_channels=[256, 512, 1024, 2048],
            hidden_dim=hidden_dim
        )
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer decoder
        self.transformer_decoder = MaskedTransformerDecoder(
            hidden_dim=hidden_dim,
            num_layers=9,
            num_heads=8
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # Multi-scale
        
        # Pixel decoder
        mask_features, multi_scale_features = self.pixel_decoder(features)
        
        # Prepare queries
        batch_size = x.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_feat = self.query_feat.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transformer decoder with masked attention
        outputs = self.transformer_decoder(
            query_feat, query_embed, multi_scale_features
        )
        
        # Predictions from each layer
        predictions = []
        for output in outputs:
            # Class prediction
            class_pred = self.class_embed(output)  # [B, N, K+1]
            
            # Mask prediction via dot product
            mask_embed = self.mask_embed(output)  # [B, N, D]
            mask_pred = torch.einsum('bnd,bdhw->bnhw', mask_embed, mask_features)
            
            predictions.append({'classes': class_pred, 'masks': mask_pred})
        
        return predictions


class MaskedCrossAttention(nn.Module):
    """Cross-attention with predicted mask as attention bias"""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query, key, value, mask_pred=None):
        B, N, D = query.shape
        _, HW, _ = key.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, D // self.num_heads)
        k = self.k_proj(key).reshape(B, HW, self.num_heads, D // self.num_heads)
        v = self.v_proj(value).reshape(B, HW, self.num_heads, D // self.num_heads)
        
        # Standard cross-attention
        attn = torch.einsum('bnhd,bmhd->bhnm', q, k) * self.scale
        
        # Apply mask as attention bias (key innovation!)
        if mask_pred is not None:
            # mask_pred: [B, N, H, W] -> [B, N, HW]
            mask_flat = mask_pred.flatten(2)
            # Only attend to predicted mask regions
            attn_mask = (mask_flat.sigmoid() < 0.5).unsqueeze(1)  # [B, 1, N, HW]
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        out = out.reshape(B, N, D)
        
        return self.out_proj(out)


# Task-specific inference
def semantic_inference(pred_classes, pred_masks):
    """Convert to semantic segmentation"""
    # [B, N, K+1] and [B, N, H, W]
    
    # Get class with highest score for each pixel
    mask_probs = pred_masks.sigmoid()  # [B, N, H, W]
    class_probs = pred_classes.softmax(-1)[:, :, :-1]  # Exclude no-object
    
    # For each class, sum contributions from all queries
    # [B, K, H, W]
    semantic_map = torch.einsum('bnhw,bnk->bkhw', mask_probs, class_probs)
    
    return semantic_map.argmax(dim=1)


def instance_inference(pred_classes, pred_masks, threshold=0.5):
    """Convert to instance segmentation"""
    # Keep high-confidence predictions
    scores = pred_classes.softmax(-1)[:, :, :-1].max(-1)[0]  # [B, N]
    
    instances = []
    for b in range(pred_classes.shape[0]):
        valid = scores[b] > threshold
        masks = pred_masks[b, valid].sigmoid() > 0.5
        classes = pred_classes[b, valid].argmax(-1)
        confs = scores[b, valid]
        
        instances.append({
            'masks': masks,
            'classes': classes,
            'scores': confs
        })
    
    return instances


def panoptic_inference(pred_classes, pred_masks, thing_ids, stuff_ids):
    """Convert to panoptic segmentation"""
    # Combine stuff (semantic) and things (instance)
    mask_probs = pred_masks.sigmoid()
    class_probs = pred_classes.softmax(-1)[:, :, :-1]
    
    # Assign each pixel to highest scoring query
    combined_scores = mask_probs * class_probs.max(-1, keepdim=True)[0]
    panoptic_seg = combined_scores.argmax(dim=1)
    
    # Create instance IDs for thing classes
    # (simplified - actual implementation more complex)
    return panoptic_seg
```

**Key Innovations:**

| Feature | Description |
|---------|-------------|
| Masked Attention | Use predicted masks to constrain attention to relevant regions |
| Multi-scale deformable attention | Efficient attention across resolution levels |
| Query sharing | Same queries work for all tasks |
| Auxiliary losses | Deep supervision at every decoder layer |

**Performance:**

| Task | Dataset | Mask2Former | Previous SOTA |
|------|---------|-------------|---------------|
| Semantic | ADE20K | 57.8 mIoU | 54.1 |
| Instance | COCO | 50.1 AP | 48.1 |
| Panoptic | COCO | 57.8 PQ | 55.1 |

**Interview Tip:** Mask2Former's key insight is that masked attention constrains each query to attend only to its predicted region, improving convergence and quality. The unified formulation shows that semantic/instance/panoptic are fundamentally the same problem with different post-processing.

---

### Question 44
- [ ] Done
**Explain SAM (Segment Anything Model) and its promptable segmentation capabilities.**

**Answer:**

SAM (Segment Anything Model) is a foundation model for segmentation that can segment any object given a prompt (point, box, or text). It was trained on 11M images with 1B+ masks, enabling zero-shot transfer to new objects and domains without fine-tuning.

**Key Capabilities:**

| Prompt Type | Input | Example |
|------------|-------|---------|
| Point | (x, y) click | Click on object to segment |
| Box | [x1, y1, x2, y2] | Draw bounding box |
| Mask | Binary mask | Refine existing mask |
| Text | Description | "the cat" (with CLIP) |

**Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                          SAM                             │
├─────────────────┬───────────────────┬───────────────────┤
│  Image Encoder  │  Prompt Encoder   │   Mask Decoder    │
│  (ViT-H: 632M)  │  (lightweight)    │  (transformer)    │
│                 │                   │                   │
│  Image →        │  Points, boxes,   │  → Masks + IoU    │
│  Embeddings     │  masks → embed    │     scores        │
└─────────────────┴───────────────────┴───────────────────┘
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    """Segment Anything Model"""
    
    def __init__(self, image_encoder, prompt_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
    
    def forward(self, image, points=None, boxes=None, masks=None):
        # Encode image (expensive, do once per image)
        image_embedding = self.image_encoder(image)
        
        # Encode prompts (cheap, can iterate)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, boxes=boxes, masks=masks
        )
        
        # Decode masks
        masks, iou_scores = self.mask_decoder(
            image_embedding, sparse_embeddings, dense_embeddings
        )
        
        return masks, iou_scores


class PromptEncoder(nn.Module):
    """Encode points, boxes, and masks into embeddings"""
    
    def __init__(self, embed_dim=256, image_size=1024, mask_in_chans=16):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Point embeddings (foreground/background)
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(2)  # fg, bg
        ])
        
        # Box corner embeddings
        self.box_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(2)  # top-left, bottom-right
        ])
        
        # Positional encoding for points
        self.pe_layer = PositionalEncoding2D(embed_dim // 2)
        
        # Mask embedding (for mask prompts)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, 2, stride=2),
            nn.LayerNorm([mask_in_chans // 4, image_size // 2, image_size // 2]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, 2, stride=2),
            nn.LayerNorm([mask_in_chans, image_size // 4, image_size // 4]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, 1),
        )
        
        # No-mask embedding
        self.no_mask_embed = nn.Embedding(1, embed_dim)
    
    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings = []
        
        # Encode points
        if points is not None:
            coords, labels = points  # coords: [B, N, 2], labels: [B, N]
            point_embed = self.pe_layer(coords)
            
            # Add foreground/background embedding
            for i, label in enumerate([0, 1]):
                point_embed[labels == label] += self.point_embeddings[label].weight
            
            sparse_embeddings.append(point_embed)
        
        # Encode boxes (as two corner points)
        if boxes is not None:
            # boxes: [B, 4] -> corner embeddings
            corners = boxes.reshape(-1, 2, 2)  # [B, 2, 2]
            corner_embed = self.pe_layer(corners)
            corner_embed[:, 0] += self.box_embeddings[0].weight
            corner_embed[:, 1] += self.box_embeddings[1].weight
            sparse_embeddings.append(corner_embed)
        
        # Concatenate sparse embeddings
        if sparse_embeddings:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        else:
            sparse_embeddings = torch.empty(1, 0, self.embed_dim)
        
        # Encode mask prompt
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
            dense_embeddings = dense_embeddings.expand(-1, -1, 64, 64)
        
        return sparse_embeddings, dense_embeddings


class MaskDecoder(nn.Module):
    """Decode image + prompt embeddings into masks"""
    
    def __init__(self, embed_dim=256, num_masks=3):
        super().__init__()
        self.num_masks = num_masks  # Output multiple mask candidates
        
        # IoU prediction token + mask tokens
        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(num_masks, embed_dim)
        
        # Two-way transformer for cross-attention
        self.transformer = TwoWayTransformer(
            depth=2, embed_dim=embed_dim, num_heads=8
        )
        
        # Upscaling
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, 2, stride=2),
            nn.LayerNorm([embed_dim // 4, 128, 128]),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, stride=2),
            nn.GELU(),
        )
        
        # Output heads
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim // 8)
            ) for _ in range(num_masks)
        ])
        
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_masks)
        )
    
    def forward(self, image_embedding, sparse_embeddings, dense_embeddings):
        B = image_embedding.shape[0]
        
        # Prepare tokens
        output_tokens = torch.cat([
            self.iou_token.weight.unsqueeze(0),
            self.mask_tokens.weight.unsqueeze(0)
        ], dim=1)
        output_tokens = output_tokens.expand(B, -1, -1)
        tokens = torch.cat([output_tokens, sparse_embeddings], dim=1)
        
        # Add dense embeddings to image
        src = image_embedding + dense_embeddings
        
        # Two-way cross-attention
        hs, src = self.transformer(src, tokens)
        
        # Extract outputs
        iou_token_out = hs[:, 0]
        mask_tokens_out = hs[:, 1:1 + self.num_masks]
        
        # Upscale image features
        upscaled = self.output_upscaling(src)
        
        # Generate masks via hypernetworks
        masks = []
        for i, mlp in enumerate(self.output_hypernetworks_mlps):
            hyper = mlp(mask_tokens_out[:, i])
            mask = torch.einsum('bd,bdhw->bhw', hyper, upscaled)
            masks.append(mask)
        
        masks = torch.stack(masks, dim=1)  # [B, num_masks, H, W]
        
        # Predict IoU
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        return masks, iou_pred


# Usage example
def segment_with_sam(sam_model, image, points):
    """Interactive segmentation with SAM"""
    # Precompute image embedding (only once)
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(image)
    
    # Interactive loop
    while True:
        # Get user click
        click_point = get_user_click()
        
        # Encode prompt
        sparse, dense = sam_model.prompt_encoder(
            points=(click_point, torch.tensor([[1]]))  # foreground
        )
        
        # Decode mask
        masks, scores = sam_model.mask_decoder(
            image_embedding, sparse, dense
        )
        
        # Select best mask
        best_idx = scores.argmax()
        best_mask = masks[0, best_idx]
        
        display(best_mask)
```

**Training Data:**

| Metric | Value |
|--------|-------|
| Images | 11 million |
| Masks | 1.1 billion |
| Annotation | Data engine (model-in-the-loop) |
| Categories | Class-agnostic |

**Key Features:**
1. **Zero-shot transfer**: Works on unseen objects/domains
2. **Multiple outputs**: Returns 3 masks with IoU scores (for ambiguity)
3. **Efficient**: Image encoder runs once, prompts are cheap
4. **Composable**: Chain SAM with other models (detection, tracking)

**Interview Tip:** SAM is a foundation model for segmentation—trained to generalize across all possible objects. The key innovations are: promptable interface, massive diverse training data via the data engine, and the ambiguity-aware multi-mask output. It's designed for interactive annotation and can be composed with task-specific models.

---

## Practical Considerations

### Question 45
- [ ] Done
**How do you handle temporal consistency in video semantic/instance segmentation?**

**Answer:**

Temporal consistency ensures segmentation masks remain stable across frames, avoiding flickering and ID switches. Key approaches include using optical flow warping, memory networks to propagate features, and tracking-based association.

**Consistency Issues in Video:**

| Problem | Manifestation |
|---------|---------------|
| Flickering | Mask appears/disappears between frames |
| Boundary jitter | Edges change erratically |
| ID switching | Same object gets different IDs |
| Missing frames | Occlusion causes gaps |

**Approaches:**

| Method | How It Works | Use Case |
|--------|-------------|----------|
| Optical flow | Warp previous mask/features | Short-term |
| Memory networks | Store and query past features | Long-term |
| Tracking | Associate detections across frames | Instance seg |
| Temporal smoothing | Post-process predictions | All |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowBasedConsistency(nn.Module):
    """Use optical flow for temporal consistency"""
    
    def __init__(self, segmentation_model, flow_model):
        super().__init__()
        self.seg_model = segmentation_model
        self.flow_model = flow_model
        self.alpha = 0.7  # Blend weight
    
    def forward(self, frames):
        """Process video with flow-based consistency"""
        B, T, C, H, W = frames.shape
        
        outputs = []
        prev_seg = None
        
        for t in range(T):
            frame = frames[:, t]
            
            # Current frame prediction
            current_seg = self.seg_model(frame)
            
            if prev_seg is not None and t > 0:
                # Estimate optical flow
                flow = self.flow_model(frames[:, t-1], frame)
                
                # Warp previous segmentation
                warped_seg = self.warp(prev_seg, flow)
                
                # Blend with current prediction
                current_seg = self.alpha * current_seg + (1 - self.alpha) * warped_seg
            
            outputs.append(current_seg)
            prev_seg = current_seg.detach()
        
        return torch.stack(outputs, dim=1)
    
    def warp(self, x, flow):
        """Warp tensor using optical flow"""
        B, C, H, W = x.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device)
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        
        # Add flow
        grid = grid + flow.permute(0, 2, 3, 1)
        
        # Normalize to [-1, 1]
        grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
        
        # Sample
        return F.grid_sample(x, grid, mode='bilinear', align_corners=True)


class MemoryBasedVideoSegmentation(nn.Module):
    """STM-style memory network for video segmentation"""
    
    def __init__(self, encoder, memory_size=5):
        super().__init__()
        self.encoder = encoder
        self.memory_size = memory_size
        
        # Memory components
        self.key_encoder = nn.Conv2d(256, 128, 3, padding=1)
        self.value_encoder = nn.Conv2d(256 + 1, 256, 3, padding=1)  # +1 for mask
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )
    
    def forward(self, frames, first_mask):
        """Video object segmentation with memory"""
        B, T, C, H, W = frames.shape
        
        # Initialize memory with first frame
        first_feat = self.encoder(frames[:, 0])
        memory_keys = [self.key_encoder(first_feat)]
        memory_values = [self.value_encoder(torch.cat([first_feat, first_mask], dim=1))]
        
        outputs = [first_mask]
        
        for t in range(1, T):
            # Encode current frame
            query_feat = self.encoder(frames[:, t])
            query_key = self.key_encoder(query_feat)
            
            # Match against memory
            matched_value = self.memory_read(query_key, memory_keys, memory_values)
            
            # Decode mask
            mask = self.decoder(matched_value)
            outputs.append(mask.sigmoid())
            
            # Update memory (every few frames)
            if t % 5 == 0 and len(memory_keys) < self.memory_size:
                memory_keys.append(query_key)
                memory_values.append(self.value_encoder(
                    torch.cat([query_feat, mask.sigmoid()], dim=1)
                ))
        
        return torch.stack(outputs, dim=1)
    
    def memory_read(self, query_key, memory_keys, memory_values):
        """Read from memory using attention"""
        B, C, H, W = query_key.shape
        
        query = query_key.flatten(2)  # [B, C, HW]
        
        similarities = []
        for mem_key in memory_keys:
            key = mem_key.flatten(2)  # [B, C, HW]
            sim = torch.bmm(query.transpose(1, 2), key)  # [B, HW, HW]
            similarities.append(sim)
        
        # Softmax over all memory positions
        all_sim = torch.cat(similarities, dim=-1)  # [B, HW, T*HW]
        attn = F.softmax(all_sim, dim=-1)
        
        # Read values
        all_values = torch.cat([v.flatten(2) for v in memory_values], dim=-1)
        out = torch.bmm(all_values, attn.transpose(1, 2))  # [B, C, HW]
        
        return out.reshape(B, -1, H, W)


class TrackingBasedInstanceSegmentation(nn.Module):
    """Track instances across frames"""
    
    def __init__(self, detector, tracker):
        super().__init__()
        self.detector = detector
        self.tracker = tracker
    
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        
        all_tracks = []
        active_tracks = {}
        next_id = 0
        
        for t in range(T):
            # Detect instances
            detections = self.detector(frames[:, t])
            
            if t == 0:
                # Initialize tracks
                for det in detections:
                    active_tracks[next_id] = det
                    next_id += 1
            else:
                # Associate with previous tracks
                matches = self.tracker.associate(
                    active_tracks, detections
                )
                
                # Update tracks
                for track_id, det_idx in matches:
                    active_tracks[track_id] = detections[det_idx]
                
                # Add new tracks for unmatched detections
                unmatched = set(range(len(detections))) - set(m[1] for m in matches)
                for det_idx in unmatched:
                    active_tracks[next_id] = detections[det_idx]
                    next_id += 1
            
            all_tracks.append(dict(active_tracks))
        
        return all_tracks


def temporal_smoothing(predictions, window_size=5):
    """Post-process temporal smoothing"""
    T = len(predictions)
    smoothed = []
    
    for t in range(T):
        start = max(0, t - window_size // 2)
        end = min(T, t + window_size // 2 + 1)
        
        # Average nearby predictions
        window = torch.stack(predictions[start:end])
        smoothed.append(window.mean(dim=0))
    
    return smoothed
```

**Comparison:**

| Method | Temporal Range | Speed | Accuracy |
|--------|---------------|-------|----------|
| Flow warping | 1 frame | Fast | Good for motion |
| Memory networks | Many frames | Medium | Best for occlusion |
| Tracking | All frames | Fast | Best for instances |
| CRF smoothing | Local | Fast | Boundary quality |

**Interview Tip:** Flow-based methods handle smooth motion well but fail for occlusions. Memory networks are better for long-term consistency but more expensive. For instance segmentation, tracking-based association (like in SORT or ByteTrack) is most practical. Combine multiple approaches for best results.

---

### Question 46
- [ ] Done
**What approaches work for domain adaptation in segmentation across different imaging modalities?**

**Answer:**

Domain adaptation for segmentation addresses the shift between source domain (e.g., synthetic data) and target domain (e.g., real images). Key approaches include adversarial training, self-training with pseudo-labels, style transfer, and feature alignment.

**Domain Shift Types:**

| Shift Type | Example |
|------------|---------|
| Synthetic → Real | GTA5 → Cityscapes |
| Day → Night | Daytime → Nighttime driving |
| Clear → Adverse | Sunny → Rainy/Foggy |
| Modality | RGB → Depth, CT → MRI |

**Adaptation Approaches:**

| Method | Idea | Strength |
|--------|------|----------|
| Adversarial | Discriminator aligns features | Global alignment |
| Self-training | Pseudo-labels on target | Uses target data |
| Style transfer | Source looks like target | Visual alignment |
| Feature alignment | Match statistics (MMD) | Simple, effective |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialDomainAdaptation(nn.Module):
    """Adversarial training for domain adaptation"""
    
    def __init__(self, segmenter, discriminator):
        super().__init__()
        self.segmenter = segmenter  # Source-trained
        self.discriminator = discriminator
    
    def forward(self, source_images, target_images, source_labels):
        # Segment both domains
        source_pred = self.segmenter(source_images)
        target_pred = self.segmenter(target_images)
        
        return source_pred, target_pred
    
    def compute_losses(self, source_images, target_images, source_labels):
        source_pred, target_pred = self(source_images, target_images, source_labels)
        
        # Segmentation loss on source
        seg_loss = F.cross_entropy(source_pred, source_labels)
        
        # Adversarial loss - fool discriminator
        source_domain = self.discriminator(source_pred)
        target_domain = self.discriminator(target_pred)
        
        # Source = 0, Target = 1; segmenter wants target to look like source
        adv_loss = F.binary_cross_entropy_with_logits(
            target_domain, torch.zeros_like(target_domain)
        )
        
        return seg_loss, adv_loss


class OutputSpaceDiscriminator(nn.Module):
    """Discriminate on segmentation output space"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_classes, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, stride=2, padding=1),
        )
    
    def forward(self, x):
        return self.conv(F.softmax(x, dim=1))


class SelfTrainingAdapter(nn.Module):
    """Self-training with pseudo-labels"""
    
    def __init__(self, segmenter, confidence_threshold=0.9):
        super().__init__()
        self.segmenter = segmenter
        self.threshold = confidence_threshold
        
        # EMA teacher
        self.teacher = copy.deepcopy(segmenter)
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def generate_pseudo_labels(self, target_images):
        """Generate confident pseudo-labels"""
        with torch.no_grad():
            teacher_pred = self.teacher(target_images)
            probs = F.softmax(teacher_pred, dim=1)
            
            # Get confidence and predictions
            confidence, pseudo_labels = probs.max(dim=1)
            
            # Mask low-confidence pixels
            mask = confidence > self.threshold
            
        return pseudo_labels, mask
    
    def train_step(self, source_images, source_labels, target_images):
        # Source supervised loss
        source_pred = self.segmenter(source_images)
        source_loss = F.cross_entropy(source_pred, source_labels)
        
        # Target self-training loss
        pseudo_labels, mask = self.generate_pseudo_labels(target_images)
        target_pred = self.segmenter(target_images)
        
        # Loss only on confident pixels
        target_loss = F.cross_entropy(
            target_pred, pseudo_labels, reduction='none'
        )
        target_loss = (target_loss * mask.float()).sum() / mask.sum().clamp(min=1)
        
        total_loss = source_loss + 0.5 * target_loss
        
        # Update EMA teacher
        self.update_teacher()
        
        return total_loss
    
    def update_teacher(self, momentum=0.999):
        for t_param, s_param in zip(self.teacher.parameters(), 
                                     self.segmenter.parameters()):
            t_param.data = momentum * t_param.data + (1 - momentum) * s_param.data


class StyleTransferAdapter(nn.Module):
    """Style transfer for domain adaptation"""
    
    def __init__(self, segmenter, style_transfer_net):
        super().__init__()
        self.segmenter = segmenter
        self.style_net = style_transfer_net
    
    def train_step(self, source_images, source_labels, target_images):
        # Transfer source to target style
        stylized_source = self.style_net(source_images, target_images)
        
        # Train on stylized images
        pred = self.segmenter(stylized_source)
        loss = F.cross_entropy(pred, source_labels)
        
        return loss


class FeatureAlignmentAdapter(nn.Module):
    """Align feature statistics between domains"""
    
    def __init__(self, segmenter):
        super().__init__()
        self.segmenter = segmenter
    
    def mmd_loss(self, source_features, target_features):
        """Maximum Mean Discrepancy"""
        # Flatten spatial dimensions
        source = source_features.flatten(2).mean(dim=2)  # [B, C]
        target = target_features.flatten(2).mean(dim=2)
        
        # MMD with RBF kernel
        xx = self.rbf_kernel(source, source)
        yy = self.rbf_kernel(target, target)
        xy = self.rbf_kernel(source, target)
        
        return xx.mean() + yy.mean() - 2 * xy.mean()
    
    def rbf_kernel(self, x, y, gamma=1.0):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-gamma * (diff ** 2).sum(dim=-1))
    
    def train_step(self, source_images, source_labels, target_images):
        # Get features from intermediate layer
        source_feat = self.segmenter.backbone(source_images)
        target_feat = self.segmenter.backbone(target_images)
        
        # Segmentation loss
        source_pred = self.segmenter.decoder(source_feat)
        seg_loss = F.cross_entropy(source_pred, source_labels)
        
        # Alignment loss
        align_loss = self.mmd_loss(source_feat, target_feat)
        
        return seg_loss + 0.1 * align_loss
```

**Best Practices:**

| Scenario | Recommended Approach |
|----------|---------------------|
| Synthetic → Real | Style transfer + self-training |
| Similar domains | Feature alignment |
| No target labels | Adversarial + pseudo-labels |
| Few target labels | Semi-supervised + adaptation |

**Performance (GTA5 → Cityscapes):**

| Method | mIoU |
|--------|------|
| Source only | 35.4 |
| AdaptSegNet | 42.4 |
| DACS (self-training) | 52.1 |
| DAFormer | 68.3 |

**Interview Tip:** Self-training with strong data augmentation (like DACS) often outperforms adversarial methods. The key is confidence thresholding for pseudo-labels and mixing source/target samples. For visual domain shift, style transfer as preprocessing is effective.

---

### Question 47
- [ ] Done
**Explain active learning strategies for efficient mask annotation in segmentation tasks.**

**Answer:**

Active learning selects the most informative samples for annotation, reducing labeling cost. For segmentation, strategies consider pixel-level uncertainty, region diversity, and annotation effort (some masks are harder to draw than others).

**Active Learning Pipeline:**

```
Initial labeled set → Train model → Select informative samples → Annotate → Repeat
         ↑                                    ↓
         └────────────────────────────────────┘
```

**Selection Strategies:**

| Strategy | Metric | What It Selects |
|----------|--------|-----------------|
| Uncertainty | Entropy, margin | Confusing samples |
| Diversity | Core-set, cluster | Representative samples |
| Expected Change | Gradient magnitude | High-impact samples |
| Combined | Hybrid | Best of both |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActiveLearningSegmentation:
    """Active learning for segmentation"""
    
    def __init__(self, model, strategy='uncertainty'):
        self.model = model
        self.strategy = strategy
    
    def select_samples(self, unlabeled_images, num_samples):
        """Select most informative samples"""
        if self.strategy == 'uncertainty':
            scores = self.uncertainty_sampling(unlabeled_images)
        elif self.strategy == 'diversity':
            return self.diversity_sampling(unlabeled_images, num_samples)
        elif self.strategy == 'badge':
            return self.badge_sampling(unlabeled_images, num_samples)
        else:
            scores = torch.rand(len(unlabeled_images))
        
        # Select top-k
        _, indices = scores.topk(num_samples)
        return indices.tolist()
    
    def uncertainty_sampling(self, images):
        """Select based on prediction uncertainty"""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for img in images:
                pred = self.model(img.unsqueeze(0))
                probs = F.softmax(pred, dim=1)
                
                # Pixel-wise entropy
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                
                # Aggregate to image-level score
                score = entropy.mean()
                scores.append(score)
        
        return torch.tensor(scores)
    
    def margin_sampling(self, images):
        """Select based on margin between top 2 predictions"""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for img in images:
                pred = self.model(img.unsqueeze(0))
                probs = F.softmax(pred, dim=1)
                
                # Margin = diff between top 2
                top2, _ = probs.topk(2, dim=1)
                margin = top2[:, 0] - top2[:, 1]
                
                # Lower margin = more uncertain
                score = 1 - margin.mean()
                scores.append(score)
        
        return torch.tensor(scores)
    
    def diversity_sampling(self, images, num_samples):
        """Core-set approach for diversity"""
        self.model.eval()
        
        # Extract features
        features = []
        with torch.no_grad():
            for img in images:
                feat = self.model.backbone(img.unsqueeze(0))
                feat = feat.mean(dim=[2, 3])  # Global pool
                features.append(feat.squeeze())
        
        features = torch.stack(features)
        
        # Greedy core-set selection
        selected = []
        remaining = list(range(len(images)))
        
        # Start with random
        first = np.random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        # Greedy farthest-first
        for _ in range(num_samples - 1):
            selected_feats = features[selected]
            remaining_feats = features[remaining]
            
            # Distance to nearest selected
            distances = torch.cdist(remaining_feats, selected_feats).min(dim=1)[0]
            
            # Select farthest
            farthest_idx = distances.argmax().item()
            selected.append(remaining[farthest_idx])
            remaining.remove(remaining[farthest_idx])
        
        return selected
    
    def badge_sampling(self, images, num_samples):
        """BADGE: Diversity in gradient space"""
        self.model.eval()
        
        # Get gradient embeddings
        grad_embeddings = []
        for img in images:
            img = img.unsqueeze(0).requires_grad_(False)
            pred = self.model(img)
            
            # Hallucinate labels (use prediction)
            pseudo_label = pred.argmax(dim=1)
            
            # Compute gradient w.r.t. last layer
            loss = F.cross_entropy(pred, pseudo_label)
            self.model.zero_grad()
            loss.backward()
            
            # Get gradient of last layer weights
            grad = self.model.head.weight.grad.flatten()
            grad_embeddings.append(grad.clone())
        
        grad_embeddings = torch.stack(grad_embeddings)
        
        # K-means++ on gradient embeddings
        return self._kmeans_plus_plus(grad_embeddings, num_samples)
    
    def _kmeans_plus_plus(self, features, k):
        """K-means++ initialization for diverse sampling"""
        n = features.shape[0]
        selected = [np.random.randint(n)]
        
        for _ in range(k - 1):
            distances = torch.cdist(features, features[selected]).min(dim=1)[0]
            probs = distances ** 2
            probs = probs / probs.sum()
            
            next_idx = torch.multinomial(probs, 1).item()
            selected.append(next_idx)
        
        return selected


class RegionBasedActiveLearning:
    """Segment-level active learning"""
    
    def __init__(self, model):
        self.model = model
    
    def select_regions(self, image, num_regions=5):
        """Select uncertain regions for annotation"""
        self.model.eval()
        
        with torch.no_grad():
            pred = self.model(image.unsqueeze(0))
            probs = F.softmax(pred, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)[0]
        
        # Superpixel segmentation
        superpixels = self._get_superpixels(image)
        
        # Score each superpixel
        region_scores = []
        for region_id in range(superpixels.max() + 1):
            mask = superpixels == region_id
            region_entropy = entropy[mask].mean()
            region_scores.append((region_id, region_entropy.item()))
        
        # Select top uncertain regions
        region_scores.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in region_scores[:num_regions]]


def annotation_cost_aware_selection(images, model, budget):
    """Consider annotation difficulty in selection"""
    scores = []
    
    for img in images:
        # Uncertainty
        pred = model(img.unsqueeze(0))
        uncertainty = compute_entropy(pred).mean()
        
        # Estimated annotation cost (more objects = harder)
        num_segments = estimate_segments(pred)
        cost = 1 + 0.1 * num_segments  # Linear cost model
        
        # Value per unit cost
        value = uncertainty / cost
        scores.append(value)
    
    # Select highest value samples within budget
    return select_within_budget(scores, images, budget)
```

**Comparison:**

| Strategy | Pros | Cons |
|----------|------|------|
| Entropy | Simple, fast | May select outliers |
| Diversity | Covers data space | May miss hard cases |
| BADGE | Best of both | Computationally expensive |
| Region-based | Fine-grained | More annotation overhead |

**Interview Tip:** Pure uncertainty sampling often selects redundant samples. Combining uncertainty with diversity (like BADGE) gives the best results. For segmentation specifically, consider region-level active learning—you can annotate only uncertain regions, not the entire image.

---

### Question 48
- [ ] Done
**How do you implement uncertainty quantification in segmentation predictions?**

**Answer:**

Uncertainty quantification estimates how confident the model is about each pixel prediction. Key approaches include MC Dropout, deep ensembles, and learned uncertainty. This helps identify unreliable predictions and areas needing human review.

**Types of Uncertainty:**

| Type | Source | Example |
|------|--------|---------|
| Aleatoric | Data noise | Blurry boundaries |
| Epistemic | Model uncertainty | Novel objects |
| Total | Both combined | Overall uncertainty |

**Approaches:**

| Method | How It Works | Pros/Cons |
|--------|-------------|-----------|
| MC Dropout | Multiple forward passes with dropout | Simple, slow inference |
| Deep Ensembles | Train multiple models | Best quality, expensive |
| Learned | Network predicts uncertainty | Fast, needs proper training |
| Evidential | Dirichlet distribution | Single pass, novel |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropoutUncertainty(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, segmenter, num_samples=10, dropout_rate=0.1):
        super().__init__()
        self.segmenter = segmenter
        self.num_samples = num_samples
        
        # Enable dropout during inference
        self.enable_dropout(dropout_rate)
    
    def enable_dropout(self, rate):
        """Add dropout layers and keep them active during eval"""
        for module in self.segmenter.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate
            elif isinstance(module, nn.Conv2d):
                # Add dropout after conv layers
                pass
    
    def forward(self, x):
        self.segmenter.train()  # Keep dropout active
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.segmenter(x)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)  # [T, B, C, H, W]
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Uncertainty: predictive entropy
        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1)
        
        # Mutual information (epistemic uncertainty)
        individual_entropy = -(predictions * torch.log(predictions + 1e-8)).sum(dim=2)
        expected_entropy = individual_entropy.mean(dim=0)
        mutual_info = entropy - expected_entropy
        
        return mean_pred, entropy, mutual_info


class DeepEnsembleUncertainty(nn.Module):
    """Ensemble of independently trained models"""
    
    def __init__(self, model_class, num_models=5, **model_kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            model_class(**model_kwargs) for _ in range(num_models)
        ])
    
    def forward(self, x):
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)  # [M, B, C, H, W]
        
        # Mean and variance
        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0).mean(dim=1)  # Average across classes
        
        # Predictive entropy
        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1)
        
        return mean_pred, entropy, variance


class LearnedUncertainty(nn.Module):
    """Network that predicts its own uncertainty"""
    
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        
        # Segmentation head
        self.seg_head = nn.Conv2d(256, num_classes, 1)
        
        # Uncertainty head (aleatoric)
        self.uncertainty_head = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Logits
        logits = self.seg_head(features)
        
        # Log variance (for numerical stability)
        log_var = self.uncertainty_head(features)
        
        return logits, log_var
    
    def loss(self, logits, log_var, target):
        """Uncertainty-aware loss"""
        # Heteroscedastic loss
        precision = torch.exp(-log_var)
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        
        # Weight by predicted precision + regularization
        loss = precision * ce_loss + 0.5 * log_var
        
        return loss.mean()
    
    def get_uncertainty(self, x):
        logits, log_var = self.forward(x)
        
        # Aleatoric uncertainty
        aleatoric = log_var.exp().mean(dim=1)
        
        # Epistemic from softmax entropy
        probs = F.softmax(logits, dim=1)
        epistemic = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        
        return F.softmax(logits, dim=1), aleatoric, epistemic


class EvidentialSegmentation(nn.Module):
    """Evidential deep learning for segmentation"""
    
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Predict Dirichlet concentration parameters
        self.evidence_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1),
            nn.Softplus()  # Evidence must be positive
        )
    
    def forward(self, x):
        features = self.backbone(x)
        evidence = self.evidence_head(features)
        
        # Dirichlet parameters
        alpha = evidence + 1  # Ensure > 1
        
        return alpha
    
    def get_predictions_and_uncertainty(self, x):
        alpha = self.forward(x)
        S = alpha.sum(dim=1, keepdim=True)
        
        # Expected probability
        prob = alpha / S
        
        # Epistemic uncertainty (Dirichlet entropy)
        uncertainty = self.num_classes / S.squeeze(1)
        
        return prob, uncertainty
    
    def loss(self, alpha, target):
        """Evidence-based loss"""
        S = alpha.sum(dim=1, keepdim=True)
        
        # Expected cross-entropy
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        loss = (target_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
        
        # KL regularization
        alpha_tilde = target_onehot + (1 - target_onehot) * alpha
        kl = self.kl_dirichlet(alpha_tilde)
        
        return (loss + 0.01 * kl).mean()


def calibration_analysis(predictions, uncertainties, targets):
    """Analyze uncertainty calibration"""
    # Expected Calibration Error
    confidences = predictions.max(dim=1)[0].flatten()
    correct = (predictions.argmax(dim=1) == targets).float().flatten()
    
    n_bins = 15
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    ece = 0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correct[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    
    ece = ece / len(confidences)
    return ece.item()
```

**Comparison:**

| Method | Quality | Speed | Training Cost |
|--------|---------|-------|---------------|
| MC Dropout | Good | Slow (T passes) | Low |
| Ensembles | Best | Slow (M models) | High |
| Learned | OK | Fast | Medium |
| Evidential | Good | Fast | Medium |

**Use Cases:**

| Application | Use Uncertainty For |
|-------------|---------------------|
| Medical imaging | Flag uncertain regions for doctor |
| Autonomous driving | Trigger human takeover |
| Active learning | Select samples for annotation |
| OOD detection | Identify novel objects |

**Interview Tip:** Ensembles give best uncertainty estimates but are impractical for real-time. MC Dropout is a good middle ground. For deployment, learned uncertainty is fastest but needs careful training (heteroscedastic loss). Always validate calibration—confident predictions should be accurate.

---

### Question 49
- [ ] Done
**What techniques help segment objects in adverse weather or lighting conditions?**

**Answer:**

Adverse conditions (rain, fog, night, glare) degrade image quality and change object appearance. Techniques include multi-sensor fusion, domain adaptation, image restoration, and training with synthetic adverse data.

**Challenges by Condition:**

| Condition | Challenge | Example |
|-----------|-----------|---------|
| Rain | Reflections, blur, streaks | Wet road confusion |
| Fog | Low contrast, limited visibility | Miss distant objects |
| Night | Low light, headlight glare | Poor visibility |
| Snow | Occlusion, white-out | Lane confusion |

**Approaches:**

| Technique | How It Helps |
|-----------|--------------|
| Multi-sensor fusion | LiDAR/radar unaffected by weather |
| Image restoration | De-rain/de-fog before segmentation |
| Domain adaptation | Train on adverse + normal together |
| Robust augmentation | Simulate weather during training |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class WeatherAugmentation:
    """Simulate adverse weather for training robustness"""
    
    def __init__(self, fog_prob=0.3, rain_prob=0.3, night_prob=0.2):
        self.fog_prob = fog_prob
        self.rain_prob = rain_prob
        self.night_prob = night_prob
    
    def __call__(self, image):
        if torch.rand(1) < self.fog_prob:
            image = self.add_fog(image)
        if torch.rand(1) < self.rain_prob:
            image = self.add_rain(image)
        if torch.rand(1) < self.night_prob:
            image = self.simulate_night(image)
        return image
    
    def add_fog(self, image, intensity=None):
        """Simulate fog using atmospheric scattering model"""
        if intensity is None:
            intensity = torch.rand(1).item() * 0.5 + 0.3
        
        C, H, W = image.shape
        
        # Depth-based transmission (approximation)
        depth = torch.linspace(0.1, 1.0, H).unsqueeze(1).expand(H, W)
        transmission = torch.exp(-intensity * depth)
        
        # Atmospheric light (whitish)
        atmospheric_light = 0.9
        
        # Scattering equation: I = J*t + A*(1-t)
        foggy = image * transmission + atmospheric_light * (1 - transmission)
        
        return foggy.clamp(0, 1)
    
    def add_rain(self, image, intensity=None):
        """Add rain streaks"""
        if intensity is None:
            intensity = torch.rand(1).item() * 0.3 + 0.1
        
        C, H, W = image.shape
        
        # Rain streaks (elongated in vertical direction)
        noise = torch.randn(1, H // 4, W // 4)
        noise = F.interpolate(noise.unsqueeze(0), size=(H, W), mode='bilinear')[0, 0]
        
        # Threshold to create sparse streaks
        streaks = (noise > 1.5).float() * intensity
        
        # Apply to image (brighten where rain)
        rainy = image + streaks.unsqueeze(0)
        
        return rainy.clamp(0, 1)
    
    def simulate_night(self, image, darkness=None):
        """Simulate nighttime conditions"""
        if darkness is None:
            darkness = torch.rand(1).item() * 0.5 + 0.3
        
        # Reduce brightness
        night = image * (1 - darkness)
        
        # Add noise (camera sensor noise in low light)
        noise = torch.randn_like(image) * 0.02
        night = night + noise
        
        return night.clamp(0, 1)


class ImageRestorationModule(nn.Module):
    """Restore degraded images before segmentation"""
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder-decoder for restoration
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        restored = self.decoder(features)
        return restored + x  # Residual connection


class MultiSensorFusion(nn.Module):
    """Fuse RGB with LiDAR/depth for robustness"""
    
    def __init__(self, num_classes):
        super().__init__()
        
        # RGB branch
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Depth/LiDAR branch
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),  # Concatenated features
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
        
        # Confidence-weighted fusion
        self.rgb_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        
        # Confidence-based weighting
        rgb_conf = self.rgb_confidence(rgb_feat)
        depth_conf = 1 - rgb_conf  # Complementary
        
        # Weighted fusion
        fused = rgb_conf.view(-1, 1, 1, 1) * rgb_feat + \
                depth_conf.view(-1, 1, 1, 1) * depth_feat
        
        # Also concatenate for richer features
        concat = torch.cat([rgb_feat, depth_feat], dim=1)
        fused = self.fusion(concat) + fused
        
        return self.decoder(fused)


class AdverseWeatherAdapter(nn.Module):
    """Domain adaptation for adverse weather"""
    
    def __init__(self, base_model, weather_conditions=['rain', 'fog', 'night']):
        super().__init__()
        self.base_model = base_model
        
        # Condition-specific normalization layers
        self.condition_norms = nn.ModuleDict({
            cond: nn.ModuleList([
                nn.InstanceNorm2d(64),
                nn.InstanceNorm2d(128),
                nn.InstanceNorm2d(256)
            ]) for cond in weather_conditions
        })
        
        # Weather condition classifier
        self.weather_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, len(weather_conditions) + 1)  # +1 for clear
        )
    
    def forward(self, x, condition=None):
        # Auto-detect condition if not provided
        if condition is None:
            features = self.base_model.encoder(x)
            condition_logits = self.weather_classifier(features[-1])
            condition = ['clear', 'rain', 'fog', 'night'][condition_logits.argmax()]
        
        # Use condition-specific normalization
        if condition in self.condition_norms:
            # Apply specialized normalization
            pass
        
        return self.base_model(x)


def train_with_adverse_augmentation():
    """Training pipeline with weather augmentation"""
    transform = T.Compose([
        T.RandomResizedCrop(512),
        T.RandomHorizontalFlip(),
        WeatherAugmentation(fog_prob=0.3, rain_prob=0.3, night_prob=0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform
```

**Best Practices:**

| Scenario | Recommended Approach |
|----------|---------------------|
| LiDAR available | Multi-sensor fusion (LiDAR robust to weather) |
| Camera only | Restoration + robust training |
| Deploy anywhere | Train with synthetic adverse data |
| Specific condition | Condition-specific adaptation |

**Performance Impact:**

| Condition | Clear-trained mIoU | Adverse-trained mIoU |
|-----------|-------------------|---------------------|
| Clear | 70.0 | 68.5 |
| Fog | 35.2 | 55.8 |
| Rain | 42.1 | 58.3 |
| Night | 28.4 | 48.9 |

**Interview Tip:** Multi-sensor fusion is most reliable—LiDAR and radar work in any weather. For camera-only systems, training with realistic adverse augmentation is crucial. Image restoration as a preprocessing step can help but adds latency. Domain adaptation from clear→adverse improves generalization.

---

### Question 50
- [ ] Done
**How do you handle segmentation with limited computational resources or memory on edge devices?**

**Answer:**

Edge deployment requires optimizing for latency, memory, and power while maintaining accuracy. Key techniques include efficient architectures, quantization, pruning, knowledge distillation, and input resolution reduction.

**Constraints on Edge Devices:**

| Device | RAM | Compute | Example |
|--------|-----|---------|---------|
| Mobile phone | 2-8 GB | GPU/NPU | Pixel, iPhone |
| Embedded | 256MB-2GB | CPU/NPU | Jetson Nano |
| Microcontroller | <1 MB | CPU only | STM32, ESP32 |

**Optimization Techniques:**

| Technique | Memory Saving | Speed Improvement |
|-----------|--------------|-------------------|
| Efficient architecture | 4-10× | 2-5× |
| Quantization (INT8) | 4× | 2-4× |
| Pruning | 2-10× | 2-5× |
| Knowledge distillation | Variable | 1× (training only) |
| Resolution reduction | 4× (per 2× scale) | 4× |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileSegmentation(nn.Module):
    """Lightweight segmentation for edge devices"""
    
    def __init__(self, num_classes=19, width_mult=0.5):
        super().__init__()
        
        base_channels = int(32 * width_mult)
        
        # Efficient encoder
        self.encoder = nn.Sequential(
            # Stem
            self._conv_bn(3, base_channels, stride=2),
            
            # Inverted residual blocks
            InvertedResidual(base_channels, base_channels * 2, stride=2, expand=6),
            InvertedResidual(base_channels * 2, base_channels * 2, stride=1, expand=6),
            
            InvertedResidual(base_channels * 2, base_channels * 4, stride=2, expand=6),
            InvertedResidual(base_channels * 4, base_channels * 4, stride=1, expand=6),
            
            InvertedResidual(base_channels * 4, base_channels * 8, stride=2, expand=6),
        )
        
        # Lightweight ASPP
        self.aspp = LightASPP(base_channels * 8, 128)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def _conv_bn(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.decoder(x)
        
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block"""
    
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_ch == out_ch
        
        hidden = in_ch * expand
        
        layers = []
        if expand != 1:
            # Pointwise expand
            layers.extend([
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True)
        ])
        
        # Pointwise project
        layers.extend([
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class LightASPP(nn.Module):
    """Lightweight ASPP"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b2 = F.interpolate(b2, size=size, mode='bilinear', align_corners=False)
        
        return self.fusion(torch.cat([b1, b2], dim=1))


def quantize_model(model, calibration_loader):
    """Post-training quantization"""
    import torch.quantization as quant
    
    # Fuse conv-bn-relu
    model.eval()
    model_fused = torch.quantization.fuse_modules(
        model, [['conv', 'bn', 'relu']]
    )
    
    # Prepare for quantization
    model_fused.qconfig = quant.get_default_qconfig('fbgemm')
    model_prepared = quant.prepare(model_fused)
    
    # Calibrate with representative data
    with torch.no_grad():
        for images, _ in calibration_loader:
            model_prepared(images)
    
    # Convert to quantized
    model_quantized = quant.convert(model_prepared)
    
    return model_quantized


def knowledge_distillation_loss(student_output, teacher_output, 
                                 target, temperature=4.0, alpha=0.5):
    """Distill large model knowledge to small model"""
    # Hard loss (cross-entropy with labels)
    hard_loss = F.cross_entropy(student_output, target)
    
    # Soft loss (KL divergence with teacher)
    student_soft = F.log_softmax(student_output / temperature, dim=1)
    teacher_soft = F.softmax(teacher_output / temperature, dim=1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    soft_loss = soft_loss * (temperature ** 2)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss


class MultiResolutionInference:
    """Trade off accuracy vs speed at runtime"""
    
    def __init__(self, model, resolutions=[256, 384, 512]):
        self.model = model
        self.resolutions = resolutions
    
    def infer(self, image, target_latency_ms=50):
        """Select resolution based on latency budget"""
        import time
        
        best_resolution = self.resolutions[0]
        
        for res in self.resolutions:
            # Test latency
            test_input = F.interpolate(
                image, size=(res, res), mode='bilinear'
            )
            
            start = time.time()
            with torch.no_grad():
                _ = self.model(test_input)
            latency = (time.time() - start) * 1000
            
            if latency < target_latency_ms:
                best_resolution = res
            else:
                break
        
        # Run at selected resolution
        input_resized = F.interpolate(image, size=(best_resolution, best_resolution))
        output = self.model(input_resized)
        output = F.interpolate(output, size=image.shape[2:])
        
        return output


def export_to_edge(model, example_input, format='onnx'):
    """Export model for edge deployment"""
    model.eval()
    
    if format == 'onnx':
        torch.onnx.export(
            model, example_input, 'model.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}}
        )
    elif format == 'tflite':
        # Convert via ONNX → TensorFlow → TFLite
        pass
    elif format == 'coreml':
        import coremltools as ct
        traced = torch.jit.trace(model, example_input)
        mlmodel = ct.convert(traced)
        mlmodel.save('model.mlpackage')
```

**Architecture Comparison:**

| Model | Params | GFLOPs | mIoU | Target |
|-------|--------|--------|------|--------|
| DeepLabV3+ | 59M | 255 | 79.3 | Server |
| BiSeNet | 12.9M | 14.8 | 69.0 | Mobile |
| DDRNet-23 | 5.7M | 8.0 | 79.4 | Real-time |
| Mobile models | <2M | <1 | 60-70 | Edge |

**Interview Tip:** The key trade-offs are model size, compute, memory, and accuracy. Start with efficient architectures (MobileNet, EfficientNet), then apply quantization (INT8 is standard). Knowledge distillation from a large teacher can recover much of the accuracy lost from model shrinking. Always profile on the target hardware.

---

## Evaluation & Medical Imaging

### Question 51
- [ ] Done
**Explain IoU, Dice coefficient, and boundary F1-score for segmentation evaluation. When to use each?**

**Answer:**

IoU (Intersection over Union), Dice coefficient, and Boundary F1-score measure different aspects of segmentation quality. IoU penalizes errors more, Dice is better for imbalanced data, and Boundary F1 focuses on edge accuracy.

**Metric Definitions:**

| Metric | Formula | Range |
|--------|---------|-------|
| IoU (Jaccard) | TP / (TP + FP + FN) | [0, 1] |
| Dice (F1) | 2TP / (2TP + FP + FN) | [0, 1] |
| Boundary F1 | Precision × Recall at boundaries | [0, 1] |

**Relationship:**

```
Dice = 2 × IoU / (1 + IoU)
IoU = Dice / (2 - Dice)

Example: If IoU = 0.5, then Dice = 0.667
         If IoU = 0.8, then Dice = 0.889
```

**Python Implementation:**
```python
import torch
import numpy as np
from scipy.ndimage import binary_erosion

def compute_iou(pred, target, num_classes):
    """Compute IoU for each class"""
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = float('nan')  # Class not present
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return ious


def compute_dice(pred, target, num_classes, smooth=1e-6):
    """Compute Dice coefficient for each class"""
    dices = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        
        dice = (2 * intersection + smooth) / (pred_cls.sum() + target_cls.sum() + smooth)
        dices.append(dice.item())
    
    return dices


def compute_boundary_f1(pred, target, thickness=1):
    """Compute F1 score on boundaries only"""
    
    def get_boundary(mask, thickness):
        eroded = binary_erosion(mask, iterations=thickness)
        return mask ^ eroded  # XOR gives boundary
    
    pred_boundary = get_boundary(pred, thickness)
    target_boundary = get_boundary(target, thickness)
    
    # Allow small tolerance in matching
    from scipy.ndimage import distance_transform_edt
    
    pred_dist = distance_transform_edt(~pred_boundary)
    target_dist = distance_transform_edt(~target_boundary)
    
    threshold = 2  # pixels
    
    # Precision: how many predicted boundary pixels are near target boundary
    precision = (target_dist[pred_boundary] <= threshold).sum() / max(pred_boundary.sum(), 1)
    
    # Recall: how many target boundary pixels are near predicted boundary
    recall = (pred_dist[target_boundary] <= threshold).sum() / max(target_boundary.sum(), 1)
    
    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return f1


def mean_iou(pred, target, num_classes, ignore_index=255):
    """Mean IoU across all classes"""
    ious = compute_iou(pred, target, num_classes)
    
    # Filter out NaN (classes not in image)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    
    return np.mean(valid_ious) if valid_ious else 0.0


class SegmentationMetrics:
    """Accumulate metrics over dataset"""
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        self.confusion_matrix += np.bincount(
            target * self.num_classes + pred,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
    
    def compute_iou(self):
        """Per-class IoU from confusion matrix"""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                 self.confusion_matrix.sum(axis=0) - intersection)
        
        iou = intersection / np.maximum(union, 1)
        return iou
    
    def compute_metrics(self):
        iou = self.compute_iou()
        
        return {
            'mIoU': np.nanmean(iou),
            'IoU_per_class': iou,
            'Pixel_Accuracy': np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        }


# Usage example
def evaluate_segmentation(model, dataloader, num_classes):
    """Full evaluation"""
    metrics = SegmentationMetrics(num_classes)
    boundary_f1s = []
    
    for images, targets in dataloader:
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        
        for pred, target in zip(preds, targets):
            metrics.update(pred.numpy(), target.numpy())
            boundary_f1s.append(compute_boundary_f1(pred.numpy(), target.numpy()))
    
    results = metrics.compute_metrics()
    results['Boundary_F1'] = np.mean(boundary_f1s)
    
    return results
```

**When to Use Each:**

| Metric | Best For | Why |
|--------|----------|-----|
| IoU (mIoU) | Standard benchmark | Penalizes FP and FN equally |
| Dice | Medical imaging | Better for small objects, imbalanced |
| Boundary F1 | Edge quality | Ignores interior, focuses on boundaries |
| Pixel Accuracy | Quick check | Simple, but biased to large classes |

**Comparison Example:**

```
Scenario: 100×100 image, 10×10 object

Prediction: Slightly larger (11×11)
- IoU = 100/121 = 0.83
- Dice = 200/221 = 0.90
- Boundary overlap = 75%

Prediction: Shifted by 1 pixel
- IoU = 81/119 = 0.68
- Dice = 162/200 = 0.81
- Boundary overlap = 30% (much worse!)
```

**Interview Tip:** IoU is stricter than Dice for the same prediction quality. Use mIoU for benchmarks, Dice for class imbalance (medical), and Boundary F1 when edge quality matters (e.g., image editing). Always report per-class metrics to identify weak classes.

---

### Question 52
- [ ] Done
**How do you optimize U-Net architectures for medical image segmentation (3D U-Net, nnU-Net)?**

**Answer:**

Medical image segmentation requires handling 3D volumes, class imbalance, limited data, and domain-specific challenges. 3D U-Net extends U-Net to volumetric data, while nnU-Net automatically configures the best architecture and training pipeline for any medical dataset.

**3D U-Net Key Changes:**

| 2D U-Net | 3D U-Net |
|----------|----------|
| 2D Conv | 3D Conv |
| 2D MaxPool | 3D MaxPool |
| 2D UpConv | 3D UpConv |
| H×W input | D×H×W input |
| Skip 2D | Skip 3D |

**nnU-Net Auto-Configuration:**

| Component | What nnU-Net Decides |
|-----------|---------------------|
| Architecture | 2D, 3D, or cascade |
| Patch size | Based on GPU memory |
| Batch size | Maximize GPU usage |
| Normalization | Instance norm (standard) |
| Loss | Dice + CE combined |
| Augmentation | Rotation, scaling, noise |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation"""
    
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256]):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._double_conv3d(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._double_conv3d(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._double_conv3d(feature * 2, feature))
        
        # Output
        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def _double_conv3d(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),  # Instance norm for medical
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear')
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)  # Conv
        
        return self.final(x)


class nnUNetArchitecture(nn.Module):
    """Simplified nnU-Net-style architecture"""
    
    def __init__(self, in_channels, out_channels, patch_size, 
                 base_features=32, max_features=320):
        super().__init__()
        
        # Auto-determine architecture based on patch size
        self.ndim = 3 if len(patch_size) == 3 else 2
        self.features = self._compute_features(patch_size, base_features, max_features)
        
        # Build network
        conv_class = nn.Conv3d if self.ndim == 3 else nn.Conv2d
        norm_class = nn.InstanceNorm3d if self.ndim == 3 else nn.InstanceNorm2d
        pool_class = nn.MaxPool3d if self.ndim == 3 else nn.MaxPool2d
        
        # Encoder
        self.encoder_stages = nn.ModuleList()
        in_ch = in_channels
        for feat in self.features:
            self.encoder_stages.append(
                self._conv_block(in_ch, feat, conv_class, norm_class)
            )
            in_ch = feat
        
        # Decoder
        self.decoder_stages = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i in range(len(self.features) - 1, 0, -1):
            self.upsample.append(
                nn.ConvTranspose3d(self.features[i], self.features[i-1], 2, 2) 
                if self.ndim == 3 else
                nn.ConvTranspose2d(self.features[i], self.features[i-1], 2, 2)
            )
            self.decoder_stages.append(
                self._conv_block(self.features[i-1] * 2, self.features[i-1], 
                                conv_class, norm_class)
            )
        
        self.output = conv_class(self.features[0], out_channels, 1)
        self.pool = pool_class(2)
    
    def _compute_features(self, patch_size, base, max_feat):
        """Compute feature channels based on patch size"""
        num_stages = min(5, int(np.log2(min(patch_size))))
        features = []
        feat = base
        for _ in range(num_stages):
            features.append(min(feat, max_feat))
            feat *= 2
        return features
    
    def _conv_block(self, in_ch, out_ch, conv_class, norm_class):
        return nn.Sequential(
            conv_class(in_ch, out_ch, 3, padding=1, bias=False),
            norm_class(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            conv_class(out_ch, out_ch, 3, padding=1, bias=False),
            norm_class(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x):
        skips = []
        
        for stage in self.encoder_stages[:-1]:
            x = stage(x)
            skips.append(x)
            x = self.pool(x)
        
        x = self.encoder_stages[-1](x)
        
        for up, stage in zip(self.upsample, self.decoder_stages):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = stage(x)
        
        return self.output(x)


def nnunet_loss(pred, target, dice_weight=1.0, ce_weight=1.0):
    """Combined Dice + Cross Entropy loss (nnU-Net default)"""
    # Dice loss
    pred_soft = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    
    dice_loss = 0
    for c in range(num_classes):
        pred_c = pred_soft[:, c]
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        dice = (2 * intersection + 1) / (pred_c.sum() + target_c.sum() + 1)
        dice_loss += 1 - dice
    
    dice_loss /= num_classes
    
    # Cross entropy
    ce_loss = F.cross_entropy(pred, target)
    
    return dice_weight * dice_loss + ce_weight * ce_loss


def medical_augmentation(volume, mask):
    """nnU-Net-style augmentation"""
    import random
    
    # Random rotation (90 degrees)
    if random.random() > 0.5:
        k = random.choice([1, 2, 3])
        volume = torch.rot90(volume, k, dims=[-2, -1])
        mask = torch.rot90(mask, k, dims=[-2, -1])
    
    # Random flip
    for dim in [-1, -2, -3]:
        if random.random() > 0.5:
            volume = torch.flip(volume, [dim])
            mask = torch.flip(mask, [dim])
    
    # Random scaling
    if random.random() > 0.5:
        scale = random.uniform(0.85, 1.15)
        new_size = [int(s * scale) for s in volume.shape[-3:]]
        volume = F.interpolate(volume.unsqueeze(0).unsqueeze(0), 
                               size=new_size, mode='trilinear')[0, 0]
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                            size=new_size, mode='nearest')[0, 0].long()
    
    # Gaussian noise
    if random.random() > 0.5:
        noise = torch.randn_like(volume) * 0.1
        volume = volume + noise
    
    return volume, mask
```

**nnU-Net Best Practices:**

| Practice | Implementation |
|----------|---------------|
| Instance normalization | Better than BN for small batches |
| LeakyReLU (0.01) | Prevents dead neurons |
| Deep supervision | Loss at multiple scales |
| Sliding window | For large volumes |
| 5-fold cross-validation | Standard evaluation |

**Performance (Medical Decathlon):**

| Task | nnU-Net Dice | Previous SOTA |
|------|-------------|---------------|
| Liver | 0.967 | 0.952 |
| Spleen | 0.968 | 0.960 |
| Hippocampus | 0.900 | 0.878 |
| Prostate | 0.761 | 0.720 |

**Interview Tip:** nnU-Net's success comes from systematic auto-configuration rather than novel architecture. Key choices: 3D convs for volumetric data, instance normalization (batch size often 1-2), Dice + CE loss for class imbalance, and heavy augmentation for limited data.

---

### Question 53
- [ ] Done
**What are the challenges of segmentation in specialized domains like satellite imagery or microscopy?**

**Answer:**

Specialized domains present unique challenges: satellite imagery has massive scale and varying resolution, microscopy has small objects with complex shapes, and both often lack large labeled datasets. Domain-specific augmentation and architectures are needed.

**Domain-Specific Challenges:**

| Domain | Key Challenges |
|--------|---------------|
| Satellite | Large images (10000×10000), multi-spectral, varying resolution, class imbalance |
| Microscopy | Small objects, crowded scenes, varying stains, 3D volumes |
| Medical CT/MRI | 3D data, intensity variations, organ variability |
| Industrial | Defect detection, rare events, high precision requirements |

**Satellite Imagery Challenges:**

| Challenge | Solution |
|-----------|----------|
| Huge images | Sliding window, patch-based processing |
| Multi-spectral (10+ bands) | Multi-channel input, band selection |
| Varying GSD (resolution) | Multi-scale training, FPN |
| Class imbalance | Focal loss, oversampling rare classes |
| Temporal changes | Multi-temporal fusion |

**Microscopy Challenges:**

| Challenge | Solution |
|-----------|----------|
| Touching objects | Instance segmentation, watershed |
| Variable staining | Color normalization, augmentation |
| Dense packing | High-resolution processing |
| 3D imaging | 3D U-Net, slice-by-slice with context |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SatelliteSegmentation:
    """Handle large satellite images"""
    
    def __init__(self, model, patch_size=512, overlap=128):
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
    
    def predict_large_image(self, image):
        """Sliding window inference with overlap"""
        C, H, W = image.shape
        stride = self.patch_size - self.overlap
        
        # Output accumulator
        output = torch.zeros(self.model.num_classes, H, W)
        counts = torch.zeros(1, H, W)
        
        for y in range(0, H - self.patch_size + 1, stride):
            for x in range(0, W - self.patch_size + 1, stride):
                patch = image[:, y:y+self.patch_size, x:x+self.patch_size]
                
                with torch.no_grad():
                    pred = self.model(patch.unsqueeze(0))[0]
                
                # Weighted blending (center higher weight)
                weight = self._get_weight_map()
                output[:, y:y+self.patch_size, x:x+self.patch_size] += pred * weight
                counts[:, y:y+self.patch_size, x:x+self.patch_size] += weight
        
        return output / counts.clamp(min=1)
    
    def _get_weight_map(self):
        """Gaussian-like weight for blending"""
        p = self.patch_size
        x = torch.linspace(-1, 1, p)
        y = torch.linspace(-1, 1, p)
        xx, yy = torch.meshgrid(x, y)
        weight = torch.exp(-(xx**2 + yy**2) / 0.5)
        return weight.unsqueeze(0)


class MultiSpectralEncoder(nn.Module):
    """Handle multi-spectral satellite data"""
    
    def __init__(self, num_bands=13, base_channels=64):
        super().__init__()
        
        # Group bands by type
        self.rgb_conv = nn.Conv2d(3, base_channels // 2, 3, padding=1)
        self.nir_conv = nn.Conv2d(1, base_channels // 4, 3, padding=1)  # Near-IR
        self.other_conv = nn.Conv2d(num_bands - 4, base_channels // 4, 3, padding=1)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: [B, bands, H, W]
        rgb = self.rgb_conv(x[:, :3])
        nir = self.nir_conv(x[:, 3:4])
        other = self.other_conv(x[:, 4:])
        
        fused = torch.cat([rgb, nir, other], dim=1)
        return self.fusion(fused)


class MicroscopyInstanceSeg(nn.Module):
    """Instance segmentation for microscopy"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.encoder = UNetEncoder()
        
        # Semantic segmentation head
        self.semantic_head = nn.Conv2d(64, num_classes, 1)
        
        # Boundary prediction head (for separating touching objects)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        
        # Distance transform head (for watershed)
        self.distance_head = nn.Conv2d(64, 1, 1)
    
    def forward(self, x):
        features = self.encoder(x)
        
        semantic = self.semantic_head(features)
        boundary = self.boundary_head(features)
        distance = self.distance_head(features)
        
        return {
            'semantic': semantic,
            'boundary': torch.sigmoid(boundary),
            'distance': distance
        }
    
    def post_process(self, outputs):
        """Watershed-based instance separation"""
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from scipy import ndimage
        
        semantic = outputs['semantic'].argmax(dim=1)[0].numpy()
        distance = outputs['distance'][0, 0].numpy()
        
        # Find markers (local maxima of distance)
        markers = peak_local_max(distance, min_distance=5, labels=semantic)
        marker_labels = np.zeros_like(semantic)
        for i, (y, x) in enumerate(markers):
            marker_labels[y, x] = i + 1
        
        # Watershed
        instances = watershed(-distance, marker_labels, mask=semantic > 0)
        
        return instances


class DomainSpecificAugmentation:
    """Domain-specific data augmentation"""
    
    def __init__(self, domain='satellite'):
        self.domain = domain
    
    def __call__(self, image, mask):
        if self.domain == 'satellite':
            return self.satellite_augment(image, mask)
        elif self.domain == 'microscopy':
            return self.microscopy_augment(image, mask)
        return image, mask
    
    def satellite_augment(self, image, mask):
        """Satellite-specific augmentations"""
        import random
        
        # Random rotation (any angle for aerial view)
        angle = random.uniform(0, 360)
        image = self.rotate(image, angle)
        mask = self.rotate(mask, angle, is_mask=True)
        
        # Simulate different acquisition times (brightness/contrast)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = image * factor
        
        # Simulate atmospheric effects (haze)
        if random.random() > 0.7:
            haze = random.uniform(0, 0.2)
            image = image * (1 - haze) + haze
        
        # Channel dropout (simulate missing bands)
        if random.random() > 0.8:
            drop_channel = random.randint(0, image.shape[0] - 1)
            image[drop_channel] = 0
        
        return image, mask
    
    def microscopy_augment(self, image, mask):
        """Microscopy-specific augmentations"""
        import random
        
        # Random 90-degree rotation
        k = random.randint(0, 3)
        image = torch.rot90(image, k, dims=[-2, -1])
        mask = torch.rot90(mask, k, dims=[-2, -1])
        
        # Color jitter (staining variations)
        if random.random() > 0.5:
            # Adjust hue/saturation
            image = self.stain_augment(image)
        
        # Elastic deformation (biological variability)
        if random.random() > 0.5:
            image, mask = self.elastic_transform(image, mask)
        
        # Blur (out-of-focus simulation)
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            image = F.avg_pool2d(image.unsqueeze(0), kernel_size, 1, 
                                 kernel_size // 2)[0]
        
        return image, mask


def handle_class_imbalance(dataloader, num_classes):
    """Compute class weights for imbalanced data"""
    class_counts = torch.zeros(num_classes)
    
    for _, masks in dataloader:
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum()
    
    # Inverse frequency weighting
    weights = 1.0 / class_counts.clamp(min=1)
    weights = weights / weights.sum() * num_classes
    
    return weights
```

**Best Practices by Domain:**

| Domain | Recommendations |
|--------|----------------|
| Satellite | Patch-based training, multi-scale FPN, handle class imbalance |
| Microscopy | Instance seg for touching objects, stain normalization |
| Medical | 3D processing, limited data augmentation, uncertainty |
| Industrial | High precision, anomaly detection approach |

**Interview Tip:** Each specialized domain requires understanding the imaging physics and annotation challenges. Satellite needs handling huge images and multi-spectral data. Microscopy needs separating touching objects (instance seg). Both benefit from domain-specific augmentations that simulate real variations.

---

### Question 54
- [ ] Done
**Explain federated learning for medical image segmentation across hospitals with privacy constraints.**

**Answer:**

Federated learning trains a shared model across multiple hospitals without sharing raw patient data. Each hospital trains locally on their data, shares only model updates (gradients/weights), and a central server aggregates these updates. This preserves patient privacy while leveraging diverse data.

**Why Federated Learning for Medical:**

| Challenge | How FL Helps |
|-----------|-------------|
| Privacy regulations (HIPAA, GDPR) | Data never leaves hospital |
| Data heterogeneity | Model learns from diverse populations |
| Rare conditions | Pool rare cases across sites |
| Annotation differences | Aggregation smooths variations |

**Federated Learning Pipeline:**

```
Round t:
┌──────────┐    ┌──────────┐    ┌──────────┐
│Hospital A│    │Hospital B│    │Hospital C│
│ Local    │    │ Local    │    │ Local    │
│ Data     │    │ Data     │    │ Data     │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     ▼               ▼               ▼
  Train on        Train on        Train on
  local data      local data      local data
     │               │               │
     ▼               ▼               ▼
  Δw_A             Δw_B            Δw_C
     │               │               │
     └───────────────┼───────────────┘
                     ▼
              ┌─────────────┐
              │   Server    │
              │  Aggregate  │
              │  w = avg(Δw)│
              └──────┬──────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
  Update w        Update w        Update w
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import copy
import numpy as np

class FederatedServer:
    """Central server for federated learning"""
    
    def __init__(self, model, num_clients):
        self.global_model = model
        self.num_clients = num_clients
    
    def aggregate(self, client_updates, weights=None):
        """Federated averaging"""
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Initialize with zeros
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key]).float()
        
        # Weighted average
        for client_update, weight in zip(client_updates, weights):
            for key in global_dict:
                global_dict[key] += weight * client_update[key].float()
        
        self.global_model.load_state_dict(global_dict)
        return self.global_model.state_dict()
    
    def broadcast(self):
        """Send global model to all clients"""
        return copy.deepcopy(self.global_model.state_dict())


class FederatedClient:
    """Hospital/client in federated learning"""
    
    def __init__(self, model, local_data, client_id, local_epochs=5):
        self.model = model
        self.local_data = local_data
        self.client_id = client_id
        self.local_epochs = local_epochs
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    def receive_global_model(self, global_weights):
        """Update local model with global weights"""
        self.model.load_state_dict(global_weights)
    
    def local_train(self):
        """Train on local data"""
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for images, masks in self.local_data:
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = nn.functional.cross_entropy(outputs, masks)
                
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()
    
    def compute_update(self, global_weights):
        """Compute weight update (delta)"""
        local_weights = self.model.state_dict()
        update = {}
        
        for key in local_weights:
            update[key] = local_weights[key] - global_weights[key]
        
        return update


class DifferentialPrivacyFL:
    """Federated learning with differential privacy"""
    
    def __init__(self, epsilon=1.0, delta=1e-5, clip_norm=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
    
    def clip_gradients(self, gradients):
        """Clip gradient norm per sample"""
        total_norm = torch.norm(
            torch.stack([torch.norm(g) for g in gradients])
        )
        
        clip_factor = min(1.0, self.clip_norm / (total_norm + 1e-6))
        
        return [g * clip_factor for g in gradients]
    
    def add_noise(self, update, num_samples):
        """Add Gaussian noise for differential privacy"""
        sensitivity = 2 * self.clip_norm / num_samples
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        noisy_update = {}
        for key, value in update.items():
            noise = torch.randn_like(value) * noise_scale
            noisy_update[key] = value + noise
        
        return noisy_update


class SecureAggregation:
    """Secure aggregation to hide individual updates"""
    
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.masks = {}
    
    def generate_masks(self, model_shape):
        """Generate pairwise masks that cancel out"""
        masks = [{} for _ in range(self.num_clients)]
        
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                for key, shape in model_shape.items():
                    # Random mask
                    mask = torch.randn(shape)
                    masks[i][key] = masks[i].get(key, 0) + mask
                    masks[j][key] = masks[j].get(key, 0) - mask  # Cancels
        
        return masks
    
    def mask_update(self, update, mask):
        """Add mask to update"""
        masked = {}
        for key in update:
            masked[key] = update[key] + mask.get(key, 0)
        return masked
    
    def aggregate(self, masked_updates):
        """Sum aggregation (masks cancel)"""
        result = {}
        for key in masked_updates[0]:
            result[key] = sum(u[key] for u in masked_updates) / len(masked_updates)
        return result


def federated_training_loop(server, clients, num_rounds=100):
    """Main federated learning loop"""
    
    for round_num in range(num_rounds):
        # Broadcast global model
        global_weights = server.broadcast()
        
        # Collect client updates
        client_updates = []
        client_sizes = []
        
        for client in clients:
            # Receive global model
            client.receive_global_model(global_weights)
            
            # Local training
            local_weights = client.local_train()
            client_updates.append(local_weights)
            client_sizes.append(len(client.local_data.dataset))
        
        # Weighted aggregation by dataset size
        total_size = sum(client_sizes)
        weights = [s / total_size for s in client_sizes]
        
        # Aggregate
        server.aggregate(client_updates, weights)
        
        # Evaluate
        if round_num % 10 == 0:
            evaluate_global_model(server.global_model)


class FedProx:
    """FedProx: Handle heterogeneous data with proximal term"""
    
    def __init__(self, mu=0.01):
        self.mu = mu
    
    def compute_loss(self, model_output, target, local_weights, global_weights):
        """Add proximal term to loss"""
        # Task loss
        task_loss = nn.functional.cross_entropy(model_output, target)
        
        # Proximal term: ||w - w_global||^2
        prox_loss = 0
        for key in local_weights:
            prox_loss += torch.norm(
                local_weights[key] - global_weights[key]
            ) ** 2
        
        return task_loss + self.mu / 2 * prox_loss
```

**Challenges and Solutions:**

| Challenge | Solution |
|-----------|----------|
| Non-IID data | FedProx, SCAFFOLD |
| Communication cost | Gradient compression, fewer rounds |
| Stragglers | Asynchronous FL, client selection |
| Privacy attacks | Differential privacy, secure aggregation |

**Performance:**

| Setting | Centralized | FedAvg | FedProx |
|---------|-------------|--------|---------|
| IID data | 0.82 Dice | 0.80 | 0.80 |
| Non-IID | 0.82 Dice | 0.72 | 0.78 |

**Interview Tip:** FedAvg works well for IID data but struggles when hospitals have different patient populations (non-IID). FedProx adds a proximal term to prevent local models from drifting too far from global. Differential privacy and secure aggregation provide formal privacy guarantees but reduce utility.

---

### Question 55
- [ ] Done
**How do you handle noisy or inconsistent annotations in segmentation ground truth?**

**Answer:**

Noisy annotations are common in segmentation due to annotator disagreement, boundary ambiguity, and labeling errors. Techniques include robust loss functions, noise-aware training, label smoothing, and learning from multiple annotators.

**Types of Annotation Noise:**

| Type | Example | Impact |
|------|---------|--------|
| Boundary uncertainty | Fuzzy object edges | Boundary jitter |
| Mislabeling | Wrong class assigned | Confusion between classes |
| Inconsistency | Different annotators disagree | Training instability |
| Missing annotations | Unlabeled objects | False negative supervision |

**Approaches:**

| Technique | How It Helps |
|-----------|--------------|
| Robust loss | Less sensitive to outliers |
| Label smoothing | Soften overconfident labels |
| Co-training | Multiple models filter noise |
| Multi-annotator modeling | Learn from disagreements |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============ Robust Loss Functions ============

class NoiseTolerantCE(nn.Module):
    """Symmetric Cross-Entropy for noisy labels"""
    
    def __init__(self, alpha=0.1, beta=1.0, num_classes=19):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Standard CE
        ce = -torch.sum(target_onehot * torch.log(pred_soft + 1e-8), dim=1)
        
        # Reverse CE (model predicts label)
        rce = -torch.sum(pred_soft * torch.log(target_onehot + 1e-8), dim=1)
        
        return (self.alpha * ce + self.beta * rce).mean()


class GeneralizedCrossEntropy(nn.Module):
    """GCE: Noise-robust through truncation"""
    
    def __init__(self, q=0.7, num_classes=19):
        super().__init__()
        self.q = q
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        
        # Get prediction at true label
        target_expanded = target.unsqueeze(1)
        pred_at_label = pred_soft.gather(1, target_expanded).squeeze(1)
        
        # Truncated loss: (1 - p^q) / q
        loss = (1 - torch.pow(pred_at_label, self.q)) / self.q
        
        return loss.mean()


class BootstrapLoss(nn.Module):
    """Bootstrap from own predictions"""
    
    def __init__(self, beta=0.8, num_classes=19):
        super().__init__()
        self.beta = beta  # Weight on hard labels
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Mix hard labels with soft predictions
        soft_target = self.beta * target_onehot + (1 - self.beta) * pred_soft.detach()
        
        # Cross-entropy with soft target
        loss = -torch.sum(soft_target * torch.log(pred_soft + 1e-8), dim=1)
        
        return loss.mean()


# ============ Label Smoothing ============

class LabelSmoothing(nn.Module):
    """Smooth hard labels to soften overconfidence"""
    
    def __init__(self, num_classes=19, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Smooth: high prob for correct, low for others
        smooth_target = target_onehot * confidence + \
                       (1 - target_onehot) * smooth_value
        
        log_pred = F.log_softmax(pred, dim=1)
        loss = -torch.sum(smooth_target * log_pred, dim=1)
        
        return loss.mean()


# ============ Multi-Annotator Learning ============

class MultiAnnotatorModel(nn.Module):
    """Learn from multiple annotators with different expertise"""
    
    def __init__(self, base_model, num_annotators, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_annotators = num_annotators
        self.num_classes = num_classes
        
        # Annotator reliability weights (learned)
        self.annotator_weights = nn.Parameter(torch.ones(num_annotators))
        
        # Annotator-specific confusion matrices
        self.confusion_matrices = nn.ParameterList([
            nn.Parameter(torch.eye(num_classes) * 0.9 + 0.1 / num_classes)
            for _ in range(num_annotators)
        ])
    
    def forward(self, x):
        return self.base_model(x)
    
    def compute_loss(self, pred, annotations):
        """
        annotations: dict {annotator_id: mask} for available annotations
        """
        pred_soft = F.softmax(pred, dim=1)
        total_loss = 0
        total_weight = 0
        
        for annotator_id, mask in annotations.items():
            # Get annotator weight
            weight = F.softmax(self.annotator_weights, dim=0)[annotator_id]
            
            # Adjust for annotator confusion
            confusion = self.confusion_matrices[annotator_id]
            adjusted_pred = torch.einsum('bchw,cd->bdhw', pred_soft, confusion)
            
            # Cross-entropy
            loss = F.cross_entropy(adjusted_pred, mask, reduction='mean')
            
            total_loss += weight * loss
            total_weight += weight
        
        return total_loss / total_weight


# ============ Co-Training for Noise Filtering ============

class CoTeaching:
    """Co-teaching: Two networks filter noise for each other"""
    
    def __init__(self, model1, model2, forget_rate=0.2):
        self.model1 = model1
        self.model2 = model2
        self.forget_rate = forget_rate
    
    def train_step(self, images, labels, optimizer1, optimizer2):
        # Forward both models
        pred1 = self.model1(images)
        pred2 = self.model2(images)
        
        # Compute per-pixel loss
        loss1 = F.cross_entropy(pred1, labels, reduction='none')
        loss2 = F.cross_entropy(pred2, labels, reduction='none')
        
        # Each model selects small-loss samples for the other
        num_keep = int((1 - self.forget_rate) * loss1.numel())
        
        # Model 1 selects for Model 2
        _, indices1 = loss1.flatten().topk(num_keep, largest=False)
        mask1 = torch.zeros_like(loss1.flatten())
        mask1[indices1] = 1
        mask1 = mask1.view_as(loss1)
        
        # Model 2 selects for Model 1
        _, indices2 = loss2.flatten().topk(num_keep, largest=False)
        mask2 = torch.zeros_like(loss2.flatten())
        mask2[indices2] = 1
        mask2 = mask2.view_as(loss2)
        
        # Train on filtered samples
        filtered_loss1 = (loss1 * mask2).sum() / mask2.sum()
        filtered_loss2 = (loss2 * mask1).sum() / mask1.sum()
        
        optimizer1.zero_grad()
        filtered_loss1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        filtered_loss2.backward()
        optimizer2.step()
        
        return filtered_loss1.item(), filtered_loss2.item()


# ============ Uncertainty-Aware Training ============

class ConfidenceWeightedLoss(nn.Module):
    """Weight loss by model confidence"""
    
    def __init__(self, warmup_epochs=10):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        
        # Confidence = max probability
        confidence = pred_soft.max(dim=1)[0]
        
        # Per-pixel loss
        loss = F.cross_entropy(pred, target, reduction='none')
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup: uniform weight
            return loss.mean()
        else:
            # Weight by inverse confidence (uncertain = potentially noisy)
            # Lower weight for low-confidence = potentially noisy
            weights = confidence.detach()
            return (loss * weights).sum() / weights.sum()


def clean_labels_with_pretrained(model, dataset, threshold=0.9):
    """Use pretrained model to clean noisy labels"""
    clean_data = []
    
    model.eval()
    for image, noisy_label in dataset:
        with torch.no_grad():
            pred = model(image.unsqueeze(0))
            confidence = F.softmax(pred, dim=1).max(dim=1)[0][0]
        
        # Keep only high-confidence pixels
        clean_mask = noisy_label.clone()
        low_conf = confidence < threshold
        clean_mask[low_conf] = 255  # Ignore index
        
        clean_data.append((image, clean_mask))
    
    return clean_data
```

**Best Practices:**

| Scenario | Recommendation |
|----------|---------------|
| Known noisy labels | GCE or symmetric CE |
| Multiple annotators | Multi-annotator modeling |
| Boundary uncertainty | Label smoothing |
| Unknown noise rate | Co-teaching |

**Interview Tip:** For noisy labels, robust loss functions (GCE, SCE) are simple and effective. When multiple annotations exist, model the annotator reliability explicitly. Co-teaching is powerful but doubles training cost. Always validate on a clean held-out set.

---

## Advanced Topics

### Question 56
- [ ] Done
**Explain few-shot segmentation in novel semantic categories without retraining.**

**Answer:**

Few-shot segmentation segments new object categories using only 1-5 example images with masks, without retraining the model. It learns to match query images against support examples using learned feature comparison, enabling generalization to unseen classes.

**Problem Setup:**

```
Support Set: K images with masks of novel class (e.g., K=1 or K=5)
Query Image: Image to segment (contains same novel class)
Output: Segmentation mask for query image
```

**Key Approaches:**

| Approach | Idea |
|----------|------|
| Prototypical | Compare query features to class prototype |
| Attention-based | Cross-attention between support and query |
| Meta-learning | Learn to adapt quickly |
| Correlation | Match support features in query |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalFewShotSeg(nn.Module):
    """Prototypical network for few-shot segmentation"""
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # Feature alignment layers
        self.query_proj = nn.Conv2d(256, 256, 1)
        self.support_proj = nn.Conv2d(256, 256, 1)
    
    def forward(self, query_img, support_imgs, support_masks):
        """
        query_img: [B, C, H, W]
        support_imgs: [B, K, C, H, W] - K support images
        support_masks: [B, K, H, W] - corresponding masks
        """
        B, K = support_imgs.shape[:2]
        
        # Extract query features
        query_feat = self.backbone(query_img)  # [B, D, h, w]
        query_feat = self.query_proj(query_feat)
        
        # Extract support features and compute prototype
        prototypes = []
        for k in range(K):
            support_feat = self.backbone(support_imgs[:, k])
            support_feat = self.support_proj(support_feat)
            
            # Mask support features
            mask_resized = F.interpolate(
                support_masks[:, k:k+1].float(), 
                size=support_feat.shape[2:],
                mode='nearest'
            )
            
            # Masked average pooling -> prototype
            masked_feat = support_feat * mask_resized
            prototype = masked_feat.sum(dim=[2, 3]) / (mask_resized.sum(dim=[2, 3]) + 1e-6)
            prototypes.append(prototype)
        
        # Average prototypes across K shots
        prototype = torch.stack(prototypes, dim=0).mean(dim=0)  # [B, D]
        
        # Compare query features to prototype (cosine similarity)
        prototype = prototype.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        
        query_norm = F.normalize(query_feat, dim=1)
        proto_norm = F.normalize(prototype, dim=1)
        
        similarity = (query_norm * proto_norm).sum(dim=1, keepdim=True)
        
        # Upsample to original resolution
        mask = F.interpolate(similarity, size=query_img.shape[2:], mode='bilinear')
        
        return mask


class AttentionFewShotSeg(nn.Module):
    """Cross-attention based few-shot segmentation"""
    
    def __init__(self, backbone, hidden_dim=256):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        
        # Cross-attention
        self.query_proj = nn.Conv2d(256, hidden_dim, 1)
        self.key_proj = nn.Conv2d(256, hidden_dim, 1)
        self.value_proj = nn.Conv2d(256, hidden_dim, 1)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )
    
    def forward(self, query_img, support_imgs, support_masks):
        B, K = support_imgs.shape[:2]
        
        # Query features
        query_feat = self.backbone(query_img)  # [B, C, h, w]
        q = self.query_proj(query_feat)  # [B, D, h, w]
        
        # Collect support features
        support_feats = []
        for k in range(K):
            feat = self.backbone(support_imgs[:, k])
            
            # Mask to foreground only
            mask = F.interpolate(
                support_masks[:, k:k+1].float(),
                size=feat.shape[2:],
                mode='nearest'
            )
            feat = feat * mask
            support_feats.append(feat)
        
        support_feat = torch.stack(support_feats, dim=1).mean(dim=1)
        
        # Key and value from support
        k = self.key_proj(support_feat)
        v = self.value_proj(support_feat)
        
        # Cross-attention
        B, D, h, w = q.shape
        q_flat = q.flatten(2)  # [B, D, hw]
        k_flat = k.flatten(2)
        v_flat = v.flatten(2)
        
        attn = torch.bmm(q_flat.transpose(1, 2), k_flat)  # [B, hw_q, hw_s]
        attn = F.softmax(attn / (D ** 0.5), dim=-1)
        
        out = torch.bmm(v_flat, attn.transpose(1, 2))  # [B, D, hw_q]
        out = out.reshape(B, D, h, w)
        
        # Decode
        mask = self.decoder(out)
        mask = F.interpolate(mask, size=query_img.shape[2:], mode='bilinear')
        
        return mask


class DenseAffinityFewShot(nn.Module):
    """Dense correlation for few-shot segmentation"""
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        self.proj = nn.Conv2d(256, 128, 1)
        
        # Learnable temperature
        self.temp = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, query_img, support_imgs, support_masks):
        B, K = support_imgs.shape[:2]
        
        # Extract features
        query_feat = self.proj(self.backbone(query_img))  # [B, D, h, w]
        
        all_correlations = []
        
        for k in range(K):
            support_feat = self.proj(self.backbone(support_imgs[:, k]))
            
            # Dense correlation (4D correlation)
            # For each query location, correlate with all support locations
            B, D, Hq, Wq = query_feat.shape
            _, _, Hs, Ws = support_feat.shape
            
            query_flat = query_feat.flatten(2)  # [B, D, HW_q]
            support_flat = support_feat.flatten(2)  # [B, D, HW_s]
            
            # Correlation: [B, HW_q, HW_s]
            corr = torch.bmm(
                F.normalize(query_flat, dim=1).transpose(1, 2),
                F.normalize(support_flat, dim=1)
            ) * self.temp
            
            # Mask correlation with support mask
            mask_flat = F.interpolate(
                support_masks[:, k:k+1].float(),
                size=(Hs, Ws), mode='nearest'
            ).flatten(2)  # [B, 1, HW_s]
            
            # Weight correlation by support mask
            corr = corr * mask_flat  # [B, HW_q, HW_s]
            
            # Aggregate: how much each query pixel correlates with foreground
            fg_score = corr.sum(dim=-1) / (mask_flat.sum(dim=-1) + 1e-6)
            fg_score = fg_score.reshape(B, 1, Hq, Wq)
            
            all_correlations.append(fg_score)
        
        # Average across shots
        output = torch.stack(all_correlations, dim=0).mean(dim=0)
        
        return F.interpolate(output, size=query_img.shape[2:], mode='bilinear')


def few_shot_episode(model, query_loader, support_set, threshold=0.5):
    """Evaluate few-shot segmentation"""
    ious = []
    
    support_imgs, support_masks = support_set
    
    for query_img, query_mask in query_loader:
        with torch.no_grad():
            pred = model(query_img, support_imgs, support_masks)
            pred_binary = (pred > threshold).float()
        
        # Compute IoU
        intersection = (pred_binary * query_mask).sum()
        union = ((pred_binary + query_mask) > 0).float().sum()
        iou = intersection / (union + 1e-6)
        ious.append(iou.item())
    
    return np.mean(ious)
```

**Training Strategy:**

| Phase | Description |
|-------|-------------|
| Episodic training | Sample random classes, create support/query |
| Base classes | Train on seen classes |
| Novel classes | Evaluate on unseen classes (zero-shot transfer) |

**Performance (Pascal-5i):**

| Method | 1-shot | 5-shot |
|--------|--------|--------|
| Prototypical | 55.2 | 58.4 |
| PANet | 59.2 | 63.5 |
| HSNet | 64.0 | 69.5 |
| VAT | 67.2 | 72.4 |

**Interview Tip:** Few-shot segmentation is about comparing query features to support features using learned similarity. Prototypical networks are simple but effective—compute foreground prototype from support, then classify query pixels by similarity. Dense correlation captures fine-grained matching but is more expensive.

---

### Question 57
- [ ] Done
**How do you implement knowledge distillation for compressing large segmentation models?**

**Answer:**

Knowledge distillation transfers knowledge from a large teacher model to a smaller student model. For segmentation, this includes distilling pixel-wise soft labels, intermediate features, and structural relationships, enabling the student to approach teacher performance with fewer parameters.

**Distillation Types:**

| Type | What's Transferred | Loss |
|------|-------------------|------|
| Response-based | Soft logits | KL divergence |
| Feature-based | Intermediate features | L2 / attention transfer |
| Relation-based | Pairwise similarities | Gram matrix / correlation |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationDistillation:
    """Knowledge distillation for segmentation models"""
    
    def __init__(self, teacher, student, temperature=4.0, 
                 alpha=0.5, feature_weight=0.1):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha  # Weight for soft loss
        self.feature_weight = feature_weight
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def compute_loss(self, images, targets):
        """Combined distillation loss"""
        
        # Get predictions
        with torch.no_grad():
            teacher_logits = self.teacher(images)
            teacher_features = self.teacher.get_features(images)
        
        student_logits = self.student(images)
        student_features = self.student.get_features(images)
        
        # 1. Hard loss (with ground truth)
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # 2. Soft loss (KL divergence with teacher)
        soft_loss = self.response_distillation(
            student_logits, teacher_logits, self.temperature
        )
        
        # 3. Feature distillation
        feature_loss = self.feature_distillation(
            student_features, teacher_features
        )
        
        # Combine
        total_loss = (
            (1 - self.alpha) * hard_loss +
            self.alpha * soft_loss +
            self.feature_weight * feature_loss
        )
        
        return total_loss, {
            'hard': hard_loss.item(),
            'soft': soft_loss.item(),
            'feature': feature_loss.item()
        }
    
    def response_distillation(self, student_logits, teacher_logits, T):
        """Soft label distillation at pixel level"""
        # Soft labels from teacher
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        
        # Log softmax from student
        student_log_soft = F.log_softmax(student_logits / T, dim=1)
        
        # KL divergence
        kl_loss = F.kl_div(
            student_log_soft, teacher_soft, reduction='batchmean'
        ) * (T ** 2)
        
        return kl_loss
    
    def feature_distillation(self, student_features, teacher_features):
        """Feature-level distillation"""
        loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Align channels if different
            if s_feat.shape[1] != t_feat.shape[1]:
                adapter = nn.Conv2d(s_feat.shape[1], t_feat.shape[1], 1).to(s_feat.device)
                s_feat = adapter(s_feat)
            
            # Align spatial size
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear')
            
            # L2 loss
            loss += F.mse_loss(s_feat, t_feat)
        
        return loss / len(student_features)


class AttentionTransfer(nn.Module):
    """Attention transfer distillation"""
    
    def __init__(self, p=2):
        super().__init__()
        self.p = p  # Power for attention map
    
    def compute_attention(self, features):
        """Compute spatial attention map"""
        # Sum over channels, normalize
        attention = torch.pow(features.abs(), self.p)
        attention = attention.mean(dim=1, keepdim=True)
        
        # Normalize
        attention = attention / attention.sum(dim=[2, 3], keepdim=True)
        
        return attention
    
    def forward(self, student_features, teacher_features):
        loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_attn = self.compute_attention(s_feat)
            t_attn = self.compute_attention(t_feat)
            
            # Align spatial size
            if s_attn.shape != t_attn.shape:
                s_attn = F.interpolate(s_attn, size=t_attn.shape[2:], mode='bilinear')
            
            loss += F.mse_loss(s_attn, t_attn)
        
        return loss / len(student_features)


class StructuredDistillation(nn.Module):
    """Distill structural relationships"""
    
    def __init__(self):
        super().__init__()
    
    def pairwise_similarity(self, features):
        """Compute pairwise similarity matrix"""
        B, C, H, W = features.shape
        
        # Sample spatial locations (full is too expensive)
        feat_flat = features.flatten(2)  # [B, C, HW]
        
        # Normalize
        feat_norm = F.normalize(feat_flat, dim=1)
        
        # Similarity matrix
        similarity = torch.bmm(feat_norm.transpose(1, 2), feat_norm)  # [B, HW, HW]
        
        return similarity
    
    def forward(self, student_features, teacher_features):
        loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Downsample for efficiency
            s_feat = F.adaptive_avg_pool2d(s_feat, (16, 16))
            t_feat = F.adaptive_avg_pool2d(t_feat, (16, 16))
            
            s_sim = self.pairwise_similarity(s_feat)
            t_sim = self.pairwise_similarity(t_feat)
            
            loss += F.mse_loss(s_sim, t_sim)
        
        return loss / len(student_features)


class ChannelDistillation(nn.Module):
    """Channel-wise knowledge distillation"""
    
    def forward(self, student_features, teacher_features):
        loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Global average pool to get channel statistics
            s_stats = s_feat.mean(dim=[2, 3])  # [B, C]
            t_stats = t_feat.mean(dim=[2, 3])
            
            # Match channel correlations
            s_corr = torch.bmm(s_stats.unsqueeze(2), s_stats.unsqueeze(1))
            t_corr = torch.bmm(t_stats.unsqueeze(2), t_stats.unsqueeze(1))
            
            loss += F.mse_loss(s_corr, t_corr)
        
        return loss / len(student_features)


class SelfDistillation(nn.Module):
    """Self-distillation within same model (deep supervision)"""
    
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        
        # Auxiliary heads at intermediate layers
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(256, num_classes, 1),
            nn.Conv2d(512, num_classes, 1)
        ])
    
    def forward(self, x, target):
        # Main prediction
        main_out = self.model(x)
        main_loss = F.cross_entropy(main_out, target)
        
        # Auxiliary predictions
        features = self.model.get_intermediate_features(x)
        
        aux_loss = 0
        for aux_head, feat in zip(self.aux_heads, features):
            aux_out = aux_head(feat)
            aux_out = F.interpolate(aux_out, size=target.shape[1:], mode='bilinear')
            
            # Distill from main head
            with torch.no_grad():
                soft_target = F.softmax(main_out, dim=1)
            
            aux_loss += F.kl_div(
                F.log_softmax(aux_out, dim=1),
                soft_target,
                reduction='batchmean'
            )
        
        return main_loss + 0.4 * aux_loss


def distillation_training(teacher, student, train_loader, epochs=100):
    """Complete distillation training loop"""
    distiller = SegmentationDistillation(
        teacher, student,
        temperature=4.0,
        alpha=0.7,
        feature_weight=0.1
    )
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for images, targets in train_loader:
            loss, metrics = distiller.compute_loss(images, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return student
```

**Performance:**

| Model | Params | mIoU | vs Teacher |
|-------|--------|------|------------|
| Teacher (ResNet-101) | 60M | 78.5 | - |
| Student (ResNet-18) | 12M | 72.1 | -6.4 |
| Student + Distill | 12M | 75.3 | -3.2 |

**Best Practices:**

| Practice | Reason |
|----------|--------|
| Temperature 3-6 | Soften distribution, not too flat |
| Multi-scale features | Transfer at multiple resolutions |
| Progressive training | Start simple, add losses |
| Feature adaptation | Handle channel mismatch |

**Interview Tip:** For segmentation distillation, pixel-wise soft labels (response-based) are most important. Feature distillation helps but requires channel adaptation layers. Temperature controls the "softness" of the teacher's output—too high makes everything uniform, too low is like hard labels.

---

### Question 58
- [ ] Done
**What techniques help with explaining segmentation decisions for model interpretability?**

**Answer:**

Interpretability in segmentation helps understand why the model assigns specific class labels to pixels. Techniques include gradient-based attribution, attention visualization, concept-based explanations, and counterfactual analysis.

**Interpretability Goals:**

| Goal | Question Answered |
|------|-------------------|
| Attribution | Which input pixels influenced this prediction? |
| Feature importance | What patterns does the model look for? |
| Concept explanation | What high-level concepts are used? |
| Failure analysis | Why did the model make this error? |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """Gradient-weighted Class Activation Mapping for segmentation"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, target_class, target_pixel=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        # Create target for backward
        if target_pixel is not None:
            # Explain specific pixel
            py, px = target_pixel
            target = output[0, target_class, py, px]
        else:
            # Average over all pixels of this class
            mask = (output.argmax(dim=1) == target_class).float()
            target = (output[:, target_class] * mask).sum()
        
        # Backward
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Compute CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize and resize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear')
        
        return cam[0, 0]


class IntegratedGradients:
    """Integrated Gradients attribution for segmentation"""
    
    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps
    
    def __call__(self, x, target_class, baseline=None):
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        # Interpolate between baseline and input
        scaled_inputs = []
        for i in range(self.steps + 1):
            alpha = i / self.steps
            scaled_input = baseline + alpha * (x - baseline)
            scaled_inputs.append(scaled_input)
        
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(scaled_inputs)
        
        # Target: sum of target class logits
        target = outputs[:, target_class].sum()
        
        # Backward
        target.backward()
        
        # Average gradients
        avg_gradients = scaled_inputs.grad.mean(dim=0)
        
        # Attribution = (input - baseline) * avg_gradients
        attribution = (x - baseline) * avg_gradients
        
        return attribution.abs().sum(dim=1)[0]


class SHAP_Segmentation:
    """SHAP values for segmentation (simplified)"""
    
    def __init__(self, model, num_samples=100):
        self.model = model
        self.num_samples = num_samples
    
    def __call__(self, x, target_class, superpixel_segments):
        """Explain using superpixel occlusion"""
        num_superpixels = superpixel_segments.max() + 1
        
        # Baseline (gray image)
        baseline = torch.ones_like(x) * 0.5
        
        # Sample random coalitions
        shapley_values = torch.zeros(num_superpixels)
        
        for _ in range(self.num_samples):
            # Random permutation of superpixels
            perm = torch.randperm(num_superpixels)
            
            prev_value = self._evaluate(baseline, target_class)
            
            current_input = baseline.clone()
            
            for i, sp_idx in enumerate(perm):
                # Add superpixel to coalition
                mask = (superpixel_segments == sp_idx).float().unsqueeze(0).unsqueeze(0)
                current_input = current_input * (1 - mask) + x * mask
                
                # Evaluate
                current_value = self._evaluate(current_input, target_class)
                
                # Marginal contribution
                shapley_values[sp_idx] += (current_value - prev_value)
                prev_value = current_value
        
        shapley_values /= self.num_samples
        
        return shapley_values
    
    def _evaluate(self, x, target_class):
        with torch.no_grad():
            output = self.model(x)
            return output[:, target_class].mean()


class ConceptActivationVectors:
    """Test for concept presence in model"""
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        
        self.cavs = {}  # Learned concept vectors
    
    def learn_concept(self, concept_name, positive_images, negative_images):
        """Learn a CAV for a concept"""
        
        # Extract features
        pos_features = self._extract_features(positive_images)
        neg_features = self._extract_features(negative_images)
        
        # Train linear classifier
        from sklearn.linear_model import LogisticRegression
        
        X = torch.cat([pos_features, neg_features], dim=0).numpy()
        y = [1] * len(pos_features) + [0] * len(neg_features)
        
        clf = LogisticRegression()
        clf.fit(X, y)
        
        # CAV = normal to decision boundary
        cav = torch.tensor(clf.coef_[0])
        self.cavs[concept_name] = cav
    
    def _extract_features(self, images):
        features = []
        for img in images:
            feat = self.model.get_layer_output(img, self.layer_name)
            feat = feat.mean(dim=[2, 3])  # Global pool
            features.append(feat)
        return torch.cat(features, dim=0)
    
    def tcav_score(self, images, target_class, concept_name):
        """Compute TCAV score: how sensitive is the class to the concept?"""
        cav = self.cavs[concept_name]
        
        positive_count = 0
        total = 0
        
        for img in images:
            # Get gradient of target class w.r.t. layer
            img.requires_grad_(True)
            feat = self.model.get_layer_output(img, self.layer_name)
            output = self.model.classify_from_features(feat)
            
            output[:, target_class].sum().backward()
            grad = img.grad.mean(dim=[0, 2, 3])
            
            # Directional derivative along CAV
            directional_deriv = (grad * cav).sum()
            
            if directional_deriv > 0:
                positive_count += 1
            total += 1
        
        return positive_count / total


class CounterfactualExplanation:
    """Generate counterfactual: minimal change to flip prediction"""
    
    def __init__(self, model, learning_rate=0.01, steps=100):
        self.model = model
        self.lr = learning_rate
        self.steps = steps
    
    def generate(self, x, target_pixel, current_class, counterfactual_class):
        """What minimal change would change pixel classification?"""
        x_cf = x.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([x_cf], lr=self.lr)
        
        for _ in range(self.steps):
            output = self.model(x_cf)
            
            py, px = target_pixel
            current_logit = output[0, current_class, py, px]
            target_logit = output[0, counterfactual_class, py, px]
            
            # Minimize: (current - target) + lambda * ||x_cf - x||
            classification_loss = current_logit - target_logit
            distance_loss = F.mse_loss(x_cf, x)
            
            loss = classification_loss + 0.1 * distance_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clamp to valid range
            x_cf.data.clamp_(0, 1)
            
            # Check if prediction flipped
            if output[0, :, py, px].argmax() == counterfactual_class:
                break
        
        # Return difference
        return (x_cf - x).abs()


def visualize_explanation(image, explanation, title="Explanation"):
    """Overlay explanation on image"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Input")
    
    # Explanation heatmap
    axes[1].imshow(explanation, cmap='jet')
    axes[1].set_title(title)
    
    # Overlay
    axes[2].imshow(image.permute(1, 2, 0))
    axes[2].imshow(explanation, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay")
    
    return fig
```

**Comparison:**

| Method | Pros | Cons |
|--------|------|------|
| GradCAM | Fast, class-specific | Coarse resolution |
| Integrated Gradients | Satisfies axioms | Expensive |
| SHAP | Theoretically grounded | Very expensive |
| CAV | High-level concepts | Needs concept examples |

**Interview Tip:** GradCAM is most practical for quick debugging. For rigorous analysis, Integrated Gradients is preferred (satisfies sensitivity and implementation invariance). TCAV is useful when you want to test if the model uses specific concepts (e.g., "does it use texture or shape?").

---

### Question 59
- [ ] Done
**How do you integrate conditional random fields (CRF) as post-processing for segmentation refinement?**

**Answer:**

CRF (Conditional Random Field) refines segmentation boundaries by considering both unary potentials (CNN predictions) and pairwise potentials (spatial/color similarity). It smooths noisy predictions while respecting image edges, producing sharper boundaries.

**Why CRF Helps:**

| CNN Output Issue | How CRF Fixes It |
|------------------|------------------|
| Noisy predictions | Smooths via pairwise terms |
| Blurry boundaries | Sharpens using image edges |
| Small holes | Fills based on neighbors |
| Isolated pixels | Removes based on context |

**CRF Energy Function:**

$$E(\mathbf{x}) = \sum_i \psi_u(x_i) + \sum_{i,j} \psi_p(x_i, x_j)$$

- **Unary** $\psi_u$: CNN log-probabilities (how likely pixel i is class $x_i$)
- **Pairwise** $\psi_p$: Penalty for different labels when pixels are similar

**Pairwise Potential:**

$$\psi_p(x_i, x_j) = \mu(x_i, x_j) \left[ w_1 \exp\left(-\frac{|p_i - p_j|^2}{2\sigma_\alpha^2} - \frac{|I_i - I_j|^2}{2\sigma_\beta^2}\right) + w_2 \exp\left(-\frac{|p_i - p_j|^2}{2\sigma_\gamma^2}\right) \right]$$

- **Appearance kernel**: Similar colors should have same label
- **Smoothness kernel**: Nearby pixels should have same label

**Python Implementation:**
```python
import numpy as np
import torch
import torch.nn.functional as F

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    PYDENSECRF_AVAILABLE = True
except ImportError:
    PYDENSECRF_AVAILABLE = False

class DenseCRF:
    """Dense CRF post-processing for segmentation"""
    
    def __init__(self, num_classes, 
                 appearance_kernel_weight=3,
                 smoothness_kernel_weight=3,
                 appearance_kernel_sigma=(3, 3),  # (spatial, color)
                 smoothness_kernel_sigma=3,
                 num_iterations=5):
        self.num_classes = num_classes
        self.w1 = appearance_kernel_weight
        self.w2 = smoothness_kernel_weight
        self.sigma_alpha = appearance_kernel_sigma[0]
        self.sigma_beta = appearance_kernel_sigma[1]
        self.sigma_gamma = smoothness_kernel_sigma
        self.num_iters = num_iterations
    
    def __call__(self, image, probs):
        """
        image: numpy array [H, W, 3], uint8, 0-255
        probs: numpy array [C, H, W], float32, softmax probabilities
        
        Returns: refined probabilities [C, H, W]
        """
        if not PYDENSECRF_AVAILABLE:
            return probs
        
        H, W = image.shape[:2]
        
        # Create CRF
        crf = dcrf.DenseCRF2D(W, H, self.num_classes)
        
        # Set unary potentials (negative log prob)
        unary = unary_from_softmax(probs)
        crf.setUnaryEnergy(unary)
        
        # Add pairwise potentials
        # Appearance kernel (bilateral: spatial + color)
        crf.addPairwiseBilateral(
            sxy=self.sigma_alpha,
            srgb=self.sigma_beta,
            rgbim=image,
            compat=self.w1,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )
        
        # Smoothness kernel (spatial only)
        crf.addPairwiseGaussian(
            sxy=self.sigma_gamma,
            compat=self.w2,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )
        
        # Inference
        Q = crf.inference(self.num_iters)
        Q = np.array(Q).reshape(self.num_classes, H, W)
        
        return Q


class DifferentiableCRF(torch.nn.Module):
    """Differentiable CRF for end-to-end training"""
    
    def __init__(self, num_classes, num_iterations=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        
        # Learnable parameters
        self.w1 = torch.nn.Parameter(torch.tensor(3.0))
        self.w2 = torch.nn.Parameter(torch.tensor(3.0))
        
        # Gaussian filter for smoothness
        self.sigma_spatial = 3
    
    def forward(self, probs, image):
        """
        probs: [B, C, H, W] - softmax probabilities
        image: [B, 3, H, W] - input image
        
        Returns: refined probabilities
        """
        B, C, H, W = probs.shape
        
        # Mean-field inference
        Q = probs.clone()
        
        for _ in range(self.num_iterations):
            # Message passing
            # 1. Appearance-based (bilateral filter approximation)
            appearance_msg = self._bilateral_filter(Q, image)
            
            # 2. Spatial smoothness
            spatial_msg = self._gaussian_filter(Q)
            
            # Combine messages
            pairwise = self.w1 * appearance_msg + self.w2 * spatial_msg
            
            # Update Q
            Q_new = probs * torch.exp(-pairwise)
            Q = F.softmax(Q_new, dim=1)
        
        return Q
    
    def _gaussian_filter(self, Q):
        """Gaussian spatial filter"""
        kernel_size = int(6 * self.sigma_spatial) | 1
        
        # Separable Gaussian
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * self.sigma_spatial ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(Q.device)
        
        # Apply per channel
        filtered = []
        for c in range(Q.shape[1]):
            f = F.conv2d(Q[:, c:c+1], kernel, padding=kernel_size // 2)
            filtered.append(f)
        
        return torch.cat(filtered, dim=1) - Q  # Message
    
    def _bilateral_filter(self, Q, image, sigma_spatial=10, sigma_color=0.1):
        """Approximated bilateral filter"""
        # For efficiency, use permutohedral lattice or grid approximation
        # Simplified version: use guided filter
        
        B, C, H, W = Q.shape
        
        # Downsample for efficiency
        scale = 4
        Q_small = F.avg_pool2d(Q, scale)
        img_small = F.avg_pool2d(image, scale)
        
        # Simple guided filter approximation
        mean_I = F.avg_pool2d(img_small, 5, 1, 2)
        mean_Q = F.avg_pool2d(Q_small, 5, 1, 2)
        
        corr_IQ = F.avg_pool2d(img_small * Q_small, 5, 1, 2)
        corr_II = F.avg_pool2d(img_small ** 2, 5, 1, 2)
        
        var_I = corr_II - mean_I ** 2 + 1e-6
        cov_IQ = corr_IQ - mean_I * mean_Q
        
        a = cov_IQ / var_I
        b = mean_Q - a * mean_I
        
        mean_a = F.avg_pool2d(a, 5, 1, 2)
        mean_b = F.avg_pool2d(b, 5, 1, 2)
        
        filtered = mean_a * img_small + mean_b
        
        # Upsample back
        filtered = F.interpolate(filtered, size=(H, W), mode='bilinear')
        
        return filtered - Q


class CRFAsRNN(torch.nn.Module):
    """CRF as RNN: learnable CRF in neural network"""
    
    def __init__(self, num_classes, num_iterations=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        
        # Learned compatibility matrix
        self.compatibility = torch.nn.Parameter(
            torch.eye(num_classes) - torch.ones(num_classes, num_classes) / num_classes
        )
        
        # Spatial filter
        self.spatial_filter = torch.nn.Conv2d(num_classes, num_classes, 5, padding=2)
        
        # Bilateral filter (learned)
        self.bilateral_filter = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes + 3, 64, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, num_classes, 5, padding=2)
        )
    
    def forward(self, unary, image):
        """
        unary: [B, C, H, W] - CNN output (logits)
        image: [B, 3, H, W] - input image
        """
        Q = F.softmax(unary, dim=1)
        
        for _ in range(self.num_iterations):
            # Spatial message passing
            spatial_msg = self.spatial_filter(Q)
            
            # Bilateral message passing
            concat = torch.cat([Q, image], dim=1)
            bilateral_msg = self.bilateral_filter(concat)
            
            # Compatibility transform
            B, C, H, W = Q.shape
            msg = (spatial_msg + bilateral_msg).permute(0, 2, 3, 1)  # [B, H, W, C]
            msg = torch.matmul(msg, self.compatibility)  # Compatibility
            msg = msg.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Update
            Q = F.softmax(unary - msg, dim=1)
        
        return Q


def apply_crf_batch(model, images, num_classes):
    """Apply CRF to batch of predictions"""
    crf = DenseCRF(num_classes)
    
    refined = []
    for img, logits in zip(images, model(images)):
        probs = F.softmax(logits, dim=0).cpu().numpy()
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        refined_probs = crf(img_np, probs)
        refined.append(torch.from_numpy(refined_probs))
    
    return torch.stack(refined)
```

**Performance Impact:**

| Model | Without CRF | With CRF | Improvement |
|-------|-------------|----------|-------------|
| DeepLab | 75.4 mIoU | 77.7 mIoU | +2.3 |
| FCN | 62.2 mIoU | 65.5 mIoU | +3.3 |

**When to Use CRF:**

| Scenario | Recommendation |
|----------|---------------|
| Boundary quality critical | Use CRF |
| Real-time needed | Skip CRF (adds ~100ms) |
| End-to-end training | CRF-RNN variant |
| Modern transformers | Often not needed |

**Interview Tip:** CRF post-processing was essential for older FCN models but modern architectures (DeepLab, Swin) often have sufficient boundary quality. If using CRF, pydensecrf with bilateral + Gaussian kernels is standard. For end-to-end training, CRF-RNN or learnable variants are needed.

---

### Question 60
- [ ] Done
**Explain multi-task learning that combines segmentation with depth estimation or other vision tasks.**

**Answer:**

Multi-task learning (MTL) trains a single network to perform multiple related tasks simultaneously (segmentation + depth + normals). Shared representations improve efficiency and often boost performance on each task through regularization and complementary learned features.

**Why Multi-Task Works:**

| Benefit | Explanation |
|---------|-------------|
| Shared features | Edges useful for both segmentation & depth |
| Regularization | Tasks regularize each other, prevent overfitting |
| Efficiency | Single backbone for multiple outputs |
| Complementary learning | Depth helps with occlusion reasoning |

**Common Task Combinations:**

| Primary Task | Complementary Tasks |
|--------------|---------------------|
| Semantic Segmentation | Depth, Surface Normals, Edges |
| Instance Segmentation | Depth, Pose Estimation |
| Scene Understanding | Segmentation, Depth, Layout |
| Autonomous Driving | Segmentation, Depth, Flow, Detection |

**Architecture Pattern:**

$$\mathcal{L}_{total} = \sum_{t=1}^{T} w_t \mathcal{L}_t$$

Where $w_t$ are task weights balancing gradient magnitudes.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedEncoder(nn.Module):
    """Shared backbone for multi-task learning"""
    
    def __init__(self, backbone='resnet50'):
        super().__init__()
        import torchvision.models as models
        
        resnet = getattr(models, backbone)(pretrained=True)
        
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        
        self.channels = [256, 512, 1024, 2048]
    
    def forward(self, x):
        features = []
        
        x = self.layer0(x)
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features


class TaskSpecificDecoder(nn.Module):
    """Decoder for a specific task"""
    
    def __init__(self, encoder_channels, output_channels, task_type='segmentation'):
        super().__init__()
        
        self.task_type = task_type
        
        # FPN-style decoder
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, 256, 1) for c in encoder_channels
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for _ in encoder_channels
        ])
        
        # Final prediction head
        if task_type == 'segmentation':
            self.head = nn.Conv2d(256, output_channels, 1)
        elif task_type == 'depth':
            self.head = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 1, 1),
                nn.Sigmoid()  # Normalized depth
            )
        elif task_type == 'normals':
            self.head = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 3, 1)  # Unit normal vectors
            )
    
    def forward(self, features):
        # Top-down pathway
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]
        
        # Feature fusion (top-down)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], scale_factor=2, mode='bilinear', align_corners=False
            )
        
        outputs = [o(l) for o, l in zip(self.output_convs, laterals)]
        
        # Use finest level
        out = outputs[0]
        out = self.head(out)
        
        return out


class MultiTaskNetwork(nn.Module):
    """Multi-task network for segmentation + depth + normals"""
    
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        
        # Shared encoder
        self.encoder = SharedEncoder(backbone)
        
        # Task-specific decoders
        self.seg_decoder = TaskSpecificDecoder(
            self.encoder.channels, num_classes, 'segmentation'
        )
        self.depth_decoder = TaskSpecificDecoder(
            self.encoder.channels, 1, 'depth'
        )
        self.normal_decoder = TaskSpecificDecoder(
            self.encoder.channels, 3, 'normals'
        )
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Shared features
        features = self.encoder(x)
        
        # Task predictions
        seg = self.seg_decoder(features)
        depth = self.depth_decoder(features)
        normals = self.normal_decoder(features)
        
        # Upsample to input resolution
        seg = F.interpolate(seg, input_size, mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, input_size, mode='bilinear', align_corners=False)
        normals = F.interpolate(normals, input_size, mode='bilinear', align_corners=False)
        
        # Normalize surface normals
        normals = F.normalize(normals, dim=1)
        
        return {
            'segmentation': seg,
            'depth': depth,
            'normals': normals
        }


class CrossTaskAttention(nn.Module):
    """Cross-task feature sharing module"""
    
    def __init__(self, channels, num_tasks=3):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Cross-attention between tasks
        self.query_convs = nn.ModuleList([
            nn.Conv2d(channels, channels // 8, 1) for _ in range(num_tasks)
        ])
        self.key_convs = nn.ModuleList([
            nn.Conv2d(channels, channels // 8, 1) for _ in range(num_tasks)
        ])
        self.value_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) for _ in range(num_tasks)
        ])
        
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_tasks)
        ])
    
    def forward(self, task_features):
        """
        task_features: list of [B, C, H, W] for each task
        """
        B, C, H, W = task_features[0].shape
        
        outputs = []
        for i, feat in enumerate(task_features):
            # Query from current task
            query = self.query_convs[i](feat).view(B, -1, H * W)  # [B, C', HW]
            
            # Keys and values from all tasks
            keys = []
            values = []
            for j, other_feat in enumerate(task_features):
                if i != j:
                    keys.append(self.key_convs[j](other_feat).view(B, -1, H * W))
                    values.append(self.value_convs[j](other_feat).view(B, -1, H * W))
            
            keys = torch.cat(keys, dim=2)    # [B, C', 2*HW]
            values = torch.cat(values, dim=2)  # [B, C, 2*HW]
            
            # Attention
            attn = torch.bmm(query.permute(0, 2, 1), keys)  # [B, HW, 2*HW]
            attn = F.softmax(attn / (C ** 0.5), dim=-1)
            
            # Aggregate
            out = torch.bmm(values, attn.permute(0, 2, 1))  # [B, C, HW]
            out = out.view(B, C, H, W)
            
            # Residual
            out = self.gamma[i] * out + feat
            outputs.append(out)
        
        return outputs


class UncertaintyWeighting(nn.Module):
    """Automatic task weighting based on uncertainty"""
    
    def __init__(self, num_tasks):
        super().__init__()
        # Log variance for each task (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        """
        losses: list of task losses
        Returns: weighted total loss
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            # Precision weighting: loss / (2 * sigma^2) + log(sigma)
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss


class GradNorm(nn.Module):
    """GradNorm: Gradient normalization for multi-task learning"""
    
    def __init__(self, num_tasks, alpha=1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.initial_losses = None
    
    def compute_grad_norms(self, shared_layer, task_losses):
        """Compute gradient norms for each task"""
        grad_norms = []
        
        for i, loss in enumerate(task_losses):
            # Get gradients w.r.t. shared layer
            grads = torch.autograd.grad(
                loss, shared_layer.parameters(),
                retain_graph=True, create_graph=True
            )
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
            grad_norms.append(grad_norm)
        
        return torch.stack(grad_norms)
    
    def forward(self, task_losses, shared_layer):
        """Compute GradNorm loss for weight update"""
        # Store initial losses for computing loss ratios
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in task_losses]
        
        # Compute weighted losses
        weighted_losses = [self.weights[i] * l for i, l in enumerate(task_losses)]
        total_loss = sum(weighted_losses)
        
        # Compute gradient norms
        grad_norms = self.compute_grad_norms(shared_layer, weighted_losses)
        
        # Target gradient norm (average)
        mean_norm = grad_norms.mean()
        
        # Loss ratios
        loss_ratios = torch.tensor([
            l.item() / self.initial_losses[i]
            for i, l in enumerate(task_losses)
        ], device=task_losses[0].device)
        relative_losses = loss_ratios / loss_ratios.mean()
        
        # Target norms based on relative training rates
        target_norms = mean_norm * (relative_losses ** self.alpha)
        
        # GradNorm loss
        gradnorm_loss = (grad_norms - target_norms.detach()).abs().sum()
        
        return total_loss, gradnorm_loss


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self, num_classes, use_uncertainty=True):
        super().__init__()
        
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.depth_loss = nn.L1Loss()
        self.normal_loss = nn.CosineEmbeddingLoss()
        
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.uncertainty = UncertaintyWeighting(3)
    
    def forward(self, predictions, targets):
        """
        predictions: dict with 'segmentation', 'depth', 'normals'
        targets: dict with 'seg_mask', 'depth_map', 'normal_map'
        """
        losses = []
        
        # Segmentation loss
        seg_loss = self.seg_loss(predictions['segmentation'], targets['seg_mask'])
        losses.append(seg_loss)
        
        # Depth loss (scale-invariant)
        valid_depth = targets['depth_map'] > 0
        if valid_depth.any():
            pred_depth = predictions['depth'][valid_depth]
            gt_depth = targets['depth_map'][valid_depth]
            depth_loss = self.depth_loss(pred_depth, gt_depth)
        else:
            depth_loss = torch.tensor(0.0, device=seg_loss.device)
        losses.append(depth_loss)
        
        # Normal loss (cosine similarity)
        valid_normal = targets['normal_map'].sum(dim=1).abs() > 0
        if valid_normal.any():
            B, C, H, W = predictions['normals'].shape
            pred_n = predictions['normals'].permute(0, 2, 3, 1)[valid_normal]
            gt_n = targets['normal_map'].permute(0, 2, 3, 1)[valid_normal]
            target = torch.ones(pred_n.shape[0], device=pred_n.device)
            normal_loss = self.normal_loss(pred_n, gt_n, target)
        else:
            normal_loss = torch.tensor(0.0, device=seg_loss.device)
        losses.append(normal_loss)
        
        # Combine losses
        if self.use_uncertainty:
            total_loss = self.uncertainty(losses)
        else:
            total_loss = sum(losses)
        
        return total_loss, {
            'seg': seg_loss.item(),
            'depth': depth_loss.item(),
            'normal': normal_loss.item()
        }


def train_multi_task():
    """Training loop for multi-task model"""
    model = MultiTaskNetwork(num_classes=21)
    criterion = MultiTaskLoss(num_classes=21, use_uncertainty=True)
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': criterion.uncertainty.parameters(), 'lr': 1e-3}
    ])
    
    # Training would go here
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Task Weighting Strategies:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| Fixed weights | Manual tuning | When task importance is known |
| Uncertainty weighting | Learn $\sigma$ per task | General purpose |
| GradNorm | Balance gradient magnitudes | Large difference in task difficulties |
| Dynamic Weight Average | Based on loss ratios | Adaptive weighting |

**Benchmark Results (NYUv2):**

| Method | Seg mIoU | Depth RMSE | Normal Error |
|--------|----------|------------|--------------|
| Single-task | 35.2 | 0.612 | 27.4° |
| MTL (equal weights) | 34.1 | 0.625 | 28.1° |
| MTL (uncertainty) | 36.8 | 0.598 | 26.2° |
| MTL (GradNorm) | 37.1 | 0.591 | 25.8° |

**Interview Tip:** Multi-task learning works best when tasks share meaningful features (segmentation + depth both need edge detection). Key challenge is task balancing—uncertainty weighting (learning log-variance per task) is simple and effective. For autonomous driving, combining segmentation + depth + flow in one network is efficient for real-time deployment.

---


---

# --- Vision Transformers (ViT) Questions (from 45_vision_transformers) ---

# Vision Transformers (ViT) Interview Questions - Theory Questions

## Question 1
- [ ] Done

**Describe the positional encoding scheme used in ViT.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**How does the self-attention mechanism work with image patches?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**What are the computational advantages of patch-based processing?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Explain the linear projection layer in ViT patch embedding.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**What is the significance of patch size in ViT performance?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Describe the pre-training strategy for large-scale ViT models.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How does ViT compare to ResNet in terms of inductive biases?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Explain the role of layer normalization in ViT blocks.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How does transfer learning work with pre-trained ViT models?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What is the computational complexity of ViT compared to CNNs?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Describe the MLP head used for classification in ViT.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**How do you visualize attention patterns in Vision Transformers?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**What are the limitations of ViT on small datasets?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Explain the scaling laws for Vision Transformers.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How does ViT handle object detection tasks (DETR)?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**What are hybrid architectures combining CNN and ViT?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Describe the DeiT (Data-efficient image Transformers) approach.**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**How does knowledge distillation improve ViT training?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**What is the role of the distillation token in DeiT?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**Explain masked image modeling in ViT (MAE).**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you implement efficient attention for high-resolution images?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Describe the fine-tuning process for downstream tasks.**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How does ViT perform on different types of visual tasks?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**What is the effect of different attention head configurations?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**Explain the role of dropout in ViT training.**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**How do you handle class imbalance in ViT classification?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**Describe the attention rollout technique for interpretability.**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**How does ViT handle multi-scale features?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**What are the optimization challenges specific to ViT training?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**Explain the concept of attention distance in ViT.**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you implement ViT for video understanding?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**What is the impact of batch size on ViT training stability?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**Describe the gradient flow characteristics in deep ViT models.**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**How does ViT compare to EfficientNet in efficiency metrics?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**What are the deployment considerations for ViT models?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**Explain the role of warmup in ViT optimization.**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you handle computational constraints in ViT inference?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**What are the failure modes of Vision Transformers?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**Describe the attention patterns learned by different ViT layers.**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**How does ViT perform on out-of-distribution images?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**What are the recent advances in efficient ViT architectures?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**Explain the relationship between ViT and CLIP models.**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**What are the future research directions for Vision Transformers?**

**Answer:** _[To be filled]_

---


---

# --- Swin Transformer Questions (from 46_swin_transformer) ---

# Swin Transformer Interview Questions - Theory Questions

## Question 1
- [ ] Done

**Explain shifted window partitioning.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Describe hierarchical representation in Swin.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Compare Swin-T, Swin-S, Swin-B and Swin-L.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Explain limitation of global attention and Swin's solution.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Discuss computational cost versus ViT.**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Describe patch merging and linear embedding.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Explain Swin in object detection pipelines (Swin + FPN).**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Describe Swin for semantic segmentation (UPerNet).**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**Discuss Swin-MST variant for super-resolution.**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Explain linear complexity to image size in Swin.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Compare window size hyperparameter effects.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**Discuss window attention masking implementation.**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**Explain gradient flow across windows.**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Describe Swin's performance on ImageNet-22K.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**Explain layer normalisation placement in Swin.**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**Discuss data augmentation differences vs. CNNs.**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Explain adapting Swin for video (Swin-V2, Swin Transformer 3-D).**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**Compare Swin to ConvNeXt performance.**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**Explain meta-former perspective on Swin.**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**Discuss Swin's transfer to medical image tasks.**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**Explain model-scaling rules in Swin-V2.**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Describe training stability improvements in Swin-V2.**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**Explain log-scaled continuous position bias.**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**Discuss memory savings via zero-redundancy optimizer.**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**Explain Swin's robustness to translation.**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**Describe window-wise activation checkpointing.**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**Discuss sparsity patterns for Swin inference acceleration.**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**Explain distillation of Swin into smaller models.**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**Describe challenges of Swin on non-square images.**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**Discuss Swin in panoptic segmentation.**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**Explain adapting Swin for multi-modal tasks.**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**Describe cross-window attention variants.**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**Discuss Swin's receptive field growth.**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**Explain efficient attention kernels for Swin on GPUs.**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**Discuss Swin for LiDAR point clouds.**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**Explain hierarchical clustering vs. fixed windows.**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**Discuss fine-tuning Swin on small datasets.**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**Explain window size search with NAS.**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**Discuss self-distillation in Swin.**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**Explain cyclic shift and overlap ratio.**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**Describe global average pooling head for classification.**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**Discuss label smoothing and Mixup for Swin.**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**Explain zero-shot robustness of Swin.**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**Discuss quantisation aware training of Swin.**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**Explain data efficient training recipe (DeiT) applied to Swin.**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**Describe 3-D Swin for video action detection.**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**Explain weakly supervised Swin pre-training.**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**Discuss open-vocabulary detection with Swin backbones.**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**Predict future Swin research directions.**

**Answer:** _[To be filled]_

---


---

# --- U-Net Questions (from 50_unet) ---

# U-Net Interview Questions - Theory Questions

## Question 1
- [ ] Done

**Explain encoder–decoder structure of U-Net.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Describe skip connections and why crucial.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Discuss U-Net for biomedical segmentation.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Explain Dice loss vs. cross-entropy for segmentation.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Describe training with patch extraction.**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Explain U-Net variants (U-Net++, UNet-3+).**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Discuss attention U-Net.**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Explain 3-D U-Net.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**Describe residual U-Net.**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Explain multi-scale supervision.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Discuss cascaded U-Nets for coarse-to-fine.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**Explain why symmetric depth helpful.**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**Describe memory optimisation via tiled prediction.**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Explain deep supervision in U-Net++.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**Discuss efficient channel attention in U-Net.**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**Explain use of dilated convolutions.**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Discuss domain adaptation in medical U-Nets.**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**Explain UNet for self-driving car lane detection.**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**Describe label smoothing for segmentation.**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**Explain data augmentation specifics (elastic deformation).**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**Discuss combined BCE+Dice loss.**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Describe group norm vs. batch norm in small batch segmentation.**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**Explain Mobile U-Net for on-device.**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**Discuss U-Net in GAN generator (Pix2Pix).**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**Explain integrating transformers into U-Net (TransUNet).**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**Describe U-NeXt and meta-former perspectives.**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**Explain probabilistic U-Net for uncertainty.**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**Discuss anisotropic receptive fields in 3-D U-Net.**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**Explain federated learning with U-Nets across hospitals.**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**Describe neural architecture search for U-Net blocks.**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**Explain morphological post-processing.**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**Discuss pyramid pooling vs. U-Net skip.**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**Explain attention gates for organ segmentation.**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**Describe class imbalance handling.**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**Explain U-Net in domain generalisation tasks.**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**Discuss adversarial training to refine segmentation.**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**Explain boundary loss for thin structures.**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**Describe ensemble of U-Nets.**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**Explain quantisation for real-time semantic segmentation.**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**Discuss semi-supervised consistency loss in U-Net.**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**Explain multi-modal input (RGB + Depth) U-Net.**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**Describe integration with conditional random fields.**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**Explain hypertuning patch size and stride.**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**Discuss memory attention U-Net.**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**Explain jointly learning segmentation and uncertainty maps.**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**Describe auto-context U-Net pipelines.**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**Explain label propagation with U-Net.**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**Discuss limitations of U-Net on very large images.**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**Predict future U-Net innovations.**

**Answer:** _[To be filled]_

---
