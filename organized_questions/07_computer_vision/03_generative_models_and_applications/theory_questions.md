# Generative Models & CV Applications - Interview Questions

## GANs Fundamentals

### Question 1
- [ ] Done
**Explain the generator and discriminator adversarial training dynamics. What is mode collapse and how do you prevent it?**

**Answer:**

GANs use a minimax game: Generator (G) creates fake images to fool Discriminator (D), while D tries to distinguish real from fake. Training alternates between updating D to classify better and G to produce more realistic images.

**Adversarial Dynamics:**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

| Component | Objective | Training Signal |
|-----------|-----------|-----------------|
| Discriminator | Maximize accuracy | BCE loss on real/fake |
| Generator | Fool discriminator | D's output on fake samples |

**Training Flow:**
1. Sample real data $x$ and noise $z$
2. Generate fake: $\hat{x} = G(z)$
3. Train D: maximize $\log D(x) + \log(1 - D(\hat{x}))$
4. Train G: maximize $\log D(G(z))$ (or minimize $\log(1 - D(G(z)))$)

**Mode Collapse:**

Generator learns to produce limited variety of outputs that fool D, ignoring data diversity.

| Type | Symptom | Example |
|------|---------|---------|
| Full collapse | All outputs identical | Same face every time |
| Partial collapse | Limited modes | Only 3-4 digit types in MNIST |
| Oscillating | Cycles between modes | Alternates between outputs |

**Prevention Strategies:**

| Technique | How It Helps |
|-----------|--------------|
| **Minibatch discrimination** | D sees batch statistics, detects lack of variety |
| **Unrolled GAN** | G considers future D updates |
| **Feature matching** | Match feature statistics, not just fool D |
| **Spectral normalization** | Stabilize D training |
| **Diverse latent sampling** | Encourage diversity in G |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim -> features*8 x 4 x 4
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # -> features*4 x 8 x 8
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # -> features*2 x 16 x 16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # -> features x 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # -> img_channels x 64 x 64
            nn.ConvTranspose2d(features, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # img_channels x 64 x 64 -> features x 32 x 32
            nn.Conv2d(img_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> features*2 x 16 x 16
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> features*4 x 8 x 8
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> features*8 x 4 x 4
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> 1 x 1 x 1
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1, 1)


class MinibatchDiscrimination(nn.Module):
    """Prevents mode collapse by examining batch diversity"""
    
    def __init__(self, in_features, out_features, kernel_dims=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        
        self.T = nn.Parameter(torch.randn(in_features, out_features * kernel_dims))
    
    def forward(self, x):
        # x: [B, in_features]
        # Compute pairwise differences
        M = x @ self.T  # [B, out_features * kernel_dims]
        M = M.view(-1, self.out_features, self.kernel_dims)  # [B, out_features, kernel_dims]
        
        # L1 distance between all pairs
        diff = M.unsqueeze(0) - M.unsqueeze(1)  # [B, B, out_features, kernel_dims]
        diff_norm = diff.abs().sum(dim=3)  # [B, B, out_features]
        
        # Negative exponential
        mb_features = torch.exp(-diff_norm).sum(dim=0) - 1  # [B, out_features]
        
        return torch.cat([x, mb_features], dim=1)


def feature_matching_loss(real_features, fake_features):
    """Match intermediate feature statistics"""
    return F.mse_loss(
        fake_features.mean(dim=0),
        real_features.mean(dim=0).detach()
    )


def train_gan_stable(G, D, dataloader, epochs=100, device='cuda'):
    """Training with mode collapse prevention"""
    
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    latent_dim = 100
    
    for epoch in range(epochs):
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # Labels with smoothing (prevents overconfident D)
            real_labels = torch.ones(batch_size, 1).to(device) * 0.9
            fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1
            
            # Train Discriminator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = G(z).detach()
            
            real_loss = F.binary_cross_entropy(D(real_imgs), real_labels)
            fake_loss = F.binary_cross_entropy(D(fake_imgs), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
            
            # Train Generator (with diversity encouragement)
            z1 = torch.randn(batch_size, latent_dim).to(device)
            z2 = torch.randn(batch_size, latent_dim).to(device)
            
            fake1 = G(z1)
            fake2 = G(z2)
            
            # Adversarial loss
            g_loss = F.binary_cross_entropy(D(fake1), real_labels)
            
            # Diversity regularization: different z should give different outputs
            z_diff = (z1 - z2).abs().mean(dim=1, keepdim=True)
            img_diff = (fake1 - fake2).abs().view(batch_size, -1).mean(dim=1, keepdim=True)
            diversity_loss = F.relu(z_diff - img_diff).mean()
            
            total_g_loss = g_loss + 0.1 * diversity_loss
            
            opt_G.zero_grad()
            total_g_loss.backward()
            opt_G.step()
```

**Monitoring for Mode Collapse:**
- Track variety in generated samples
- Use FID/IS scores over training
- Visualize latent space interpolations

**Interview Tip:** Mode collapse is the key GAN failure mode. Prevention combines architectural choices (spectral normalization, progressive training), loss modifications (WGAN, feature matching), and training tricks (different learning rates, label smoothing). Modern GANs like StyleGAN2 rarely suffer from mode collapse due to these accumulated techniques.

---

### Question 2
- [ ] Done
**Compare GAN loss functions: vanilla GAN, WGAN, WGAN-GP, and hinge loss. When would you use each?**

**Answer:**

Different GAN losses address training stability and gradient issues. Vanilla GAN suffers from vanishing gradients; WGAN uses Wasserstein distance for stable training; WGAN-GP improves with gradient penalty; Hinge loss is fast and stable for large-scale models.

**Loss Comparison:**

| Loss | Formula (D/Critic) | Advantages | Disadvantages |
|------|-------------------|------------|---------------|
| **Vanilla GAN** | $-\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1-D(G(z)))]$ | Simple | Vanishing gradients, mode collapse |
| **WGAN** | $\mathbb{E}[D(G(z))] - \mathbb{E}[D(x)]$ | Meaningful loss, stable | Requires weight clipping |
| **WGAN-GP** | WGAN + $\lambda \mathbb{E}[(\|\nabla D(\hat{x})\|_2 - 1)^2]$ | No clipping, smooth gradients | Slower (gradient computation) |
| **Hinge** | $\mathbb{E}[\max(0, 1-D(x))] + \mathbb{E}[\max(0, 1+D(G(z)))]$ | Fast, stable, saturates well | Less theoretical justification |

**Mathematical Details:**

**Vanilla GAN:**
$$\mathcal{L}_D = -\mathbb{E}_{x}[\log D(x)] - \mathbb{E}_{z}[\log(1 - D(G(z)))]$$
$$\mathcal{L}_G = -\mathbb{E}_{z}[\log D(G(z))]$$

**WGAN (Wasserstein Distance):**
$$\mathcal{L}_{critic} = \mathbb{E}_{z}[D(G(z))] - \mathbb{E}_{x}[D(x)]$$
$$\mathcal{L}_G = -\mathbb{E}_{z}[D(G(z))]$$

**WGAN-GP (Gradient Penalty):**
$$\mathcal{L}_{critic} = \text{WGAN} + \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$
where $\hat{x} = \epsilon x + (1-\epsilon)G(z)$, $\epsilon \sim U(0,1)$

**Hinge Loss:**
$$\mathcal{L}_D = \mathbb{E}_{x}[\max(0, 1 - D(x))] + \mathbb{E}_{z}[\max(0, 1 + D(G(z)))]$$
$$\mathcal{L}_G = -\mathbb{E}_{z}[D(G(z))]$$

**Python Implementation:**
```python
import torch
import torch.nn.functional as F

class GANLosses:
    """Different GAN loss functions"""
    
    @staticmethod
    def vanilla_d_loss(real_pred, fake_pred):
        """Standard GAN discriminator loss"""
        real_loss = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
        return real_loss + fake_loss
    
    @staticmethod
    def vanilla_g_loss(fake_pred):
        """Standard GAN generator loss (non-saturating)"""
        return F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )
    
    @staticmethod
    def wgan_d_loss(real_pred, fake_pred):
        """WGAN critic loss (maximize real - fake)"""
        return fake_pred.mean() - real_pred.mean()
    
    @staticmethod
    def wgan_g_loss(fake_pred):
        """WGAN generator loss (maximize critic score)"""
        return -fake_pred.mean()
    
    @staticmethod
    def gradient_penalty(critic, real, fake, device):
        """Gradient penalty for WGAN-GP"""
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Interpolated samples
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)
        
        # Critic score on interpolated
        mixed_scores = critic(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=mixed_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    @staticmethod
    def wgan_gp_d_loss(real_pred, fake_pred, gp, lambda_gp=10):
        """WGAN-GP critic loss"""
        return fake_pred.mean() - real_pred.mean() + lambda_gp * gp
    
    @staticmethod
    def hinge_d_loss(real_pred, fake_pred):
        """Hinge loss for discriminator"""
        real_loss = F.relu(1.0 - real_pred).mean()
        fake_loss = F.relu(1.0 + fake_pred).mean()
        return real_loss + fake_loss
    
    @staticmethod
    def hinge_g_loss(fake_pred):
        """Hinge loss for generator"""
        return -fake_pred.mean()
    
    @staticmethod
    def ls_d_loss(real_pred, fake_pred):
        """Least squares GAN discriminator loss"""
        real_loss = ((real_pred - 1) ** 2).mean()
        fake_loss = (fake_pred ** 2).mean()
        return 0.5 * (real_loss + fake_loss)
    
    @staticmethod
    def ls_g_loss(fake_pred):
        """Least squares GAN generator loss"""
        return 0.5 * ((fake_pred - 1) ** 2).mean()


def train_with_different_losses(G, D, dataloader, loss_type='wgan-gp'):
    """Training loop with configurable loss"""
    
    losses = GANLosses()
    
    if loss_type == 'wgan-gp':
        opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0, 0.9))
        opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0, 0.9))
        n_critic = 5  # Train D more often
    else:
        opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        n_critic = 1
    
    for real_imgs in dataloader:
        device = real_imgs.device
        batch_size = real_imgs.size(0)
        
        # Train Discriminator/Critic
        for _ in range(n_critic):
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = G(z).detach()
            
            real_pred = D(real_imgs)
            fake_pred = D(fake_imgs)
            
            if loss_type == 'vanilla':
                d_loss = losses.vanilla_d_loss(real_pred, fake_pred)
            elif loss_type == 'wgan':
                d_loss = losses.wgan_d_loss(real_pred, fake_pred)
                # Weight clipping
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)
            elif loss_type == 'wgan-gp':
                gp = losses.gradient_penalty(D, real_imgs, fake_imgs, device)
                d_loss = losses.wgan_gp_d_loss(real_pred, fake_pred, gp)
            elif loss_type == 'hinge':
                d_loss = losses.hinge_d_loss(real_pred, fake_pred)
            elif loss_type == 'lsgan':
                d_loss = losses.ls_d_loss(real_pred, fake_pred)
            
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
        
        # Train Generator
        z = torch.randn(batch_size, 100, device=device)
        fake_imgs = G(z)
        fake_pred = D(fake_imgs)
        
        if loss_type == 'vanilla':
            g_loss = losses.vanilla_g_loss(fake_pred)
        elif loss_type in ['wgan', 'wgan-gp']:
            g_loss = losses.wgan_g_loss(fake_pred)
        elif loss_type == 'hinge':
            g_loss = losses.hinge_g_loss(fake_pred)
        elif loss_type == 'lsgan':
            g_loss = losses.ls_g_loss(fake_pred)
        
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
```

**When to Use Each:**

| Scenario | Recommended Loss |
|----------|------------------|
| Starting out/baseline | WGAN-GP |
| Large-scale (BigGAN, StyleGAN) | Hinge + spectral norm |
| Need stable training | WGAN-GP |
| Speed matters | Hinge or LSGAN |
| Simple datasets | Vanilla (with tricks) |

**Training Stability Comparison:**

| Loss | Gradient Behavior | Training Stability |
|------|-------------------|-------------------|
| Vanilla | Vanishes when D is strong | Unstable |
| LSGAN | Smooth gradients | Moderate |
| WGAN | Continuous, meaningful | Good |
| WGAN-GP | Smooth + 1-Lipschitz | Excellent |
| Hinge | Saturates at margin | Very good |

**Interview Tip:** WGAN-GP is the safest choice for stable training. Hinge loss with spectral normalization is standard for large-scale models (BigGAN, StyleGAN). The key insight is that WGAN measures Earth Mover's distance, which provides meaningful gradients even when distributions don't overlap—vanilla GAN fails here.

---

### Question 3
- [ ] Done
**Explain progressive growing in GANs and how it enables high-resolution image generation.**

**Answer:**

Progressive Growing trains GANs by starting at low resolution (4×4) and gradually adding layers to reach high resolution (1024×1024). This curriculum approach stabilizes training—the model first learns coarse structure, then progressively adds fine details.

**Core Idea:**

| Stage | Resolution | What's Learned |
|-------|------------|----------------|
| 1 | 4×4 | Basic colors, rough shapes |
| 2 | 8×8 | Major structure |
| 3 | 16×16 | Object layout |
| 4 | 32×32 | Features, parts |
| 5 | 64×64 | Details |
| 6 | 128×128 | Fine textures |
| 7-9 | 256-1024 | High-frequency details |

**Why It Works:**

1. **Simpler problem first**: Low-res images are easier to model
2. **Stable gradients**: Fewer layers = more stable training
3. **Faster iterations**: Small images train quickly
4. **Knowledge transfer**: Learned weights help higher resolutions

**Smooth Transition (Fade-In):**

When adding a new layer, blend between resized output and new layer output:

$$\text{out} = (1 - \alpha) \cdot \text{upsampled\_prev} + \alpha \cdot \text{new\_layer}$$

$\alpha$ increases from 0 to 1 over training iterations.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressiveGenerator(nn.Module):
    """Progressive Growing Generator"""
    
    def __init__(self, latent_dim=512, max_resolution=1024):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_resolution = max_resolution
        
        # Starting constant 4x4
        self.initial = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Synthesis blocks for each resolution
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        # 4x4 -> initial to_rgb
        self.to_rgb.append(nn.Conv2d(512, 3, 1))
        
        # Progressive blocks: 8x8, 16x16, ..., max_resolution
        channels = [512, 512, 512, 512, 256, 128, 64, 32, 16]
        
        num_blocks = int(torch.log2(torch.tensor(max_resolution))) - 2
        
        for i in range(num_blocks):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            self.blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.2)
            ))
            self.to_rgb.append(nn.Conv2d(out_ch, 3, 1))
        
        # Current training stage
        self.current_stage = 0
        self.alpha = 1.0  # Blend factor for fade-in
    
    def forward(self, z):
        # Map latent to initial constant
        batch_size = z.size(0)
        x = self.initial.expand(batch_size, -1, -1, -1)
        
        if self.current_stage == 0:
            return self.to_rgb[0](x)
        
        # Process through blocks up to current stage
        for i in range(self.current_stage):
            if i == self.current_stage - 1 and self.alpha < 1.0:
                # Fade-in: blend between upsampled previous and new block
                prev_rgb = F.interpolate(
                    self.to_rgb[i](x), scale_factor=2, mode='bilinear'
                )
                x = self.blocks[i](x)
                new_rgb = self.to_rgb[i + 1](x)
                return (1 - self.alpha) * prev_rgb + self.alpha * new_rgb
            else:
                x = self.blocks[i](x)
        
        return self.to_rgb[self.current_stage](x)
    
    def grow(self):
        """Move to next resolution stage"""
        if self.current_stage < len(self.blocks):
            self.current_stage += 1
            self.alpha = 0.0  # Start fade-in
    
    def update_alpha(self, progress):
        """Update fade-in alpha (0 to 1)"""
        self.alpha = min(1.0, progress)


class ProgressiveDiscriminator(nn.Module):
    """Progressive Growing Discriminator (mirrors Generator)"""
    
    def __init__(self, max_resolution=1024):
        super().__init__()
        
        # From RGB for each resolution
        self.from_rgb = nn.ModuleList()
        
        # Downsampling blocks
        self.blocks = nn.ModuleList()
        
        channels = [16, 32, 64, 128, 256, 512, 512, 512, 512]
        
        num_blocks = int(torch.log2(torch.tensor(max_resolution))) - 2
        
        for i in range(num_blocks):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            self.from_rgb.append(nn.Conv2d(3, in_ch, 1))
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ))
        
        # Final from_rgb for 4x4
        self.from_rgb.append(nn.Conv2d(3, 512, 1))
        
        # Final layers at 4x4
        self.final = nn.Sequential(
            nn.Conv2d(513, 512, 3, padding=1),  # +1 for minibatch std
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
        
        self.current_stage = 0
        self.alpha = 1.0
    
    def minibatch_std(self, x):
        """Minibatch standard deviation layer"""
        std = x.std(dim=0, keepdim=True).mean()
        std_feature = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std_feature], dim=1)
    
    def forward(self, x):
        # Entry point based on current stage
        block_idx = len(self.blocks) - self.current_stage - 1
        
        if self.current_stage == 0:
            x = self.from_rgb[-1](x)
        else:
            if self.alpha < 1.0:
                # Fade-in
                new_x = self.from_rgb[block_idx](x)
                new_x = self.blocks[block_idx](new_x)
                
                old_x = F.avg_pool2d(x, 2)
                old_x = self.from_rgb[block_idx + 1](old_x)
                
                x = self.alpha * new_x + (1 - self.alpha) * old_x
                block_idx += 1
            else:
                x = self.from_rgb[block_idx](x)
                x = self.blocks[block_idx](x)
                block_idx += 1
            
            # Process remaining blocks
            for i in range(block_idx, len(self.blocks)):
                x = self.blocks[i](x)
        
        # Final layers
        x = self.minibatch_std(x)
        return self.final(x)


def train_progressive_gan(G, D, dataloader, epochs_per_stage=10):
    """Progressive training loop"""
    
    num_stages = len(G.blocks)
    
    for stage in range(num_stages + 1):
        resolution = 4 * (2 ** stage)
        print(f"Training at {resolution}x{resolution}")
        
        for epoch in range(epochs_per_stage):
            # Fade-in period (first half of epochs)
            if epoch < epochs_per_stage // 2:
                alpha = epoch / (epochs_per_stage // 2)
                G.update_alpha(alpha)
                D.alpha = alpha
            else:
                G.update_alpha(1.0)
                D.alpha = 1.0
            
            for real_imgs in dataloader:
                # Resize real images to current resolution
                real_imgs = F.interpolate(real_imgs, size=resolution)
                
                # Standard GAN training step...
                pass
        
        # Grow to next resolution
        if stage < num_stages:
            G.grow()
            D.current_stage += 1
```

**Key Techniques in ProGAN:**

| Technique | Purpose |
|-----------|---------|
| Fade-in ($\alpha$ blending) | Smooth layer transition |
| Minibatch std | Prevent mode collapse |
| Pixel normalization | Stabilize G |
| Equalized learning rate | Uniform layer updates |

**Resolution Timeline (Typical):**

| Resolution | Training Images |
|------------|-----------------|
| 4×4 | ~600K |
| 8×8 | ~600K |
| 16×16 | ~600K |
| 32×32 | ~800K |
| 64×64 | ~800K |
| 128×128 | ~1M |
| 256×256 | ~1M |
| 512×512 | ~1M |
| 1024×1024 | ~2M |

**Interview Tip:** Progressive growing was revolutionary for high-res GANs (2017). The key insight is curriculum learning—start simple, add complexity gradually. StyleGAN2 removed progressive growing in favor of better architecture, but the curriculum concept remains influential. Fade-in prevents abrupt changes that destabilize training.

---

### Question 4
- [ ] Done
**What are FID (Fréchet Inception Distance) and Inception Score? How do you evaluate generative model quality?**

**Answer:**

FID and IS are standard metrics for evaluating generative models. IS measures quality and diversity using a classifier; FID compares feature statistics between real and generated distributions—lower FID means closer to real data.

**Inception Score (IS):**

Measures two properties:
1. **Quality**: Generated images should be classifiable (low entropy $p(y|x)$)
2. **Diversity**: All classes should appear (high entropy $p(y)$)

$$\text{IS} = \exp\left(\mathbb{E}_x \left[D_{KL}(p(y|x) \| p(y))\right]\right)$$

| IS Value | Interpretation |
|----------|----------------|
| ~1 | Poor (random noise) |
| ~10 | Moderate |
| ~50+ | CIFAR-10 quality |
| ~150+ | ImageNet quality |

**Fréchet Inception Distance (FID):**

Compares Inception feature distributions (real vs. generated) as multivariate Gaussians:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

| FID Value | Interpretation |
|-----------|----------------|
| 0 | Identical to real |
| <10 | Excellent |
| 10-50 | Good |
| 50-100 | Moderate |
| >100 | Poor |

**Comparison:**

| Metric | Pros | Cons |
|--------|------|------|
| **IS** | Simple, captures quality+diversity | Doesn't use real data, ImageNet-biased |
| **FID** | Uses real data, captures distribution | Assumes Gaussian, sample-dependent |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision import models
import torch.nn.functional as F

class InceptionV3Features(nn.Module):
    """Extract features from InceptionV3 for FID/IS"""
    
    def __init__(self, output_blocks=[3], resize_input=True):
        super().__init__()
        
        inception = models.inception_v3(pretrained=True)
        inception.eval()
        
        self.resize_input = resize_input
        self.blocks = nn.ModuleList()
        
        # Block 0: up to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        
        # Block 1: up to maxpool2
        block1 = [
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2)
        ]
        self.blocks.append(nn.Sequential(*block1))
        
        # Block 2: Mixed layers up to Mixed_5d
        block2 = [
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        ]
        self.blocks.append(nn.Sequential(*block2))
        
        # Block 3: Final layers (2048-dim features for FID)
        block3 = [
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(1)
        ]
        self.blocks.append(nn.Sequential(*block3))
        
        self.output_blocks = output_blocks
        
        # Freeze
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        outputs = []
        
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1] if needed
        if x.min() >= 0:
            x = 2 * x - 1
        
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outputs.append(x)
        
        return outputs


def calculate_inception_score(imgs, model, batch_size=32, splits=10):
    """Calculate Inception Score"""
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get predictions
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size].to(device)
            logits = model(batch)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Split and calculate
    scores = []
    for i in range(splits):
        part = preds[i * len(preds) // splits : (i+1) * len(preds) // splits]
        
        # p(y|x) - per-image class probabilities
        py_given_x = part
        
        # p(y) - marginal class distribution
        py = np.mean(part, axis=0, keepdims=True)
        
        # KL divergence
        kl = py_given_x * (np.log(py_given_x + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1)
        
        scores.append(np.exp(np.mean(kl)))
    
    return float(np.mean(scores)), float(np.std(scores))


def calculate_fid(real_features, fake_features):
    """Calculate Fréchet Inception Distance"""
    
    # Calculate statistics
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    
    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    
    return float(fid)


def calculate_kid(real_features, fake_features, subset_size=1000, num_subsets=100):
    """Kernel Inception Distance - more reliable for small samples"""
    
    n = real_features.shape[1]
    
    def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1):
        if gamma is None:
            gamma = 1.0 / n
        return (gamma * (x @ y.T) + coef0) ** degree
    
    kids = []
    
    for _ in range(num_subsets):
        # Random subsets
        real_idx = np.random.choice(len(real_features), subset_size, replace=False)
        fake_idx = np.random.choice(len(fake_features), subset_size, replace=False)
        
        real_subset = real_features[real_idx]
        fake_subset = fake_features[fake_idx]
        
        # Kernel matrices
        k_rr = polynomial_kernel(real_subset, real_subset)
        k_ff = polynomial_kernel(fake_subset, fake_subset)
        k_rf = polynomial_kernel(real_subset, fake_subset)
        
        # MMD
        mmd = (k_rr.sum() - np.diag(k_rr).sum()) / (subset_size * (subset_size - 1))
        mmd += (k_ff.sum() - np.diag(k_ff).sum()) / (subset_size * (subset_size - 1))
        mmd -= 2 * k_rf.mean()
        
        kids.append(mmd)
    
    return float(np.mean(kids)), float(np.std(kids))


class GenerativeModelEvaluator:
    """Complete evaluation suite for generative models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = InceptionV3Features(output_blocks=[3]).to(device)
        self.inception.eval()
        
        # For IS calculation
        self.classifier = models.inception_v3(pretrained=True).to(device)
        self.classifier.eval()
    
    @torch.no_grad()
    def extract_features(self, images, batch_size=32):
        """Extract Inception features"""
        features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            feat = self.inception(batch)[0]
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def evaluate(self, real_images, fake_images, num_samples=10000):
        """Full evaluation"""
        
        # Sample subset
        real_idx = np.random.choice(len(real_images), min(num_samples, len(real_images)), replace=False)
        fake_idx = np.random.choice(len(fake_images), min(num_samples, len(fake_images)), replace=False)
        
        real_subset = real_images[real_idx]
        fake_subset = fake_images[fake_idx]
        
        # Extract features
        real_features = self.extract_features(real_subset)
        fake_features = self.extract_features(fake_subset)
        
        # Calculate metrics
        fid = calculate_fid(real_features, fake_features)
        is_mean, is_std = calculate_inception_score(fake_subset, self.classifier)
        kid_mean, kid_std = calculate_kid(real_features, fake_features)
        
        return {
            'fid': fid,
            'is_mean': is_mean,
            'is_std': is_std,
            'kid_mean': kid_mean,
            'kid_std': kid_std
        }
```

**Other Metrics:**

| Metric | Use Case |
|--------|----------|
| **KID** | Better for small samples |
| **Precision/Recall** | Separate quality vs. coverage |
| **LPIPS** | Perceptual similarity |
| **SSIM/PSNR** | Pixel-level (paired data) |

**Benchmark FID Scores (FFHQ 1024×1024):**

| Model | FID |
|-------|-----|
| StyleGAN | 4.40 |
| StyleGAN2 | 2.84 |
| StyleGAN3 | 2.79 |
| Diffusion (ADM) | ~3.0 |

**Interview Tip:** FID is the gold standard—always report it. Use ≥50K samples for stable estimates. IS is useful but ImageNet-biased. For small datasets or quick iteration, KID is more reliable. Precision/Recall helps diagnose mode collapse (high precision, low recall) vs. quality issues (low precision).

---

## StyleGAN Family

### Question 5
- [ ] Done
**Explain StyleGAN's mapping network and how the W latent space differs from Z space.**

**Answer:**

StyleGAN's mapping network is an 8-layer MLP that transforms random noise Z into an intermediate latent space W. W space is more disentangled—each dimension controls more independent features, enabling better image manipulation and interpolation.

**Architecture:**

```
Z space (512-dim) → Mapping Network (8 FC layers) → W space (512-dim) → Synthesis Network → Image
```

**Why W Space is Better:**

| Property | Z Space | W Space |
|----------|---------|---------|
| Distribution | Fixed Gaussian | Learned, data-adapted |
| Disentanglement | Entangled | More disentangled |
| Interpolation | May cross invalid regions | Smoother, more meaningful |
| Editing | Unpredictable | Predictable attribute changes |

**Mathematical Perspective:**

Z is sampled from $\mathcal{N}(0, I)$—a hypersphere where corners are far from training data. The mapping network learns $f: Z \rightarrow W$ that "warps" this space to match the data distribution.

**Key Insight:** Natural images don't uniformly fill the Gaussian space. Some Z values produce unrealistic images. W is shaped to avoid these regions.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np

class MappingNetwork(nn.Module):
    """StyleGAN Mapping Network: Z -> W"""
    
    def __init__(self, z_dim=512, w_dim=512, num_layers=8, 
                 lr_multiplier=0.01):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_features = z_dim if i == 0 else w_dim
            
            layers.append(EqualizedLinear(
                in_features, w_dim, 
                lr_multiplier=lr_multiplier
            ))
            layers.append(nn.LeakyReLU(0.2))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        # Normalize z to unit length (PixelNorm)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return self.net(z)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate"""
    
    def __init__(self, in_features, out_features, lr_multiplier=1.0, bias=True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # He initialization scale
        self.scale = lr_multiplier / np.sqrt(in_features)
    
    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight * self.scale, self.bias)
        return out


class StyleGANGenerator(nn.Module):
    """Simplified StyleGAN Generator structure"""
    
    def __init__(self, z_dim=512, w_dim=512, img_resolution=1024):
        super().__init__()
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim)
        
        # Number of synthesis layers
        self.num_layers = int(np.log2(img_resolution)) * 2 - 2
        
        # Synthesis network (simplified)
        self.synthesis = SynthesisNetwork(w_dim, img_resolution)
    
    def forward(self, z, truncation_psi=1.0, w_avg=None):
        # Map Z to W
        w = self.mapping(z)
        
        # Truncation trick
        if truncation_psi < 1.0 and w_avg is not None:
            w = w_avg + truncation_psi * (w - w_avg)
        
        # Broadcast W to all layers (W space)
        # Or use different W per layer (W+ space)
        w_broadcast = w.unsqueeze(1).repeat(1, self.num_layers, 1)
        
        # Generate image
        img = self.synthesis(w_broadcast)
        
        return img


class LatentSpaceAnalysis:
    """Tools for analyzing StyleGAN latent spaces"""
    
    def __init__(self, generator, device='cuda'):
        self.G = generator.to(device)
        self.device = device
        self.w_avg = None
    
    def compute_w_statistics(self, num_samples=10000):
        """Compute mean W for truncation trick"""
        self.G.eval()
        
        ws = []
        with torch.no_grad():
            for _ in range(num_samples // 100):
                z = torch.randn(100, 512, device=self.device)
                w = self.G.mapping(z)
                ws.append(w)
        
        ws = torch.cat(ws, dim=0)
        self.w_avg = ws.mean(dim=0, keepdim=True)
        self.w_std = ws.std()
        
        return self.w_avg, self.w_std
    
    def interpolate_z(self, z1, z2, steps=10):
        """Interpolate in Z space (slerp for Gaussian)"""
        # Spherical interpolation for unit sphere
        z1_norm = z1 / z1.norm(dim=1, keepdim=True)
        z2_norm = z2 / z2.norm(dim=1, keepdim=True)
        
        omega = torch.acos((z1_norm * z2_norm).sum(dim=1, keepdim=True))
        
        images = []
        for t in np.linspace(0, 1, steps):
            z_interp = (torch.sin((1-t) * omega) * z1_norm + 
                       torch.sin(t * omega) * z2_norm) / torch.sin(omega)
            
            with torch.no_grad():
                img = self.G(z_interp)
            images.append(img)
        
        return images
    
    def interpolate_w(self, w1, w2, steps=10):
        """Interpolate in W space (linear)"""
        images = []
        
        for t in np.linspace(0, 1, steps):
            w_interp = (1 - t) * w1 + t * w2
            
            with torch.no_grad():
                w_broadcast = w_interp.unsqueeze(1).repeat(
                    1, self.G.num_layers, 1
                )
                img = self.G.synthesis(w_broadcast)
            images.append(img)
        
        return images
    
    def measure_ppl(self, num_samples=10000, epsilon=1e-4):
        """Perceptual Path Length - measures W smoothness"""
        # PPL measures how much image changes for small latent changes
        # Lower PPL = smoother, more disentangled space
        
        # Sample pairs of nearby latents
        z1 = torch.randn(num_samples, 512, device=self.device)
        z2 = z1 + epsilon * torch.randn_like(z1)
        
        with torch.no_grad():
            w1 = self.G.mapping(z1)
            w2 = self.G.mapping(z2)
            
            # W space distance
            w_dist = (w1 - w2).norm(dim=1)
            
            # Generate images
            img1 = self.G(z1)
            img2 = self.G(z2)
            
            # Perceptual distance (would use LPIPS)
            # Simplified: use L2 as proxy
            img_dist = (img1 - img2).view(num_samples, -1).norm(dim=1)
        
        ppl = (img_dist / w_dist).mean()
        
        return ppl.item()


def compare_z_vs_w_interpolation(G, device='cuda'):
    """Demonstrate Z vs W interpolation difference"""
    
    analyzer = LatentSpaceAnalysis(G, device)
    
    # Sample two random latents
    z1 = torch.randn(1, 512, device=device)
    z2 = torch.randn(1, 512, device=device)
    
    # Z interpolation
    z_images = analyzer.interpolate_z(z1, z2, steps=5)
    
    # W interpolation
    with torch.no_grad():
        w1 = G.mapping(z1)
        w2 = G.mapping(z2)
    w_images = analyzer.interpolate_w(w1, w2, steps=5)
    
    # W interpolation is typically smoother
    return z_images, w_images
```

**W vs W+ vs S Space:**

| Space | Description | Use Case |
|-------|-------------|----------|
| **W** | Single 512-dim vector | Generation |
| **W+** | Different W per layer | More flexible editing |
| **S** | Post-affine transform | Fine-grained control |

**Interview Tip:** The mapping network is StyleGAN's key innovation—it creates a disentangled space where linear operations correspond to meaningful image edits. When editing real images (GAN inversion), W+ space allows more faithful reconstruction but may lose disentanglement. The truncation trick ($\psi < 1$) trades diversity for quality by moving W toward the mean.

---

### Question 6
- [ ] Done
**Describe AdaIN (Adaptive Instance Normalization) and how it injects style at each layer.**

**Answer:**

AdaIN transfers style by normalizing content features, then scaling and shifting with style statistics. In StyleGAN, learned affine transforms from W vector control scale ($\gamma$) and shift ($\beta$) at each layer, modulating feature maps to inject style.

**AdaIN Formula:**

$$\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$$

For StyleGAN (learned from W):

$$\text{out} = \gamma_s \cdot \frac{x - \mu(x)}{\sigma(x)} + \beta_s$$

Where $\gamma_s, \beta_s$ are learned from style vector $w$.

**Why It Works:**

| Step | Effect |
|------|--------|
| Normalize content | Remove original statistics |
| Apply style scale | Control feature magnitude |
| Apply style bias | Shift feature distribution |

**Style Injection in StyleGAN:**

```
W (512-dim) → Affine Transform → (γ, β) per channel → Modulate conv features
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    
    def __init__(self, style_dim, num_features):
        super().__init__()
        
        # Affine transform from style to scale/shift
        self.fc = nn.Linear(style_dim, num_features * 2)
        
        # Initialize to identity (gamma=1, beta=0)
        self.fc.bias.data[:num_features] = 1  # gamma
        self.fc.bias.data[num_features:] = 0  # beta
    
    def forward(self, x, style):
        """
        x: content features [B, C, H, W]
        style: style vector [B, style_dim]
        """
        # Get style parameters
        params = self.fc(style)  # [B, 2*C]
        gamma, beta = params.chunk(2, dim=1)  # Each [B, C]
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        # Instance normalize content
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        x_normalized = (x - mean) / std
        
        # Apply style
        return gamma * x_normalized + beta


class StyleConvBlock(nn.Module):
    """Convolution block with style modulation (StyleGAN style)"""
    
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=3):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.adain = AdaIN(w_dim, out_channels)
        self.noise_scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x, w, noise=None):
        # Convolution
        x = self.conv(x)
        
        # Add noise (per-pixel variation)
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        x = x + self.noise_scale * noise
        
        # Style modulation
        x = self.adain(x, w)
        
        # Activation
        x = self.activation(x)
        
        return x


class ModulatedConv2d(nn.Module):
    """StyleGAN2 Modulated Convolution (replaces AdaIN)"""
    
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=3, demodulate=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        
        # Convolution weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        
        # Style to modulation
        self.modulation = nn.Linear(w_dim, in_channels)
    
    def forward(self, x, w):
        """
        x: [B, C_in, H, W]
        w: [B, w_dim]
        """
        B, C_in, H, W = x.shape
        
        # Get modulation weights from style
        style = self.modulation(w)  # [B, C_in]
        style = style.view(B, 1, C_in, 1, 1)  # [B, 1, C_in, 1, 1]
        
        # Modulate weights
        weight = self.weight.unsqueeze(0) * style  # [B, C_out, C_in, K, K]
        
        # Demodulation (normalize to prevent magnitude explosion)
        if self.demodulate:
            demod = torch.rsqrt(
                weight.pow(2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8
            )
            weight = weight * demod
        
        # Group convolution (each sample has its own kernel)
        x = x.view(1, B * C_in, H, W)
        weight = weight.view(B * self.out_channels, C_in, self.kernel_size, self.kernel_size)
        
        out = F.conv2d(x, weight, padding=self.kernel_size // 2, groups=B)
        out = out.view(B, self.out_channels, H, W)
        
        return out


class StyleGANSynthesisLayer(nn.Module):
    """Complete synthesis layer with all components"""
    
    def __init__(self, in_channels, out_channels, w_dim, resolution, 
                 use_noise=True, use_modulated_conv=True):
        super().__init__()
        
        self.resolution = resolution
        self.use_noise = use_noise
        
        # Main convolution
        if use_modulated_conv:
            self.conv = ModulatedConv2d(in_channels, out_channels, w_dim)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.adain = AdaIN(w_dim, out_channels)
        
        self.use_modulated_conv = use_modulated_conv
        
        # Noise injection
        if use_noise:
            self.noise_weight = nn.Parameter(torch.zeros(1))
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x, w, noise=None):
        # Convolution with style
        if self.use_modulated_conv:
            x = self.conv(x, w)
        else:
            x = self.conv(x)
            x = self.adain(x, w)
        
        # Add noise
        if self.use_noise:
            if noise is None:
                noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            x = x + self.noise_weight * noise
        
        x = self.activation(x)
        
        return x


def visualize_style_effect():
    """Demonstrate how different styles affect generation"""
    
    # Two random style vectors
    w1 = torch.randn(1, 512)
    w2 = torch.randn(1, 512)
    
    # Same content, different styles
    content = torch.randn(1, 256, 32, 32)
    
    adain = AdaIN(512, 256)
    
    # Apply different styles
    styled1 = adain(content, w1)
    styled2 = adain(content, w2)
    
    print(f"Style 1 stats: mean={styled1.mean():.3f}, std={styled1.std():.3f}")
    print(f"Style 2 stats: mean={styled2.mean():.3f}, std={styled2.std():.3f}")
    
    return styled1, styled2
```

**StyleGAN1 vs StyleGAN2:**

| Aspect | StyleGAN1 (AdaIN) | StyleGAN2 (ModConv) |
|--------|-------------------|---------------------|
| Normalization | Instance norm | Weight demodulation |
| Artifacts | Water droplet artifacts | Cleaner |
| Disentanglement | Good | Better |

**Why AdaIN at Each Layer:**

| Layer Level | What Style Controls |
|-------------|---------------------|
| Early (4-8 px) | Pose, face shape |
| Middle (16-32 px) | Features, expression |
| Late (64-1024 px) | Colors, textures |

**Interview Tip:** AdaIN is the mechanism that makes StyleGAN controllable—it provides entry points at each resolution to inject style information. StyleGAN2 replaced AdaIN with weight demodulation to fix water droplet artifacts. The key insight is that style (statistics: mean/variance) and content (structure) can be separated and recombined.

---

### Question 7
- [ ] Done
**Explain the separation of coarse, middle, and fine styles in StyleGAN layers.**

**Answer:**

StyleGAN's multi-resolution synthesis allows different W vectors to control different aspects. Early layers (low resolution) control coarse features like pose; middle layers control facial structure; late layers control fine details like hair texture and colors.

**Layer-Attribute Correspondence:**

| Resolution | Layers | Controls |
|------------|--------|----------|
| **Coarse** (4-8 px) | 0-3 | Pose, face shape, gender, age |
| **Middle** (16-32 px) | 4-7 | Features, expression, hair style |
| **Fine** (64-1024 px) | 8-17 | Color scheme, textures, micro-details |

**Style Mixing Experiment:**

Swap W vectors at different layer cutoffs:
- Swap at coarse → keep structure of source A, features/colors of B
- Swap at fine → keep structure and features of A, colors of B

**Mathematical View:**

With W+ space, each layer $l$ gets its own style vector $w_l$:

$$\text{Layer}_l(x) = f(x, w_l)$$

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np

class StyleMixer:
    """Analyze and manipulate style across layers"""
    
    def __init__(self, generator, device='cuda'):
        self.G = generator.to(device)
        self.device = device
        
        # Layer groupings for StyleGAN2 (1024x1024)
        self.coarse_layers = list(range(0, 4))    # 4x4 to 8x8
        self.middle_layers = list(range(4, 8))    # 16x16 to 32x32
        self.fine_layers = list(range(8, 18))     # 64x64 to 1024x1024
    
    def generate_w_plus(self, z, num_layers=18):
        """Generate W+ (different W per layer)"""
        with torch.no_grad():
            w = self.G.mapping(z)  # [B, 512]
            w_plus = w.unsqueeze(1).repeat(1, num_layers, 1)  # [B, 18, 512]
        return w_plus
    
    def style_mixing(self, w_source, w_target, mixing_level='coarse'):
        """Mix styles from two W+ vectors"""
        w_mixed = w_source.clone()
        
        if mixing_level == 'coarse':
            # Use target's coarse styles
            w_mixed[:, self.coarse_layers] = w_target[:, self.coarse_layers]
        elif mixing_level == 'middle':
            # Use target's middle styles
            w_mixed[:, self.middle_layers] = w_target[:, self.middle_layers]
        elif mixing_level == 'fine':
            # Use target's fine styles
            w_mixed[:, self.fine_layers] = w_target[:, self.fine_layers]
        elif isinstance(mixing_level, int):
            # Crossover at specific layer
            w_mixed[:, mixing_level:] = w_target[:, mixing_level:]
        
        return w_mixed
    
    def create_style_mixing_figure(self, num_rows=4, num_cols=4):
        """Create style mixing grid visualization"""
        
        # Generate source (rows) and destination (columns) latents
        z_sources = torch.randn(num_rows, 512, device=self.device)
        z_dests = torch.randn(num_cols, 512, device=self.device)
        
        w_sources = [self.generate_w_plus(z.unsqueeze(0)) for z in z_sources]
        w_dests = [self.generate_w_plus(z.unsqueeze(0)) for z in z_dests]
        
        # Generate pure images
        images = {}
        
        with torch.no_grad():
            # Row images (sources)
            for i, w in enumerate(w_sources):
                images[('row', i)] = self.G.synthesis(w)
            
            # Column images (destinations)
            for j, w in enumerate(w_dests):
                images[('col', j)] = self.G.synthesis(w)
            
            # Mixed images (source structure + destination style)
            for i, w_src in enumerate(w_sources):
                for j, w_dst in enumerate(w_dests):
                    # Use source for coarse, destination for middle+fine
                    w_mixed = self.style_mixing(w_src, w_dst, mixing_level=4)
                    images[(i, j)] = self.G.synthesis(w_mixed)
        
        return images


class LayerWiseStyleEditor:
    """Edit styles at specific layer groups"""
    
    def __init__(self, generator, num_layers=18):
        self.G = generator
        self.num_layers = num_layers
        
        # Discovered semantic directions in W space
        self.directions = {}
    
    def find_direction(self, attribute_name, positive_ws, negative_ws):
        """Find direction for an attribute (simplified linear SVM)"""
        # In practice, use InterFaceGAN or GANSpace
        positive_mean = positive_ws.mean(dim=0)
        negative_mean = negative_ws.mean(dim=0)
        
        direction = positive_mean - negative_mean
        direction = direction / direction.norm()
        
        self.directions[attribute_name] = direction
        return direction
    
    def edit_at_layers(self, w_plus, direction, strength, layers):
        """Apply direction only at specific layers"""
        w_edited = w_plus.clone()
        
        for layer_idx in layers:
            w_edited[:, layer_idx] += strength * direction
        
        return w_edited
    
    def coarse_edit(self, w_plus, direction, strength):
        """Edit only coarse features (pose, shape)"""
        layers = list(range(0, 4))
        return self.edit_at_layers(w_plus, direction, strength, layers)
    
    def middle_edit(self, w_plus, direction, strength):
        """Edit middle features (expression, features)"""
        layers = list(range(4, 8))
        return self.edit_at_layers(w_plus, direction, strength, layers)
    
    def fine_edit(self, w_plus, direction, strength):
        """Edit fine features (colors, textures)"""
        layers = list(range(8, self.num_layers))
        return self.edit_at_layers(w_plus, direction, strength, layers)
    
    def progressive_edit(self, w_plus, direction, strength):
        """Apply direction progressively across all layers"""
        w_edited = w_plus.clone()
        
        for i in range(self.num_layers):
            # Varying strength across layers
            layer_strength = strength * (1 - i / self.num_layers)
            w_edited[:, i] += layer_strength * direction
        
        return w_edited


class StyleTransferBetweenFaces:
    """Transfer specific attributes between faces"""
    
    def __init__(self, generator):
        self.G = generator
    
    def transfer_pose(self, w_content, w_style):
        """Keep content's features, use style's pose"""
        w_result = w_content.clone()
        w_result[:, :4] = w_style[:, :4]  # Coarse layers
        return w_result
    
    def transfer_expression(self, w_content, w_style):
        """Keep content's identity, use style's expression"""
        w_result = w_content.clone()
        w_result[:, 4:8] = w_style[:, 4:8]  # Middle layers
        return w_result
    
    def transfer_colors(self, w_content, w_style):
        """Keep content's structure, use style's colors"""
        w_result = w_content.clone()
        w_result[:, 8:] = w_style[:, 8:]  # Fine layers
        return w_result
    
    def interpolate_layers(self, w1, w2, coarse_ratio=0.0, middle_ratio=0.5, fine_ratio=1.0):
        """Interpolate with different ratios per layer group"""
        w_result = w1.clone()
        
        # Coarse interpolation
        w_result[:, :4] = (1 - coarse_ratio) * w1[:, :4] + coarse_ratio * w2[:, :4]
        
        # Middle interpolation
        w_result[:, 4:8] = (1 - middle_ratio) * w1[:, 4:8] + middle_ratio * w2[:, 4:8]
        
        # Fine interpolation
        w_result[:, 8:] = (1 - fine_ratio) * w1[:, 8:] + fine_ratio * w2[:, 8:]
        
        return w_result


def demo_layer_effects():
    """Demonstrate what each layer group controls"""
    
    layer_groups = {
        'Coarse (4-8px)': {
            'layers': [0, 1, 2, 3],
            'controls': ['Head pose', 'Face shape', 'Eyeglasses', 'Age', 'Gender']
        },
        'Middle (16-32px)': {
            'layers': [4, 5, 6, 7],
            'controls': ['Facial features', 'Expression', 'Hair style', 'Eyes open/closed']
        },
        'Fine (64-1024px)': {
            'layers': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'controls': ['Hair color', 'Skin tone', 'Lighting', 'Background', 'Texture details']
        }
    }
    
    for group, info in layer_groups.items():
        print(f"\n{group}:")
        print(f"  Layers: {info['layers']}")
        print(f"  Controls: {', '.join(info['controls'])}")
```

**Discovered Attribute-Layer Mapping (FFHQ):**

| Attribute | Primary Layers | Best Editing Range |
|-----------|---------------|-------------------|
| Pose | 0-3 | Coarse only |
| Age | 0-5 | Coarse + early middle |
| Gender | 0-4 | Coarse |
| Expression | 4-7 | Middle |
| Hair style | 3-7 | Coarse + middle |
| Hair color | 8-11 | Early fine |
| Background | 10-17 | Fine |
| Lighting | 8-17 | All fine |

**Interview Tip:** This coarse-middle-fine separation is what makes StyleGAN so powerful for editing. When inverting real images, W+ space (different W per layer) allows more faithful reconstruction than W space. Style mixing creates controllable "chimera" faces by combining attributes from different sources at specific resolution levels.

---

### Question 8
- [ ] Done
**What are StyleGAN2's key improvements: weight demodulation, path-length regularization, no progressive growing?**

**Answer:**

StyleGAN2 fixes StyleGAN1's artifacts (water droplets, phase artifacts) through: (1) weight demodulation instead of AdaIN to eliminate blob artifacts, (2) path-length regularization for smoother latent space, and (3) removing progressive growing for simpler training.

**Key Improvements:**

| Issue in StyleGAN1 | StyleGAN2 Solution | Effect |
|-------------------|-------------------|--------|
| Water droplet artifacts | Weight demodulation | Cleaner textures |
| Latent space irregularity | Path-length regularization | Smoother edits |
| Complex training | No progressive growing | Simpler pipeline |
| Feature map artifacts | Skip connections redesign | Better image quality |

**1. Weight Demodulation (replaces AdaIN):**

Instead of normalizing activations, normalize the convolution weights:

$$w'_{ijk} = \frac{w_{ijk}}{\sqrt{\sum_{i,k} (s_i \cdot w_{ijk})^2 + \epsilon}}$$

Where $s_i$ is the per-channel modulation from style.

**2. Path-Length Regularization:**

Encourages that small changes in W cause proportional changes in image:

$$\mathcal{L}_{PPL} = \mathbb{E}_{w,y}\left[\left(\|J_w^T y\|_2 - a\right)^2\right]$$

Where $J_w$ is the Jacobian and $a$ is the running average of path lengths.

**3. No Progressive Growing:**

- Use skip connections and residual connections instead
- Train at full resolution from the start
- More stable, fewer training stages

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ModulatedConv2d(nn.Module):
    """StyleGAN2 Modulated Convolution with Weight Demodulation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 w_dim, demodulate=True, upsample=False, downsample=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        
        # Convolution weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        # Initialize with He scaling
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        
        # Style to modulation
        self.modulation = nn.Linear(w_dim, in_channels)
        self.modulation.bias.data.fill_(1)  # Initialize to no modulation
    
    def forward(self, x, style):
        """
        x: [B, C_in, H, W]
        style: [B, w_dim]
        """
        B, C_in, H, W = x.shape
        
        # Step 1: Get modulation scales from style
        scales = self.modulation(style)  # [B, C_in]
        
        # Step 2: Modulate weights
        # weight: [C_out, C_in, K, K]
        # scales: [B, C_in]
        weight = self.weight.unsqueeze(0)  # [1, C_out, C_in, K, K]
        weight = weight * scales.view(B, 1, C_in, 1, 1)  # [B, C_out, C_in, K, K]
        
        # Step 3: Demodulate (normalize)
        if self.demodulate:
            # Compute normalization factor per output channel
            demod = torch.rsqrt(
                weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8
            )  # [B, C_out]
            weight = weight * demod.view(B, self.out_channels, 1, 1, 1)
        
        # Step 4: Apply convolution
        # Reshape for grouped convolution
        x = x.view(1, B * C_in, H, W)
        weight = weight.view(
            B * self.out_channels, C_in, self.kernel_size, self.kernel_size
        )
        
        if self.upsample:
            # Transposed convolution for upsampling
            weight = weight.view(B, self.out_channels, C_in, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2)  # [B, C_in, C_out, K, K]
            weight = weight.reshape(B * C_in, self.out_channels, self.kernel_size, self.kernel_size)
            
            out = F.conv_transpose2d(x, weight, stride=2, padding=self.kernel_size//2, 
                                     output_padding=1, groups=B)
            out = out.view(B, self.out_channels, H * 2, W * 2)
        elif self.downsample:
            out = F.conv2d(x, weight, stride=2, padding=self.kernel_size//2, groups=B)
            out = out.view(B, self.out_channels, H // 2, W // 2)
        else:
            out = F.conv2d(x, weight, padding=self.kernel_size//2, groups=B)
            out = out.view(B, self.out_channels, H, W)
        
        return out


class StyleGAN2Generator(nn.Module):
    """Simplified StyleGAN2 Generator"""
    
    def __init__(self, z_dim=512, w_dim=512, img_resolution=1024, img_channels=3):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, num_layers=8)
        
        # Synthesis network
        self.synthesis = StyleGAN2Synthesis(w_dim, img_resolution, img_channels)
    
    def forward(self, z, truncation_psi=1.0, truncation_cutoff=None):
        # Map z to w
        w = self.mapping(z)
        
        # Truncation
        if truncation_psi != 1.0:
            w_avg = self.mapping.w_avg
            if truncation_cutoff is None:
                w = w_avg + truncation_psi * (w - w_avg)
            else:
                w[:, :truncation_cutoff] = (
                    w_avg + truncation_psi * (w[:, :truncation_cutoff] - w_avg)
                )
        
        # Synthesis
        img = self.synthesis(w)
        
        return img


class StyleGAN2Synthesis(nn.Module):
    """StyleGAN2 Synthesis Network with residual connections"""
    
    def __init__(self, w_dim, img_resolution, img_channels):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.num_layers = int(np.log2(img_resolution)) * 2 - 2
        
        # Channel configs
        channels = {
            4: 512, 8: 512, 16: 512, 32: 512,
            64: 256, 128: 128, 256: 64, 512: 32, 1024: 16
        }
        
        # Constant input
        self.const = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        
        # Build layers
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        
        in_ch = channels[4]
        for res in [4, 8, 16, 32, 64, 128, 256, 512, 1024][:int(np.log2(img_resolution))]:
            out_ch = channels[res]
            
            # Two conv layers per resolution
            self.layers.append(
                StyleGAN2Block(in_ch, out_ch, w_dim, upsample=(res > 4))
            )
            
            # To RGB
            self.to_rgbs.append(ToRGB(out_ch, img_channels, w_dim))
            
            in_ch = out_ch
    
    def forward(self, w):
        x = self.const.repeat(w.shape[0], 1, 1, 1)
        
        rgb = None
        
        for i, (layer, to_rgb) in enumerate(zip(self.layers, self.to_rgbs)):
            x = layer(x, w[:, i * 2], w[:, i * 2 + 1] if i * 2 + 1 < w.shape[1] else w[:, i * 2])
            
            # Skip connections to RGB
            if rgb is None:
                rgb = to_rgb(x, w[:, i * 2])
            else:
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
                rgb = rgb + to_rgb(x, w[:, i * 2])
        
        return rgb


class PathLengthRegularization:
    """Path-Length Regularization for smoother W space"""
    
    def __init__(self, decay=0.99):
        self.mean_path_length = 0
        self.decay = decay
    
    def __call__(self, fake_imgs, w, mean_path_length=None):
        """
        fake_imgs: generated images [B, C, H, W]
        w: latent vectors [B, L, W_dim] (W+ space)
        """
        # Random noise for image perturbation direction
        noise = torch.randn_like(fake_imgs) / np.sqrt(fake_imgs.shape[2] * fake_imgs.shape[3])
        
        # Compute gradient of output w.r.t. input styles
        grad_outputs = noise
        
        gradients = torch.autograd.grad(
            outputs=fake_imgs,
            inputs=w,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Path length
        path_lengths = torch.sqrt(
            gradients.pow(2).sum(dim=[1, 2]).mean()
        )
        
        # Update running average
        if mean_path_length is None:
            mean_path_length = self.mean_path_length
        
        if self.mean_path_length == 0:
            self.mean_path_length = path_lengths.detach().mean()
        else:
            self.mean_path_length = (
                self.decay * self.mean_path_length + 
                (1 - self.decay) * path_lengths.detach().mean()
            )
        
        # Regularization loss
        path_penalty = (path_lengths - self.mean_path_length).pow(2).mean()
        
        return path_penalty, self.mean_path_length


def train_stylegan2(G, D, dataloader, device='cuda'):
    """Training loop with StyleGAN2 improvements"""
    
    path_length_reg = PathLengthRegularization()
    
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0, 0.99))
    
    r1_gamma = 10.0  # R1 regularization weight
    pl_weight = 2.0  # Path length weight
    
    for real_imgs in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        
        # --- Discriminator step ---
        z = torch.randn(batch_size, 512, device=device)
        fake_imgs = G(z).detach()
        
        real_pred = D(real_imgs)
        fake_pred = D(fake_imgs)
        
        # Logistic loss
        d_loss = F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()
        
        # R1 regularization (every 16 steps in practice)
        real_imgs.requires_grad_(True)
        real_pred = D(real_imgs)
        
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_imgs,
            create_graph=True
        )[0]
        
        r1_penalty = grad_real.pow(2).view(batch_size, -1).sum(dim=1).mean()
        d_loss = d_loss + r1_gamma / 2 * r1_penalty
        
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()
        
        # --- Generator step ---
        z = torch.randn(batch_size, 512, device=device)
        
        # Get W with gradients for path length
        w = G.mapping(z)
        w.requires_grad_(True)
        
        fake_imgs = G.synthesis(w)
        fake_pred = D(fake_imgs)
        
        g_loss = F.softplus(-fake_pred).mean()
        
        # Path length regularization (every 8 steps in practice)
        pl_penalty, _ = path_length_reg(fake_imgs, w)
        g_loss = g_loss + pl_weight * pl_penalty
        
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
```

**Performance Comparison:**

| Metric | StyleGAN1 | StyleGAN2 |
|--------|-----------|-----------|
| FID (FFHQ) | 4.40 | 2.84 |
| PPL (W) | 412 | 109 |
| Artifacts | Water droplets | Clean |
| Training | Progressive | Direct |

**Interview Tip:** Weight demodulation is the key architectural change—it bakes normalization into the convolution weights rather than normalizing activations. This prevents the per-sample normalization artifacts (blobs). Path-length regularization creates a smoother latent space where small changes in W produce proportional image changes—critical for editing applications.

---

### Question 9
- [ ] Done
**Explain StyleGAN3's alias-free design and how it eliminates texture sticking artifacts.**

**Answer:**

StyleGAN3 solves "texture sticking"—where fine details stay fixed to pixel coordinates during animation/interpolation instead of moving with the object. The solution is alias-free operations throughout the network, treating images as continuous signals rather than discrete pixels.

**The Texture Sticking Problem:**

In StyleGAN1/2, when interpolating in latent space:
- Coarse features (pose) change smoothly
- Fine textures (hair strands, wrinkles) "stick" to fixed screen positions
- Creates unnatural "sliding" effect in animations

**Root Cause:** Aliasing from discrete sampling operations (upsampling, nonlinearities) that don't respect Nyquist sampling theorem.

**StyleGAN3 Solutions:**

| Issue | Cause | StyleGAN3 Fix |
|-------|-------|---------------|
| Texture sticking | Position encoding leakage | Remove all positional references |
| Aliasing in upsample | Nearest/bilinear artifacts | Ideal low-pass filters |
| Nonlinearity aliasing | ReLU creates high frequencies | Filtered nonlinearities |
| Edge artifacts | Boundary effects | Proper margin handling |

**Key Design Principles:**

1. **Continuous Signal Representation**: Treat feature maps as band-limited continuous signals
2. **Equivariance**: Output should transform predictably with input transformations
3. **Nyquist-Compliant Operations**: All operations must respect bandwidth limits

**Mathematical Framework:**

For translation equivariance:
$$f(T_t(x)) = T_t(f(x))$$

Where $T_t$ is translation by $t$.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LowPassFilter(nn.Module):
    """Ideal low-pass filter for alias-free operations"""
    
    def __init__(self, cutoff, kernel_size=None, filter_type='kaiser'):
        super().__init__()
        
        self.cutoff = cutoff
        
        if kernel_size is None:
            kernel_size = int(np.ceil(6 / cutoff)) | 1  # Ensure odd
        
        self.kernel_size = kernel_size
        
        # Create filter kernel
        if filter_type == 'sinc':
            kernel = self._sinc_filter(kernel_size, cutoff)
        elif filter_type == 'kaiser':
            kernel = self._kaiser_filter(kernel_size, cutoff)
        else:
            kernel = self._gaussian_filter(kernel_size, cutoff)
        
        self.register_buffer('kernel', kernel)
    
    def _sinc_filter(self, size, cutoff):
        """Ideal sinc low-pass filter"""
        x = torch.arange(size) - size // 2
        kernel = torch.sinc(2 * cutoff * x)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, 1, -1)
    
    def _kaiser_filter(self, size, cutoff, beta=6):
        """Kaiser-windowed sinc filter"""
        x = torch.arange(size) - size // 2
        sinc = torch.sinc(2 * cutoff * x)
        
        # Kaiser window
        n = torch.arange(size)
        alpha = (size - 1) / 2
        window = torch.special.i0(beta * torch.sqrt(1 - ((n - alpha) / alpha) ** 2))
        window = window / torch.special.i0(torch.tensor(beta))
        
        kernel = sinc * window
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, 1, -1)
    
    def _gaussian_filter(self, size, cutoff):
        """Gaussian low-pass filter"""
        sigma = 0.5 / cutoff
        x = torch.arange(size) - size // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, 1, -1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Separable filtering (horizontal)
        x = x.view(B * C, 1, H, W)
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode='reflect')
        x = F.conv2d(x, self.kernel)
        
        # Vertical
        x = x.permute(0, 1, 3, 2)  # Swap H and W
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode='reflect')
        x = F.conv2d(x, self.kernel)
        x = x.permute(0, 1, 3, 2)
        
        x = x.view(B, C, H, W)
        return x


class FilteredNonlinearity(nn.Module):
    """Apply nonlinearity without introducing aliasing"""
    
    def __init__(self, in_cutoff, out_cutoff):
        super().__init__()
        
        # Upsample before nonlinearity
        self.upsample_factor = 2
        
        # Lowpass filters
        self.pre_filter = LowPassFilter(in_cutoff)
        self.post_filter = LowPassFilter(out_cutoff)
    
    def forward(self, x):
        # Upsample to higher resolution
        x = F.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
        
        # Apply nonlinearity at higher resolution (less aliasing)
        x = F.leaky_relu(x, 0.2)
        
        # Filter to remove high frequencies introduced by nonlinearity
        x = self.post_filter(x)
        
        # Downsample back
        x = F.interpolate(x, scale_factor=1/self.upsample_factor, mode='bilinear', align_corners=False)
        
        return x


class AliasFreeSynthesisLayer(nn.Module):
    """Alias-free synthesis layer (StyleGAN3 style)"""
    
    def __init__(self, in_channels, out_channels, w_dim, 
                 in_cutoff=1.0, out_cutoff=1.0, 
                 upsample=1, use_rotation=True):
        super().__init__()
        
        self.upsample = upsample
        self.use_rotation = use_rotation
        
        # Modulated convolution
        self.conv = ModulatedConv2d(in_channels, out_channels, 3, w_dim)
        
        # Alias-free components
        self.filter = LowPassFilter(min(in_cutoff, out_cutoff))
        
        # Affine transform parameters from style (for rotation equivariance)
        if use_rotation:
            self.affine = nn.Linear(w_dim, 6)  # 2D affine matrix params
            self.affine.weight.data.zero_()
            self.affine.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))  # Identity
        
        self.magnitude_ema = None
    
    def forward(self, x, w, noise=None):
        # Apply modulated convolution
        x = self.conv(x, w)
        
        # Upsample with filtering
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode='bilinear', align_corners=False)
            x = self.filter(x)
        
        # Filtered nonlinearity
        x = self.filter(F.leaky_relu(x, 0.2))
        
        return x


class FourierFeatures(nn.Module):
    """Fourier features for position-free coordinate encoding"""
    
    def __init__(self, num_channels, bandwidth=1.0):
        super().__init__()
        
        # Random frequencies (learned or fixed)
        self.register_buffer(
            'frequencies',
            torch.randn(num_channels // 2, 2) * bandwidth
        )
    
    def forward(self, coords):
        """
        coords: [B, 2] or [B, H, W, 2] normalized coordinates
        """
        # Project coordinates to frequency domain
        projected = coords @ self.frequencies.T * 2 * np.pi
        
        # Sine and cosine features
        features = torch.cat([
            torch.sin(projected),
            torch.cos(projected)
        ], dim=-1)
        
        return features


class StyleGAN3Generator(nn.Module):
    """Simplified StyleGAN3 Generator concept"""
    
    def __init__(self, z_dim=512, w_dim=512, img_resolution=1024):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        
        # Mapping network (same as StyleGAN2)
        self.mapping = MappingNetwork(z_dim, w_dim)
        
        # Fourier features as input (instead of learned constant)
        self.input = FourierFeatures(512)
        
        # Synthesis with alias-free layers
        self.synthesis_layers = nn.ModuleList()
        
        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        channels = [512, 512, 512, 256, 128, 64, 32, 16, 8]
        cutoffs = [0.5, 0.5, 0.5, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        
        for i, (res, ch, cutoff) in enumerate(zip(resolutions, channels, cutoffs)):
            if res <= img_resolution:
                upsample = 2 if i > 0 else 1
                in_ch = channels[i-1] if i > 0 else 512
                
                self.synthesis_layers.append(
                    AliasFreeSynthesisLayer(in_ch, ch, w_dim, 
                                           in_cutoff=cutoffs[max(0, i-1)],
                                           out_cutoff=cutoff,
                                           upsample=upsample)
                )
        
        # To RGB
        self.to_rgb = nn.Conv2d(channels[-1], 3, 1)
    
    def forward(self, z, truncation_psi=1.0):
        # Map to W
        w = self.mapping(z)
        
        # Generate coordinate grid
        B = z.shape[0]
        coords = self._make_coords(4, 4, z.device)  # Start at 4x4
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Initial features from Fourier
        x = self.input(coords)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Synthesis
        for i, layer in enumerate(self.synthesis_layers):
            x = layer(x, w)
        
        # To RGB
        img = self.to_rgb(x)
        
        return img
    
    def _make_coords(self, H, W, device):
        """Create normalized coordinate grid"""
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        coords = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
        return coords


def visualize_texture_sticking():
    """Demonstrate texture sticking vs alias-free"""
    print("StyleGAN2 (with sticking):")
    print("  - Interpolate pose: hair strands stay in fixed positions")
    print("  - Unnatural 'sliding' effect")
    print("")
    print("StyleGAN3 (alias-free):")
    print("  - Interpolate pose: hair moves naturally with head")
    print("  - Smooth, consistent animations")
```

**StyleGAN3 Configurations:**

| Variant | Focus | Use Case |
|---------|-------|----------|
| StyleGAN3-T | Translation equivariance | General images |
| StyleGAN3-R | Rotation + translation | Faces (natural rotations) |

**Comparison:**

| Aspect | StyleGAN2 | StyleGAN3 |
|--------|-----------|-----------|
| FID | 2.84 | 2.79 |
| Animation quality | Texture sticking | Smooth |
| Computation | Baseline | ~1.5x slower |
| Equivariance | None | Translation/Rotation |

**Interview Tip:** StyleGAN3's contribution is primarily for animation/video applications. For static image generation, StyleGAN2 may be preferred due to simpler architecture. The key insight is treating neural networks as signal processing systems—respecting Nyquist limits prevents aliasing artifacts that cause texture sticking.

---

### Question 10
- [ ] Done
**Describe the truncation trick in StyleGAN. How does ψ parameter trade diversity for quality?**

**Answer:**

The truncation trick moves latent vectors toward the mean of W space during inference. Lower ψ values produce higher quality but less diverse images; ψ=1 uses the full distribution. This trades exploration (diversity) for exploitation (quality).

**Formula:**

$$w' = \bar{w} + \psi (w - \bar{w})$$

Where:
- $\bar{w}$ = mean W vector (computed over many samples)
- $\psi$ = truncation parameter (0 to 1+)
- $w$ = original latent vector

**Effect of ψ:**

| ψ Value | Effect | Use Case |
|---------|--------|----------|
| 0 | All images = average face | Never used |
| 0.5 | High quality, low diversity | Quality demos |
| 0.7 | Good balance | Common default |
| 1.0 | Full diversity, some artifacts | Training metrics |
| >1.0 | Extra diverse, more artifacts | Exploration |

**Why It Works:**

The W space has a complex learned distribution. Extreme latent vectors (far from mean) often map to unusual or low-quality images. Truncating toward the mean keeps samples in the "high-quality" region.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np

class TruncatedStyleGAN:
    """StyleGAN with truncation trick"""
    
    def __init__(self, generator, device='cuda'):
        self.G = generator.to(device)
        self.device = device
        self.w_avg = None
    
    def compute_w_avg(self, num_samples=10000):
        """Compute mean W vector for truncation"""
        self.G.eval()
        
        ws = []
        batch_size = 100
        
        with torch.no_grad():
            for _ in range(num_samples // batch_size):
                z = torch.randn(batch_size, self.G.z_dim, device=self.device)
                w = self.G.mapping(z)
                ws.append(w)
        
        ws = torch.cat(ws, dim=0)
        self.w_avg = ws.mean(dim=0, keepdim=True)
        
        return self.w_avg
    
    def generate(self, z, truncation_psi=0.7, truncation_cutoff=None):
        """Generate with truncation"""
        
        if self.w_avg is None:
            self.compute_w_avg()
        
        with torch.no_grad():
            # Get W
            w = self.G.mapping(z)
            
            # Apply truncation
            if truncation_psi != 1.0:
                if truncation_cutoff is None:
                    # Apply to all layers
                    w = self.w_avg + truncation_psi * (w - self.w_avg)
                else:
                    # Apply only to first N layers (coarse features)
                    w[:, :truncation_cutoff] = (
                        self.w_avg[:, :truncation_cutoff] + 
                        truncation_psi * (w[:, :truncation_cutoff] - self.w_avg[:, :truncation_cutoff])
                    )
            
            # Generate image
            img = self.G.synthesis(w)
        
        return img
    
    def interpolate_truncation(self, z, psi_values=[0.0, 0.3, 0.5, 0.7, 1.0]):
        """Show same latent at different truncation levels"""
        
        images = []
        for psi in psi_values:
            img = self.generate(z, truncation_psi=psi)
            images.append(img)
        
        return images


class AdaptiveTruncation:
    """More sophisticated truncation strategies"""
    
    def __init__(self, generator, device='cuda'):
        self.G = generator
        self.device = device
        self.w_avg = None
        self.w_std = None
    
    def compute_statistics(self, num_samples=50000):
        """Compute W distribution statistics"""
        ws = []
        batch_size = 100
        
        with torch.no_grad():
            for _ in range(num_samples // batch_size):
                z = torch.randn(batch_size, 512, device=self.device)
                w = self.G.mapping(z)
                ws.append(w)
        
        ws = torch.cat(ws, dim=0)
        self.w_avg = ws.mean(dim=0)
        self.w_std = ws.std(dim=0)
        
        return self.w_avg, self.w_std
    
    def layer_adaptive_truncation(self, w, psi_coarse=0.7, psi_fine=1.0, cutoff=8):
        """Different truncation for different layers"""
        w_truncated = w.clone()
        
        # Stronger truncation for coarse layers (more impact on quality)
        w_truncated[:, :cutoff] = (
            self.w_avg[:cutoff] + 
            psi_coarse * (w[:, :cutoff] - self.w_avg[:cutoff])
        )
        
        # Weaker truncation for fine layers (preserve detail diversity)
        w_truncated[:, cutoff:] = (
            self.w_avg[cutoff:] + 
            psi_fine * (w[:, cutoff:] - self.w_avg[cutoff:])
        )
        
        return w_truncated
    
    def distance_adaptive_truncation(self, w, max_distance=2.0):
        """Truncate samples that are too far from mean"""
        distance = (w - self.w_avg).norm(dim=-1) / self.w_std.norm()
        
        # Compute adaptive psi based on distance
        psi = torch.clamp(max_distance / distance, max=1.0)
        psi = psi.unsqueeze(-1)
        
        w_truncated = self.w_avg + psi * (w - self.w_avg)
        
        return w_truncated
    
    def quality_aware_truncation(self, w, discriminator, target_score=0.5):
        """Truncate until discriminator score reaches target"""
        # Binary search for optimal psi
        psi_low, psi_high = 0.0, 1.0
        
        for _ in range(10):  # Binary search iterations
            psi_mid = (psi_low + psi_high) / 2
            w_trunc = self.w_avg + psi_mid * (w - self.w_avg)
            
            img = self.G.synthesis(w_trunc)
            score = discriminator(img).mean().item()
            
            if score < target_score:
                psi_high = psi_mid  # Need more truncation
            else:
                psi_low = psi_mid  # Can afford less truncation
        
        return self.w_avg + psi_low * (w - self.w_avg)


def demonstrate_truncation_tradeoff():
    """Visualize quality-diversity tradeoff"""
    
    print("Truncation Trick Effects:")
    print("=" * 50)
    print(f"{'ψ':^6} | {'Quality':^12} | {'Diversity':^12} | {'Use Case':^15}")
    print("-" * 50)
    print(f"{'0.0':^6} | {'Maximum':^12} | {'None':^12} | {'Average face':^15}")
    print(f"{'0.5':^6} | {'Very High':^12} | {'Low':^12} | {'Demos':^15}")
    print(f"{'0.7':^6} | {'High':^12} | {'Medium':^12} | {'General use':^15}")
    print(f"{'1.0':^6} | {'Variable':^12} | {'Maximum':^12} | {'FID eval':^15}")
    print(f"{'>1.0':^6} | {'Low':^12} | {'Very High':^12} | {'Exploration':^15}")


def fid_vs_truncation_analysis():
    """Analyze FID at different truncation levels"""
    
    # Typical FID values for StyleGAN2 FFHQ
    truncation_fid = {
        0.5: 4.5,   # Slightly worse (less diversity)
        0.7: 3.5,   # Often optimal for FID
        1.0: 2.84,  # Best FID (full distribution)
        1.1: 3.2,   # Slight degradation
        1.5: 5.0    # Notable degradation
    }
    
    print("\nFID vs Truncation (StyleGAN2 FFHQ):")
    for psi, fid in truncation_fid.items():
        print(f"  ψ={psi}: FID = {fid}")
```

**Truncation Cutoff:**

Apply truncation only to first N layers (coarse features):
- Coarse layers: truncate strongly (ψ=0.5-0.7)
- Fine layers: keep full diversity (ψ=1.0)

This preserves texture variation while controlling structural quality.

**FID Considerations:**

| Metric Goal | Recommended ψ |
|-------------|---------------|
| Best FID | 1.0 (full distribution) |
| Best perceptual quality | 0.7-0.8 |
| Marketing/demos | 0.5-0.7 |
| Diversity studies | 1.0+ |

**Interview Tip:** Always report ψ when presenting StyleGAN results—it significantly affects both FID and visual quality. FID should be computed at ψ=1.0 for fair comparison. The truncation trick exploits that the W distribution has a "sweet spot" near the mean where most high-quality images lie. It's essentially trading the tails of the distribution for concentrated quality.

---

### Question 11
- [ ] Done
**Explain StyleGAN latent spaces: Z, W, W+, and S space. Which is best for image editing?**

**Answer:**

StyleGAN has multiple latent spaces with increasing expressiveness: Z (initial random), W (after mapping network), W+ (per-layer W), and S (post-affine). For editing, W+ offers the best balance of editability and reconstruction quality.

**Latent Space Hierarchy:**

| Space | Dimensionality | Description | Disentanglement |
|-------|---------------|-------------|-----------------|
| **Z** | 512 | Random Gaussian input | Low |
| **W** | 512 | Single vector after mapping | High |
| **W+** | 18×512 = 9216 | Different W per layer | Medium |
| **S** | ~9000+ | Per-channel style codes | Low |

**Comparison:**

| Space | Reconstruction | Editability | Use Case |
|-------|---------------|-------------|----------|
| Z | Poor | Poor | Generation only |
| W | Moderate | Excellent | Semantic editing |
| W+ | Excellent | Good | Real image editing |
| S | Perfect | Difficult | Fine-grained control |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentSpaces:
    """Work with different StyleGAN latent spaces"""
    
    def __init__(self, generator, num_layers=18):
        self.G = generator
        self.num_layers = num_layers
        self.w_dim = 512
    
    def z_to_w(self, z):
        """Map Z to W space"""
        return self.G.mapping(z)  # [B, 512]
    
    def w_to_w_plus(self, w):
        """Broadcast W to W+ (same W for all layers)"""
        return w.unsqueeze(1).repeat(1, self.num_layers, 1)  # [B, 18, 512]
    
    def w_plus_to_s(self, w_plus, synthesis_network):
        """
        Convert W+ to S space (style codes after affine transform)
        S space has per-layer, per-channel style parameters
        """
        s_codes = []
        
        for layer_idx, layer in enumerate(synthesis_network.layers):
            w = w_plus[:, layer_idx]  # [B, 512]
            
            # Get style after affine transform (gamma, beta)
            style = layer.affine(w)  # [B, num_features]
            s_codes.append(style)
        
        return s_codes  # List of [B, C] for each layer
    
    def generate_from_z(self, z):
        """Standard generation from Z"""
        return self.G(z)
    
    def generate_from_w(self, w):
        """Generate from W space"""
        w_plus = self.w_to_w_plus(w)
        return self.G.synthesis(w_plus)
    
    def generate_from_w_plus(self, w_plus):
        """Generate from W+ space"""
        return self.G.synthesis(w_plus)


class GANInversion:
    """Invert real images to latent space"""
    
    def __init__(self, generator, device='cuda'):
        self.G = generator.to(device)
        self.device = device
        
        # VGG for perceptual loss
        from torchvision import models
        vgg = models.vgg16(pretrained=True).features[:16].to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
    
    def optimize_w(self, target_image, num_steps=1000, lr=0.1):
        """Optimize in W space"""
        # Initialize at mean
        w = torch.zeros(1, 512, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([w], lr=lr)
        
        for step in range(num_steps):
            w_plus = w.unsqueeze(1).repeat(1, 18, 1)
            generated = self.G.synthesis(w_plus)
            
            loss = self._reconstruction_loss(generated, target_image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return w.detach()
    
    def optimize_w_plus(self, target_image, num_steps=1000, lr=0.1):
        """Optimize in W+ space (better reconstruction)"""
        # Initialize each layer independently
        w_plus = torch.zeros(1, 18, 512, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([w_plus], lr=lr)
        
        for step in range(num_steps):
            generated = self.G.synthesis(w_plus)
            
            # Combined loss
            loss = self._reconstruction_loss(generated, target_image)
            
            # W+ regularization (keep close to W manifold)
            w_mean = w_plus.mean(dim=1, keepdim=True)
            reg_loss = (w_plus - w_mean).pow(2).mean()
            
            total_loss = loss + 0.01 * reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return w_plus.detach()
    
    def _reconstruction_loss(self, generated, target):
        """Combined pixel + perceptual loss"""
        # Pixel loss
        pixel_loss = F.mse_loss(generated, target)
        
        # Perceptual loss
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)
        perceptual_loss = F.mse_loss(gen_features, target_features)
        
        return pixel_loss + 0.1 * perceptual_loss


class LatentSpaceEditor:
    """Edit images in different latent spaces"""
    
    def __init__(self, generator):
        self.G = generator
    
    def edit_in_w(self, w, direction, strength):
        """
        W space editing: highly disentangled
        Direction affects all layers equally
        """
        w_edited = w + strength * direction
        w_plus = w_edited.unsqueeze(1).repeat(1, 18, 1)
        return self.G.synthesis(w_plus)
    
    def edit_in_w_plus(self, w_plus, direction, strength, layers=None):
        """
        W+ space editing: more control, can target specific layers
        """
        w_edited = w_plus.clone()
        
        if layers is None:
            layers = list(range(18))
        
        for layer_idx in layers:
            w_edited[:, layer_idx] += strength * direction
        
        return self.G.synthesis(w_edited)
    
    def layer_swap(self, w_plus_source, w_plus_target, swap_layers):
        """Swap specific layers between two latent codes"""
        w_result = w_plus_source.clone()
        w_result[:, swap_layers] = w_plus_target[:, swap_layers]
        return self.G.synthesis(w_result)


class EncoderTypes:
    """Different encoder architectures for GAN inversion"""
    
    @staticmethod
    def psp_encoder():
        """
        pSp: Encodes directly to W+ space
        - Fast (single forward pass)
        - Good reconstruction
        - May lose some editability
        """
        pass
    
    @staticmethod
    def e4e_encoder():
        """
        e4e: Encoder for Editing
        - Encodes closer to W space
        - Better editability than pSp
        - Slightly worse reconstruction
        """
        pass
    
    @staticmethod
    def restyle_encoder():
        """
        ReStyle: Iterative refinement
        - Multiple forward passes
        - Best reconstruction
        - Can control quality/speed tradeoff
        """
        pass


def compare_latent_spaces():
    """Summary of latent space properties"""
    
    print("StyleGAN Latent Spaces Comparison:")
    print("=" * 70)
    print(f"{'Space':^6} | {'Dim':^8} | {'Reconstruct':^12} | {'Edit':^10} | {'Best For':^20}")
    print("-" * 70)
    print(f"{'Z':^6} | {'512':^8} | {'Poor':^12} | {'Poor':^10} | {'Generation':^20}")
    print(f"{'W':^6} | {'512':^8} | {'Moderate':^12} | {'Excellent':^10} | {'Semantic editing':^20}")
    print(f"{'W+':^6} | {'9216':^8} | {'Excellent':^12} | {'Good':^10} | {'Real image editing':^20}")
    print(f"{'S':^6} | {'~9000':^8} | {'Perfect':^12} | {'Difficult':^10} | {'Fine control':^20}")
    
    print("\n\nRecommendations:")
    print("- For random generation: Z or W")
    print("- For editing synthetic images: W")
    print("- For editing real photos: W+ (via encoder or optimization)")
    print("- For maximum reconstruction: S (but loses editability)")


# Practical example
def real_image_editing_pipeline(image, generator, edit_direction, strength):
    """Complete pipeline for editing real images"""
    
    # 1. Invert to W+ (good balance)
    inverter = GANInversion(generator)
    w_plus = inverter.optimize_w_plus(image)
    
    # 2. Edit in W+ space
    editor = LatentSpaceEditor(generator)
    
    # Apply edit to relevant layers only
    coarse_layers = list(range(0, 4))
    middle_layers = list(range(4, 8))
    
    edited_image = editor.edit_in_w_plus(
        w_plus, edit_direction, strength,
        layers=coarse_layers + middle_layers  # Skip fine layers
    )
    
    return edited_image
```

**Which Space for Editing?**

| Task | Recommended Space | Reason |
|------|-------------------|--------|
| Semantic editing (age, smile) | W | Most disentangled |
| Real photo editing | W+ | Best reconstruction |
| Style transfer | W+ with layer targeting | Control coarse/fine |
| Perfect reconstruction | S or optimization | Maximum expressiveness |
| Fast inference | W (via e4e) | Single forward pass |

**Interview Tip:** W space is the most disentangled (best for editing), but W+ allows better reconstruction of real images. The key insight is the editability-reconstruction tradeoff: more expressive spaces (W+ > W) fit images better but may lose the semantic structure that makes editing work. e4e encoder specifically optimizes for this tradeoff.

---

### Question 12
- [ ] Done
**How do GAN inversion methods (e4e, pSp, optimization-based) project real images to latent space?**

**Answer:**

GAN inversion finds a latent code that reconstructs a given real image when passed through the generator. Methods include: optimization (slow, accurate), pSp encoder (fast, W+), and e4e (fast, more editable). Each trades off speed, reconstruction quality, and editability.

**Method Comparison:**

| Method | Speed | Reconstruction | Editability | Approach |
|--------|-------|---------------|-------------|----------|
| **Optimization** | Slow (minutes) | Excellent | Variable | Iterative gradient descent |
| **pSp** | Fast (ms) | Good | Moderate | Direct encoder to W+ |
| **e4e** | Fast (ms) | Good | Better | Encoder with edit-friendly constraints |
| **ReStyle** | Medium (sec) | Excellent | Good | Iterative encoder refinement |
| **HyperStyle** | Fast | Excellent | Good | Hypernetwork-based |

**The Inversion Problem:**

Given image $x^*$, find $w$ such that:
$$w^* = \arg\min_w \mathcal{L}(G(w), x^*) + \lambda R(w)$$

Where $\mathcal{L}$ is reconstruction loss and $R$ is regularization.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class OptimizationInversion:
    """Optimization-based GAN inversion (most flexible)"""
    
    def __init__(self, generator, device='cuda'):
        self.G = generator.to(device).eval()
        self.device = device
        
        # Perceptual loss network
        vgg = models.vgg16(pretrained=True).features.to(device)
        self.vgg_layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23] # relu4_3
        ])
        for p in vgg.parameters():
            p.requires_grad = False
    
    def invert(self, target, space='w+', num_steps=1000, lr=0.1):
        """
        Invert image to latent space
        
        Args:
            target: [1, 3, H, W] normalized to [-1, 1]
            space: 'w' or 'w+'
        """
        if space == 'w':
            latent = torch.zeros(1, 512, device=self.device, requires_grad=True)
        else:  # w+
            # Initialize from random z
            z = torch.randn(1, 512, device=self.device)
            with torch.no_grad():
                w = self.G.mapping(z)
            latent = w.unsqueeze(1).repeat(1, 18, 1).requires_grad_(True)
        
        optimizer = torch.optim.Adam([latent], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 300, 0.5)
        
        best_loss = float('inf')
        best_latent = None
        
        for step in range(num_steps):
            # Generate
            if space == 'w':
                w_plus = latent.unsqueeze(1).repeat(1, 18, 1)
            else:
                w_plus = latent
            
            generated = self.G.synthesis(w_plus)
            
            # Losses
            pixel_loss = F.mse_loss(generated, target)
            perceptual_loss = self._perceptual_loss(generated, target)
            
            # W+ regularization (stay close to W manifold)
            if space == 'w+':
                w_mean = latent.mean(dim=1, keepdim=True)
                reg_loss = (latent - w_mean).pow(2).mean()
            else:
                reg_loss = 0
            
            loss = pixel_loss + 0.8 * perceptual_loss + 0.01 * reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_latent = latent.clone().detach()
        
        return best_latent
    
    def _perceptual_loss(self, generated, target):
        """Multi-scale perceptual loss"""
        loss = 0
        
        x = (generated + 1) / 2  # [-1,1] to [0,1]
        y = (target + 1) / 2
        
        for layer in self.vgg_layers:
            x = layer(x)
            y = layer(y)
            loss += F.mse_loss(x, y)
        
        return loss


class pSpEncoder(nn.Module):
    """
    pSp: Pixel2Style2Pixel Encoder
    Directly encodes image to W+ space
    """
    
    def __init__(self, num_layers=18):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Backbone (typically ResNet or similar)
        from torchvision.models import resnet50
        backbone = resnet50(pretrained=True)
        
        # Remove FC layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Feature pyramid for multi-scale W prediction
        self.coarse_pool = nn.AdaptiveAvgPool2d(1)
        self.medium_pool = nn.AdaptiveAvgPool2d(4)
        self.fine_pool = nn.AdaptiveAvgPool2d(16)
        
        # Map2style: convert features to W vectors
        # Coarse (first 4 layers)
        self.coarse_mappers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512)
            ) for _ in range(4)
        ])
        
        # Medium (layers 4-8)
        self.medium_mappers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(512 * 16, 512)
            ) for _ in range(4)
        ])
        
        # Fine (layers 8-18)
        self.fine_mappers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 256, 1),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(256 * 256, 512)
            ) for _ in range(10)
        ])
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # [B, 2048, H/32, W/32]
        
        w_plus = []
        
        # Coarse styles from global features
        coarse_feat = self.coarse_pool(features).flatten(1)  # [B, 2048]
        for mapper in self.coarse_mappers:
            w_plus.append(mapper(coarse_feat))
        
        # Medium styles from 4x4 features
        medium_feat = self.medium_pool(features)  # [B, 2048, 4, 4]
        for mapper in self.medium_mappers:
            w_plus.append(mapper(medium_feat))
        
        # Fine styles from 16x16 features
        fine_feat = self.fine_pool(features)  # [B, 2048, 16, 16]
        for mapper in self.fine_mappers:
            w_plus.append(mapper(fine_feat))
        
        # Stack to W+
        w_plus = torch.stack(w_plus, dim=1)  # [B, 18, 512]
        
        return w_plus


class e4eEncoder(nn.Module):
    """
    e4e: Encoder for Editing
    Similar to pSp but with constraints for better editability
    """
    
    def __init__(self, num_layers=18, w_avg=None):
        super().__init__()
        
        self.num_layers = num_layers
        self.register_buffer('w_avg', w_avg if w_avg is not None else torch.zeros(512))
        
        # Same backbone as pSp
        from torchvision.models import resnet50
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # But predict OFFSETS from w_avg instead of absolute W
        self.offset_predictors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512)
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        
        offsets = []
        for predictor in self.offset_predictors:
            offset = predictor(features)
            offsets.append(offset)
        
        offsets = torch.stack(offsets, dim=1)  # [B, 18, 512]
        
        # W+ = w_avg + small offset
        w_plus = self.w_avg.unsqueeze(0).unsqueeze(0) + offsets
        
        return w_plus
    
    def get_loss(self, generated, target, w_plus):
        """e4e loss includes editability constraint"""
        
        # Reconstruction losses
        pixel_loss = F.mse_loss(generated, target)
        # + perceptual loss, identity loss, etc.
        
        # Delta regularization (keep W+ close to W manifold)
        w_mean = w_plus.mean(dim=1, keepdim=True)
        delta_loss = (w_plus - w_mean).norm(dim=2).mean()
        
        # Progressive training: allow more deviation gradually
        
        return pixel_loss + 0.1 * delta_loss


class ReStyleEncoder(nn.Module):
    """
    ReStyle: Iterative refinement encoder
    Multiple forward passes for better reconstruction
    """
    
    def __init__(self, generator, base_encoder):
        super().__init__()
        
        self.G = generator
        self.encoder = base_encoder
        self.num_iterations = 5
    
    def forward(self, target, num_iters=None):
        if num_iters is None:
            num_iters = self.num_iterations
        
        # Start from w_avg
        w_plus = self.encoder.w_avg.unsqueeze(0).unsqueeze(0).expand(
            target.size(0), 18, 512
        ).clone()
        
        for i in range(num_iters):
            # Generate current reconstruction
            with torch.no_grad():
                current_img = self.G.synthesis(w_plus)
            
            # Encoder predicts residual from (target, current)
            combined = torch.cat([target, current_img], dim=1)
            delta = self.encoder(combined)
            
            # Update W+
            w_plus = w_plus + delta
        
        return w_plus


def compare_inversion_methods():
    """Summarize method tradeoffs"""
    
    print("GAN Inversion Methods Comparison:")
    print("=" * 80)
    print(f"{'Method':^15} | {'Time':^10} | {'MSE':^8} | {'LPIPS':^8} | {'Edit':^8}")
    print("-" * 80)
    print(f"{'Optimization':^15} | {'~5 min':^10} | {'0.01':^8} | {'0.05':^8} | {'★★★':^8}")
    print(f"{'pSp':^15} | {'~40 ms':^10} | {'0.03':^8} | {'0.12':^8} | {'★★☆':^8}")
    print(f"{'e4e':^15} | {'~40 ms':^10} | {'0.04':^8} | {'0.14':^8} | {'★★★':^8}")
    print(f"{'ReStyle':^15} | {'~200 ms':^10} | {'0.02':^8} | {'0.08':^8} | {'★★★':^8}")
    print(f"{'HyperStyle':^15} | {'~100 ms':^10} | {'0.02':^8} | {'0.07':^8} | {'★★☆':^8}")
```

**Key Design Decisions:**

| Aspect | pSp | e4e |
|--------|-----|-----|
| Target space | W+ | W+ (closer to W) |
| Prediction | Absolute W | Offset from w_avg |
| Regularization | Minimal | Strong delta reg |
| Editability | Moderate | Better |
| Reconstruction | Better | Slightly worse |

**Interview Tip:** The core tradeoff is reconstruction vs. editability. pSp produces latent codes that may be "out of distribution" for the W+ manifold—they reconstruct well but edits may fail. e4e constrains predictions to stay closer to the W manifold (via delta regularization), sacrificing some reconstruction quality for better semantic editing. For production, ReStyle or HyperStyle offer the best balance.

---

### Question 13
- [ ] Done
**Explain semantic editing in StyleGAN latent space (InterFaceGAN, GANSpace, StyleCLIP).**

**Answer:**

Semantic editing finds directions in latent space that correspond to meaningful attributes (age, smile, pose). InterFaceGAN uses labeled data with linear SVM; GANSpace finds unsupervised directions via PCA; StyleCLIP uses CLIP to find text-guided directions.

**Method Comparison:**

| Method | Supervision | Approach | Flexibility |
|--------|-------------|----------|-------------|
| **InterFaceGAN** | Labels needed | Linear SVM boundaries | Fixed attributes |
| **GANSpace** | Unsupervised | PCA on W space | Discover latent factors |
| **StyleCLIP** | Text prompts | CLIP-guided optimization | Open vocabulary |
| **SeFa** | Unsupervised | Eigenvectors of G weights | Layer-specific |

**Core Idea:**

For a learned direction $n$:
$$w_{edited} = w + \alpha \cdot n$$

Where $\alpha$ controls edit strength (positive/negative).

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.svm import LinearSVC

class InterFaceGAN:
    """Find semantic directions using labeled data"""
    
    def __init__(self, generator, device='cuda'):
        self.G = generator.to(device)
        self.device = device
        self.directions = {}
    
    def train_boundary(self, attribute_name, positive_ws, negative_ws):
        """
        Train linear SVM to find attribute direction
        
        Args:
            positive_ws: W vectors with attribute (e.g., smiling)
            negative_ws: W vectors without attribute
        """
        # Prepare data
        X = torch.cat([positive_ws, negative_ws], dim=0).cpu().numpy()
        y = np.concatenate([
            np.ones(len(positive_ws)),
            np.zeros(len(negative_ws))
        ])
        
        # Train SVM
        svm = LinearSVC(C=1.0, max_iter=10000)
        svm.fit(X, y)
        
        # Normal vector to decision boundary = edit direction
        direction = svm.coef_[0]
        direction = direction / np.linalg.norm(direction)
        
        self.directions[attribute_name] = torch.tensor(
            direction, dtype=torch.float32, device=self.device
        )
        
        return self.directions[attribute_name]
    
    def edit(self, w, attribute_name, strength):
        """Edit W vector along attribute direction"""
        direction = self.directions[attribute_name]
        w_edited = w + strength * direction
        return w_edited
    
    def conditional_edit(self, w, attribute_name, strength, preserve=None):
        """
        Edit one attribute while preserving others
        Uses orthogonalization
        """
        direction = self.directions[attribute_name].clone()
        
        if preserve is not None:
            for attr in preserve:
                preserve_dir = self.directions[attr]
                # Orthogonalize: remove component along preserved direction
                direction = direction - (direction @ preserve_dir) * preserve_dir
            direction = direction / direction.norm()
        
        return w + strength * direction


class GANSpace:
    """Discover semantic directions via PCA (unsupervised)"""
    
    def __init__(self, generator, device='cuda'):
        self.G = generator.to(device)
        self.device = device
        self.components = None
        self.explained_variance = None
    
    def compute_components(self, num_samples=10000, layer_idx=None):
        """
        Find principal components in W or intermediate feature space
        
        layer_idx: If specified, analyze features at that layer
        """
        samples = []
        batch_size = 100
        
        self.G.eval()
        with torch.no_grad():
            for _ in range(num_samples // batch_size):
                z = torch.randn(batch_size, 512, device=self.device)
                
                if layer_idx is None:
                    # Analyze W space
                    w = self.G.mapping(z)
                    samples.append(w.cpu())
                else:
                    # Analyze intermediate features (for layer-specific edits)
                    w = self.G.mapping(z)
                    # Would need to hook into synthesis layers
                    samples.append(w.cpu())
        
        samples = torch.cat(samples, dim=0).numpy()
        
        # Center and compute PCA
        mean = samples.mean(axis=0)
        centered = samples - mean
        
        # SVD
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        
        self.components = torch.tensor(Vh, dtype=torch.float32, device=self.device)
        self.explained_variance = S ** 2 / (num_samples - 1)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        
        return self.components, self.explained_variance
    
    def edit(self, w, component_idx, strength):
        """Edit along principal component direction"""
        direction = self.components[component_idx]
        return w + strength * direction
    
    def explore_components(self, w, num_components=20, strength=3.0):
        """Generate images varying each component"""
        images = []
        
        for i in range(num_components):
            direction = self.components[i]
            
            # Negative, neutral, positive
            for s in [-strength, 0, strength]:
                w_edited = w + s * direction
                w_plus = w_edited.unsqueeze(1).repeat(1, 18, 1)
                
                with torch.no_grad():
                    img = self.G.synthesis(w_plus)
                images.append((i, s, img))
        
        return images


class StyleCLIP:
    """Text-guided image editing using CLIP"""
    
    def __init__(self, generator, clip_model, device='cuda'):
        self.G = generator.to(device).eval()
        self.clip = clip_model.to(device).eval()
        self.device = device
    
    def find_direction_optimization(self, text_prompt, neutral_prompt="a face",
                                     num_samples=100, num_steps=300):
        """
        Find direction that maximizes CLIP similarity to text
        
        Method: Global Direction (optimize direction in W space)
        """
        # Encode text targets
        import clip
        text_target = clip.tokenize([text_prompt]).to(self.device)
        text_neutral = clip.tokenize([neutral_prompt]).to(self.device)
        
        with torch.no_grad():
            target_features = self.clip.encode_text(text_target)
            neutral_features = self.clip.encode_text(text_neutral)
        
        # Direction to optimize
        direction = torch.randn(512, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([direction], lr=0.1)
        
        for step in range(num_steps):
            # Sample random W vectors
            z = torch.randn(num_samples, 512, device=self.device)
            with torch.no_grad():
                w = self.G.mapping(z)
            
            # Edit
            w_edited = w + direction.unsqueeze(0)
            
            # Generate and get CLIP features
            w_plus = w_edited.unsqueeze(1).repeat(1, 18, 1)
            with torch.no_grad():
                images = self.G.synthesis(w_plus)
            
            # Resize for CLIP
            images_clip = F.interpolate(images, size=(224, 224), mode='bilinear')
            images_clip = (images_clip + 1) / 2  # [-1,1] to [0,1]
            
            image_features = self.clip.encode_image(images_clip)
            
            # Cosine similarity
            target_sim = F.cosine_similarity(image_features, target_features)
            neutral_sim = F.cosine_similarity(image_features, neutral_features)
            
            # Loss: maximize target sim, minimize neutral sim change
            loss = -target_sim.mean() + 0.5 * (neutral_sim.mean() - 1).abs()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize direction
            with torch.no_grad():
                direction.data = direction.data / direction.data.norm() * 5
        
        return direction.detach()
    
    def text_guided_edit(self, w, text_prompt, strength=1.0, method='latent_optimization'):
        """Edit image based on text prompt"""
        
        if method == 'latent_optimization':
            # Optimize this specific W to match text
            return self._optimize_single(w, text_prompt, strength)
        elif method == 'mapper':
            # Use trained text-conditioned mapper
            return self._mapper_edit(w, text_prompt, strength)
        elif method == 'global':
            # Use pre-computed global direction
            direction = self.find_direction_optimization(text_prompt)
            return w + strength * direction
    
    def _optimize_single(self, w, text_prompt, strength):
        """Optimize single image to match text"""
        import clip
        
        w_edit = w.clone().requires_grad_(True)
        
        text = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(text)
        
        optimizer = torch.optim.Adam([w_edit], lr=0.1)
        
        for _ in range(100):
            w_plus = w_edit.unsqueeze(1).repeat(1, 18, 1)
            image = self.G.synthesis(w_plus)
            
            # CLIP loss
            image_clip = F.interpolate(image, (224, 224))
            image_features = self.clip.encode_image((image_clip + 1) / 2)
            clip_loss = -F.cosine_similarity(image_features, text_features).mean()
            
            # Identity preservation
            id_loss = (w_edit - w).pow(2).mean()
            
            loss = clip_loss + 0.1 * id_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return w_edit.detach()


class SeFa:
    """Semantic Factorization - find directions from generator weights"""
    
    def __init__(self, generator, layer_idx=0):
        self.G = generator
        self.layer_idx = layer_idx
    
    def get_directions(self, num_directions=20):
        """
        Find semantically meaningful directions by analyzing
        the weights of the first linear layer after style injection
        """
        # Get weight matrix from synthesis network
        # This is the affine transform that converts W to style
        weight = self.G.synthesis.layers[self.layer_idx].conv.modulation.weight
        
        # Eigenvectors of W^T W give important directions
        # Or directly use SVD of weight
        U, S, Vh = torch.linalg.svd(weight.T)
        
        # Top eigenvectors are meaningful edit directions
        directions = Vh[:num_directions]
        
        return directions, S[:num_directions]


def semantic_editing_example():
    """Complete example of semantic editing workflow"""
    
    print("Semantic Editing Approaches:")
    print("=" * 60)
    print("\n1. InterFaceGAN (Supervised)")
    print("   - Need labeled data (1000+ samples per attribute)")
    print("   - Attributes: age, gender, smile, glasses, pose")
    print("   - Pro: Clean, orthogonal directions")
    print("   - Con: Limited to labeled attributes")
    
    print("\n2. GANSpace (Unsupervised)")
    print("   - No labels needed")
    print("   - PCA discovers latent factors")
    print("   - Pro: Discover unexpected controls")
    print("   - Con: Directions may mix multiple attributes")
    
    print("\n3. StyleCLIP (Text-guided)")
    print("   - Use any text description")
    print("   - 'Make it look older', 'Add glasses'")
    print("   - Pro: Open vocabulary, intuitive")
    print("   - Con: May not preserve identity perfectly")
    
    print("\n4. SeFa (Weight Analysis)")
    print("   - Analyze generator weights directly")
    print("   - No data or optimization needed")
    print("   - Pro: Fast, interpretable")
    print("   - Con: Limited control over what's found")
```

**Discovered Directions (StyleGAN-FFHQ):**

| Attribute | Layer Sensitivity | Direction Type |
|-----------|-------------------|----------------|
| Age | Coarse + Middle | InterFaceGAN |
| Smile | Middle | InterFaceGAN |
| Pose (yaw) | Coarse | GANSpace PC1 |
| Gender | Coarse | InterFaceGAN |
| Hair color | Fine | StyleCLIP |

**Interview Tip:** InterFaceGAN is the gold standard for known attributes—it finds clean, interpretable directions. GANSpace is great for exploration. StyleCLIP is most flexible but may require careful prompt engineering and identity preservation. The key insight is that StyleGAN's W space is sufficiently linear that simple arithmetic (w + α·n) produces meaningful edits.

---

## CycleGAN & Unpaired Translation

### Question 14
- [ ] Done
**Explain image-to-image translation without paired data. How does cycle-consistency loss work?**

**Answer:**

CycleGAN enables translation between domains (horses↔zebras, summer↔winter) without paired examples. Cycle-consistency loss ensures that translating to another domain and back recovers the original: $F(G(x)) ≈ x$. This constraint enables learning meaningful mappings without supervision.

**The Problem:**

| Paired Data | Unpaired Data |
|-------------|---------------|
| Edge ↔ Photo (same scene) | Horses ↔ Zebras |
| Exact correspondence | Just two collections |
| Pix2Pix works | Need CycleGAN |

**CycleGAN Architecture:**

```
Domain A                    Domain B
   x -----> G -----> ŷ
             ↓
             D_B (real vs fake B)
             
   x̂ <----- F <----- ŷ
   ↓
   Compare x vs x̂ (cycle-consistency)
```

**Losses:**

1. **Adversarial Loss:**
$$\mathcal{L}_{GAN}(G, D_B) = \mathbb{E}_y[\log D_B(y)] + \mathbb{E}_x[\log(1 - D_B(G(x)))]$$

2. **Cycle-Consistency Loss:**
$$\mathcal{L}_{cyc} = \mathbb{E}_x[\|F(G(x)) - x\|_1] + \mathbb{E}_y[\|G(F(y)) - y\|_1]$$

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ResNet block for generator"""
    
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """Generator for CycleGAN (ResNet-based)"""
    
    def __init__(self, in_channels=3, out_channels=3, num_residuals=9):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(num_residuals):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN Discriminator (70x70 receptive field)"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)


class CycleGAN(nn.Module):
    """Complete CycleGAN model"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Generators: A->B and B->A
        self.G_AB = CycleGANGenerator(in_channels, in_channels)
        self.G_BA = CycleGANGenerator(in_channels, in_channels)
        
        # Discriminators
        self.D_A = PatchDiscriminator(in_channels)
        self.D_B = PatchDiscriminator(in_channels)
    
    def forward(self, real_A, real_B):
        # Generate fake images
        fake_B = self.G_AB(real_A)  # A -> B
        fake_A = self.G_BA(real_B)  # B -> A
        
        # Cycle reconstruction
        rec_A = self.G_BA(fake_B)   # A -> B -> A
        rec_B = self.G_AB(fake_A)   # B -> A -> B
        
        return fake_A, fake_B, rec_A, rec_B


class CycleGANLoss:
    """Compute all CycleGAN losses"""
    
    def __init__(self, lambda_cyc=10.0, lambda_id=0.5):
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id
        
        self.criterion_GAN = nn.MSELoss()  # LSGAN
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
    
    def discriminator_loss(self, real_pred, fake_pred):
        """Discriminator loss (LSGAN)"""
        real_loss = self.criterion_GAN(real_pred, torch.ones_like(real_pred))
        fake_loss = self.criterion_GAN(fake_pred, torch.zeros_like(fake_pred))
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, model, real_A, real_B):
        """Complete generator loss"""
        
        # Generate
        fake_B = model.G_AB(real_A)
        fake_A = model.G_BA(real_B)
        
        # Adversarial loss
        pred_fake_B = model.D_B(fake_B)
        pred_fake_A = model.D_A(fake_A)
        
        loss_GAN_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
        loss_GAN_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
        loss_GAN = loss_GAN_AB + loss_GAN_BA
        
        # Cycle consistency loss
        rec_A = model.G_BA(fake_B)
        rec_B = model.G_AB(fake_A)
        
        loss_cycle_A = self.criterion_cycle(rec_A, real_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) * self.lambda_cyc
        
        # Identity loss (optional but helps preserve color)
        if self.lambda_id > 0:
            # G_AB should be identity for B
            same_B = model.G_AB(real_B)
            loss_id_B = self.criterion_identity(same_B, real_B)
            
            # G_BA should be identity for A
            same_A = model.G_BA(real_A)
            loss_id_A = self.criterion_identity(same_A, real_A)
            
            loss_identity = (loss_id_A + loss_id_B) * self.lambda_cyc * self.lambda_id
        else:
            loss_identity = 0
        
        total_loss = loss_GAN + loss_cycle + loss_identity
        
        return total_loss, {
            'GAN': loss_GAN.item(),
            'cycle': loss_cycle.item(),
            'identity': loss_identity.item() if isinstance(loss_identity, torch.Tensor) else 0
        }


class ReplayBuffer:
    """Buffer to store generated images for discriminator training"""
    
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if torch.rand(1).item() > 0.5:
                    i = torch.randint(0, self.max_size, (1,)).item()
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def train_cyclegan(model, dataloader_A, dataloader_B, epochs=200, device='cuda'):
    """Training loop for CycleGAN"""
    
    model = model.to(device)
    
    # Optimizers
    opt_G = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )
    opt_D_A = torch.optim.Adam(model.D_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(model.D_B.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # LR schedulers (linear decay after half training)
    lr_lambda = lambda epoch: 1.0 - max(0, epoch - epochs//2) / (epochs//2 + 1)
    
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda)
    
    loss_fn = CycleGANLoss()
    
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    
    for epoch in range(epochs):
        for real_A, real_B in zip(dataloader_A, dataloader_B):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Train Generators
            opt_G.zero_grad()
            g_loss, loss_dict = loss_fn.generator_loss(model, real_A, real_B)
            g_loss.backward()
            opt_G.step()
            
            # Generate for discriminator training
            with torch.no_grad():
                fake_B = model.G_AB(real_A)
                fake_A = model.G_BA(real_B)
            
            # Use buffer
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            
            # Train Discriminator A
            opt_D_A.zero_grad()
            d_A_loss = loss_fn.discriminator_loss(
                model.D_A(real_A),
                model.D_A(fake_A.detach())
            )
            d_A_loss.backward()
            opt_D_A.step()
            
            # Train Discriminator B
            opt_D_B.zero_grad()
            d_B_loss = loss_fn.discriminator_loss(
                model.D_B(real_B),
                model.D_B(fake_B.detach())
            )
            d_B_loss.backward()
            opt_D_B.step()
        
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
```

**Why Cycle-Consistency Works:**

Without constraints, G could map all inputs to a single output that fools D. Cycle-consistency forces:
- $G$ must encode enough information to allow $F$ to reconstruct
- Mapping must be reversible → meaningful correspondence

**Interview Tip:** Cycle-consistency is elegant but has limitations—it assumes mappings are approximately invertible. This fails for significant domain gaps (e.g., dog→cat) where information must be invented. The identity loss helps preserve colors when domains share color distributions (e.g., painting styles).

---

### Question 15
- [ ] Done
**What is the identity loss in CycleGAN and when is it necessary?**

**Answer:**

Identity loss ensures that when the generator receives an image already in the target domain, it outputs the same image unchanged. This preserves color and tonal characteristics, preventing unwanted color shifts during translation.

**Identity Loss Formula:**

$$\mathcal{L}_{identity} = \mathbb{E}_y[\|G_{A \rightarrow B}(y) - y\|_1] + \mathbb{E}_x[\|G_{B \rightarrow A}(x) - x\|_1]$$

**Why It Helps:**

| Without Identity Loss | With Identity Loss |
|----------------------|-------------------|
| May shift colors arbitrarily | Preserves original colors |
| Blue sky → orange sky | Blue sky stays blue |
| Background changes | Only target object changes |

**When to Use:**

| Scenario | Identity Loss Needed? |
|----------|----------------------|
| Photo → Painting | Yes (preserve color palette) |
| Horse → Zebra | Yes (zebra patterns, not colors) |
| Day → Night | No (color should change) |
| Summer → Winter | Depends on goal |

**Python Implementation:**
```python
import torch
import torch.nn as nn

class IdentityLoss(nn.Module):
    """Identity loss for CycleGAN"""
    
    def __init__(self, lambda_id=0.5):
        super().__init__()
        self.lambda_id = lambda_id
        self.criterion = nn.L1Loss()
    
    def forward(self, G_AB, G_BA, real_A, real_B):
        """
        G_AB should act as identity on domain B
        G_BA should act as identity on domain A
        """
        # G_AB(real_B) should equal real_B
        same_B = G_AB(real_B)
        loss_id_B = self.criterion(same_B, real_B)
        
        # G_BA(real_A) should equal real_A
        same_A = G_BA(real_A)
        loss_id_A = self.criterion(same_A, real_A)
        
        return (loss_id_A + loss_id_B) * self.lambda_id


class CycleGANWithIdentity:
    """CycleGAN training with identity loss"""
    
    def __init__(self, lambda_cyc=10.0, lambda_id=5.0):
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id
        
        # Identity loss is typically 0.5 * lambda_cyc
        # Some use 5.0 directly
    
    def generator_loss(self, G_AB, G_BA, D_A, D_B, real_A, real_B):
        # Forward
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        
        # Adversarial loss
        loss_GAN = (
            nn.MSELoss()(D_B(fake_B), torch.ones_like(D_B(fake_B))) +
            nn.MSELoss()(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        )
        
        # Cycle consistency
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        loss_cycle = (
            nn.L1Loss()(rec_A, real_A) +
            nn.L1Loss()(rec_B, real_B)
        ) * self.lambda_cyc
        
        # Identity loss
        same_A = G_BA(real_A)  # Should be identity
        same_B = G_AB(real_B)  # Should be identity
        loss_identity = (
            nn.L1Loss()(same_A, real_A) +
            nn.L1Loss()(same_B, real_B)
        ) * self.lambda_id
        
        return loss_GAN + loss_cycle + loss_identity


def ablation_identity_loss():
    """Show effect of identity loss"""
    
    print("Identity Loss Ablation:")
    print("=" * 50)
    
    results = {
        'Horse→Zebra': {
            'with_id': 'Zebra patterns added, horse color preserved',
            'without_id': 'May turn brown horse to white/black'
        },
        'Photo→Monet': {
            'with_id': 'Monet brushstrokes, original colors',
            'without_id': 'Colors may shift to Monet palette'
        },
        'Summer→Winter': {
            'with_id': 'Green stays greenish in winter',
            'without_id': 'Green becomes brown/white (more realistic)'
        }
    }
    
    for task, effect in results.items():
        print(f"\n{task}:")
        print(f"  With identity: {effect['with_id']}")
        print(f"  Without: {effect['without_id']}")
```

**Typical Weighting:**

```
Total Loss = λ_adv * L_adv + λ_cyc * L_cyc + λ_id * L_identity

Common values:
λ_adv = 1.0
λ_cyc = 10.0  
λ_id = 5.0 (or 0.5 * λ_cyc)
```

**Interview Tip:** Identity loss is crucial when you want to change texture/style but preserve colors (horse→zebra, photo→painting). Skip it when color change is the goal (day→night). It's essentially regularization that prevents the generator from learning unnecessary transformations.

---

### Question 16
- [ ] Done
**Explain the PatchGAN discriminator and why patch-level discrimination works better than image-level.**

**Answer:**

PatchGAN (Markovian discriminator) classifies NxN image patches as real/fake rather than the entire image. This enforces local texture/style consistency and provides denser gradient signal. A 70×70 receptive field captures local structure while remaining computationally efficient.

**Image-Level vs Patch-Level:**

| Aspect | Image-Level D | PatchGAN |
|--------|---------------|----------|
| Output | Single real/fake | Grid of real/fake |
| Receptive field | Entire image | Local patches |
| Parameters | Many | Fewer |
| Gradient signal | Sparse | Dense |
| Focus | Global structure | Local textures |

**Why Patches Work Better:**

1. **High-frequency detail**: Patches capture textures better than global assessment
2. **Dense gradients**: Every patch provides feedback → faster training
3. **Parameter efficiency**: Fully convolutional, reuses weights
4. **Translation invariance**: Same discriminator works across image

**Receptive Field Sizes:**

| Config | Receptive Field | Best For |
|--------|-----------------|----------|
| 1×1 | Per-pixel | Color matching |
| 16×16 | Small patches | Very local textures |
| 70×70 | Medium patches | General purpose |
| 286×286 | Large patches | Structural consistency |

**Python Implementation:**
```python
import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with ~70x70 receptive field
    Output is NxN grid of real/fake predictions
    """
    
    def __init__(self, in_channels=3, num_filters=64, num_layers=3):
        super().__init__()
        
        # First layer (no normalization)
        layers = [
            nn.Conv2d(in_channels, num_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            layers += [
                nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult,
                         4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(num_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Second to last layer (stride=1)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        
        layers += [
            nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult,
                     4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_filters * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer (1 channel for real/fake)
        layers += [
            nn.Conv2d(num_filters * nf_mult, 1, 4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Returns [B, 1, H', W'] where each position is real/fake prediction
        for corresponding image patch
        """
        return self.model(x)


class MultiScalePatchGAN(nn.Module):
    """
    Multi-scale discriminator (used in Pix2PixHD)
    Multiple PatchGANs at different scales
    """
    
    def __init__(self, in_channels=3, num_discriminators=3):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        
        for i in range(num_discriminators):
            self.discriminators.append(
                PatchGANDiscriminator(in_channels)
            )
        
        # Downsample for coarser scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        outputs = []
        
        for i, D in enumerate(self.discriminators):
            outputs.append(D(x))
            
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        
        return outputs


def compute_receptive_field():
    """Calculate receptive field size"""
    
    # For standard PatchGAN configuration
    # 4x4 conv, stride 2, padding 1
    # Repeated 4 times + final stride 1 layers
    
    rf = 1  # Start with 1 pixel
    stride = 1
    
    layers = [
        (4, 2),  # k=4, s=2
        (4, 2),  # k=4, s=2
        (4, 2),  # k=4, s=2
        (4, 1),  # k=4, s=1
        (4, 1),  # k=4, s=1
    ]
    
    for k, s in layers:
        rf = rf + (k - 1) * stride
        stride = stride * s
    
    print(f"Receptive field: {rf}x{rf} pixels")
    return rf


class PatchGANLoss:
    """Loss computation for PatchGAN"""
    
    def __init__(self, use_lsgan=True):
        if use_lsgan:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
    
    def __call__(self, prediction, target_is_real):
        """
        prediction: [B, 1, H, W] patch predictions
        target_is_real: bool
        """
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        
        # Loss is averaged over all patches
        return self.criterion(prediction, target)
    
    def discriminator_loss(self, real_pred, fake_pred):
        """Standard D loss"""
        real_loss = self(real_pred, True)
        fake_loss = self(fake_pred, False)
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, fake_pred):
        """G wants D to think fake is real"""
        return self(fake_pred, True)


def visualize_patch_output():
    """Show what PatchGAN output looks like"""
    
    D = PatchGANDiscriminator()
    
    # 256x256 input
    x = torch.randn(1, 3, 256, 256)
    out = D(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Each output position evaluates a 70x70 patch")
    print(f"Total: {out.shape[2] * out.shape[3]} patch predictions")
    
    # Output is typically 30x30 for 256x256 input
    # 256 / 8 - 2 = 30 (due to striding and final conv)
```

**Comparison:**

| Discriminator Type | Parameters | FID | Training |
|-------------------|------------|-----|----------|
| Image-level | ~10M | Higher | Slower |
| PatchGAN (70×70) | ~3M | Lower | Faster |
| Multi-scale PatchGAN | ~6M | Lowest | Medium |

**Interview Tip:** PatchGAN is almost always preferred over image-level discriminators for image-to-image tasks. The key insight is that high-level structure is enforced by the L1 reconstruction loss, while the discriminator focuses on local texture realism. This division of labor is more effective than asking one network to do both.

---

### Question 17
- [ ] Done
**What are CycleGAN's limitations (geometry changes, semantic consistency) and how does CUT improve on them?**

**Answer:**

CycleGAN struggles with: (1) geometry changes—assumes shape-preserving mappings, (2) semantic inconsistency—may change unrelated regions, (3) heavy computation—two generators and discriminators. CUT (Contrastive Unpaired Translation) uses contrastive learning for faster, more consistent translation.

**CycleGAN Limitations:**

| Limitation | Example | Cause |
|------------|---------|-------|
| No geometry change | Can't turn dog→cat | Cycle-consistency requires reversibility |
| Semantic drift | Horse→zebra changes background | No spatial correspondence |
| Training cost | 2 generators, 2 discriminators | Bidirectional requirement |
| Mode collapse | Single zebra pattern | GAN training instability |

**CUT Key Idea:**

Instead of cycle-consistency, use contrastive learning to enforce correspondence:
- Patches that should match (same location, before/after) → pull together
- Patches that shouldn't match (different locations) → push apart

**CUT Loss (PatchNCE):**

$$\mathcal{L}_{PatchNCE} = -\log \frac{\exp(v \cdot v^+ / \tau)}{\exp(v \cdot v^+ / \tau) + \sum_{n} \exp(v \cdot v_n^- / \tau)}$$

Where $v$ is query patch, $v^+$ is corresponding positive, $v_n^-$ are negatives.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchNCELoss(nn.Module):
    """
    Contrastive loss for CUT
    Ensures corresponding patches match, non-corresponding don't
    """
    
    def __init__(self, num_patches=256, temperature=0.07):
        super().__init__()
        self.num_patches = num_patches
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, feat_q, feat_k, feat_k_pool=None):
        """
        feat_q: query features from generated image [B, C, H, W]
        feat_k: key features from input image [B, C, H, W]
        
        Corresponding spatial locations should be similar
        """
        B, C, H, W = feat_q.shape
        
        # Sample random patch locations
        num_patches = min(self.num_patches, H * W)
        patch_ids = torch.randperm(H * W, device=feat_q.device)[:num_patches]
        
        # Extract patches at same locations
        feat_q_flat = feat_q.view(B, C, -1)  # [B, C, HW]
        feat_k_flat = feat_k.view(B, C, -1)
        
        feat_q_patches = feat_q_flat[:, :, patch_ids]  # [B, C, num_patches]
        feat_k_patches = feat_k_flat[:, :, patch_ids]
        
        # Normalize
        feat_q_patches = F.normalize(feat_q_patches, dim=1)
        feat_k_patches = F.normalize(feat_k_patches, dim=1)
        
        # Positive: same location (diagonal)
        # Negative: different locations
        
        # Compute similarities [B, num_patches, num_patches]
        similarities = torch.bmm(
            feat_q_patches.transpose(1, 2),  # [B, num_patches, C]
            feat_k_patches                    # [B, C, num_patches]
        ) / self.temperature
        
        # Labels: positive is on diagonal (index i)
        labels = torch.arange(num_patches, device=feat_q.device)
        labels = labels.unsqueeze(0).expand(B, -1)  # [B, num_patches]
        
        # Cross-entropy loss (treats diagonal as correct class)
        loss = self.cross_entropy(
            similarities.view(B * num_patches, num_patches),
            labels.view(-1)
        ).mean()
        
        return loss


class CUTGenerator(nn.Module):
    """Generator for CUT (similar to CycleGAN but single direction)"""
    
    def __init__(self, in_channels=3, out_channels=3, num_residuals=9):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(*[
            ResidualBlock(256) for _ in range(num_residuals)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        )
    
    def forward(self, x, return_features=False):
        # Encode
        feat1 = self.encoder[:4](x)   # After first conv
        feat2 = self.encoder[4:7](feat1)  # After downsample 1
        feat3 = self.encoder[7:](feat2)   # After downsample 2
        
        # Transform
        feat_transformed = self.residuals(feat3)
        
        # Decode
        out = self.decoder(feat_transformed)
        
        if return_features:
            return out, [feat1, feat2, feat3]
        return out


class FeatureProjector(nn.Module):
    """Project features to embedding space for contrastive learning"""
    
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, features):
        """
        features: [B, C, H, W]
        returns: [B, out_channels, H, W]
        """
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        features = features.reshape(B * H * W, C)
        
        projected = self.mlp(features)
        
        projected = projected.reshape(B, H, W, -1)
        projected = projected.permute(0, 3, 1, 2)  # [B, C', H, W]
        
        return projected


class CUT(nn.Module):
    """Contrastive Unpaired Translation"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Only ONE generator (A → B)
        self.G = CUTGenerator(in_channels)
        
        # Discriminator (same as CycleGAN)
        self.D = PatchGANDiscriminator(in_channels)
        
        # Feature projectors for NCE loss
        self.projectors = nn.ModuleList([
            FeatureProjector(64),
            FeatureProjector(128),
            FeatureProjector(256)
        ])
        
        self.nce_loss = PatchNCELoss()
    
    def forward(self, real_A):
        return self.G(real_A)
    
    def compute_losses(self, real_A, real_B, lambda_nce=1.0):
        """Compute all CUT losses"""
        
        # Generate fake B
        fake_B, feats_A = self.G(real_A, return_features=True)
        _, feats_fake = self.G(fake_B, return_features=True)
        
        # Adversarial loss
        pred_fake = self.D(fake_B)
        loss_G_adv = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
        
        # PatchNCE loss at multiple scales
        loss_nce = 0
        for i, (feat_A, feat_fake, proj) in enumerate(
            zip(feats_A, feats_fake, self.projectors)
        ):
            # Project features
            feat_A_proj = proj(feat_A)
            feat_fake_proj = proj(feat_fake)
            
            # NCE loss
            loss_nce += self.nce_loss(feat_fake_proj, feat_A_proj)
        
        loss_nce = loss_nce / len(self.projectors)
        
        # Identity NCE (optional - helps with color preservation)
        idt_B, feats_idt = self.G(real_B, return_features=True)
        _, feats_B = self.G(real_B, return_features=True)  # Get features
        
        loss_idt_nce = 0
        for feat_B, feat_idt, proj in zip(feats_B, feats_idt, self.projectors):
            feat_B_proj = proj(feat_B)
            feat_idt_proj = proj(feat_idt)
            loss_idt_nce += self.nce_loss(feat_idt_proj, feat_B_proj)
        
        loss_idt_nce = loss_idt_nce / len(self.projectors)
        
        total_loss = loss_G_adv + lambda_nce * (loss_nce + loss_idt_nce)
        
        return total_loss, {
            'adv': loss_G_adv.item(),
            'nce': loss_nce.item(),
            'idt_nce': loss_idt_nce.item()
        }


def compare_cyclegan_cut():
    """Comparison of CycleGAN vs CUT"""
    
    print("CycleGAN vs CUT Comparison:")
    print("=" * 60)
    print(f"{'Aspect':^20} | {'CycleGAN':^18} | {'CUT':^18}")
    print("-" * 60)
    print(f"{'Generators':^20} | {'2':^18} | {'1':^18}")
    print(f"{'Training time':^20} | {'~2 days':^18} | {'~1 day':^18}")
    print(f"{'Memory':^20} | {'Higher':^18} | {'Lower':^18}")
    print(f"{'Correspondence':^20} | {'Cycle loss':^18} | {'Contrastive':^18}")
    print(f"{'Geometry change':^20} | {'Limited':^18} | {'Better':^18}")
    print(f"{'Quality':^20} | {'Good':^18} | {'Similar/Better':^18}")
```

**Key Improvements in CUT:**

| Aspect | CycleGAN | CUT |
|--------|----------|-----|
| Generators | 2 | 1 |
| GPU memory | ~11GB | ~6GB |
| Training time | Baseline | 40% faster |
| Semantic consistency | Implicit | Explicit (contrastive) |
| Geometry flexibility | Poor | Better |

**Interview Tip:** CUT replaces the expensive cycle-consistency constraint with a contrastive learning objective. The key insight is that spatial correspondence can be learned by matching features at the same location—no need to reconstruct the original image. This also allows more geometric flexibility since we're not requiring exact pixel reconstruction.

---

### Question 18
- [ ] Done
**Explain multi-domain translation with StarGAN vs. training separate CycleGAN models.**

**Answer:**

StarGAN handles N domains with a single model by conditioning on target domain labels, while CycleGAN requires N(N-1)/2 separate models for pairwise translation. StarGAN is more efficient and enables sharing learned features across domains.

**Scaling Comparison:**

| Domains | CycleGAN Models | StarGAN Models |
|---------|-----------------|----------------|
| 2 | 1 | 1 |
| 3 | 3 | 1 |
| 4 | 6 | 1 |
| 5 | 10 | 1 |
| N | N(N-1)/2 | 1 |

**StarGAN Architecture:**

Generator takes image + target domain label:
$$G(x, c) \rightarrow y$$

Discriminator has two outputs:
1. Real/Fake classification
2. Domain classification

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StarGANGenerator(nn.Module):
    """Generator conditioned on target domain"""
    
    def __init__(self, img_channels=3, num_domains=5, num_residuals=6):
        super().__init__()
        
        self.num_domains = num_domains
        
        # Initial convolution (image + domain one-hot concatenated spatially)
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels + num_domains, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = self._down_block(64, 128)
        self.down2 = self._down_block(128, 256)
        
        # Residual blocks
        self.residuals = nn.Sequential(*[
            ResidualBlock(256) for _ in range(num_residuals)
        ])
        
        # Upsampling
        self.up1 = self._up_block(256, 128)
        self.up2 = self._up_block(128, 64)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(64, img_channels, 7, padding=3),
            nn.Tanh()
        )
    
    def _down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, target_domain):
        """
        x: input image [B, C, H, W]
        target_domain: target domain index [B] or one-hot [B, num_domains]
        """
        # Convert to one-hot if needed
        if target_domain.dim() == 1:
            domain_onehot = F.one_hot(target_domain, self.num_domains).float()
        else:
            domain_onehot = target_domain
        
        # Expand domain to spatial dimensions
        B, _, H, W = x.shape
        domain_spatial = domain_onehot.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # Concatenate with input
        x = torch.cat([x, domain_spatial], dim=1)
        
        # Forward pass
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residuals(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        
        return x


class StarGANDiscriminator(nn.Module):
    """Discriminator with real/fake + domain classification"""
    
    def __init__(self, img_channels=3, num_domains=5, img_size=128):
        super().__init__()
        
        self.num_domains = num_domains
        
        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        )
        
        # Real/Fake output (PatchGAN style)
        self.adv_output = nn.Conv2d(2048, 1, 3, padding=1)
        
        # Domain classification output
        final_size = img_size // 64  # After 6 stride-2 convs
        self.domain_output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_domains)
        )
    
    def forward(self, x):
        features = self.features(x)
        
        # Real/fake prediction
        adv_out = self.adv_output(features)
        
        # Domain classification
        domain_out = self.domain_output(features)
        
        return adv_out, domain_out


class StarGAN(nn.Module):
    """Complete StarGAN model"""
    
    def __init__(self, img_channels=3, num_domains=5, img_size=128):
        super().__init__()
        
        self.G = StarGANGenerator(img_channels, num_domains)
        self.D = StarGANDiscriminator(img_channels, num_domains, img_size)
        self.num_domains = num_domains
    
    def forward(self, x, target_domain):
        return self.G(x, target_domain)


class StarGANLoss:
    """Losses for StarGAN training"""
    
    def __init__(self, lambda_cls=1.0, lambda_rec=10.0, lambda_gp=10.0):
        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_gp = lambda_gp
        
        self.adv_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.rec_criterion = nn.L1Loss()
    
    def discriminator_loss(self, model, real_img, real_domain):
        """D loss: real/fake + domain classification"""
        
        # Real image
        real_adv, real_cls = model.D(real_img)
        
        # Adversarial loss (real should be 1)
        loss_adv_real = self.adv_criterion(
            real_adv, torch.ones_like(real_adv)
        )
        
        # Domain classification loss (classify real images)
        loss_cls = self.cls_criterion(real_cls, real_domain)
        
        # Generate fake
        target_domain = torch.randint(
            0, model.num_domains, (real_img.size(0),), 
            device=real_img.device
        )
        fake_img = model.G(real_img, target_domain).detach()
        
        fake_adv, _ = model.D(fake_img)
        loss_adv_fake = self.adv_criterion(
            fake_adv, torch.zeros_like(fake_adv)
        )
        
        # Gradient penalty
        loss_gp = self._gradient_penalty(model.D, real_img, fake_img)
        
        total = loss_adv_real + loss_adv_fake + self.lambda_cls * loss_cls + self.lambda_gp * loss_gp
        
        return total
    
    def generator_loss(self, model, real_img, real_domain):
        """G loss: fool D + correct domain + reconstruction"""
        
        # Random target domain
        target_domain = torch.randint(
            0, model.num_domains, (real_img.size(0),),
            device=real_img.device
        )
        
        # Generate
        fake_img = model.G(real_img, target_domain)
        
        # Adversarial loss
        fake_adv, fake_cls = model.D(fake_img)
        loss_adv = self.adv_criterion(fake_adv, torch.ones_like(fake_adv))
        
        # Domain classification (fake should be classified as target)
        loss_cls = self.cls_criterion(fake_cls, target_domain)
        
        # Reconstruction loss (translate back to original domain)
        rec_img = model.G(fake_img, real_domain)
        loss_rec = self.rec_criterion(rec_img, real_img)
        
        total = loss_adv + self.lambda_cls * loss_cls + self.lambda_rec * loss_rec
        
        return total
    
    def _gradient_penalty(self, D, real, fake):
        """WGAN-GP gradient penalty"""
        B = real.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=real.device)
        
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        adv_out, _ = D(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=adv_out,
            inputs=interpolated,
            grad_outputs=torch.ones_like(adv_out),
            create_graph=True
        )[0]
        
        gradients = gradients.view(B, -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return penalty


def stargan_applications():
    """Example StarGAN use cases"""
    
    applications = {
        'Facial Attributes': {
            'domains': ['Black Hair', 'Blond Hair', 'Brown Hair', 'Male', 'Young'],
            'dataset': 'CelebA',
            'note': 'Multi-label: can combine attributes'
        },
        'Expression Transfer': {
            'domains': ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised'],
            'dataset': 'RaFD',
            'note': 'Single-label: one expression at a time'
        },
        'Age Progression': {
            'domains': ['Child', 'Young Adult', 'Middle Age', 'Senior'],
            'dataset': 'MORPH',
            'note': 'Can interpolate between ages'
        }
    }
    
    for app, details in applications.items():
        print(f"\n{app}:")
        print(f"  Domains: {details['domains']}")
        print(f"  Dataset: {details['dataset']}")
        print(f"  Note: {details['note']}")
```

**StarGAN v2 Improvements:**

| Aspect | StarGAN v1 | StarGAN v2 |
|--------|------------|------------|
| Domain representation | One-hot label | Learned style code |
| Diversity | Limited | Multiple styles per domain |
| Architecture | Single path | Style encoder + mapping |

**When to Use Each:**

| Scenario | Recommendation |
|----------|---------------|
| 2 domains | CycleGAN (simpler) |
| 3+ domains | StarGAN |
| High diversity needed | StarGAN v2 |
| Different modalities | CycleGAN per pair |

**Interview Tip:** StarGAN's key innovation is the domain-conditioned generator with an auxiliary domain classifier. The reconstruction loss (A→B→A) replaces cycle-consistency when we know the domain labels. StarGAN v2 further improves by learning style codes instead of one-hot labels, allowing multiple diverse outputs per domain.

---

## Pix2Pix & Paired Translation

### Question 19
- [ ] Done
**Explain conditional GAN architecture in Pix2Pix and the role of L1 reconstruction loss.**

**Answer:**

Pix2Pix is a conditional GAN for paired image-to-image translation (edges→photos, labels→images). The generator is conditioned on input image; L1 loss ensures structural accuracy while adversarial loss adds realistic textures.

**Architecture:**

```
Input Image → Generator (U-Net) → Output Image
                    ↓
         Discriminator (PatchGAN)
              ↓
         Real/Fake + L1 to ground truth
```

**Why Both Losses:**

| Loss | Contribution |
|------|--------------|
| **L1 (Reconstruction)** | Correct structure, prevent mode collapse |
| **Adversarial** | Sharp textures, realistic details |
| **L1 only** | Blurry but accurate |
| **Adversarial only** | Sharp but may hallucinate |

**Objective Function:**

$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x, z)))]$$

$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z}[\|y - G(x, z)\|_1]$$

$$G^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    """U-Net Generator for Pix2Pix"""
    
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._encoder_block(in_channels, features, normalize=False)  # 64
        self.enc2 = self._encoder_block(features, features * 2)      # 128
        self.enc3 = self._encoder_block(features * 2, features * 4)  # 256
        self.enc4 = self._encoder_block(features * 4, features * 8)  # 512
        self.enc5 = self._encoder_block(features * 8, features * 8)  # 512
        self.enc6 = self._encoder_block(features * 8, features * 8)  # 512
        self.enc7 = self._encoder_block(features * 8, features * 8)  # 512
        self.enc8 = self._encoder_block(features * 8, features * 8, normalize=False)  # 512
        
        # Decoder (upsampling with skip connections)
        self.dec1 = self._decoder_block(features * 8, features * 8, dropout=True)
        self.dec2 = self._decoder_block(features * 16, features * 8, dropout=True)
        self.dec3 = self._decoder_block(features * 16, features * 8, dropout=True)
        self.dec4 = self._decoder_block(features * 16, features * 8)
        self.dec5 = self._decoder_block(features * 16, features * 4)
        self.dec6 = self._decoder_block(features * 8, features * 2)
        self.dec7 = self._decoder_block(features * 4, features)
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def _encoder_block(self, in_ch, out_ch, normalize=True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _decoder_block(self, in_ch, out_ch, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder with stored features for skip connections
        e1 = self.enc1(x)    # 128
        e2 = self.enc2(e1)   # 64
        e3 = self.enc3(e2)   # 32
        e4 = self.enc4(e3)   # 16
        e5 = self.enc5(e4)   # 8
        e6 = self.enc6(e5)   # 4
        e7 = self.enc7(e6)   # 2
        e8 = self.enc8(e7)   # 1
        
        # Decoder with skip connections (concatenate)
        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        
        return self.final(d7)


class ConditionalDiscriminator(nn.Module):
    """PatchGAN Discriminator conditioned on input image"""
    
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        
        # Input: concatenation of input and output (or target)
        total_channels = in_channels + out_channels
        
        self.model = nn.Sequential(
            # No normalization on first layer
            nn.Conv2d(total_channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features, features * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features * 2, features * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features * 4, features * 8, 4, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output 1 channel for real/fake
            nn.Conv2d(features * 8, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, input_img, output_img):
        """
        Discriminator sees both input and output together
        """
        x = torch.cat([input_img, output_img], dim=1)
        return self.model(x)


class Pix2PixLoss:
    """Combined loss for Pix2Pix training"""
    
    def __init__(self, lambda_L1=100.0):
        self.lambda_L1 = lambda_L1
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
    
    def discriminator_loss(self, D, input_img, real_img, fake_img):
        """Standard conditional GAN discriminator loss"""
        
        # Real pair
        real_pred = D(input_img, real_img)
        real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))
        
        # Fake pair
        fake_pred = D(input_img, fake_img.detach())
        fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
        
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, G, D, input_img, real_img):
        """Generator loss: fool D + L1 reconstruction"""
        
        fake_img = G(input_img)
        
        # Adversarial loss
        fake_pred = D(input_img, fake_img)
        adv_loss = self.bce_loss(fake_pred, torch.ones_like(fake_pred))
        
        # L1 reconstruction loss
        l1_loss = self.l1_loss(fake_img, real_img)
        
        total_loss = adv_loss + self.lambda_L1 * l1_loss
        
        return total_loss, fake_img


def train_pix2pix(G, D, dataloader, epochs=200, device='cuda'):
    """Pix2Pix training loop"""
    
    G = G.to(device)
    D = D.to(device)
    
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    loss_fn = Pix2PixLoss(lambda_L1=100.0)
    
    for epoch in range(epochs):
        for input_img, real_img in dataloader:
            input_img = input_img.to(device)
            real_img = real_img.to(device)
            
            # Train Discriminator
            fake_img = G(input_img)
            
            opt_D.zero_grad()
            d_loss = loss_fn.discriminator_loss(D, input_img, real_img, fake_img)
            d_loss.backward()
            opt_D.step()
            
            # Train Generator
            opt_G.zero_grad()
            g_loss, _ = loss_fn.generator_loss(G, D, input_img, real_img)
            g_loss.backward()
            opt_G.step()


def ablation_l1_weight():
    """Effect of L1 weight on results"""
    
    print("L1 Weight Ablation:")
    print("=" * 50)
    print(f"{'λ_L1':^10} | {'Structure':^15} | {'Sharpness':^15}")
    print("-" * 50)
    print(f"{'0 (GAN only)':^10} | {'Poor':^15} | {'Sharp but wrong':^15}")
    print(f"{'1':^10} | {'Moderate':^15} | {'Sharp':^15}")
    print(f"{'10':^10} | {'Good':^15} | {'Good':^15}")
    print(f"{'100':^10} | {'Excellent':^15} | {'Slightly blurry':^15}")
    print(f"{'1000':^10} | {'Excellent':^15} | {'Blurry':^15}")
```

**Why Conditioning Matters:**

The discriminator sees both input and output:
$$D(x, y) \text{ vs } D(x, G(x))$$

This ensures the output is **consistent with the input**, not just realistic in isolation.

**λ_L1 Trade-off:**

| Higher λ_L1 | Lower λ_L1 |
|-------------|------------|
| More accurate | More creative |
| Blurrier | Sharper |
| Safer | Riskier |

**Interview Tip:** The L1 loss is critical—without it, the generator may produce realistic but unrelated outputs (mode collapse to common patterns). λ=100 is the standard choice, meaning L1 dominates early training for structure, while adversarial loss refines textures. The conditional discriminator is what makes Pix2Pix different from regular GANs.

---

### Question 20
- [ ] Done
**Describe the U-Net generator in Pix2Pix and why skip connections help paired translation.**

**Answer:**

U-Net uses an encoder-decoder architecture with skip connections that directly connect encoder layers to corresponding decoder layers. For image translation, this preserves low-level details (edges, textures) that would be lost through the bottleneck, enabling precise spatial alignment.

**U-Net Structure:**

```
Input [256x256]
   ↓ Conv, stride 2
[128x128, 64ch] ────────────────────────────→ Concat → [128x128, 128ch]
   ↓ Conv, stride 2                                        ↑ ConvT, stride 2
[64x64, 128ch]  ──────────────────────────→ Concat → [64x64, 256ch]
   ↓ Conv, stride 2                                        ↑
   ...                  Bottleneck                        ...
   ↓                      [1x1]                            ↑
```

**Why Skip Connections Help:**

| Without Skips | With Skips |
|---------------|------------|
| All info through bottleneck | Bypass fine details |
| Blurry edges | Sharp edges |
| Lost local patterns | Preserved textures |
| Must re-synthesize details | Copy and refine |

**Information Flow:**

| Path | Information Type |
|------|-----------------|
| Through bottleneck | High-level semantics, global structure |
| Skip connections | Low-level details, edges, textures |

**Python Implementation:**
```python
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """Basic U-Net encoder/decoder block"""
    
    def __init__(self, in_channels, out_channels, down=True, 
                 use_dropout=False, use_bn=True):
        super().__init__()
        
        if down:
            conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
            act = nn.LeakyReLU(0.2, inplace=True)
        else:
            conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
            act = nn.ReLU(inplace=True)
        
        layers = [conv]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(act)
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """Complete U-Net with skip connections"""
    
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super().__init__()
        
        # Encoder
        self.e1 = UNetBlock(in_channels, base_filters, down=True, use_bn=False)
        self.e2 = UNetBlock(base_filters, base_filters * 2, down=True)
        self.e3 = UNetBlock(base_filters * 2, base_filters * 4, down=True)
        self.e4 = UNetBlock(base_filters * 4, base_filters * 8, down=True)
        self.e5 = UNetBlock(base_filters * 8, base_filters * 8, down=True)
        self.e6 = UNetBlock(base_filters * 8, base_filters * 8, down=True)
        self.e7 = UNetBlock(base_filters * 8, base_filters * 8, down=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (note: input channels doubled due to skip connection concatenation)
        self.d1 = UNetBlock(base_filters * 8, base_filters * 8, down=False, use_dropout=True)
        self.d2 = UNetBlock(base_filters * 16, base_filters * 8, down=False, use_dropout=True)  # 8+8
        self.d3 = UNetBlock(base_filters * 16, base_filters * 8, down=False, use_dropout=True)
        self.d4 = UNetBlock(base_filters * 16, base_filters * 8, down=False)
        self.d5 = UNetBlock(base_filters * 16, base_filters * 4, down=False)
        self.d6 = UNetBlock(base_filters * 8, base_filters * 2, down=False)
        self.d7 = UNetBlock(base_filters * 4, base_filters, down=False)
        
        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder (save for skip connections)
        e1 = self.e1(x)      # 128, 64ch
        e2 = self.e2(e1)     # 64, 128ch
        e3 = self.e3(e2)     # 32, 256ch
        e4 = self.e4(e3)     # 16, 512ch
        e5 = self.e5(e4)     # 8, 512ch
        e6 = self.e6(e5)     # 4, 512ch
        e7 = self.e7(e6)     # 2, 512ch
        
        # Bottleneck
        b = self.bottleneck(e7)  # 1, 512ch
        
        # Decoder with skip connections
        d1 = self.d1(b)
        d1 = torch.cat([d1, e7], dim=1)  # Concatenate skip
        
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        
        return self.final(d7)


class ResNetGenerator(nn.Module):
    """Alternative: ResNet-based generator (no skips)"""
    
    def __init__(self, in_channels=3, out_channels=3, num_residuals=9):
        super().__init__()
        
        # Downsample
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(*[
            ResidualBlock(256) for _ in range(num_residuals)
        ])
        
        # Upsample
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(self.residuals(self.encoder(x)))


def compare_generators():
    """U-Net vs ResNet for image translation"""
    
    print("Generator Architecture Comparison:")
    print("=" * 60)
    print(f"{'Aspect':^20} | {'U-Net':^18} | {'ResNet':^18}")
    print("-" * 60)
    print(f"{'Skip connections':^20} | {'Yes (concat)':^18} | {'No':^18}")
    print(f"{'Detail preservation':^20} | {'Excellent':^18} | {'Good':^18}")
    print(f"{'Global changes':^20} | {'Harder':^18} | {'Easier':^18}")
    print(f"{'Best for':^20} | {'Edges→Photo':^18} | {'Style transfer':^18}")
    print(f"{'Memory':^20} | {'Higher':^18} | {'Lower':^18}")


def skip_connection_ablation():
    """Show importance of skip connections"""
    
    results = {
        'No skips': 'Blurry, loses fine details, poor edge alignment',
        'Some skips': 'Better edges, some details preserved',
        'All skips': 'Sharp edges, excellent detail preservation'
    }
    
    print("\nSkip Connection Ablation:")
    for config, result in results.items():
        print(f"  {config}: {result}")
```

**When U-Net vs ResNet:**

| Task | Best Generator | Why |
|------|---------------|-----|
| Edges → Photo | U-Net | Need exact edge positions |
| Segmentation → Image | U-Net | Preserve boundaries |
| Style transfer | ResNet | Global changes, less copying |
| Day → Night | Either | Moderate spatial change |

**Interview Tip:** Skip connections are essential when the input and output should be spatially aligned (edges, segmentation, maps). They "shuttle" low-level information around the bottleneck. For tasks requiring more transformation (different structure), ResNet generators without skips are preferred. The concatenation (vs. addition) allows the decoder to learn when to use each source.

---

### Question 21
- [ ] Done
**How does Pix2PixHD achieve high-resolution translation with multi-scale discriminators?**

**Answer:**

Pix2PixHD generates 2048×1024 images using: (1) coarse-to-fine generators that progressively add detail, (2) multi-scale discriminators that evaluate at 3 different scales, and (3) feature matching loss for training stability. This hierarchical approach enables high-resolution synthesis where regular Pix2Pix fails.

**Key Improvements over Pix2Pix:**

| Aspect | Pix2Pix | Pix2PixHD |
|--------|---------|-----------|
| Resolution | 256×256 | 2048×1024 |
| Generator | Single U-Net | Coarse-to-fine |
| Discriminator | Single scale | Multi-scale (3) |
| Loss | L1 + GAN | + Feature matching + VGG |

**Multi-Scale Discriminator:**

Three discriminators operating at different scales:
- D1: Full resolution (2048×1024)
- D2: Downsampled 2× (1024×512)
- D3: Downsampled 4× (512×256)

**Why Multiple Scales:**

| Scale | Focus |
|-------|-------|
| Full (D1) | Fine textures, high-frequency details |
| Medium (D2) | Object parts, medium features |
| Coarse (D3) | Global structure, layout |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale PatchGAN discriminator for Pix2PixHD"""
    
    def __init__(self, in_channels, num_discriminators=3, num_layers=4):
        super().__init__()
        
        self.num_discriminators = num_discriminators
        
        self.discriminators = nn.ModuleList()
        for _ in range(num_discriminators):
            self.discriminators.append(
                NLayerDiscriminator(in_channels, num_layers)
            )
        
        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=1, count_include_pad=False
        )
    
    def forward(self, input_img, target_img):
        """
        Returns list of outputs at each scale
        Also returns intermediate features for feature matching
        """
        results = []
        features = []
        
        # Concatenate input and target
        x = torch.cat([input_img, target_img], dim=1)
        
        for i, D in enumerate(self.discriminators):
            # Get output and features
            out, feats = D(x)
            results.append(out)
            features.append(feats)
            
            # Downsample for next scale
            if i < self.num_discriminators - 1:
                x = self.downsample(x)
        
        return results, features


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with feature extraction"""
    
    def __init__(self, in_channels, num_layers=4, base_filters=64):
        super().__init__()
        
        layers = []
        current_filters = base_filters
        
        # First layer (no normalization)
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, current_filters, 4, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        # Intermediate layers
        for i in range(1, num_layers):
            prev_filters = current_filters
            current_filters = min(current_filters * 2, 512)
            stride = 2 if i < num_layers - 1 else 1
            
            layers.append(nn.Sequential(
                nn.Conv2d(prev_filters, current_filters, 4, stride=stride, padding=2),
                nn.InstanceNorm2d(current_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        # Output layer
        layers.append(nn.Conv2d(current_filters, 1, 4, stride=1, padding=2))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        features = []
        
        for layer in self.layers[:-1]:
            x = layer(x)
            features.append(x)
        
        output = self.layers[-1](x)
        
        return output, features


class GlobalGenerator(nn.Module):
    """Coarse (global) generator for Pix2PixHD"""
    
    def __init__(self, in_channels=3, out_channels=3, ngf=64, num_downs=4, num_residuals=9):
        super().__init__()
        
        # Initial conv
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        for i in range(num_downs):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # Residual blocks
        mult = 2 ** num_downs
        for _ in range(num_residuals):
            layers += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(num_downs):
            mult = 2 ** (num_downs - i)
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True)
            ]
        
        # Output
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class LocalEnhancer(nn.Module):
    """Local (fine) generator that enhances global output"""
    
    def __init__(self, in_channels=3, out_channels=3, ngf=32):
        super().__init__()
        
        # Downsample input
        self.down = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(*[
            ResidualBlock(ngf * 2) for _ in range(3)
        ])
        
        # Upsample
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()
        )
    
    def forward(self, x, global_features):
        """
        x: high-resolution input
        global_features: features from global generator (upsampled)
        """
        local_features = self.down(x)
        
        # Add global features
        combined = local_features + global_features
        
        refined = self.residuals(combined)
        output = self.up(refined)
        
        return output


class Pix2PixHDGenerator(nn.Module):
    """Complete Pix2PixHD coarse-to-fine generator"""
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Global (coarse) generator
        self.global_generator = GlobalGenerator(in_channels, out_channels)
        
        # Local (fine) enhancer
        self.local_enhancer = LocalEnhancer(in_channels, out_channels)
        
        # Feature downsampler for combining scales
        self.downsample = nn.AvgPool2d(2)
    
    def forward(self, x, return_intermediate=False):
        # Downsample for global generator
        x_down = self.downsample(x)
        
        # Global (coarse) output
        global_output = self.global_generator(x_down)
        
        # Upsample global output for local enhancer
        global_upsampled = F.interpolate(
            global_output, scale_factor=2, mode='bilinear', align_corners=True
        )
        
        # Local (fine) enhancement
        final_output = self.local_enhancer(x, global_upsampled)
        
        if return_intermediate:
            return final_output, global_output
        return final_output


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for stable training"""
    
    def __init__(self, num_D=3, num_layers=4, lambda_feat=10.0):
        super().__init__()
        self.num_D = num_D
        self.num_layers = num_layers
        self.lambda_feat = lambda_feat
        self.criterion = nn.L1Loss()
    
    def forward(self, real_features, fake_features):
        """
        Match intermediate features from discriminator
        """
        loss = 0
        
        for d in range(self.num_D):
            for i in range(len(real_features[d])):
                loss += self.criterion(
                    fake_features[d][i],
                    real_features[d][i].detach()
                )
        
        return loss * self.lambda_feat / (self.num_D * self.num_layers)


class Pix2PixHDLoss:
    """Complete loss for Pix2PixHD training"""
    
    def __init__(self, lambda_feat=10.0, lambda_vgg=10.0):
        self.lambda_feat = lambda_feat
        self.lambda_vgg = lambda_vgg
        
        self.criterion_gan = nn.MSELoss()
        self.criterion_feat = nn.L1Loss()
        
        # VGG for perceptual loss
        from torchvision import models
        vgg = models.vgg19(pretrained=True).features[:36]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
    
    def discriminator_loss(self, D, real_outputs, fake_outputs):
        """Multi-scale discriminator loss"""
        loss = 0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            loss += self.criterion_gan(real_out, torch.ones_like(real_out))
            loss += self.criterion_gan(fake_out, torch.zeros_like(fake_out))
        
        return loss / len(real_outputs)
    
    def generator_loss(self, D, input_img, real_img, fake_img):
        """Generator loss: GAN + feature matching + VGG"""
        
        # Get discriminator outputs and features
        real_outputs, real_features = D(input_img, real_img)
        fake_outputs, fake_features = D(input_img, fake_img)
        
        # GAN loss (multi-scale)
        loss_gan = 0
        for fake_out in fake_outputs:
            loss_gan += self.criterion_gan(fake_out, torch.ones_like(fake_out))
        loss_gan /= len(fake_outputs)
        
        # Feature matching loss
        loss_feat = 0
        for i in range(len(real_features)):
            for j in range(len(real_features[i])):
                loss_feat += self.criterion_feat(
                    fake_features[i][j], 
                    real_features[i][j].detach()
                )
        loss_feat = loss_feat * self.lambda_feat / (len(real_features) * len(real_features[0]))
        
        # VGG perceptual loss
        real_vgg = self.vgg(real_img)
        fake_vgg = self.vgg(fake_img)
        loss_vgg = self.criterion_feat(fake_vgg, real_vgg) * self.lambda_vgg
        
        return loss_gan + loss_feat + loss_vgg
```

**Training Strategy:**

| Phase | What's Trained | Resolution |
|-------|---------------|------------|
| 1 | Global G only | 1024×512 |
| 2 | Global + Local | 2048×1024 |
| 3 | Fine-tune all | 2048×1024 |

**Interview Tip:** Pix2PixHD's key insight is that high-resolution synthesis requires multi-scale supervision. A single discriminator at full resolution only sees local patches, missing global coherence. The coarse-to-fine generator ensures global structure is correct before adding fine details. Feature matching loss provides more stable gradients than just GAN loss at high resolution.

---

### Question 22
- [ ] Done
**Compare applications: edges-to-photo, semantic-to-photo, day-to-night. What determines task difficulty?**

**Answer:**

Task difficulty depends on: (1) amount of information in the input, (2) ambiguity in the mapping, and (3) domain gap. Edges→photo is hardest (sparse input), semantic→photo is moderate (structured but ambiguous), day→night is easier (dense input, mostly color change).

**Difficulty Comparison:**

| Task | Input Info | Ambiguity | Difficulty |
|------|-----------|-----------|------------|
| Edges → Photo | Very sparse | Very high | Hard |
| Segmentation → Photo | Medium | High | Medium-Hard |
| Day → Night | Dense | Low-Medium | Medium |
| Colorization | Dense | Medium | Medium |
| Super-resolution | Dense | Low | Easy |

**What Makes Tasks Harder:**

| Factor | Easier | Harder |
|--------|--------|--------|
| Input density | Dense (photos) | Sparse (sketches) |
| Mapping ambiguity | One-to-one | One-to-many |
| Domain gap | Similar | Very different |
| Required synthesis | Little | Lots |
| Color/texture info | Preserved | Must invent |

**Analysis by Task:**

**1. Edges → Photo (Hardest):**
```
Input: Binary edge map (very sparse)
Challenge: Must hallucinate textures, colors, materials
Ambiguity: Same edge could be many objects
Success: Requires strong priors (conditional on class)
```

**2. Semantic Segmentation → Photo (Hard):**
```
Input: Class labels per pixel (no appearance info)
Challenge: Must generate textures, lighting, specific instances
Ambiguity: "Building" could be any building
Success: Better than edges (knows what objects exist)
```

**3. Day → Night (Moderate):**
```
Input: Full RGB image
Challenge: Lighting changes, add lights/reflections
Ambiguity: Which lights turn on? Where are reflections?
Success: Most structure preserved (mainly color/lighting)
```

**Python Implementation:**
```python
import torch
import torch.nn as nn

class TaskDifficultyAnalyzer:
    """Analyze image translation task difficulty"""
    
    @staticmethod
    def compute_input_entropy(input_tensor):
        """
        Higher entropy = more information = easier task
        """
        # Flatten and normalize
        flat = input_tensor.view(-1).float()
        flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)
        
        # Histogram
        hist = torch.histc(flat, bins=256, min=0, max=1)
        hist = hist / hist.sum()
        
        # Entropy
        entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
        
        return entropy.item()
    
    @staticmethod
    def compute_sparsity(input_tensor):
        """
        Higher sparsity = less information = harder task
        """
        non_zero = (input_tensor.abs() > 0.01).float().mean()
        sparsity = 1 - non_zero
        return sparsity.item()
    
    @staticmethod
    def compute_structural_similarity(input_tensor, output_tensor):
        """
        Higher SSIM = more preserved = easier task
        """
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure()
        return ssim(input_tensor, output_tensor).item()


class TaskSpecificGenerator(nn.Module):
    """Generator adaptations for different task difficulties"""
    
    def __init__(self, task_type='day2night'):
        super().__init__()
        
        if task_type == 'edges2photo':
            # Hardest: need more capacity, more residual blocks
            self.generator = self._build_heavy_generator()
        elif task_type == 'semantic2photo':
            # Medium: standard architecture with conditioning
            self.generator = self._build_conditional_generator()
        elif task_type == 'day2night':
            # Easier: lighter architecture, skip connections important
            self.generator = self._build_light_generator()
    
    def _build_heavy_generator(self):
        """For sparse input tasks (edges, sketches)"""
        return nn.Sequential(
            # More initial processing to extract structure
            nn.Conv2d(1, 128, 7, padding=3),  # 1 channel input
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            # Heavy encoder
            self._make_encoder(128, 512, num_downs=4),
            # Many residual blocks to build representation
            *[ResidualBlock(512) for _ in range(12)],
            # Decoder
            self._make_decoder(512, 64, num_ups=4),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh()
        )
    
    def _build_conditional_generator(self):
        """For semantic segmentation input"""
        return SPADEGenerator(num_classes=35, img_channels=3)
    
    def _build_light_generator(self):
        """For dense input tasks (photo to photo)"""
        return nn.Sequential(
            # Light encoder
            self._make_encoder(3, 256, num_downs=3),
            # Fewer residual blocks (just need color transform)
            *[ResidualBlock(256) for _ in range(6)],
            # Decoder with skip connections (U-Net style)
            self._make_decoder(256, 64, num_ups=3),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh()
        )
    
    def _make_encoder(self, in_ch, out_ch, num_downs):
        layers = []
        ch = in_ch
        for i in range(num_downs):
            out = min(ch * 2, out_ch)
            layers.append(nn.Conv2d(ch, out, 4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(out))
            layers.append(nn.ReLU(True))
            ch = out
        return nn.Sequential(*layers)
    
    def _make_decoder(self, in_ch, out_ch, num_ups):
        layers = []
        ch = in_ch
        for i in range(num_ups):
            out = max(ch // 2, out_ch)
            layers.append(nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(out))
            layers.append(nn.ReLU(True))
            ch = out
        return nn.Sequential(*layers)


class SPADEGenerator(nn.Module):
    """SPADE generator for semantic-to-photo (handles one-to-many)"""
    
    def __init__(self, num_classes, img_channels=3, base_ch=64):
        super().__init__()
        
        # SPADE uses semantic map to modulate normalization
        # Allows same segmentation → multiple realistic outputs
        
        self.fc = nn.Linear(256, 16 * 16 * base_ch * 16)
        
        self.spade_blocks = nn.ModuleList([
            SPADEResBlock(base_ch * 16, base_ch * 16, num_classes),  # 16x16
            SPADEResBlock(base_ch * 16, base_ch * 16, num_classes),  # 32x32
            SPADEResBlock(base_ch * 16, base_ch * 8, num_classes),   # 64x64
            SPADEResBlock(base_ch * 8, base_ch * 4, num_classes),    # 128x128
            SPADEResBlock(base_ch * 4, base_ch * 2, num_classes),    # 256x256
            SPADEResBlock(base_ch * 2, base_ch, num_classes),        # 512x512
        ])
        
        self.final = nn.Sequential(
            nn.Conv2d(base_ch, img_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, segmap, z=None):
        """
        segmap: semantic segmentation map [B, H, W] or [B, C, H, W]
        z: random noise for diversity
        """
        if z is None:
            z = torch.randn(segmap.size(0), 256, device=segmap.device)
        
        x = self.fc(z).view(-1, 1024, 16, 16)
        
        for i, block in enumerate(self.spade_blocks):
            # Resize segmap to match current resolution
            seg_resized = F.interpolate(
                segmap.float().unsqueeze(1) if segmap.dim() == 3 else segmap.float(),
                size=x.shape[2:],
                mode='nearest'
            )
            x = block(x, seg_resized)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        return self.final(x)


def task_difficulty_ranking():
    """Comprehensive difficulty ranking"""
    
    tasks = [
        ('Super-resolution', 'Dense RGB', 'Very Low', 'Easy'),
        ('Denoising', 'Dense RGB', 'Very Low', 'Easy'),
        ('Day→Night', 'Dense RGB', 'Low', 'Easy-Medium'),
        ('Summer→Winter', 'Dense RGB', 'Medium', 'Medium'),
        ('Colorization', 'Dense grayscale', 'Medium', 'Medium'),
        ('Photo→Painting', 'Dense RGB', 'Medium', 'Medium'),
        ('Segmentation→Photo', 'Class labels', 'High', 'Hard'),
        ('Layout→Photo', 'Bounding boxes', 'Very High', 'Hard'),
        ('Edges→Photo', 'Binary edges', 'Very High', 'Very Hard'),
        ('Sketch→Photo', 'Sparse lines', 'Extreme', 'Very Hard'),
    ]
    
    print("Image Translation Task Difficulty Ranking:")
    print("=" * 70)
    print(f"{'Task':^25} | {'Input Type':^15} | {'Ambiguity':^10} | {'Difficulty':^12}")
    print("-" * 70)
    for task, input_type, ambiguity, difficulty in tasks:
        print(f"{task:^25} | {input_type:^15} | {ambiguity:^10} | {difficulty:^12}")
```

**Architecture Recommendations:**

| Task Type | Generator | Special Components |
|-----------|-----------|-------------------|
| Sparse→Dense | Heavy ResNet | More capacity, no skips |
| Dense→Dense | U-Net | Skip connections |
| Semantic→Photo | SPADE | Semantic conditioning |
| One-to-many | + Noise input | VAE or noise injection |

**Interview Tip:** Task difficulty is primarily determined by the information gap between input and output. When the input is sparse (edges), the model must learn strong priors about the visual world. SPADE (Spatially Adaptive Denormalization) is specifically designed for semantic→photo because it modulates features based on semantic class, allowing different textures for different object categories.

---

## Style Transfer

### Question 23
- [ ] Done
**Explain neural style transfer: content loss, style loss (Gram matrices), and the optimization process.**

**Answer:**

Neural style transfer combines the content of one image with the style of another by optimizing a generated image to minimize: (1) content loss (feature similarity to content image), and (2) style loss (Gram matrix similarity to style image). Gram matrices capture texture statistics by measuring feature correlations.

**Core Idea:**

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{content} + \beta \cdot \mathcal{L}_{style}$$

**Content Loss:**

Match high-level features (deep layer activations):
$$\mathcal{L}_{content} = \frac{1}{2}\sum_{i,j}(F^l_{ij} - P^l_{ij})^2$$

Where $F^l$ = generated image features, $P^l$ = content image features at layer $l$.

**Style Loss (Gram Matrix):**

Gram matrix captures correlations between feature channels:
$$G^l_{ij} = \sum_k F^l_{ik} F^l_{jk}$$

Style loss matches Gram matrices across multiple layers:
$$\mathcal{L}_{style} = \sum_l w_l \cdot \frac{1}{4N_l^2 M_l^2}\sum_{i,j}(G^l_{ij} - A^l_{ij})^2$$

Where $A^l$ = style image Gram matrix, $N_l$ = number of channels, $M_l$ = height × width.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class VGGFeatures(nn.Module):
    """Extract features from VGG19 for style transfer"""
    
    def __init__(self):
        super().__init__()
        
        vgg = models.vgg19(pretrained=True).features
        
        # Content layer (deep - captures structure)
        self.content_layers = ['conv4_2']
        
        # Style layers (multiple depths - captures textures)
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        
        # Build feature extractors
        self.slices = nn.ModuleList()
        
        layer_names = {
            '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
            '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
        }
        
        prev_idx = 0
        for idx, name in sorted(layer_names.items(), key=lambda x: int(x[0])):
            idx = int(idx)
            self.slices.append(vgg[prev_idx:idx+1])
            prev_idx = idx + 1
        
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = {}
        layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
        
        for slice_module, name in zip(self.slices, layer_names):
            x = slice_module(x)
            features[name] = x
        
        return features


def gram_matrix(features):
    """
    Compute Gram matrix for style representation
    
    features: [B, C, H, W]
    returns: [B, C, C]
    """
    B, C, H, W = features.shape
    
    # Reshape to [B, C, H*W]
    features = features.view(B, C, -1)
    
    # Gram matrix: correlation between channels
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by size
    gram = gram / (C * H * W)
    
    return gram


class ContentLoss(nn.Module):
    """Content loss using feature matching"""
    
    def __init__(self, target_features):
        super().__init__()
        self.target = target_features.detach()
    
    def forward(self, input_features):
        return F.mse_loss(input_features, self.target)


class StyleLoss(nn.Module):
    """Style loss using Gram matrix matching"""
    
    def __init__(self, target_features):
        super().__init__()
        self.target_gram = gram_matrix(target_features).detach()
    
    def forward(self, input_features):
        input_gram = gram_matrix(input_features)
        return F.mse_loss(input_gram, self.target_gram)


class NeuralStyleTransfer:
    """Complete neural style transfer implementation"""
    
    def __init__(self, content_img, style_img, device='cuda'):
        self.device = device
        
        # Feature extractor
        self.vgg = VGGFeatures().to(device).eval()
        
        # Preprocess images
        self.content = self._preprocess(content_img).to(device)
        self.style = self._preprocess(style_img).to(device)
        
        # Extract target features
        content_features = self.vgg(self.content)
        style_features = self.vgg(self.style)
        
        # Content targets (deep layer)
        self.content_target = content_features['conv4_2'].detach()
        
        # Style targets (Gram matrices at multiple layers)
        self.style_targets = {}
        for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
            self.style_targets[layer] = gram_matrix(style_features[layer]).detach()
    
    def _preprocess(self, img):
        """Normalize for VGG"""
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(img).unsqueeze(0)
    
    def transfer(self, num_steps=300, content_weight=1, style_weight=1e6):
        """Run optimization-based style transfer"""
        
        # Initialize generated image (from content or noise)
        generated = self.content.clone().requires_grad_(True)
        
        # Optimizer (LBFGS works well for style transfer)
        optimizer = torch.optim.LBFGS([generated], lr=1.0)
        
        step = [0]
        
        while step[0] < num_steps:
            def closure():
                optimizer.zero_grad()
                
                # Extract features from generated image
                features = self.vgg(generated)
                
                # Content loss
                content_loss = F.mse_loss(
                    features['conv4_2'], 
                    self.content_target
                )
                
                # Style loss (multi-layer)
                style_loss = 0
                for layer in self.style_targets:
                    generated_gram = gram_matrix(features[layer])
                    style_loss += F.mse_loss(
                        generated_gram, 
                        self.style_targets[layer]
                    )
                style_loss /= len(self.style_targets)
                
                # Total loss
                total_loss = (content_weight * content_loss + 
                             style_weight * style_loss)
                
                total_loss.backward()
                
                step[0] += 1
                if step[0] % 50 == 0:
                    print(f"Step {step[0]}: Content={content_loss.item():.4f}, "
                          f"Style={style_loss.item():.4f}")
                
                return total_loss
            
            optimizer.step(closure)
        
        return generated.detach()


def style_transfer_simple(content_img, style_img, num_steps=300):
    """Simplified style transfer using Adam"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vgg = VGGFeatures().to(device).eval()
    
    # Get target features
    content_features = vgg(content_img)['conv4_2']
    style_features = vgg(style_img)
    style_grams = {k: gram_matrix(v) for k, v in style_features.items() 
                   if 'conv' in k}
    
    # Initialize from content
    generated = content_img.clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([generated], lr=0.01)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        gen_features = vgg(generated)
        
        # Content loss
        content_loss = F.mse_loss(gen_features['conv4_2'], content_features)
        
        # Style loss
        style_loss = 0
        for layer in style_grams:
            if layer in gen_features:
                gen_gram = gram_matrix(gen_features[layer])
                style_loss += F.mse_loss(gen_gram, style_grams[layer])
        
        # Total
        loss = content_loss + 1e6 * style_loss
        loss.backward()
        optimizer.step()
        
        # Clamp to valid range
        with torch.no_grad():
            generated.clamp_(-2.5, 2.5)
    
    return generated
```

**Why Gram Matrix Captures Style:**

| Property | Explanation |
|----------|-------------|
| Channel correlation | Which patterns co-occur |
| Location invariant | Ignores where patterns appear |
| Texture statistics | Captures brushstrokes, colors |
| Multi-scale | Different layers = different scales |

**Layer Roles:**

| Layer | Captures |
|-------|----------|
| conv1_1 | Colors, simple textures |
| conv2_1 | Small patterns |
| conv3_1 | Medium patterns |
| conv4_1 | Complex textures |
| conv5_1 | Large-scale style |

**Interview Tip:** The key insight is that style = correlations between features (Gram matrix), while content = the features themselves. Optimization-based transfer is slow (~30 seconds) but high quality. The content/style weight ratio (typically 1:1e6) balances structure preservation vs. style adoption.

---

### Question 24
- [ ] Done
**Compare optimization-based vs. feed-forward style transfer. What are the trade-offs?**

**Answer:**

Optimization-based style transfer iteratively updates pixels to minimize content+style loss (slow but flexible). Feed-forward uses a trained network that applies style in one pass (fast but fixed style). The trade-off is speed vs. flexibility.

**Comparison:**

| Aspect | Optimization-Based | Feed-Forward |
|--------|-------------------|--------------|
| Speed | ~30 seconds | ~50 milliseconds |
| Flexibility | Any style | Fixed style per model |
| Quality | Higher | Good but slightly lower |
| Memory | Low (just image) | High (store network) |
| Real-time | No | Yes |

**How Each Works:**

**Optimization-Based:**
```
Initialize: generated = content_image
For each iteration:
    features = VGG(generated)
    loss = content_loss + style_loss
    generated -= lr * gradient(loss)
```

**Feed-Forward:**
```
Train: network to minimize content_loss + style_loss
Inference: styled = network(content_image)  # Single forward pass
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardStyleTransfer(nn.Module):
    """Fast style transfer network (Johnson et al.)"""
    
    def __init__(self):
        super().__init__()
        
        # Encoder (downsample)
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, 9, 1),          # 256 -> 256
            ConvBlock(32, 64, 3, 2),          # 256 -> 128
            ConvBlock(64, 128, 3, 2)          # 128 -> 64
        )
        
        # Residual blocks (transform)
        self.residuals = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # Decoder (upsample)
        self.decoder = nn.Sequential(
            UpsampleBlock(128, 64),           # 64 -> 128
            UpsampleBlock(64, 32),            # 128 -> 256
            nn.Conv2d(32, 3, 9, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        return x


class ConvBlock(nn.Module):
    """Conv + InstanceNorm + ReLU"""
    
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Upsample + Conv (avoids checkerboard artifacts)"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class PerceptualLoss(nn.Module):
    """Loss for training feed-forward style transfer"""
    
    def __init__(self, style_img, content_weight=1.0, style_weight=1e5):
        super().__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # VGG features
        vgg = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg.children())[:23]).eval()
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Pre-compute style Gram matrices
        style_features = self._get_features(style_img)
        self.style_grams = {k: gram_matrix(v).detach() 
                          for k, v in style_features.items()}
    
    def _get_features(self, x):
        features = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in [3, 8, 15, 22]:  # relu1_2, relu2_2, relu3_3, relu4_3
                features[f'relu{i}'] = x
        return features
    
    def forward(self, generated, content):
        """
        generated: output of style network
        content: original content image
        """
        gen_features = self._get_features(generated)
        content_features = self._get_features(content)
        
        # Content loss (from deep layer)
        content_loss = F.mse_loss(
            gen_features['relu22'], 
            content_features['relu22']
        )
        
        # Style loss
        style_loss = 0
        for key in self.style_grams:
            gen_gram = gram_matrix(gen_features[key])
            style_loss += F.mse_loss(gen_gram, self.style_grams[key])
        style_loss /= len(self.style_grams)
        
        # Total variation loss (smoothness)
        tv_loss = self._total_variation(generated)
        
        total = (self.content_weight * content_loss + 
                 self.style_weight * style_loss +
                 1e-6 * tv_loss)
        
        return total
    
    def _total_variation(self, x):
        """Encourage spatial smoothness"""
        return (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
                torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))


def train_feed_forward_model(style_img, train_loader, epochs=2):
    """Train a fast style transfer model"""
    
    device = 'cuda'
    
    model = FeedForwardStyleTransfer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    criterion = PerceptualLoss(style_img.to(device))
    
    for epoch in range(epochs):
        for batch_idx, (content_imgs, _) in enumerate(train_loader):
            content_imgs = content_imgs.to(device)
            
            # Generate styled images
            styled = model(content_imgs)
            
            # Compute perceptual loss
            loss = criterion(styled, content_imgs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return model


def compare_methods():
    """Speed and quality comparison"""
    
    print("Style Transfer Method Comparison:")
    print("=" * 60)
    print(f"{'Method':^20} | {'Time':^12} | {'Flexibility':^12} | {'Quality':^10}")
    print("-" * 60)
    print(f"{'Optimization':^20} | {'30-60 sec':^12} | {'Any style':^12} | {'Best':^10}")
    print(f"{'Feed-forward':^20} | {'50 ms':^12} | {'1 style':^12} | {'Good':^10}")
    print(f"{'AdaIN (arbitrary)':^20} | {'100 ms':^12} | {'Any style':^12} | {'Good':^10}")
    print(f"{'MSG-Net':^20} | {'50 ms':^12} | {'Multiple':^12} | {'Good':^10}")
```

**Evolution of Methods:**

| Year | Method | Innovation |
|------|--------|------------|
| 2015 | Gatys | Optimization-based (original) |
| 2016 | Johnson | Feed-forward per-style |
| 2017 | AdaIN | Arbitrary style, feed-forward |
| 2017 | WCT | Whitening/coloring transform |

**When to Use Each:**

| Use Case | Recommended |
|----------|-------------|
| Real-time video | Feed-forward |
| Mobile app (single style) | Feed-forward |
| Arbitrary styles needed | AdaIN |
| Highest quality | Optimization |
| Interactive exploration | Optimization |

**Interview Tip:** Feed-forward networks "distill" the optimization process into a single forward pass. The network learns to predict what the optimization would produce. Training uses the same perceptual losses as optimization. The key innovation was realizing that the optimization result is deterministic given content image + style, so it can be learned.

---

### Question 25
- [ ] Done
**How do you implement real-time arbitrary style transfer (AdaIN, WCT)?**

**Answer:**

AdaIN (Adaptive Instance Normalization) enables real-time arbitrary style transfer by aligning content feature statistics (mean, variance) to style feature statistics. WCT (Whitening and Coloring Transform) uses full covariance matching for richer style transfer. Both work in a single forward pass.

**AdaIN Formula:**

$$\text{AdaIN}(x, y) = \sigma(y) \left(\frac{x - \mu(x)}{\sigma(x)}\right) + \mu(y)$$

Where $x$ = content features, $y$ = style features.

**Key Insight:** Style = feature statistics. Transfer style by transferring mean and variance.

**Comparison:**

| Method | Statistics Matched | Speed | Quality |
|--------|-------------------|-------|---------|
| AdaIN | Mean + Variance | ~100ms | Good |
| WCT | Full covariance | ~500ms | Better |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    
    def forward(self, content_feat, style_feat):
        """
        content_feat: [B, C, H, W] - features from content image
        style_feat: [B, C, H', W'] - features from style image
        """
        # Content statistics
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        content_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-8
        
        # Style statistics
        style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        style_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-8
        
        # Normalize content, then apply style statistics
        normalized = (content_feat - content_mean) / content_std
        stylized = normalized * style_std + style_mean
        
        return stylized


class AdaINStyleTransfer(nn.Module):
    """Complete AdaIN-based style transfer model"""
    
    def __init__(self):
        super().__init__()
        
        # VGG encoder (up to relu4_1)
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # AdaIN layer
        self.adain = AdaIN()
        
        # Decoder (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3)
        )
    
    def forward(self, content, style, alpha=1.0):
        """
        content: content image [B, 3, H, W]
        style: style image [B, 3, H, W]
        alpha: style strength (0 = content, 1 = full style)
        """
        # Encode
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        
        # Apply AdaIN
        stylized_feat = self.adain(content_feat, style_feat)
        
        # Interpolate for style strength control
        if alpha < 1.0:
            stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat
        
        # Decode
        output = self.decoder(stylized_feat)
        
        return output


class WCT(nn.Module):
    """Whitening and Coloring Transform for style transfer"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, content_feat, style_feat, alpha=1.0):
        """
        Full covariance matching for richer style transfer
        """
        B, C, H, W = content_feat.shape
        
        # Reshape to [B, C, HW]
        content_flat = content_feat.view(B, C, -1)
        style_flat = style_feat.view(B, C, -1)
        
        stylized = []
        for b in range(B):
            cf = content_flat[b]  # [C, HW]
            sf = style_flat[b]
            
            # Whiten content features
            whitened = self._whiten(cf)
            
            # Color with style statistics
            colored = self._color(whitened, sf)
            
            # Blend with original
            if alpha < 1.0:
                colored = alpha * colored + (1 - alpha) * cf
            
            stylized.append(colored)
        
        stylized = torch.stack(stylized)
        return stylized.view(B, C, H, W)
    
    def _whiten(self, feat):
        """Remove correlations (whiten to identity covariance)"""
        # feat: [C, HW]
        mean = feat.mean(dim=1, keepdim=True)
        feat_centered = feat - mean
        
        # Covariance
        cov = feat_centered @ feat_centered.T / (feat.size(1) - 1)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), device=feat.device))
        
        # Whitening matrix: D^(-1/2) @ E^T
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues + 1e-5))
        whiten_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T
        
        whitened = whiten_matrix @ feat_centered
        
        return whitened
    
    def _color(self, whitened, style_feat):
        """Apply style covariance and mean"""
        # style_feat: [C, HW]
        style_mean = style_feat.mean(dim=1, keepdim=True)
        style_centered = style_feat - style_mean
        
        # Style covariance
        cov = style_centered @ style_centered.T / (style_feat.size(1) - 1)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), device=style_feat.device))
        
        # Coloring matrix: E @ D^(1/2) @ E^T
        D_sqrt = torch.diag(torch.sqrt(eigenvalues + 1e-5))
        color_matrix = eigenvectors @ D_sqrt @ eigenvectors.T
        
        colored = color_matrix @ whitened + style_mean
        
        return colored


class MultiLevelWCT(nn.Module):
    """Multi-level WCT for progressive style transfer"""
    
    def __init__(self):
        super().__init__()
        
        # Multiple VGG encoders/decoders at different levels
        vgg = models.vgg19(pretrained=True).features
        
        # Level 5: relu5_1
        self.encoder5 = nn.Sequential(*list(vgg.children())[:30])
        self.decoder5 = self._make_decoder(512, 512)
        
        # Level 4: relu4_1
        self.encoder4 = nn.Sequential(*list(vgg.children())[:21])
        self.decoder4 = self._make_decoder(512, 256)
        
        # Level 3: relu3_1
        self.encoder3 = nn.Sequential(*list(vgg.children())[:12])
        self.decoder3 = self._make_decoder(256, 128)
        
        # Level 2: relu2_1
        self.encoder2 = nn.Sequential(*list(vgg.children())[:7])
        self.decoder2 = self._make_decoder(128, 64)
        
        # Level 1: relu1_1
        self.encoder1 = nn.Sequential(*list(vgg.children())[:2])
        self.decoder1 = self._make_decoder(64, 3)
        
        self.wct = WCT()
    
    def _make_decoder(self, in_ch, out_ch):
        """Simple decoder block"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest') if in_ch != out_ch else nn.Identity()
        )
    
    def forward(self, content, style, alpha=1.0):
        """Progressive coarse-to-fine style transfer"""
        
        # Extract features at all levels
        c5 = self.encoder5(content)
        s5 = self.encoder5(style)
        
        # Apply WCT at deepest level
        cs5 = self.wct(c5, s5, alpha)
        
        # Decode and repeat at each level
        out = self.decoder5(cs5)
        
        # Continue at levels 4, 3, 2, 1...
        # (simplified for brevity)
        
        return out


def real_time_style_transfer_demo():
    """Demonstrate real-time capability"""
    import time
    
    model = AdaINStyleTransfer()
    model.eval()
    
    # Dummy inputs
    content = torch.randn(1, 3, 512, 512)
    style = torch.randn(1, 3, 512, 512)
    
    # Warmup
    with torch.no_grad():
        _ = model(content, style)
    
    # Benchmark
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model(content, style)
        times.append(time.time() - start)
    
    print(f"Average inference time: {sum(times)/len(times)*1000:.1f} ms")
    print(f"FPS: {1.0 / (sum(times)/len(times)):.1f}")
```

**Comparison:**

| Aspect | AdaIN | WCT |
|--------|-------|-----|
| Statistics | Mean + Std | Full covariance |
| Speed | Faster | Slower |
| Style richness | Good | Better |
| Implementation | Simple | Complex |

**Interview Tip:** AdaIN's simplicity is its strength—matching mean and variance is fast and often sufficient. WCT captures more style by matching full covariance but requires eigendecomposition which is slower. For real-time applications, AdaIN is preferred. The alpha parameter enables smooth interpolation between content and fully stylized output.

---

### Question 26
- [ ] Done
**What techniques maintain temporal consistency in video style transfer?**

**Answer:**

Video style transfer must ensure consistency across frames to avoid flickering. Techniques include: (1) optical flow warping to propagate styles, (2) temporal losses penalizing frame differences, (3) recurrent networks with memory, and (4) feature-level consistency constraints.

**The Problem:**

Per-frame style transfer causes:
- Flickering (style changes frame-to-frame)
- Temporal artifacts (inconsistent textures)
- Jittery motion

**Solutions:**

| Technique | How It Works | Pros/Cons |
|-----------|--------------|-----------|
| Optical flow | Warp previous styled frame | Accurate but slow |
| Temporal loss | Penalize frame differences | Simple, may blur motion |
| Recurrent | Hidden state carries info | Elegant, training complex |
| Feature flow | Warp features, not pixels | Better for large motion |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConsistencyLoss(nn.Module):
    """Loss for temporal consistency in video style transfer"""
    
    def __init__(self, lambda_temporal=1.0):
        super().__init__()
        self.lambda_temporal = lambda_temporal
    
    def forward(self, current_output, previous_output, flow):
        """
        current_output: styled frame t [B, 3, H, W]
        previous_output: styled frame t-1 [B, 3, H, W]
        flow: optical flow from t-1 to t [B, 2, H, W]
        """
        # Warp previous output to current frame using flow
        warped_previous = self._warp(previous_output, flow)
        
        # Compute consistency loss (should match where flow is valid)
        consistency_loss = F.l1_loss(current_output, warped_previous)
        
        return self.lambda_temporal * consistency_loss
    
    def _warp(self, img, flow):
        """
        Backward warp image using optical flow
        """
        B, C, H, W = img.shape
        
        # Create base grid
        xx = torch.linspace(-1, 1, W, device=img.device)
        yy = torch.linspace(-1, 1, H, device=img.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        # Normalize flow to [-1, 1]
        flow_norm = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        flow_norm[..., 0] = flow_norm[..., 0] / (W / 2)
        flow_norm[..., 1] = flow_norm[..., 1] / (H / 2)
        
        # Add flow to base grid
        sample_grid = base_grid + flow_norm
        
        # Warp
        warped = F.grid_sample(img, sample_grid, mode='bilinear', 
                              padding_mode='border', align_corners=True)
        
        return warped


class OpticalFlowEstimator(nn.Module):
    """Lightweight optical flow for style transfer"""
    
    def __init__(self):
        super().__init__()
        
        # Simple flow network (or use pretrained RAFT)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 7, padding=3),  # Two frames concatenated
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 7, padding=3)  # 2D flow
        )
    
    def forward(self, frame1, frame2):
        """Estimate flow from frame1 to frame2"""
        concat = torch.cat([frame1, frame2], dim=1)
        features = self.encoder(concat)
        flow = self.decoder(features)
        return flow


class VideoStyleTransfer(nn.Module):
    """Video style transfer with temporal consistency"""
    
    def __init__(self, style_model, use_flow=True):
        super().__init__()
        
        self.style_model = style_model
        self.use_flow = use_flow
        
        if use_flow:
            self.flow_estimator = OpticalFlowEstimator()
            self.temporal_loss = TemporalConsistencyLoss()
        
        # Previous frame storage
        self.prev_styled = None
        self.prev_frame = None
    
    def forward(self, frame, style_img):
        """Process single video frame"""
        
        # Apply style transfer
        styled = self.style_model(frame, style_img)
        
        if self.use_flow and self.prev_styled is not None:
            # Estimate optical flow
            flow = self.flow_estimator(self.prev_frame, frame)
            
            # Warp previous styled frame
            warped_prev = self._warp(self.prev_styled, flow)
            
            # Blend with current styled (temporal smoothing)
            alpha = 0.8  # How much to trust current vs. warped previous
            styled = alpha * styled + (1 - alpha) * warped_prev
        
        # Store for next frame
        self.prev_styled = styled.detach()
        self.prev_frame = frame.detach()
        
        return styled
    
    def _warp(self, img, flow):
        """Warp using optical flow"""
        # Same as in TemporalConsistencyLoss
        B, C, H, W = img.shape
        
        xx = torch.linspace(-1, 1, W, device=img.device)
        yy = torch.linspace(-1, 1, H, device=img.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        flow_norm = flow.permute(0, 2, 3, 1)
        flow_norm[..., 0] /= (W / 2)
        flow_norm[..., 1] /= (H / 2)
        
        sample_grid = base_grid + flow_norm
        
        return F.grid_sample(img, sample_grid, mode='bilinear', 
                            padding_mode='border', align_corners=True)
    
    def reset(self):
        """Reset state for new video"""
        self.prev_styled = None
        self.prev_frame = None


class RecurrentVideoStyleTransfer(nn.Module):
    """Use recurrent connections for temporal consistency"""
    
    def __init__(self, base_model, hidden_channels=64):
        super().__init__()
        
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        
        # Recurrent connections
        self.rnn = nn.ConvLSTM2d(
            input_channels=512,  # Feature channels
            hidden_channels=hidden_channels,
            kernel_size=3
        )
        
        self.hidden = None
    
    def forward(self, frame, style_img):
        # Encode
        features = self.encoder(frame)
        style_features = self.encoder(style_img)
        
        # Apply style (e.g., AdaIN)
        styled_features = self._apply_style(features, style_features)
        
        # Recurrent temporal modeling
        rnn_out, self.hidden = self.rnn(styled_features.unsqueeze(0), self.hidden)
        styled_features = rnn_out.squeeze(0)
        
        # Decode
        output = self.decoder(styled_features)
        
        return output
    
    def _apply_style(self, content, style):
        # AdaIN
        c_mean = content.mean(dim=[2, 3], keepdim=True)
        c_std = content.std(dim=[2, 3], keepdim=True) + 1e-8
        s_mean = style.mean(dim=[2, 3], keepdim=True)
        s_std = style.std(dim=[2, 3], keepdim=True) + 1e-8
        
        return (content - c_mean) / c_std * s_std + s_mean
    
    def reset(self):
        self.hidden = None


class FeatureLevelConsistency(nn.Module):
    """Consistency at feature level rather than pixel level"""
    
    def __init__(self):
        super().__init__()
        
        # VGG for feature extraction
        vgg = models.vgg19(pretrained=True).features[:12]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
    
    def forward(self, current, previous, flow):
        """
        Feature-space consistency loss
        More robust to lighting/color changes
        """
        # Extract features
        current_feat = self.vgg(current)
        previous_feat = self.vgg(previous)
        
        # Downsample flow to match feature resolution
        H, W = current_feat.shape[2:]
        flow_small = F.interpolate(flow, size=(H, W), mode='bilinear')
        flow_small = flow_small * (H / flow.shape[2])  # Scale flow values
        
        # Warp previous features
        warped_feat = self._warp_features(previous_feat, flow_small)
        
        # Feature consistency loss
        loss = F.l1_loss(current_feat, warped_feat)
        
        return loss


def training_with_temporal_loss(model, video_loader, style_img, epochs=10):
    """Train video style transfer with temporal consistency"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    style_loss_fn = StyleLoss()  # Gram matrix loss
    content_loss_fn = ContentLoss()
    temporal_loss_fn = TemporalConsistencyLoss(lambda_temporal=10.0)
    
    for epoch in range(epochs):
        for video_batch in video_loader:
            # video_batch: [B, T, C, H, W] - batch of video clips
            B, T, C, H, W = video_batch.shape
            
            total_loss = 0
            prev_styled = None
            
            for t in range(T):
                frame = video_batch[:, t]
                
                # Apply style
                styled = model(frame, style_img)
                
                # Style loss
                total_loss += style_loss_fn(styled, style_img)
                
                # Content loss
                total_loss += content_loss_fn(styled, frame)
                
                # Temporal loss (except first frame)
                if prev_styled is not None:
                    flow = compute_optical_flow(video_batch[:, t-1], frame)
                    total_loss += temporal_loss_fn(styled, prev_styled, flow)
                
                prev_styled = styled
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

**Method Comparison:**

| Approach | Flickering | Speed | Motion Handling |
|----------|-----------|-------|-----------------|
| Per-frame | Severe | Fast | N/A |
| Optical flow | Minimal | Medium | Good |
| Recurrent | Low | Fast | Moderate |
| Feature flow | Low | Medium | Better |

**Interview Tip:** Optical flow warping is the most reliable technique—it explicitly models where pixels move between frames. The temporal loss ensures styled regions follow this motion. For real-time, simpler approaches like blending with warped previous frame (0.8:0.2 ratio) work well. Occlusion handling is the main challenge—use occlusion masks to avoid warping artifacts.

---

### Question 27
- [ ] Done
**How do you balance content preservation vs. style adoption? What controls this trade-off?**

**Answer:**

The content-style balance is controlled by: (1) loss weight ratio ($\alpha/\beta$), (2) style strength parameter in AdaIN, (3) layer selection for losses, and (4) multi-scale blending. Higher content weight preserves structure; higher style weight adopts more artistic style.

**Control Parameters:**

| Parameter | Effect of Increasing |
|-----------|---------------------|
| Content weight ($\alpha$) | More structure preserved |
| Style weight ($\beta$) | Stronger style patterns |
| Deeper content layer | More abstract preservation |
| More style layers | Richer style capture |
| AdaIN alpha | More stylization |

**Mathematical Trade-off:**

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{content} + \beta \cdot \mathcal{L}_{style}$$

Typical ratios:
- Light stylization: $\alpha/\beta = 1:10^4$
- Balanced: $\alpha/\beta = 1:10^5$
- Strong stylization: $\alpha/\beta = 1:10^6$

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedStyleTransfer:
    """Style transfer with configurable content-style balance"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.vgg = VGGFeatures().to(device).eval()
    
    def transfer(self, content_img, style_img, 
                 content_weight=1.0,
                 style_weight=1e6,
                 content_layers=['conv4_2'],
                 style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                 num_steps=300):
        """
        Configurable style transfer
        
        content_weight: Weight for content preservation
        style_weight: Weight for style adoption
        content_layers: Deeper = more abstract preservation
        style_layers: More layers = richer style
        """
        # Extract target features
        content_features = self.vgg(content_img)
        style_features = self.vgg(style_img)
        
        # Initialize generated image
        generated = content_img.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([generated], lr=0.01)
        
        for step in range(num_steps):
            gen_features = self.vgg(generated)
            
            # Content loss
            content_loss = 0
            for layer in content_layers:
                content_loss += F.mse_loss(
                    gen_features[layer], 
                    content_features[layer]
                )
            content_loss /= len(content_layers)
            
            # Style loss
            style_loss = 0
            for layer in style_layers:
                gen_gram = self._gram_matrix(gen_features[layer])
                style_gram = self._gram_matrix(style_features[layer])
                style_loss += F.mse_loss(gen_gram, style_gram)
            style_loss /= len(style_layers)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                generated.clamp_(-2.5, 2.5)
        
        return generated.detach()
    
    def _gram_matrix(self, features):
        B, C, H, W = features.shape
        features = features.view(B, C, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)


class AdaptiveBalancing:
    """Automatic or region-based balancing"""
    
    def __init__(self):
        pass
    
    def content_aware_weights(self, content_img, mask=None):
        """
        Different weights for different regions
        e.g., preserve faces more, stylize background more
        """
        if mask is not None:
            # Higher content weight in masked regions
            content_weights = torch.where(mask > 0.5, 10.0, 1.0)
            style_weights = torch.where(mask > 0.5, 0.1, 1.0)
        else:
            # Default uniform weights
            content_weights = torch.ones_like(content_img[:, 0:1])
            style_weights = torch.ones_like(content_img[:, 0:1])
        
        return content_weights, style_weights
    
    def semantic_balancing(self, content_img, style_img, segmentation):
        """
        Different styles for different semantic regions
        """
        # Example: preserve sky, stylize buildings more
        sky_mask = (segmentation == 0)  # Assuming 0 = sky
        building_mask = (segmentation == 1)
        
        # Higher content preservation for sky
        region_weights = {
            'sky': {'content': 10.0, 'style': 0.1},
            'building': {'content': 0.5, 'style': 2.0},
            'default': {'content': 1.0, 'style': 1.0}
        }
        
        return region_weights


class MultiScaleBlending:
    """Blend stylization at multiple scales for better control"""
    
    def __init__(self, style_model):
        self.style_model = style_model
    
    def blend(self, content, style, coarse_alpha=0.3, fine_alpha=0.8):
        """
        Different style strength at different frequencies
        
        coarse_alpha: Style strength for low frequencies (structure)
        fine_alpha: Style strength for high frequencies (textures)
        """
        # Apply style at full strength
        fully_styled = self.style_model(content, style, alpha=1.0)
        
        # Separate into frequency bands (using Gaussian pyramid)
        content_low = self._low_pass(content)
        content_high = content - content_low
        
        styled_low = self._low_pass(fully_styled)
        styled_high = fully_styled - styled_low
        
        # Blend each frequency band separately
        blended_low = coarse_alpha * styled_low + (1 - coarse_alpha) * content_low
        blended_high = fine_alpha * styled_high + (1 - fine_alpha) * content_high
        
        return blended_low + blended_high
    
    def _low_pass(self, img, sigma=3):
        """Gaussian low-pass filter"""
        kernel_size = int(6 * sigma) | 1
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size) - kernel_size // 2
        kernel = torch.exp(-x.float() ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        # Separable convolution
        kernel_h = kernel.view(1, 1, 1, -1).to(img.device)
        kernel_v = kernel.view(1, 1, -1, 1).to(img.device)
        
        # Apply per channel
        B, C, H, W = img.shape
        img_pad = F.pad(img, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
        
        filtered = []
        for c in range(C):
            ch = img_pad[:, c:c+1]
            ch = F.conv2d(ch, kernel_h)
            ch = F.conv2d(ch, kernel_v)
            filtered.append(ch)
        
        return torch.cat(filtered, dim=1)


class LayerSelection:
    """Study effect of layer selection on content-style trade-off"""
    
    @staticmethod
    def analyze_content_layers():
        """
        Content layer selection affects abstraction level
        """
        layer_effects = {
            'conv1_2': 'Preserves pixel-level details, colors',
            'conv2_2': 'Preserves textures, small patterns',
            'conv3_2': 'Preserves edges, medium features',
            'conv4_2': 'Preserves objects, semantic layout (DEFAULT)',
            'conv5_2': 'Preserves only high-level structure'
        }
        
        return layer_effects
    
    @staticmethod
    def analyze_style_layers():
        """
        Style layer selection affects style richness
        """
        layer_effects = {
            'conv1_1 only': 'Colors only',
            'conv1-2': 'Colors + simple textures',
            'conv1-3': 'Colors + medium patterns',
            'conv1-4': 'Rich textures (common choice)',
            'conv1-5': 'Full style including global patterns'
        }
        
        return layer_effects


def interactive_balancing_demo():
    """Demonstrate different balance settings"""
    
    settings = [
        {'content': 1.0, 'style': 1e4, 'name': 'Subtle style'},
        {'content': 1.0, 'style': 1e5, 'name': 'Balanced'},
        {'content': 1.0, 'style': 1e6, 'name': 'Strong style'},
        {'content': 1.0, 'style': 1e7, 'name': 'Very artistic'},
    ]
    
    print("Content-Style Balance Settings:")
    print("=" * 50)
    for s in settings:
        ratio = s['style'] / s['content']
        print(f"{s['name']:15} | Content: {s['content']:5.1f} | Style: {s['style']:.0e} | Ratio: 1:{ratio:.0e}")


def layer_experiment():
    """Show effect of layer selection"""
    
    print("\nContent Layer Effects:")
    print("-" * 40)
    for layer, effect in LayerSelection.analyze_content_layers().items():
        print(f"{layer}: {effect}")
    
    print("\nStyle Layer Effects:")
    print("-" * 40)
    for layer, effect in LayerSelection.analyze_style_layers().items():
        print(f"{layer}: {effect}")
```

**Practical Guidelines:**

| Goal | Content Weight | Style Weight | Content Layer |
|------|---------------|--------------|---------------|
| Slight filter | 1.0 | 1e4 | conv4_2 |
| Artistic but recognizable | 1.0 | 1e5 | conv4_2 |
| Heavy artistic | 1.0 | 1e6 | conv5_2 |
| Preserve structure | 10.0 | 1e5 | conv3_2 |

**Interview Tip:** The ratio matters more than absolute values. conv4_2 is the standard content layer because it captures semantic structure without pixel-level detail. For faces or detailed preservation, use earlier layers. AdaIN's alpha parameter (0-1) is the simplest way to control balance in feed-forward models—it directly interpolates between content and stylized features.

---

## 3D Reconstruction (NeRF, Gaussian Splatting)

### Question 28
- [ ] Done
**Explain NeRF's core idea: representing scenes as neural radiance fields with MLPs.**

**Answer:**

NeRF (Neural Radiance Field) represents a 3D scene as a continuous function that maps 5D input (3D position + 2D viewing direction) to color and density. An MLP learns this mapping from posed 2D images, enabling novel view synthesis by rendering from any viewpoint.

**Core Concept:**

$$F_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

Where:
- $\mathbf{x} = (x, y, z)$: 3D position
- $\mathbf{d} = (\theta, \phi)$: viewing direction
- $\mathbf{c} = (r, g, b)$: color (view-dependent)
- $\sigma$: volume density (view-independent)

**Architecture Design:**

| Component | Purpose |
|-----------|---------|
| Position input | Determine where in 3D space |
| Direction input | Capture view-dependent effects (specular) |
| Density output | Scene geometry (solid vs empty) |
| Color output | Appearance at that point from that angle |

```
Position (x,y,z) → [Positional Encoding] → MLP (8 layers) → σ (density)
                                                           ↓
                                                    + Direction (θ,φ) → MLP (1 layer) → RGB
```

**Why View Direction Matters:**
- Lambertian (matte) surfaces: color independent of view
- Specular surfaces: color changes with viewing angle (highlights)
- Direction only affects color, not density

**Python Implementation:**
```python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Fourier feature encoding for high-frequency details"""
    
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
    
    def forward(self, x):
        """
        x: (batch, 3) for position or (batch, 3) for direction
        output: (batch, 3 + 3 * 2 * num_freqs) if include_input
        """
        encoded = []
        
        if self.include_input:
            encoded.append(x)
        
        for freq in self.freq_bands.to(x.device):
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        
        return torch.cat(encoded, dim=-1)
    
    @property
    def output_dim(self):
        dim = 3 * 2 * self.num_freqs
        if self.include_input:
            dim += 3
        return dim


class NeRFMLP(nn.Module):
    """Original NeRF architecture"""
    
    def __init__(self, pos_freqs=10, dir_freqs=4, hidden_dim=256):
        super().__init__()
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(pos_freqs)
        self.dir_encoder = PositionalEncoding(dir_freqs)
        
        pos_dim = self.pos_encoder.output_dim
        dir_dim = self.dir_encoder.output_dim
        
        # First half of network (processes position)
        self.layers1 = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Skip connection at layer 5
        self.layers2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim),  # Skip connection
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Density output (view-independent)
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Feature for color
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        
        # Color MLP (view-dependent)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  # RGB in [0, 1]
        )
    
    def forward(self, positions, directions):
        """
        positions: (batch, 3) xyz coordinates
        directions: (batch, 3) viewing directions (normalized)
        
        Returns:
            rgb: (batch, 3) color
            sigma: (batch, 1) density
        """
        # Encode inputs
        pos_enc = self.pos_encoder(positions)
        dir_enc = self.dir_encoder(directions)
        
        # Process position
        h = self.layers1(pos_enc)
        h = self.layers2(torch.cat([h, pos_enc], dim=-1))  # Skip connection
        
        # Density (view-independent)
        sigma = self.density_head(h)
        sigma = torch.relu(sigma)  # Density must be non-negative
        
        # Features for color
        features = self.feature_head(h)
        
        # Color (view-dependent)
        rgb = self.color_head(torch.cat([features, dir_enc], dim=-1))
        
        return rgb, sigma


class NeRF:
    """Complete NeRF with coarse-fine sampling"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Two networks: coarse and fine
        self.coarse_net = NeRFMLP().to(device)
        self.fine_net = NeRFMLP().to(device)
    
    def render_rays(self, ray_origins, ray_directions, 
                    near=2.0, far=6.0, num_coarse=64, num_fine=128):
        """
        Render colors for a batch of rays
        """
        # Coarse sampling
        t_coarse = self.stratified_sampling(near, far, num_coarse)
        points_coarse = ray_origins[:, None] + ray_directions[:, None] * t_coarse[:, :, None]
        
        # Query coarse network
        rgb_coarse, sigma_coarse = self.query_network(
            self.coarse_net, points_coarse, ray_directions
        )
        
        # Volume rendering (coarse)
        colors_coarse, weights_coarse = self.volume_render(
            rgb_coarse, sigma_coarse, t_coarse
        )
        
        # Importance sampling for fine network
        t_fine = self.importance_sampling(t_coarse, weights_coarse, num_fine)
        t_combined = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1)[0]
        
        points_fine = ray_origins[:, None] + ray_directions[:, None] * t_combined[:, :, None]
        
        # Query fine network
        rgb_fine, sigma_fine = self.query_network(
            self.fine_net, points_fine, ray_directions
        )
        
        # Volume rendering (fine)
        colors_fine, _ = self.volume_render(rgb_fine, sigma_fine, t_combined)
        
        return colors_coarse, colors_fine
    
    def query_network(self, network, points, directions):
        """Query network at sampled points"""
        batch_size, num_samples, _ = points.shape
        
        # Flatten for network
        points_flat = points.reshape(-1, 3)
        directions_flat = directions[:, None].expand(-1, num_samples, -1).reshape(-1, 3)
        
        rgb, sigma = network(points_flat, directions_flat)
        
        # Reshape back
        rgb = rgb.reshape(batch_size, num_samples, 3)
        sigma = sigma.reshape(batch_size, num_samples)
        
        return rgb, sigma
    
    def stratified_sampling(self, near, far, num_samples):
        """Uniform sampling with stratification"""
        t = torch.linspace(near, far, num_samples + 1, device=self.device)[:-1]
        # Add random offset within each bin
        t = t + torch.rand(num_samples, device=self.device) * (far - near) / num_samples
        return t.unsqueeze(0)  # (1, num_samples)
    
    def importance_sampling(self, t_vals, weights, num_samples):
        """Sample more densely where weights are high"""
        # Normalize weights to PDF
        weights = weights + 1e-5
        pdf = weights / weights.sum(dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        
        # Inverse CDF sampling
        u = torch.rand(weights.shape[0], num_samples, device=self.device)
        indices = torch.searchsorted(cdf, u)
        
        # Get t values
        t_samples = torch.gather(t_vals, 1, indices.clamp(0, t_vals.shape[1] - 1))
        
        return t_samples
    
    def volume_render(self, rgb, sigma, t_vals):
        """Integrate colors along rays (see Q29 for details)"""
        # Distances between samples
        deltas = t_vals[:, 1:] - t_vals[:, :-1]
        deltas = torch.cat([deltas, torch.ones_like(deltas[:, :1]) * 1e10], dim=-1)
        
        # Alpha = 1 - exp(-sigma * delta)
        alpha = 1 - torch.exp(-sigma * deltas)
        
        # Transmittance
        T = torch.cumprod(1 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=-1)
        
        # Weights
        weights = alpha * T
        
        # Final color
        rgb_final = (weights.unsqueeze(-1) * rgb).sum(dim=1)
        
        return rgb_final, weights


def train_nerf():
    """Training loop overview"""
    # Inputs: images + camera poses
    # For each iteration:
    # 1. Sample random rays from training images
    # 2. Render rays through NeRF
    # 3. Compare to ground truth pixel colors
    # 4. Backprop MSE loss
    
    loss_fn = nn.MSELoss()
    # loss = loss_fn(rendered_colors, gt_colors)
```

**Key Design Decisions:**

| Choice | Reason |
|--------|--------|
| 8-layer MLP | Balance capacity vs training time |
| Skip connection at layer 5 | Better gradient flow |
| Density before direction | Geometry is view-independent |
| Positional encoding | MLPs struggle with high frequencies |

**Training Data:**
- 50-150 posed images (camera intrinsics + extrinsics)
- Known camera parameters (from COLMAP or similar)
- Training time: ~1-2 days on single GPU

**Interview Tip:** NeRF's key insight is that a simple MLP can represent arbitrarily complex scenes if given positional encodings. The 5D input (position + direction) enables view-dependent effects like reflections. The two-stage coarse-to-fine sampling focuses computation where geometry exists.

---

### Question 29
- [ ] Done
**Describe volumetric rendering in NeRF. How are colors and densities integrated along rays?**

**Answer:**

Volumetric rendering accumulates color and density along camera rays using the rendering equation. Each ray is sampled at multiple points, and colors are weighted by transmittance (how much light reaches each point) and alpha (opacity at that point). The final pixel color is the weighted sum of all sample contributions.

**Rendering Equation:**

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

Where:
- $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$: transmittance (probability ray travels without hitting anything)
- $\sigma$: volume density
- $\mathbf{c}$: color at point
- $t_n, t_f$: near and far bounds

**Discrete Approximation:**

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot \mathbf{c}_i$$

Where:
- $\alpha_i = 1 - \exp(-\sigma_i \cdot \delta_i)$: opacity at sample $i$
- $\delta_i = t_{i+1} - t_i$: distance between samples
- $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$: accumulated transmittance

**Intuition:**

| Term | Physical Meaning |
|------|------------------|
| $T_i$ | How much light can reach sample $i$ |
| $\alpha_i$ | How much light is absorbed at $i$ |
| $T_i \cdot \alpha_i$ | Weight of sample $i$ in final color |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VolumetricRenderer:
    """NeRF volumetric rendering implementation"""
    
    def __init__(self, near=2.0, far=6.0, num_samples=64):
        self.near = near
        self.far = far
        self.num_samples = num_samples
    
    def sample_along_rays(self, ray_origins, ray_directions, device='cuda'):
        """
        Generate sample points along rays
        
        ray_origins: (batch, 3)
        ray_directions: (batch, 3)
        
        Returns:
            points: (batch, num_samples, 3)
            t_vals: (batch, num_samples)
        """
        batch_size = ray_origins.shape[0]
        
        # Stratified sampling: divide [near, far] into bins
        t_vals = torch.linspace(
            self.near, self.far, self.num_samples, device=device
        )
        t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)  # (batch, samples)
        
        # Add noise within each bin (stratification)
        bin_size = (self.far - self.near) / self.num_samples
        noise = torch.rand_like(t_vals) * bin_size
        t_vals = t_vals + noise
        
        # Compute 3D points: origin + t * direction
        points = ray_origins[:, None, :] + t_vals[:, :, None] * ray_directions[:, None, :]
        
        return points, t_vals
    
    def render(self, rgb, sigma, t_vals, white_background=False):
        """
        Volumetric rendering integral
        
        rgb: (batch, num_samples, 3) - color at each sample
        sigma: (batch, num_samples) - density at each sample
        t_vals: (batch, num_samples) - depth values
        
        Returns:
            color: (batch, 3) - final pixel color
            depth: (batch,) - expected depth
            weights: (batch, num_samples) - sample weights for importance sampling
        """
        # Compute distances between samples
        deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (batch, samples-1)
        # Last delta is infinity (ray extends forever)
        delta_inf = torch.full((deltas.shape[0], 1), 1e10, device=deltas.device)
        deltas = torch.cat([deltas, delta_inf], dim=-1)  # (batch, samples)
        
        # Compute alpha (opacity) from density
        # alpha = 1 - exp(-sigma * delta)
        alpha = 1.0 - torch.exp(-sigma * deltas)
        
        # Compute transmittance T
        # T_i = prod(1 - alpha_j) for j < i
        # Using cumulative product with shift
        T = self._compute_transmittance(alpha)
        
        # Compute weights
        weights = T * alpha  # (batch, samples)
        
        # Weighted sum of colors
        color = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # (batch, 3)
        
        # Expected depth
        depth = (weights * t_vals).sum(dim=1)  # (batch,)
        
        # Accumulated opacity (for alpha compositing)
        acc = weights.sum(dim=1)  # (batch,)
        
        # Optional white background
        if white_background:
            color = color + (1.0 - acc.unsqueeze(-1))
        
        return color, depth, weights, acc
    
    def _compute_transmittance(self, alpha):
        """
        Compute transmittance from alpha values
        T_i = prod_{j<i}(1 - alpha_j)
        """
        # Shift alpha: T_i doesn't include alpha_i
        one_minus_alpha = 1.0 - alpha + 1e-10  # Add epsilon for stability
        
        # Cumulative product, shifted by 1
        # T_1 = 1, T_2 = (1-a_1), T_3 = (1-a_1)(1-a_2), ...
        T = torch.cumprod(one_minus_alpha, dim=-1)
        
        # Shift: T_i = product of (1-alpha_j) for j < i
        T = torch.cat([
            torch.ones_like(T[:, :1]),  # T_1 = 1
            T[:, :-1]  # Remove last, shift right
        ], dim=-1)
        
        return T


class HierarchicalSampling:
    """Coarse-to-fine sampling for efficient rendering"""
    
    @staticmethod
    def importance_sample(t_vals, weights, num_fine_samples):
        """
        Sample more points where weights are high
        
        t_vals: (batch, num_coarse) - coarse sample depths
        weights: (batch, num_coarse) - weights from coarse rendering
        num_fine_samples: int
        
        Returns:
            t_fine: (batch, num_fine_samples) - fine sample depths
        """
        batch_size = weights.shape[0]
        device = weights.device
        
        # Add small constant to prevent zero division
        weights = weights + 1e-5
        
        # Normalize to PDF
        pdf = weights / weights.sum(dim=-1, keepdim=True)
        
        # Compute CDF
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
        
        # Sample uniform values
        u = torch.rand(batch_size, num_fine_samples, device=device)
        
        # Inverse CDF: find bins
        indices = torch.searchsorted(cdf, u, right=True)
        indices = indices.clamp(1, t_vals.shape[1]) - 1
        
        # Get t values at those indices
        below = torch.clamp(indices - 1, min=0)
        above = torch.clamp(indices, max=t_vals.shape[1] - 1)
        
        t_below = torch.gather(t_vals, 1, below)
        t_above = torch.gather(t_vals, 1, above)
        
        # Linear interpolation
        cdf_below = torch.gather(cdf, 1, below)
        cdf_above = torch.gather(cdf, 1, indices)
        
        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t_weight = (u - cdf_below) / denom
        
        t_fine = t_below + t_weight * (t_above - t_below)
        
        return t_fine


def rendering_example():
    """Complete rendering pipeline example"""
    
    renderer = VolumetricRenderer(near=2.0, far=6.0, num_samples=64)
    
    # Example rays
    batch_size = 1024
    ray_origins = torch.zeros(batch_size, 3, device='cuda')
    ray_origins[:, 2] = -4  # Camera at z=-4
    
    ray_directions = torch.randn(batch_size, 3, device='cuda')
    ray_directions = F.normalize(ray_directions, dim=-1)
    
    # Sample points
    points, t_vals = renderer.sample_along_rays(ray_origins, ray_directions)
    
    # Query NeRF MLP (placeholder)
    # rgb, sigma = nerf_mlp(points, ray_directions)
    
    # For demo: random values
    rgb = torch.rand(batch_size, 64, 3, device='cuda')
    sigma = torch.rand(batch_size, 64, device='cuda') * 10
    
    # Render
    color, depth, weights, acc = renderer.render(rgb, sigma, t_vals)
    
    print(f"Rendered colors shape: {color.shape}")  # (1024, 3)
    print(f"Depth map shape: {depth.shape}")  # (1024,)
    print(f"Weights sum (should be ~1): {weights.sum(dim=-1).mean():.3f}")


def visualize_rendering():
    """Visualize the rendering process"""
    
    print("Volumetric Rendering Steps:")
    print("=" * 50)
    print("1. Cast ray from camera through pixel")
    print("2. Sample N points along ray (stratified)")
    print("3. Query MLP: (x,y,z,θ,φ) → (r,g,b,σ)")
    print("4. Compute alpha: α = 1 - exp(-σ·δ)")
    print("5. Compute transmittance: T = Π(1-α)")
    print("6. Weight colors: w = T·α")
    print("7. Sum: C = Σ(w·c)")
```

**Rendering Summary:**

| Step | Formula | Purpose |
|------|---------|---------|
| Delta | $\delta_i = t_{i+1} - t_i$ | Distance between samples |
| Alpha | $\alpha_i = 1 - e^{-\sigma_i \delta_i}$ | Opacity at sample |
| Transmittance | $T_i = \prod_{j<i}(1-\alpha_j)$ | Light reaching sample |
| Weights | $w_i = T_i \alpha_i$ | Sample contribution |
| Color | $C = \sum_i w_i c_i$ | Final pixel |

**Interview Tip:** The weights $w_i = T_i \alpha_i$ sum to 1 (or less if ray passes through empty space). Transmittance $T$ decreases along the ray as light is absorbed. High density $\sigma$ creates opaque regions that occlude things behind. The integral is approximated with quadrature, so more samples = more accurate but slower.

---

### Question 30
- [ ] Done
**What are positional encodings in NeRF and why do they help capture high-frequency details?**

**Answer:**

Positional encodings transform low-dimensional coordinates into high-dimensional features using sinusoidal functions. MLPs have a spectral bias toward learning low-frequency functions, so without encoding, they produce blurry results. Positional encoding allows the network to learn high-frequency details like edges and textures.

**The Spectral Bias Problem:**

| Input Type | MLP Learns | Result |
|------------|-----------|--------|
| Raw $(x,y,z)$ | Low frequencies only | Blurry, smooth surfaces |
| Encoded $(x,y,z)$ | High + low frequencies | Sharp details, textures |

**Encoding Formula:**

$$\gamma(p) = \left(\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\right)$$

For 3D position $(x,y,z)$, encoding dimension = $3 \times 2 \times L$

Typical: $L=10$ for position → 60D, $L=4$ for direction → 24D

**Why Sinusoids Work:**

1. **Create varying frequencies**: Low $2^0$ captures smooth variations, high $2^{L-1}$ captures fine details
2. **Smooth periodicity**: sin/cos are continuous and differentiable
3. **Unique mapping**: Different positions get different encodings (not periodic aliasing issue because scene is bounded)

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np

class FourierPositionalEncoding(nn.Module):
    """Original NeRF positional encoding with fixed frequencies"""
    
    def __init__(self, num_freqs=10, include_input=True, log_sampling=True):
        """
        num_freqs: L in the paper (10 for position, 4 for direction)
        include_input: whether to include original coordinates
        log_sampling: use 2^i frequencies (True) or linear (False)
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        if log_sampling:
            # 2^0, 2^1, ..., 2^(L-1)
            self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            # Linear: 1, 2, ..., L
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)
    
    def forward(self, x):
        """
        x: (batch, input_dim) - typically 3 for position/direction
        Returns: (batch, output_dim)
        """
        encoded = []
        
        if self.include_input:
            encoded.append(x)
        
        for freq in self.freq_bands.to(x.device):
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        
        return torch.cat(encoded, dim=-1)
    
    def output_dim(self, input_dim):
        """Calculate output dimension"""
        dim = input_dim * 2 * self.num_freqs
        if self.include_input:
            dim += input_dim
        return dim


class HashEncoding(nn.Module):
    """
    Instant-NGP style multi-resolution hash encoding
    Much faster training than Fourier encoding
    """
    
    def __init__(self, n_levels=16, n_features_per_level=2, 
                 log2_hashmap_size=19, base_resolution=16,
                 finest_resolution=512):
        super().__init__()
        
        self.n_levels = n_levels
        self.n_features = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Growth factor
        b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
        self.resolutions = [int(base_resolution * (b ** i)) for i in range(n_levels)]
        
        # Learnable hash tables
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, n_features_per_level) * 0.01)
            for _ in range(n_levels)
        ])
    
    def forward(self, x):
        """
        x: (batch, 3) normalized positions in [0, 1]
        Returns: (batch, n_levels * n_features)
        """
        features = []
        
        for level, resolution in enumerate(self.resolutions):
            # Scale position to grid resolution
            scaled = x * resolution
            
            # Get corner indices
            floor_idx = torch.floor(scaled).long()
            ceil_idx = floor_idx + 1
            
            # Trilinear interpolation weights
            w = scaled - floor_idx.float()
            
            # Hash and lookup (simplified - real implementation more complex)
            corner_features = self._hash_lookup(floor_idx, level)
            
            features.append(corner_features)
        
        return torch.cat(features, dim=-1)
    
    def _hash_lookup(self, indices, level):
        """Hash 3D indices to 1D and lookup in table"""
        # Spatial hash function
        primes = torch.tensor([1, 2654435761, 805459861], device=indices.device)
        hash_idx = (indices * primes).sum(dim=-1) % self.hashmap_size
        
        return self.hash_tables[level][hash_idx]


class IntegratedPositionalEncoding(nn.Module):
    """
    Mip-NeRF integrated positional encoding
    Handles anti-aliasing by encoding frustum rather than point
    """
    
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
    
    def forward(self, mean, variance):
        """
        mean: (batch, 3) - center of the sample
        variance: (batch, 3) - variance of the sample (from cone)
        
        Returns integrated encoding that handles scale
        """
        encoded = []
        
        for freq in self.freq_bands.to(mean.device):
            # Expected value of sin/cos over Gaussian
            # E[sin(σx)] ≈ exp(-σ²v/2) * sin(σμ)
            scale = torch.exp(-0.5 * (freq ** 2) * variance)
            
            encoded.append(scale * torch.sin(freq * np.pi * mean))
            encoded.append(scale * torch.cos(freq * np.pi * mean))
        
        return torch.cat(encoded, dim=-1)


def visualize_encoding():
    """Show effect of positional encoding"""
    
    # Sample positions
    x = torch.linspace(-1, 1, 100).unsqueeze(-1)
    
    encoder = FourierPositionalEncoding(num_freqs=6, include_input=True)
    encoded = encoder(x)
    
    print(f"Input dim: {x.shape[-1]}")
    print(f"Output dim: {encoded.shape[-1]}")
    print(f"Expansion factor: {encoded.shape[-1] / x.shape[-1]}x")


def frequency_analysis():
    """Analyze frequency components"""
    
    print("NeRF Positional Encoding Frequencies:")
    print("=" * 50)
    
    L_position = 10
    L_direction = 4
    
    print(f"\nPosition encoding (L={L_position}):")
    for i in range(L_position):
        freq = 2 ** i
        wavelength = 2.0 / freq
        print(f"  Level {i}: freq=2^{i}={freq:4d}, wavelength={wavelength:.4f}")
    
    print(f"\nDirection encoding (L={L_direction}):")
    for i in range(L_direction):
        freq = 2 ** i
        print(f"  Level {i}: freq=2^{i}={freq}")
    
    print(f"\nDimensions:")
    print(f"  Position: 3 + 3×2×{L_position} = {3 + 60}")
    print(f"  Direction: 3 + 3×2×{L_direction} = {3 + 24}")


def compare_encodings():
    """Compare different encoding strategies"""
    
    encodings = {
        'None (raw xyz)': {'quality': 'Blurry', 'speed': 'N/A', 'memory': 'Low'},
        'Fourier (NeRF)': {'quality': 'Sharp', 'speed': 'Days', 'memory': 'Low'},
        'Hash (Instant-NGP)': {'quality': 'Sharp', 'speed': 'Minutes', 'memory': 'Medium'},
        'Integrated (Mip-NeRF)': {'quality': 'Anti-aliased', 'speed': 'Days', 'memory': 'Low'},
    }
    
    print("Encoding Comparison:")
    print("=" * 60)
    for name, props in encodings.items():
        print(f"{name:25} | Quality: {props['quality']:12} | "
              f"Train: {props['speed']:10} | Mem: {props['memory']}")
```

**Encoding Comparison:**

| Method | Key Idea | Train Time | Quality |
|--------|----------|------------|---------|
| None | Raw coordinates | - | Blurry |
| Fourier | Fixed sin/cos frequencies | Days | Sharp |
| Hash (Instant-NGP) | Learnable multi-resolution hash | Minutes | Sharp |
| Integrated (Mip-NeRF) | Gaussian-weighted for anti-aliasing | Days | Anti-aliased |

**Why Not Just Deeper Networks?**

Deeper MLPs still have spectral bias—they learn low frequencies first and high frequencies slowly or never. Positional encoding explicitly provides high-frequency basis functions that the MLP can combine.

**Interview Tip:** The key insight is that coordinates $(x,y,z) \in [-1,1]^3$ are too "smooth" for MLPs to represent sharp edges. Encoding maps to high-dimensional space where nearby points have varying similarity based on frequency. Think of it as providing the network with "texture" primitives it can learn to combine.

---

### Question 31
- [ ] Done
**How does 3D Gaussian Splatting achieve real-time rendering compared to NeRF's slow inference?**

**Answer:**

3D Gaussian Splatting uses explicit primitives (3D Gaussians) instead of implicit neural fields. Rendering is a simple forward pass—project Gaussians to 2D, sort by depth, and alpha-blend—with no MLP queries per ray. This enables 100+ FPS vs NeRF's seconds per frame, while maintaining comparable quality.

**Fundamental Difference:**

| Aspect | NeRF | Gaussian Splatting |
|--------|------|-------------------|
| Representation | Implicit (MLP) | Explicit (point clouds) |
| Rendering | Ray marching + MLP queries | Rasterization |
| Query complexity | O(rays × samples × MLP) | O(Gaussians) |
| Rendering speed | ~30s per frame | 100+ FPS |
| Training speed | Days | Minutes |

**Gaussian Splatting Pipeline:**

```
3D Gaussians → Project to 2D → Sort by depth → Tile-based rasterization → Alpha blend → Image
```

**3D Gaussian Parameters:**

Each Gaussian has:
- **Position**: $\mu \in \mathbb{R}^3$ (3D center)
- **Covariance**: $\Sigma \in \mathbb{R}^{3\times3}$ (shape/orientation)
- **Opacity**: $\alpha \in [0,1]$
- **Color**: Spherical harmonics coefficients for view-dependent color

**Why It's Fast:**

1. **No neural network at render time**: Just project and blend
2. **Tile-based sorting**: GPU-efficient parallel sorting
3. **Early termination**: Stop when alpha saturates
4. **Sparse representation**: Only render visible Gaussians

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GaussianCloud:
    """Collection of 3D Gaussians"""
    positions: torch.Tensor      # (N, 3) - xyz centers
    scales: torch.Tensor         # (N, 3) - scale factors
    rotations: torch.Tensor      # (N, 4) - quaternions
    opacities: torch.Tensor      # (N,) - alpha values
    sh_coeffs: torch.Tensor      # (N, C, 3) - spherical harmonics for RGB


class GaussianSplatRenderer:
    """Differentiable Gaussian splatting renderer"""
    
    def __init__(self, image_height=800, image_width=800, device='cuda'):
        self.H = image_height
        self.W = image_width
        self.device = device
    
    def render(self, gaussians: GaussianCloud, camera):
        """
        Render Gaussians to image
        
        Returns:
            image: (H, W, 3)
            depth: (H, W)
        """
        # 1. Project 3D Gaussians to 2D
        means2d, depths, cov2d = self._project_gaussians(
            gaussians.positions,
            gaussians.scales,
            gaussians.rotations,
            camera
        )
        
        # 2. Compute view-dependent colors from spherical harmonics
        colors = self._sh_to_rgb(gaussians.sh_coeffs, camera.viewdir)
        
        # 3. Sort by depth (front to back)
        sorted_indices = torch.argsort(depths)
        
        # 4. Rasterize with alpha blending
        image, depth_map = self._rasterize(
            means2d[sorted_indices],
            cov2d[sorted_indices],
            colors[sorted_indices],
            gaussians.opacities[sorted_indices]
        )
        
        return image, depth_map
    
    def _project_gaussians(self, positions, scales, rotations, camera):
        """Project 3D Gaussians to 2D screen space"""
        
        # Transform to camera coordinates
        world_to_cam = camera.world_to_camera_matrix
        positions_cam = (world_to_cam[:3, :3] @ positions.T + world_to_cam[:3, 3:4]).T
        
        # Perspective projection
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        
        z = positions_cam[:, 2]
        x = (positions_cam[:, 0] * fx / z) + cx
        y = (positions_cam[:, 1] * fy / z) + cy
        means2d = torch.stack([x, y], dim=-1)
        
        # Project covariance to 2D
        # Σ_2D = J @ R @ S @ S^T @ R^T @ J^T
        # (Simplified - actual implementation uses EWA splatting)
        cov2d = self._project_covariance(scales, rotations, positions_cam, camera)
        
        return means2d, z, cov2d
    
    def _project_covariance(self, scales, rotations, positions_cam, camera):
        """Project 3D covariance to 2D using EWA splatting"""
        
        # Build rotation matrix from quaternion
        R = self._quaternion_to_matrix(rotations)
        
        # Build scaling matrix
        S = torch.diag_embed(scales)
        
        # 3D covariance: Σ = R @ S @ S^T @ R^T
        cov3d = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        
        # Jacobian of projection
        z = positions_cam[:, 2]
        J = torch.zeros(len(z), 2, 3, device=self.device)
        J[:, 0, 0] = camera.fx / z
        J[:, 1, 1] = camera.fy / z
        J[:, 0, 2] = -positions_cam[:, 0] * camera.fx / (z ** 2)
        J[:, 1, 2] = -positions_cam[:, 1] * camera.fy / (z ** 2)
        
        # 2D covariance
        cov2d = J @ cov3d @ J.transpose(-1, -2)
        
        return cov2d
    
    def _quaternion_to_matrix(self, quaternions):
        """Convert quaternions to rotation matrices"""
        # Normalize
        q = F.normalize(quaternions, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Build rotation matrix
        R = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).reshape(-1, 3, 3)
        
        return R
    
    def _sh_to_rgb(self, sh_coeffs, viewdir):
        """Evaluate spherical harmonics for view-dependent color"""
        # Simplified: just use DC term for now
        # Full implementation evaluates SH basis functions
        return torch.sigmoid(sh_coeffs[:, 0, :])
    
    def _rasterize(self, means2d, cov2d, colors, opacities):
        """
        Tile-based rasterization with alpha blending
        """
        image = torch.zeros(self.H, self.W, 3, device=self.device)
        depth = torch.zeros(self.H, self.W, device=self.device)
        accumulated_alpha = torch.zeros(self.H, self.W, device=self.device)
        
        # Create pixel grid
        y_coords = torch.arange(self.H, device=self.device).float()
        x_coords = torch.arange(self.W, device=self.device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixels = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        
        # For each Gaussian (front to back)
        for i in range(len(means2d)):
            if accumulated_alpha.min() > 0.99:
                break  # Early termination
            
            # Compute 2D Gaussian contribution
            diff = pixels - means2d[i]  # (H, W, 2)
            
            # Mahalanobis distance
            cov_inv = torch.inverse(cov2d[i] + torch.eye(2, device=self.device) * 1e-4)
            dist = (diff @ cov_inv * diff).sum(dim=-1)  # (H, W)
            
            # Gaussian weight
            weight = torch.exp(-0.5 * dist)
            
            # Alpha for this Gaussian
            alpha = opacities[i] * weight
            
            # Blend (front-to-back compositing)
            transmittance = 1 - accumulated_alpha
            contribution = alpha * transmittance
            
            image += contribution.unsqueeze(-1) * colors[i]
            accumulated_alpha += contribution
        
        return image, depth


class GaussianOptimizer:
    """Training loop for Gaussian Splatting"""
    
    def __init__(self, num_gaussians, device='cuda'):
        # Initialize from point cloud (e.g., from COLMAP)
        self.gaussians = GaussianCloud(
            positions=torch.randn(num_gaussians, 3, device=device, requires_grad=True),
            scales=torch.ones(num_gaussians, 3, device=device, requires_grad=True) * 0.01,
            rotations=torch.zeros(num_gaussians, 4, device=device, requires_grad=True),
            opacities=torch.sigmoid(torch.zeros(num_gaussians, device=device, requires_grad=True)),
            sh_coeffs=torch.zeros(num_gaussians, 16, 3, device=device, requires_grad=True)
        )
        # Initialize quaternions to identity
        self.gaussians.rotations[:, 0] = 1.0
        
        self.renderer = GaussianSplatRenderer(device=device)
    
    def train_step(self, gt_image, camera):
        """One training iteration"""
        # Render
        rendered, _ = self.renderer.render(self.gaussians, camera)
        
        # L1 + SSIM loss
        loss = F.l1_loss(rendered, gt_image) + (1 - self._ssim(rendered, gt_image))
        
        return loss
    
    def _ssim(self, img1, img2):
        """Structural similarity (simplified)"""
        return 1.0  # Placeholder
    
    def densify_and_prune(self):
        """
        Adaptive density control:
        - Split large Gaussians with high gradient
        - Clone small Gaussians with high gradient
        - Remove transparent Gaussians
        """
        pass


def speed_comparison():
    """Compare rendering speed"""
    
    print("Rendering Speed Comparison:")
    print("=" * 50)
    print(f"{'Method':<25} {'Resolution':<12} {'FPS':<10}")
    print("-" * 50)
    print(f"{'NeRF (original)':<25} {'800×800':<12} {'~0.03':<10}")
    print(f"{'Instant-NGP':<25} {'800×800':<12} {'~1-5':<10}")
    print(f"{'3D Gaussian Splatting':<25} {'1080p':<12} {'100+':<10}")
    print(f"{'3D Gaussian Splatting':<25} {'4K':<12} {'30+':<10}")
```

**Key Speedup Factors:**

| Factor | NeRF | Gaussian Splatting |
|--------|------|-------------------|
| Per-pixel work | ~200 MLP queries | Simple blending |
| Parallelism | Limited by ray marching | Highly parallel rasterization |
| Memory access | Random (MLP weights) | Coalesced (sorted Gaussians) |
| Computation | Dense MLP eval | Sparse Gaussian eval |

**Interview Tip:** The core insight is representation: NeRF is implicit (function gives density everywhere), Gaussian Splatting is explicit (just store the Gaussians). Explicit representations enable classical graphics techniques—projection, sorting, blending—that GPUs are optimized for. Training is also faster because gradients flow through simple differentiable operations, not deep MLPs.

---

### Question 32
- [ ] Done
**Explain spherical harmonics for view-dependent color in Gaussian Splatting.**

**Answer:**

Spherical harmonics (SH) are a set of basis functions on the sphere that efficiently represent view-dependent color. Each Gaussian stores SH coefficients instead of a single RGB—when rendered, the viewing direction is used to evaluate the SH, producing different colors from different angles (specular highlights, reflections).

**Why View-Dependent Color?**

| Surface Type | View Dependence | Example |
|--------------|-----------------|---------|
| Lambertian (matte) | None | Paper, cloth |
| Specular | High | Metal, glass |
| Mixed | Moderate | Plastic, painted surfaces |

**Spherical Harmonics Basics:**

SH are orthonormal basis functions $Y_l^m(\theta, \phi)$ defined on the unit sphere.

- **Degree $l$**: Controls frequency (higher = more detail)
- **Order $m$**: $-l \leq m \leq l$ (so $2l+1$ functions per degree)
- **Total functions**: $(l_{max}+1)^2$

Common choices:
- $l_{max} = 0$: 1 coefficient (constant color, Lambertian)
- $l_{max} = 2$: 9 coefficients (captures soft shadows)
- $l_{max} = 3$: 16 coefficients (captures specular highlights)

**Evaluation:**

$$c(\mathbf{d}) = \sum_{l=0}^{l_{max}} \sum_{m=-l}^{l} c_l^m \cdot Y_l^m(\mathbf{d})$$

Where $\mathbf{d}$ is viewing direction and $c_l^m$ are learned coefficients.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np

class SphericalHarmonics:
    """Spherical harmonics for view-dependent color"""
    
    # Precomputed constants for SH basis functions
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    
    @staticmethod
    def evaluate_sh(degree, directions, sh_coeffs):
        """
        Evaluate spherical harmonics at given viewing directions
        
        degree: int, maximum SH degree (0, 1, 2, or 3)
        directions: (N, 3) normalized viewing directions
        sh_coeffs: (N, num_sh, 3) SH coefficients per Gaussian, per color channel
        
        Returns: (N, 3) RGB colors
        """
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        
        result = torch.zeros_like(directions)
        
        # Degree 0 (constant term)
        result += SphericalHarmonics.C0 * sh_coeffs[:, 0]
        
        if degree >= 1:
            # Degree 1 (3 terms)
            result += SphericalHarmonics.C1 * (
                y.unsqueeze(-1) * sh_coeffs[:, 1] +
                z.unsqueeze(-1) * sh_coeffs[:, 2] +
                x.unsqueeze(-1) * sh_coeffs[:, 3]
            )
        
        if degree >= 2:
            # Degree 2 (5 terms)
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            
            result += SphericalHarmonics.C2[0] * xy.unsqueeze(-1) * sh_coeffs[:, 4]
            result += SphericalHarmonics.C2[1] * yz.unsqueeze(-1) * sh_coeffs[:, 5]
            result += SphericalHarmonics.C2[2] * (2*zz - xx - yy).unsqueeze(-1) * sh_coeffs[:, 6]
            result += SphericalHarmonics.C2[3] * xz.unsqueeze(-1) * sh_coeffs[:, 7]
            result += SphericalHarmonics.C2[4] * (xx - yy).unsqueeze(-1) * sh_coeffs[:, 8]
        
        if degree >= 3:
            # Degree 3 (7 terms) - captures specular highlights
            result += SphericalHarmonics.C3[0] * y*(3*x*x - y*y).unsqueeze(-1) * sh_coeffs[:, 9]
            result += SphericalHarmonics.C3[1] * xy*z.unsqueeze(-1) * sh_coeffs[:, 10]
            result += SphericalHarmonics.C3[2] * y*(4*z*z - x*x - y*y).unsqueeze(-1) * sh_coeffs[:, 11]
            result += SphericalHarmonics.C3[3] * z*(2*z*z - 3*x*x - 3*y*y).unsqueeze(-1) * sh_coeffs[:, 12]
            result += SphericalHarmonics.C3[4] * x*(4*z*z - x*x - y*y).unsqueeze(-1) * sh_coeffs[:, 13]
            result += SphericalHarmonics.C3[5] * z*(x*x - y*y).unsqueeze(-1) * sh_coeffs[:, 14]
            result += SphericalHarmonics.C3[6] * x*(x*x - 3*y*y).unsqueeze(-1) * sh_coeffs[:, 15]
        
        # Convert to valid RGB
        result = torch.sigmoid(result)
        
        return result


class SHColorGaussian(nn.Module):
    """Gaussian with SH-based view-dependent color"""
    
    def __init__(self, num_gaussians, sh_degree=3, device='cuda'):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh = (sh_degree + 1) ** 2
        
        # Initialize SH coefficients
        # DC term initialized to gray, higher orders to zero
        self.sh_coeffs = nn.Parameter(
            torch.zeros(num_gaussians, self.num_sh, 3, device=device)
        )
        # Initialize DC term to some base color
        self.sh_coeffs.data[:, 0, :] = 0.5
    
    def get_color(self, view_directions):
        """
        Get colors for each Gaussian from a viewing direction
        
        view_directions: (N, 3) normalized directions
        Returns: (N, 3) RGB colors
        """
        return SphericalHarmonics.evaluate_sh(
            self.sh_degree, 
            view_directions, 
            self.sh_coeffs
        )


class ViewDependentRendering:
    """Demonstrate view-dependent effects"""
    
    @staticmethod
    def visualize_sh_basis():
        """Show what each SH basis function represents"""
        
        sh_explanation = {
            'Degree 0 (1 term)': 'Constant color in all directions',
            'Degree 1 (3 terms)': 'Linear variation (soft gradients)',
            'Degree 2 (5 terms)': 'Quadratic (soft shadows, ambient occlusion)',
            'Degree 3 (7 terms)': 'Cubic (specular highlights, reflections)'
        }
        
        print("Spherical Harmonics Degrees:")
        print("=" * 60)
        for degree, description in sh_explanation.items():
            print(f"{degree}: {description}")
    
    @staticmethod
    def example_specular():
        """Example: Representing a specular highlight"""
        
        # A specular highlight is bright in one direction
        # We can represent this with high-degree SH
        
        # Light direction
        light_dir = torch.tensor([0.5, 0.5, 0.707])
        
        # View directions to evaluate
        angles = torch.linspace(0, np.pi, 90)
        view_dirs = torch.stack([
            torch.cos(angles),
            torch.zeros_like(angles),
            torch.sin(angles)
        ], dim=-1)
        
        # High-degree SH would show peak when view_dir aligns with reflection
        print("Specular highlights require degree 2+ to represent accurately")
        print("Higher degree = sharper specular peaks")


def sh_comparison():
    """Compare SH degrees for quality vs memory"""
    
    print("SH Degree Comparison:")
    print("=" * 60)
    print(f"{'Degree':<10} {'Coeffs':<10} {'Memory (per Gaussian)':<25} {'Use Case'}")
    print("-" * 60)
    
    for degree in range(4):
        num_coeffs = (degree + 1) ** 2
        memory = num_coeffs * 3 * 4  # 3 channels, 4 bytes per float
        
        use_cases = {
            0: 'Matte surfaces only',
            1: 'Soft lighting variations',
            2: 'Soft specular, ambient occlusion',
            3: 'Full specular, reflections (default)'
        }
        
        print(f"{degree:<10} {num_coeffs:<10} {memory} bytes{' ':<15} {use_cases[degree]}")


def training_with_sh():
    """How SH coefficients are learned"""
    
    print("\nTraining SH Coefficients:")
    print("=" * 60)
    print("1. Initialize DC term (degree 0) from point cloud colors")
    print("2. Higher degree terms initialized to zero")
    print("3. Photometric loss gradients flow through SH evaluation")
    print("4. Network learns to match view-dependent appearance")
    print("5. Regularization may penalize high-degree terms for smoothness")
```

**SH Degree Trade-offs:**

| Degree | # Coefficients | Memory/Gaussian | Capability |
|--------|---------------|-----------------|------------|
| 0 | 1 | 12 bytes | Constant color |
| 1 | 4 | 48 bytes | Soft gradients |
| 2 | 9 | 108 bytes | Soft shadows |
| 3 | 16 | 192 bytes | Sharp specular |

**Why SH Instead of MLP?**

- **Explicit**: No network evaluation, just polynomial
- **Compact**: 16 coefficients vs large MLP
- **Differentiable**: Simple gradient computation
- **Physically motivated**: Represents radiance fields naturally

**Interview Tip:** Spherical harmonics are the "Fourier transform for spheres." Just like Fourier decomposes 1D signals into frequencies, SH decomposes spherical functions into angular frequencies. Degree 0 is constant, degree 1 is linear (X, Y, Z components), and higher degrees capture increasingly detailed angular variations. For most scenes, degree 3 (16 coefficients) is sufficient.

---

### Question 33
- [ ] Done
**Compare NeRF vs. Gaussian Splatting for quality, speed, and memory requirements.**

**Answer:**

NeRF offers slightly better quality for complex view-dependent effects but requires seconds per frame. Gaussian Splatting achieves comparable quality with 100+ FPS rendering but uses more memory for storage. NeRF trains for days; Gaussian Splatting trains in minutes. For real-time applications, Gaussian Splatting wins; for offline high-quality rendering, NeRF may still have advantages.

**Comprehensive Comparison:**

| Aspect | NeRF | Gaussian Splatting |
|--------|------|-------------------|
| **Representation** | Implicit MLP | Explicit point cloud |
| **Rendering speed** | 10-30s/frame | 100+ FPS |
| **Training time** | 1-2 days | 5-30 minutes |
| **Memory (scene)** | ~5 MB (MLP weights) | 100-500 MB (Gaussians) |
| **Memory (rendering)** | High (many samples) | Moderate (sorted list) |
| **Quality** | Excellent | Excellent (slightly lower on edges) |
| **View-dependent** | MLP (flexible) | SH (limited degree) |
| **Editable** | Difficult | Easy (move/delete Gaussians) |

**Quality Breakdown:**

| Quality Aspect | NeRF | Gaussian Splatting | Winner |
|---------------|------|-------------------|--------|
| Specular surfaces | ★★★★★ | ★★★★☆ | NeRF (MLP more flexible) |
| Sharp edges | ★★★★☆ | ★★★★★ | GS (explicit primitives) |
| Thin structures | ★★★☆☆ | ★★★★☆ | GS (less aliasing) |
| Large scenes | ★★★★☆ | ★★★★★ | GS (faster training) |
| Novel view synthesis | ★★★★★ | ★★★★★ | Tie |

**Python Implementation:**
```python
import torch
from dataclasses import dataclass
from enum import Enum

class RenderingMethod(Enum):
    NERF = "nerf"
    GAUSSIAN_SPLATTING = "gaussian_splatting"


@dataclass
class PerformanceMetrics:
    """Performance comparison metrics"""
    render_time_ms: float
    training_time_hours: float
    memory_mb: float
    psnr_db: float
    ssim: float
    lpips: float


class MethodComparison:
    """Compare NeRF and Gaussian Splatting"""
    
    @staticmethod
    def get_typical_metrics():
        """Return typical benchmark results"""
        
        metrics = {
            'nerf': PerformanceMetrics(
                render_time_ms=30000,  # 30 seconds
                training_time_hours=24,
                memory_mb=50,  # MLP weights only
                psnr_db=31.0,
                ssim=0.95,
                lpips=0.08
            ),
            'instant_ngp': PerformanceMetrics(
                render_time_ms=200,  # ~5 FPS
                training_time_hours=0.1,  # 6 minutes
                memory_mb=100,
                psnr_db=30.5,
                ssim=0.94,
                lpips=0.09
            ),
            'gaussian_splatting': PerformanceMetrics(
                render_time_ms=8,  # 120+ FPS
                training_time_hours=0.3,  # 20 minutes
                memory_mb=300,
                psnr_db=30.8,
                ssim=0.94,
                lpips=0.08
            )
        }
        
        return metrics
    
    @staticmethod
    def choose_method(requirements):
        """
        Recommend method based on requirements
        
        requirements: dict with keys like 'real_time', 'quality_priority', etc.
        """
        if requirements.get('real_time', False):
            return 'gaussian_splatting'
        
        if requirements.get('memory_constrained', False):
            return 'nerf'  # Smaller model
        
        if requirements.get('editable_scene', False):
            return 'gaussian_splatting'  # Easy to manipulate
        
        if requirements.get('complex_materials', False):
            return 'nerf'  # MLP more flexible
        
        if requirements.get('fast_training', False):
            return 'gaussian_splatting'
        
        # Default for quality
        return 'nerf'


def detailed_comparison():
    """Print detailed feature comparison"""
    
    print("NeRF vs Gaussian Splatting - Detailed Comparison")
    print("=" * 70)
    
    features = [
        ("Core Representation", "MLP: F(x,y,z,θ,φ)→(rgb,σ)", "Explicit 3D Gaussians"),
        ("Rendering Approach", "Ray marching + MLP queries", "Rasterization + alpha blend"),
        ("Training Input", "Posed images", "Posed images + point cloud"),
        ("GPU Utilization", "Compute-bound (MLP)", "Memory-bound (primitives)"),
        ("Anti-aliasing", "Multi-scale (Mip-NeRF)", "EWA splatting (built-in)"),
        ("Dynamic Scenes", "Separate time MLP", "Deform Gaussians per frame"),
        ("Mesh Export", "Marching cubes", "Poisson reconstruction"),
        ("VR/AR Ready", "No (too slow)", "Yes (real-time)"),
    ]
    
    print(f"{'Feature':<25} {'NeRF':<30} {'Gaussian Splatting':<30}")
    print("-" * 85)
    for feature, nerf, gs in features:
        print(f"{feature:<25} {nerf:<30} {gs:<30}")


def use_case_recommendations():
    """When to use each method"""
    
    print("\nUse Case Recommendations:")
    print("=" * 60)
    
    recommendations = {
        'VR/AR applications': 'Gaussian Splatting',
        'Real-time game assets': 'Gaussian Splatting',
        'Film VFX (offline)': 'NeRF (quality matters most)',
        'Cultural heritage scanning': 'Either (quality + speed trade-off)',
        'Telepresence': 'Gaussian Splatting',
        'Scientific visualization': 'NeRF (accurate representations)',
        'Mobile deployment': 'Gaussian Splatting (faster inference)',
        'Quick prototyping': 'Gaussian Splatting (faster training)',
    }
    
    for use_case, recommendation in recommendations.items():
        print(f"  {use_case:<30} → {recommendation}")


def hybrid_approaches():
    """Emerging hybrid methods"""
    
    print("\nHybrid and Future Directions:")
    print("=" * 60)
    
    hybrids = [
        ("3DGS + Neural Texture", "Use Gaussians for geometry, neural network for appearance"),
        ("NeRF → GS Conversion", "Train NeRF, extract Gaussians for deployment"),
        ("Scaffold-GS", "Neural features on structured Gaussians"),
        ("GS + Diffusion", "Generate Gaussians from text/images"),
    ]
    
    for name, description in hybrids:
        print(f"  {name}: {description}")


def benchmark_scene(scene_type='indoor'):
    """Typical benchmark results by scene type"""
    
    benchmarks = {
        'indoor': {
            'nerf_psnr': 32.5, 'nerf_fps': 0.03,
            'gs_psnr': 31.8, 'gs_fps': 130
        },
        'outdoor': {
            'nerf_psnr': 28.0, 'nerf_fps': 0.02,
            'gs_psnr': 27.5, 'gs_fps': 100
        },
        'object': {
            'nerf_psnr': 35.0, 'nerf_fps': 0.04,
            'gs_psnr': 34.5, 'gs_fps': 150
        }
    }
    
    return benchmarks.get(scene_type, benchmarks['indoor'])
```

**When to Choose Each:**

| Scenario | Choose | Reason |
|----------|--------|--------|
| Real-time application | Gaussian Splatting | 100+ FPS |
| Maximum quality | NeRF | More flexible MLP |
| Fast iteration | Gaussian Splatting | Minutes to train |
| Limited storage | NeRF | Smaller model size |
| Scene editing needed | Gaussian Splatting | Explicit primitives |
| Complex materials | NeRF | Arbitrary appearance functions |

**Recent Developments:**

- **Instant-NGP**: Bridges gap (NeRF speed with hash encoding)
- **Mip-Splatting**: Anti-aliasing for Gaussian Splatting
- **4D Gaussian Splatting**: Dynamic scenes
- **LERF**: Language-embedded NeRF for semantic queries

**Interview Tip:** Don't frame it as "NeRF vs Gaussian Splatting" but rather "implicit vs explicit representations." NeRF's implicit MLP offers flexibility but at computational cost. Gaussian Splatting's explicit primitives enable classical graphics optimizations. The trend is toward hybrid methods that combine the best of both—neural features on explicit structures.

---

### Question 34
- [ ] Done
**How do you handle dynamic scenes in NeRF (D-NeRF, Nerfies)?**

**Answer:**

Dynamic NeRF extends the 5D input $(x,y,z,\theta,\phi)$ to 6D by adding time $t$. The network learns a deformation field that warps each frame to a canonical space, or directly learns time-conditioned radiance. D-NeRF models rigid motion; Nerfies handles non-rigid deformations like facial expressions.

**Two Main Approaches:**

| Approach | Method | Best For |
|----------|--------|----------|
| Deformation-based | Learn warp field to canonical | Non-rigid (faces, bodies) |
| Time-conditioned | Add time to radiance MLP | General dynamic scenes |

**D-NeRF Architecture:**

$$F_{\theta}: (x, y, z, t) \rightarrow (\Delta x, \Delta y, \Delta z)$$
$$G_{\theta}: (x + \Delta x, y + \Delta y, z + \Delta z, \theta, \phi) \rightarrow (rgb, \sigma)$$

1. Deformation network maps (position, time) → displacement
2. Apply displacement to get canonical coordinates
3. Query canonical NeRF for color and density

**Nerfies (Non-rigid):**

Uses SE(3) deformation field with elastic regularization:
- Learned per-frame latent codes
- As-rigid-as-possible regularization
- Coarse-to-fine training

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformationNetwork(nn.Module):
    """Learns time-dependent deformation to canonical space"""
    
    def __init__(self, pos_dim=63, time_dim=10, hidden_dim=128):
        super().__init__()
        
        self.time_encoder = PositionalEncoding(time_dim)
        
        self.network = nn.Sequential(
            nn.Linear(pos_dim + time_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Δx, Δy, Δz
        )
        
        # Initialize to identity (no deformation)
        self.network[-1].weight.data.zero_()
        self.network[-1].bias.data.zero_()
    
    def forward(self, positions_encoded, time):
        """
        positions_encoded: (batch, pos_dim) - already positionally encoded
        time: (batch, 1) - normalized time [0, 1]
        
        Returns: (batch, 3) - displacement
        """
        time_encoded = self.time_encoder(time)
        inputs = torch.cat([positions_encoded, time_encoded], dim=-1)
        
        delta = self.network(inputs)
        
        return delta


class DynamicNeRF(nn.Module):
    """D-NeRF: NeRF for dynamic scenes"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(num_freqs=10)
        self.dir_encoder = PositionalEncoding(num_freqs=4)
        
        # Deformation network
        self.deformation = DeformationNetwork(
            pos_dim=self.pos_encoder.output_dim(3)
        )
        
        # Canonical NeRF (static scene in canonical pose)
        self.canonical_nerf = NeRFMLP()
    
    def forward(self, positions, directions, time):
        """
        positions: (batch, 3) - xyz
        directions: (batch, 3) - viewing direction
        time: (batch, 1) - frame time
        
        Returns:
            rgb: (batch, 3)
            sigma: (batch, 1)
        """
        # Encode position
        pos_encoded = self.pos_encoder(positions)
        
        # Get deformation
        delta = self.deformation(pos_encoded, time)
        
        # Warp to canonical space
        canonical_positions = positions + delta
        
        # Query canonical NeRF
        rgb, sigma = self.canonical_nerf(canonical_positions, directions)
        
        return rgb, sigma


class Nerfies(nn.Module):
    """
    Nerfies: Deformable Neural Radiance Fields
    Handles non-rigid deformations with latent codes
    """
    
    def __init__(self, num_frames=100, latent_dim=128, device='cuda'):
        super().__init__()
        self.device = device
        
        # Per-frame latent codes (optimized during training)
        self.frame_codes = nn.Embedding(num_frames, latent_dim)
        
        # SE(3) deformation field
        self.deformation = SE3DeformationField(latent_dim)
        
        # Canonical NeRF
        self.canonical_nerf = NeRFMLP()
        
        # Elastic regularization weight
        self.elastic_weight = 0.01
    
    def forward(self, positions, directions, frame_idx):
        """Render with frame-specific deformation"""
        
        # Get frame-specific latent code
        latent = self.frame_codes(frame_idx)
        
        # Compute SE(3) deformation
        R, t = self.deformation(positions, latent)
        
        # Apply transformation
        canonical_positions = torch.bmm(R, positions.unsqueeze(-1)).squeeze(-1) + t
        
        # Query canonical NeRF
        rgb, sigma = self.canonical_nerf(canonical_positions, directions)
        
        return rgb, sigma
    
    def elastic_loss(self, positions, frame_idx):
        """
        As-rigid-as-possible regularization
        Encourages smooth, locally rigid deformations
        """
        # Sample nearby points
        eps = 0.01
        neighbors = positions + torch.randn_like(positions) * eps
        
        latent = self.frame_codes(frame_idx)
        
        # Get deformations
        R1, t1 = self.deformation(positions, latent)
        R2, t2 = self.deformation(neighbors, latent)
        
        # Enforce similar local transformation
        loss = F.mse_loss(R1, R2) + F.mse_loss(t1, t2)
        
        return loss * self.elastic_weight


class SE3DeformationField(nn.Module):
    """SE(3) field for rigid body transformations"""
    
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(num_freqs=6)
        
        input_dim = self.pos_encoder.output_dim(3) + latent_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output rotation (as axis-angle) and translation
        self.rotation_head = nn.Linear(hidden_dim, 3)  # axis-angle
        self.translation_head = nn.Linear(hidden_dim, 3)
    
    def forward(self, positions, latent):
        """
        Returns rotation matrix and translation
        """
        pos_enc = self.pos_encoder(positions)
        
        # Expand latent to match batch
        latent_exp = latent.expand(positions.shape[0], -1)
        
        features = self.network(torch.cat([pos_enc, latent_exp], dim=-1))
        
        # Get rotation (axis-angle to matrix)
        axis_angle = self.rotation_head(features)
        R = self._axis_angle_to_matrix(axis_angle)
        
        # Get translation
        t = self.translation_head(features)
        
        return R, t
    
    def _axis_angle_to_matrix(self, axis_angle):
        """Convert axis-angle to rotation matrix using Rodrigues"""
        theta = torch.norm(axis_angle, dim=-1, keepdim=True)
        axis = axis_angle / (theta + 1e-6)
        
        # Rodrigues formula (simplified)
        K = self._skew_symmetric(axis)
        R = torch.eye(3, device=axis.device) + \
            torch.sin(theta).unsqueeze(-1) * K + \
            (1 - torch.cos(theta)).unsqueeze(-1) * K @ K
        
        return R
    
    def _skew_symmetric(self, v):
        """Create skew-symmetric matrix from vector"""
        batch = v.shape[0]
        K = torch.zeros(batch, 3, 3, device=v.device)
        K[:, 0, 1] = -v[:, 2]
        K[:, 0, 2] = v[:, 1]
        K[:, 1, 0] = v[:, 2]
        K[:, 1, 2] = -v[:, 0]
        K[:, 2, 0] = -v[:, 1]
        K[:, 2, 1] = v[:, 0]
        return K


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
    
    def forward(self, x):
        encoded = [x]
        for freq in self.freq_bands.to(x.device):
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)
    
    def output_dim(self, input_dim):
        return input_dim * (1 + 2 * self.num_freqs)


class NeRFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified canonical NeRF
        self.network = nn.Sequential(
            nn.Linear(63, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4)  # rgb + sigma
        )
    
    def forward(self, positions, directions):
        pos_enc = PositionalEncoding(10)(positions)
        out = self.network(pos_enc)
        rgb = torch.sigmoid(out[:, :3])
        sigma = F.relu(out[:, 3:4])
        return rgb, sigma


def training_considerations():
    """Key training considerations for dynamic NeRF"""
    
    considerations = {
        'Regularization': [
            'Elastic loss (as-rigid-as-possible)',
            'Deformation smoothness in time',
            'Background regularization (static background)'
        ],
        'Training Strategy': [
            'Coarse-to-fine (start with low-frequency deformations)',
            'Anneal positional encoding frequencies',
            'Separate background model'
        ],
        'Challenges': [
            'Ambiguity between geometry and deformation',
            'Topology changes (opening/closing)',
            'Fast motion blur'
        ]
    }
    
    return considerations
```

**Method Comparison:**

| Method | Deformation Type | Input | Regularization |
|--------|-----------------|-------|----------------|
| D-NeRF | Translation only | Position + time | Smooth in time |
| Nerfies | SE(3) per-point | Frame latent | Elastic (ARAP) |
| HyperNeRF | Ambient dim | Slicing hyperspace | Dimension smoothness |
| Neural Body | SMPL-guided | Pose parameters | Skeleton prior |

**Interview Tip:** The key insight for dynamic NeRF is that instead of learning a separate NeRF per frame (infeasible), we learn a canonical NeRF plus a deformation field. This assumes the scene content is consistent across time—only position changes. For more complex dynamics (appearance changes, topology), methods like HyperNeRF use higher-dimensional ambient spaces.

---

### Question 35
- [ ] Done
**What techniques reduce NeRF training time (Instant-NGP, TensoRF, Plenoxels)?**

**Answer:**

These methods replace the slow MLP with explicit or hybrid data structures: Instant-NGP uses multi-resolution hash tables, TensoRF uses tensor decomposition, and Plenoxels uses a voxel grid. All reduce training from days to minutes while maintaining quality by trading MLP computation for optimized spatial lookups.

**Core Acceleration Ideas:**

| Method | Representation | Key Speedup |
|--------|---------------|-------------|
| Instant-NGP | Multi-resolution hash encoding | Learnable features, tiny MLP |
| TensoRF | Tensor decomposition (CP/VM) | Compact factorized representation |
| Plenoxels | Dense voxel grid + SH | No neural network at all |
| DVGO | Explicit voxel + tiny MLP | Direct optimization of voxels |

**Training Time Comparison:**

| Method | Training Time | Quality (PSNR) |
|--------|--------------|----------------|
| Original NeRF | 1-2 days | ~31 dB |
| Instant-NGP | 5-15 minutes | ~31 dB |
| TensoRF | 15-30 minutes | ~33 dB |
| Plenoxels | 10-20 minutes | ~31 dB |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiResolutionHashEncoding(nn.Module):
    """
    Instant-NGP style hash encoding
    Key insight: Replace Fourier encoding with learnable multi-resolution features
    """
    
    def __init__(self, n_levels=16, n_features=2, log2_hashmap_size=19,
                 base_resolution=16, max_resolution=2048, device='cuda'):
        super().__init__()
        
        self.n_levels = n_levels
        self.n_features = n_features
        self.hashmap_size = 2 ** log2_hashmap_size
        self.device = device
        
        # Compute resolution for each level
        growth_factor = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1))
        self.resolutions = [int(base_resolution * (growth_factor ** i)) for i in range(n_levels)]
        
        # Hash tables (learnable parameters)
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, n_features, device=device) * 0.01)
            for _ in range(n_levels)
        ])
        
        # Primes for spatial hashing
        self.primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long, device=device)
    
    def forward(self, positions):
        """
        positions: (batch, 3) normalized to [0, 1]
        Returns: (batch, n_levels * n_features)
        """
        features = []
        
        for level, resolution in enumerate(self.resolutions):
            # Scale position to grid
            scaled = positions * resolution
            
            # Get 8 corners of voxel
            floor_coords = torch.floor(scaled).long()
            
            # Trilinear interpolation weights
            weights = scaled - floor_coords.float()
            
            # Hash and lookup with interpolation
            corner_features = self._trilinear_lookup(
                floor_coords, weights, self.hash_tables[level]
            )
            
            features.append(corner_features)
        
        return torch.cat(features, dim=-1)
    
    def _trilinear_lookup(self, floor_coords, weights, table):
        """Trilinear interpolation of hashed features"""
        
        # 8 corners
        offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], device=self.device)
        
        result = torch.zeros(floor_coords.shape[0], self.n_features, device=self.device)
        
        for i, offset in enumerate(offsets):
            corner = floor_coords + offset
            
            # Spatial hash
            hash_idx = self._hash(corner)
            corner_feat = table[hash_idx]
            
            # Trilinear weight
            w = 1.0
            for d in range(3):
                w = w * (weights[:, d] if offset[d] == 1 else (1 - weights[:, d]))
            
            result += w.unsqueeze(-1) * corner_feat
        
        return result
    
    def _hash(self, coords):
        """Spatial hash function"""
        return ((coords[:, 0] * self.primes[0]) ^ 
                (coords[:, 1] * self.primes[1]) ^ 
                (coords[:, 2] * self.primes[2])) % self.hashmap_size


class InstantNGP(nn.Module):
    """Simplified Instant-NGP architecture"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Multi-resolution hash encoding (replaces Fourier)
        self.hash_encoding = MultiResolutionHashEncoding(device=device)
        
        # Tiny MLP (only 2 layers!)
        hash_dim = 16 * 2  # n_levels * n_features
        self.density_mlp = nn.Sequential(
            nn.Linear(hash_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 15 features + 1 density
        )
        
        # Direction encoding + color MLP
        self.dir_encoding = SphericalHarmonicsEncoding(degree=4)
        self.color_mlp = nn.Sequential(
            nn.Linear(16 + 16, 64),  # features + SH-encoded direction
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
    
    def forward(self, positions, directions):
        """
        Much faster than original NeRF due to:
        1. Hash encoding instead of Fourier (learnable, faster)
        2. Tiny MLP (2 layers instead of 8)
        """
        # Hash encoding
        h = self.hash_encoding(positions)
        
        # Density features
        features = self.density_mlp(h)
        density = F.relu(features[:, 0:1])
        geo_features = features[:, 1:]
        
        # Direction-dependent color
        dir_enc = self.dir_encoding(directions)
        color = self.color_mlp(torch.cat([geo_features, dir_enc], dim=-1))
        
        return color, density


class TensoRF(nn.Module):
    """
    TensoRF: Tensor decomposition for radiance fields
    Decomposes 3D volume into vector-matrix products
    """
    
    def __init__(self, grid_size=128, n_components=48, device='cuda'):
        super().__init__()
        self.device = device
        self.grid_size = grid_size
        
        # VM decomposition: V1⊗M1 + V2⊗M2 + V3⊗M3
        # Vectors (1D)
        self.vec_x = nn.Parameter(torch.randn(n_components, grid_size, 1, 1, device=device) * 0.1)
        self.vec_y = nn.Parameter(torch.randn(n_components, 1, grid_size, 1, device=device) * 0.1)
        self.vec_z = nn.Parameter(torch.randn(n_components, 1, 1, grid_size, device=device) * 0.1)
        
        # Matrices (2D)
        self.mat_xy = nn.Parameter(torch.randn(n_components, grid_size, grid_size, 1, device=device) * 0.1)
        self.mat_xz = nn.Parameter(torch.randn(n_components, grid_size, 1, grid_size, device=device) * 0.1)
        self.mat_yz = nn.Parameter(torch.randn(n_components, 1, grid_size, grid_size, device=device) * 0.1)
        
        # Decoder
        self.density_decoder = nn.Linear(n_components, 1)
        self.color_decoder = nn.Sequential(
            nn.Linear(n_components + 27, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    
    def forward(self, positions, directions):
        """
        positions: (batch, 3) in [-1, 1]
        """
        # Normalize to grid coordinates
        grid_coords = (positions + 1) / 2 * (self.grid_size - 1)
        
        # Sample from factorized volume
        features = self._sample_vm(grid_coords)
        
        # Decode
        density = F.relu(self.density_decoder(features))
        
        # Color with direction
        dir_enc = self._sh_encoding(directions)
        color = self.color_decoder(torch.cat([features, dir_enc], dim=-1))
        
        return color, density
    
    def _sample_vm(self, coords):
        """Sample from VM decomposition"""
        # Bilinear sampling from each component
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        
        # This is simplified - real implementation uses grid_sample
        features = torch.zeros(coords.shape[0], self.vec_x.shape[0], device=self.device)
        
        return features
    
    def _sh_encoding(self, directions):
        """Spherical harmonics encoding for directions"""
        # Simplified SH degree 2
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        return torch.stack([
            torch.ones_like(x),
            x, y, z,
            x*y, y*z, 2*z*z - x*x - y*y, x*z, x*x - y*y
        ] + [torch.zeros_like(x)] * 18, dim=-1)  # Pad to 27


class Plenoxels(nn.Module):
    """
    Plenoxels: No neural network at all!
    Direct optimization of sparse voxel grid with SH
    """
    
    def __init__(self, grid_size=128, sh_degree=2, device='cuda'):
        super().__init__()
        self.device = device
        self.grid_size = grid_size
        self.n_sh = (sh_degree + 1) ** 2
        
        # Sparse voxel grid (density + SH coefficients)
        # In practice, use sparse data structure
        self.density = nn.Parameter(
            torch.zeros(grid_size, grid_size, grid_size, 1, device=device)
        )
        self.sh_coeffs = nn.Parameter(
            torch.zeros(grid_size, grid_size, grid_size, self.n_sh * 3, device=device)
        )
    
    def forward(self, positions, directions):
        """
        Direct trilinear interpolation - no MLP!
        """
        # Normalize positions to [0, grid_size-1]
        grid_coords = (positions + 1) / 2 * (self.grid_size - 1)
        
        # Trilinear interpolation
        density = self._trilinear_sample(self.density, grid_coords)
        sh_coeffs = self._trilinear_sample(self.sh_coeffs, grid_coords)
        
        # Evaluate SH for view-dependent color
        color = self._evaluate_sh(sh_coeffs, directions)
        
        return color, F.relu(density)
    
    def _trilinear_sample(self, volume, coords):
        """Sample from 3D volume with trilinear interpolation"""
        # Use F.grid_sample for efficient implementation
        grid = coords.view(1, 1, 1, -1, 3)
        grid = grid / (self.grid_size - 1) * 2 - 1  # Normalize to [-1, 1]
        
        volume_5d = volume.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
        sampled = F.grid_sample(volume_5d, grid, align_corners=True, mode='bilinear')
        
        return sampled.squeeze().T
    
    def _evaluate_sh(self, sh_coeffs, directions):
        """Evaluate spherical harmonics"""
        # Reshape and evaluate (simplified)
        sh_rgb = sh_coeffs.view(-1, self.n_sh, 3)
        
        # SH basis functions (degree 0 and 1 shown)
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        basis = torch.stack([
            torch.ones_like(x) * 0.282,
            y * 0.489, z * 0.489, x * 0.489,
            x*y * 1.093, y*z * 1.093, (2*z*z - x*x - y*y) * 0.315,
            x*z * 1.093, (x*x - y*y) * 0.546
        ], dim=-1)
        
        # Dot product
        color = (basis.unsqueeze(-1) * sh_rgb).sum(dim=1)
        
        return torch.sigmoid(color)


class SphericalHarmonicsEncoding(nn.Module):
    def __init__(self, degree=4):
        super().__init__()
        self.degree = degree
    
    def forward(self, directions):
        # Returns SH features for direction
        return directions  # Simplified


def speedup_summary():
    """Summary of speedup techniques"""
    
    print("NeRF Speedup Techniques Summary:")
    print("=" * 70)
    
    techniques = [
        ("Instant-NGP", "Hash encoding", "Learnable multi-resolution hash tables", "5 min"),
        ("TensoRF", "Tensor factorization", "VM decomposition of 3D volume", "15 min"),
        ("Plenoxels", "Sparse voxels", "Direct optimization, no MLP", "10 min"),
        ("DVGO", "Voxel + MLP", "Coarse voxel + fine MLP", "15 min"),
        ("K-Planes", "Factored planes", "Hexplane factorization", "20 min"),
    ]
    
    print(f"{'Method':<15} {'Representation':<20} {'Key Idea':<35} {'Train Time'}")
    print("-" * 80)
    for method, rep, idea, time in techniques:
        print(f"{method:<15} {rep:<20} {idea:<35} {time}")
```

**Why Each Method is Fast:**

| Method | Speedup Source |
|--------|---------------|
| Instant-NGP | Hash lookup O(1) vs MLP forward pass |
| TensoRF | Compact factorized storage, fewer parameters |
| Plenoxels | No neural network, direct optimization |
| DVGO | Coarse voxel skips empty space |

**Trade-offs:**

| Method | Memory | Quality | Editability |
|--------|--------|---------|-------------|
| Original NeRF | Low | High | Poor |
| Instant-NGP | Medium | High | Poor |
| TensoRF | Medium | Very High | Medium |
| Plenoxels | High | Medium-High | Good |

**Interview Tip:** The fundamental insight is that MLPs are slow but compact, while explicit structures (voxels, hash tables) are fast but memory-intensive. Modern methods achieve both by using learned encodings (Instant-NGP) or efficient factorizations (TensoRF). The trend is toward hybrid methods that get the best of both worlds.

---

### Question 36
- [ ] Done
**How do you handle sparse or unevenly distributed camera viewpoints in 3D reconstruction?**

**Answer:**

Sparse views cause ambiguity in geometry and appearance. Solutions include: (1) depth supervision from monocular depth estimators, (2) regularization priors (smoothness, sparsity), (3) semantic/feature constraints from pretrained models, (4) diffusion-based priors for plausible inpainting, and (5) data augmentation with virtual views. The key is adding constraints that reduce the ill-posedness.

**Challenges with Sparse Views:**

| Issue | Symptom | Cause |
|-------|---------|-------|
| Underconstrained geometry | Floaters, holes | Not enough triangulation |
| Overfitting | Good on training, bad on novel | Memorizing training views |
| Ambiguous depth | Flat surfaces | Insufficient parallax |
| Missing regions | Blank areas | Unobserved geometry |

**Solution Strategies:**

| Strategy | Method | When to Use |
|----------|--------|-------------|
| Depth supervision | Monocular depth estimator | Always helps |
| Semantic features | DINO/CLIP features | Consistent appearance |
| Diffusion prior | SDS loss, Zero-1-to-3 | Few-shot (<10 views) |
| Regularization | Total variation, sparsity | General |
| Geometry priors | Normal estimation, symmetry | Objects |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseViewNeRF(nn.Module):
    """NeRF with additional priors for sparse views"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Base NeRF
        self.nerf = NeRFMLP()
        
        # Pretrained depth estimator (frozen)
        self.depth_estimator = self._load_depth_estimator()
        
        # Pretrained feature extractor (e.g., DINO)
        self.feature_extractor = self._load_feature_extractor()
        
        # Loss weights
        self.weights = {
            'rgb': 1.0,
            'depth': 0.1,
            'feature': 0.01,
            'smoothness': 0.001
        }
    
    def _load_depth_estimator(self):
        """Load pretrained monocular depth (e.g., MiDaS, DPT)"""
        # Placeholder - use torchvision or timm
        return None
    
    def _load_feature_extractor(self):
        """Load pretrained feature extractor (e.g., DINO)"""
        return None
    
    def compute_losses(self, rays, gt_rgb, gt_depth=None):
        """
        Compute all losses for sparse view training
        """
        losses = {}
        
        # Render
        rendered_rgb, rendered_depth, weights = self.render_rays(rays)
        
        # RGB loss (always)
        losses['rgb'] = F.mse_loss(rendered_rgb, gt_rgb)
        
        # Depth loss (if available or from estimator)
        if gt_depth is not None:
            # Scale-invariant depth loss
            losses['depth'] = self._scale_invariant_depth_loss(rendered_depth, gt_depth)
        
        # Smoothness regularization
        losses['smoothness'] = self._depth_smoothness_loss(rendered_depth)
        
        # Total loss
        total = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        
        return total, losses
    
    def _scale_invariant_depth_loss(self, pred, gt):
        """
        Scale-invariant depth loss (robust to unknown scale)
        """
        # Mask valid depths
        valid = (gt > 0) & (pred > 0)
        
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Log space
        log_pred = torch.log(pred[valid])
        log_gt = torch.log(gt[valid])
        
        diff = log_pred - log_gt
        
        # Scale-invariant loss
        loss = (diff ** 2).mean() - 0.5 * (diff.mean() ** 2)
        
        return loss
    
    def _depth_smoothness_loss(self, depth):
        """
        Total variation smoothness on depth
        """
        # Assuming depth is (H, W)
        if depth.dim() == 1:
            return torch.tensor(0.0, device=self.device)
        
        tv_h = torch.abs(depth[1:] - depth[:-1]).mean()
        tv_w = torch.abs(depth[:, 1:] - depth[:, :-1]).mean()
        
        return tv_h + tv_w
    
    def render_rays(self, rays):
        """Render with volumetric rendering"""
        # Placeholder
        return torch.zeros(rays.shape[0], 3), torch.zeros(rays.shape[0]), None


class DepthRegularization:
    """Monocular depth for supervision"""
    
    @staticmethod
    def get_monocular_depth(images, model='midas'):
        """
        Get depth estimates from pretrained monocular model
        
        These provide relative depth, not absolute scale
        """
        # In practice: 
        # from transformers import DPTForDepthEstimation
        # or use MiDaS
        pass
    
    @staticmethod
    def pearson_depth_loss(pred_depth, mono_depth):
        """
        Pearson correlation loss for relative depth
        Handles unknown scale and shift
        """
        pred = pred_depth.flatten()
        mono = mono_depth.flatten()
        
        # Normalize
        pred_norm = (pred - pred.mean()) / (pred.std() + 1e-6)
        mono_norm = (mono - mono.mean()) / (mono.std() + 1e-6)
        
        # Correlation (want to maximize, so negative)
        correlation = (pred_norm * mono_norm).mean()
        
        return 1 - correlation


class SemanticFeatureRegularization(nn.Module):
    """Use pretrained features for consistency"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        # Load DINO or CLIP
        # self.dino = torch.hub.load('facebookresearch/dino', 'dino_vits16')
        self.feature_dim = 384  # DINO-S
    
    def feature_consistency_loss(self, rendered_features, gt_features):
        """
        Ensure rendered view has consistent semantic features
        """
        return F.cosine_embedding_loss(
            rendered_features, 
            gt_features,
            torch.ones(rendered_features.shape[0], device=rendered_features.device)
        )


class DiffusionPrior(nn.Module):
    """
    Score Distillation Sampling (SDS) for very sparse views
    Uses diffusion model as 3D prior
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        # Load stable diffusion or specialized 3D diffusion
        # self.diffusion = StableDiffusion()
        self.guidance_scale = 100
    
    def sds_loss(self, rendered_image, text_prompt=None):
        """
        Score Distillation Sampling loss
        Guides NeRF optimization using diffusion model gradients
        """
        # Add noise
        t = torch.randint(20, 980, (1,))  # Random timestep
        noise = torch.randn_like(rendered_image)
        noisy_image = self._add_noise(rendered_image, noise, t)
        
        # Get diffusion model prediction
        # noise_pred = self.diffusion.predict_noise(noisy_image, t, text_prompt)
        
        # SDS gradient: (noise_pred - noise) weighted by sigma
        # grad = guidance_scale * sigma * (noise_pred - noise)
        
        # This provides gradients that push NeRF toward images
        # that look plausible to the diffusion model
        pass
    
    def _add_noise(self, x, noise, t):
        """Add noise according to diffusion schedule"""
        # Simplified
        alpha = 1 - t.float() / 1000
        return alpha.sqrt() * x + (1 - alpha).sqrt() * noise


class VirtualViewAugmentation:
    """Generate virtual training views"""
    
    @staticmethod
    def interpolate_cameras(cam1, cam2, num_steps=5):
        """
        Create interpolated camera poses between sparse views
        """
        # Interpolate position
        positions = torch.lerp(
            cam1['position'].unsqueeze(0).expand(num_steps, -1),
            cam2['position'].unsqueeze(0).expand(num_steps, -1),
            torch.linspace(0, 1, num_steps).unsqueeze(-1)
        )
        
        # SLERP for rotation
        rotations = VirtualViewAugmentation._slerp(
            cam1['rotation'], cam2['rotation'], 
            torch.linspace(0, 1, num_steps)
        )
        
        return [{'position': p, 'rotation': r} 
                for p, r in zip(positions, rotations)]
    
    @staticmethod
    def _slerp(q1, q2, t):
        """Spherical linear interpolation for quaternions"""
        # Simplified
        return q1 * (1 - t) + q2 * t


class RegNeRF(nn.Module):
    """
    RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs
    """
    
    def __init__(self, nerf, device='cuda'):
        super().__init__()
        self.nerf = nerf
        self.device = device
    
    def patch_regularization(self, rendered_patches):
        """
        Regularize patches from unseen viewpoints
        Uses normalizing flow to measure plausibility
        """
        # In practice: train normalizing flow on natural image patches
        # Low likelihood → high penalty
        pass
    
    def depth_smoothness(self, rendered_depth):
        """
        Encourage smooth depth in textureless regions
        """
        # Bilateral smoothness
        pass


def sparse_view_pipeline():
    """Recommended pipeline for sparse view reconstruction"""
    
    pipeline = [
        ("1. Preprocess", [
            "Run COLMAP for camera poses",
            "Compute monocular depth with MiDaS/DPT",
            "Extract DINO features for each view"
        ]),
        ("2. Training Losses", [
            "RGB reconstruction (MSE or L1)",
            "Depth ranking/correlation loss",
            "Feature consistency loss",
            "Smoothness regularization"
        ]),
        ("3. Augmentation", [
            "Virtual views from camera interpolation",
            "Random perturbations of camera pose",
            "Patch-based training"
        ]),
        ("4. Optional: Diffusion Prior", [
            "SDS loss for very sparse (<5 views)",
            "Zero-1-to-3 for single-view",
            "Text-conditioned generation"
        ])
    ]
    
    print("Sparse View Reconstruction Pipeline:")
    print("=" * 60)
    for stage, steps in pipeline:
        print(f"\n{stage}:")
        for step in steps:
            print(f"  • {step}")


class NeRFMLP(nn.Module):
    """Placeholder NeRF MLP"""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 256), nn.ReLU(),
            nn.Linear(256, 4)
        )
```

**View Count Guidelines:**

| # Views | Challenge Level | Key Techniques |
|---------|-----------------|----------------|
| 50+ | Easy | Standard NeRF works |
| 20-50 | Medium | Depth supervision |
| 5-20 | Hard | Depth + features + regularization |
| 1-5 | Very Hard | Diffusion priors essential |
| 1 | Single-view | Zero-1-to-3, Wonder3D |

**Interview Tip:** The fundamental problem is that 3D from sparse 2D is ill-posed—many 3D scenes can explain the same few images. Priors reduce ambiguity by telling the model what "reasonable" scenes look like. Monocular depth provides geometric priors, semantic features provide appearance consistency, and diffusion models provide general "what looks real" priors. For production, always use monocular depth—it's cheap and very effective.

---

## OCR (Optical Character Recognition)

### Question 37
- [ ] Done
**Compare traditional OCR pipeline (detection + recognition) vs. end-to-end approaches.**

**Answer:**

Traditional OCR uses separate detection (find text regions) and recognition (read characters) stages, allowing optimization of each. End-to-end approaches jointly detect and recognize in a single model, eliminating error propagation and handling diverse layouts better. Traditional is more modular and debuggable; end-to-end achieves higher accuracy on complex scenes.

**Pipeline Comparison:**

| Aspect | Traditional (Two-Stage) | End-to-End |
|--------|------------------------|------------|
| Architecture | Detector → Recognizer | Single unified model |
| Error propagation | Detection errors cascade | Jointly optimized |
| Modularity | High (swap components) | Low (monolithic) |
| Training data | Can train separately | Needs paired annotations |
| Speed | May be faster (parallel) | Often slower |
| Scene text | Challenging | Better handling |

**Traditional Pipeline:**

```
Image → Text Detection (EAST, CRAFT) → Crop boxes → Recognition (CRNN) → Text output
                ↓
        Bounding boxes for each word/line
```

**End-to-End Pipeline:**

```
Image → Single Model (TROCR, PaddleOCR, Donut) → Text + positions
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# ============ Traditional Two-Stage Pipeline ============

class TraditionalOCRPipeline:
    """Two-stage: Detection + Recognition"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.detector = TextDetector()
        self.recognizer = CRNNRecognizer()
    
    def __call__(self, image):
        """
        image: (C, H, W) tensor
        Returns: list of (text, bbox)
        """
        # Stage 1: Detect text regions
        boxes = self.detector(image)
        
        # Stage 2: Recognize each region
        results = []
        for box in boxes:
            cropped = self._crop_and_rectify(image, box)
            text = self.recognizer(cropped)
            results.append({'text': text, 'box': box})
        
        return results
    
    def _crop_and_rectify(self, image, box):
        """Crop and perspective-transform text region"""
        # Apply perspective transform if quadrilateral
        # Resize to fixed height, variable width
        return image[:, box[1]:box[3], box[0]:box[2]]


class TextDetector(nn.Module):
    """EAST/CRAFT-style text detector"""
    
    def __init__(self):
        super().__init__()
        # Feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        
        # Predict score map and geometry
        self.score_head = nn.Conv2d(256, 1, 1)  # Text/non-text
        self.geometry_head = nn.Conv2d(256, 5, 1)  # RBOX: 4 distances + angle
    
    def forward(self, x):
        features = self.backbone(x)
        score = torch.sigmoid(self.score_head(features))
        geometry = self.geometry_head(features)
        
        # Post-process to get boxes (NMS, etc.)
        boxes = self._decode_predictions(score, geometry)
        return boxes
    
    def _decode_predictions(self, score, geometry, threshold=0.5):
        """Convert pixel-level predictions to bounding boxes"""
        # Find high-confidence pixels
        # Decode distances to get box corners
        # Apply NMS
        return []  # List of boxes


class CRNNRecognizer(nn.Module):
    """
    CRNN: CNN + RNN + CTC for sequence recognition
    Classic architecture for text recognition
    """
    
    def __init__(self, num_classes=97, hidden_size=256):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(),  # 1xN output
        )
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, 
                           num_layers=2, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        """
        x: (B, 1, H, W) grayscale image (fixed height, variable width)
        Returns: (B, T, num_classes) logits
        """
        # CNN: (B, 1, 32, W) → (B, 512, 1, W')
        conv = self.cnn(x)
        
        # Reshape for RNN: (B, 512, 1, W') → (B, W', 512)
        B, C, H, W = conv.shape
        conv = conv.squeeze(2).permute(0, 2, 1)
        
        # RNN
        rnn_out, _ = self.rnn(conv)
        
        # Classifier
        output = self.fc(rnn_out)
        
        return output
    
    def decode_ctc(self, output):
        """
        Decode CTC output using greedy decoding
        """
        # Get best path
        _, indices = output.max(dim=-1)
        
        # Remove duplicates and blanks
        decoded = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != 0:  # 0 is blank
                decoded.append(idx)
            prev = idx
        
        return decoded


# ============ End-to-End Approaches ============

class TrOCRStyle(nn.Module):
    """
    Transformer-based end-to-end OCR
    Encoder-decoder like TrOCR, Donut
    """
    
    def __init__(self, vocab_size=50000, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        # Vision encoder (ViT-style)
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model))
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_encoder_layers
        )
        
        # Text decoder
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True),
            num_decoder_layers
        )
        
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, images, target_ids=None):
        """
        images: (B, 3, H, W)
        target_ids: (B, T) for training
        """
        # Encode image
        patches = self.patch_embed(images)  # (B, d_model, H', W')
        B, C, H, W = patches.shape
        patches = patches.flatten(2).permute(0, 2, 1)  # (B, H'*W', d_model)
        patches = patches + self.pos_embed[:, :patches.shape[1]]
        
        memory = self.encoder(patches)
        
        if target_ids is not None:
            # Training: teacher forcing
            target_emb = self.token_embed(target_ids)
            causal_mask = self._generate_square_subsequent_mask(target_ids.shape[1])
            
            output = self.decoder(target_emb, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output)
            
            return logits
        else:
            # Inference: autoregressive generation
            return self._generate(memory)
    
    def _generate(self, memory, max_len=100):
        """Autoregressive text generation"""
        B = memory.shape[0]
        generated = torch.zeros(B, 1, dtype=torch.long)  # Start token
        
        for _ in range(max_len):
            target_emb = self.token_embed(generated)
            output = self.decoder(target_emb, memory)
            logits = self.output_proj(output[:, -1:])
            next_token = logits.argmax(dim=-1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == 2).all():  # End token
                break
        
        return generated
    
    def _generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))


def compare_approaches():
    """Compare traditional vs end-to-end"""
    
    comparison = {
        'Accuracy (printed)': ('High', 'Very High'),
        'Accuracy (scene text)': ('Medium', 'High'),
        'Speed': ('Fast (parallel)', 'Slower'),
        'Training complexity': ('Simpler (modular)', 'Complex (joint)'),
        'Data requirements': ('Can pretrain separately', 'Needs full pipeline data'),
        'Flexibility': ('Easy to swap components', 'Monolithic'),
        'Error analysis': ('Clear bottleneck', 'Black box'),
    }
    
    print("Traditional vs End-to-End OCR:")
    print("=" * 60)
    print(f"{'Aspect':<25} {'Traditional':<20} {'End-to-End'}")
    print("-" * 60)
    for aspect, (trad, e2e) in comparison.items():
        print(f"{aspect:<25} {trad:<20} {e2e}")
```

**Popular Models:**

| Type | Model | Key Feature |
|------|-------|-------------|
| Traditional | EAST + CRNN | Fast, modular |
| Traditional | CRAFT + Attention | Better curved text |
| End-to-End | TrOCR | Transformer, pretrained |
| End-to-End | PaddleOCR | Production-ready, multilingual |
| End-to-End | Donut | Document understanding |

**Interview Tip:** Traditional pipelines are still preferred for production when you need: (1) speed (batch detection then parallel recognition), (2) interpretability (know which stage failed), or (3) customization (swap recognizer for different languages). End-to-end models excel when dealing with complex layouts, curved text, or when you have abundant paired training data.

---

### Question 38
- [ ] Done
**How do transformer-based OCR models (TrOCR) improve upon CNN-RNN-CTC approaches?**

**Answer:**

TrOCR replaces CNN+RNN+CTC with Vision Transformer encoder + Transformer decoder. Improvements include: (1) global context from self-attention vs local CNN, (2) no CTC alignment issues, (3) pretrained language model decoder for better context, and (4) natural handling of variable-length output through autoregressive generation.

**Architecture Comparison:**

| Component | CNN-RNN-CTC | TrOCR |
|-----------|-------------|-------|
| Image encoding | CNN (local) | ViT (global attention) |
| Sequence modeling | BiLSTM | Transformer encoder |
| Decoding | CTC (monotonic) | Autoregressive (flexible) |
| Alignment | CTC forced alignment | Cross-attention |
| Language model | None built-in | Pretrained decoder |

**CTC Limitations vs Transformer Advantages:**

| CTC Limitation | Transformer Solution |
|----------------|---------------------|
| Assumes monotonic alignment | Cross-attention learns flexible alignment |
| Independent character predictions | Autoregressive considers previous tokens |
| No language model | Pretrained decoder has language knowledge |
| Fixed output per input length | Variable length generation |
| Struggles with repeated chars | Natural handling |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNWithCTC(nn.Module):
    """Traditional CNN-RNN-CTC approach"""
    
    def __init__(self, num_classes=97, hidden_size=256):
        super().__init__()
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        
        # BiLSTM for sequence modeling
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, 
                           num_layers=2, batch_first=True)
        
        # CTC output
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        """
        x: (B, 1, 32, W) 
        Returns: (B, T, num_classes) log probabilities for CTC
        """
        conv = self.cnn(x)  # (B, 512, 4, W')
        B, C, H, W = conv.shape
        
        # Collapse height, permute for RNN
        conv = conv.view(B, C * H, W).permute(0, 2, 1)  # (B, W', C*H)
        
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        
        return F.log_softmax(output, dim=-1)
    
    def ctc_loss(self, output, targets, input_lengths, target_lengths):
        """CTC loss for training"""
        # output: (B, T, C) → (T, B, C) for CTC
        output = output.permute(1, 0, 2)
        return F.ctc_loss(output, targets, input_lengths, target_lengths, 
                          blank=0, zero_infinity=True)


class TrOCR(nn.Module):
    """
    TrOCR: Transformer-based OCR
    Uses ViT encoder + GPT-style decoder
    """
    
    def __init__(self, vocab_size=50265, d_model=768, nhead=12,
                 encoder_layers=12, decoder_layers=6, img_size=384, patch_size=16):
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # ViT Encoder
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        
        # Text Decoder (pretrained from language model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.decoder_pos_embed = nn.Embedding(512, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def encode_image(self, images):
        """Encode image with ViT"""
        B = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed[:, :x.shape[1]]
        
        # Transformer encoder
        memory = self.encoder(x)
        
        return memory
    
    def forward(self, images, target_ids=None):
        """
        images: (B, 3, H, W)
        target_ids: (B, T) text token ids for teacher forcing
        """
        # Encode image
        memory = self.encode_image(images)
        
        if target_ids is not None:
            # Training with teacher forcing
            positions = torch.arange(target_ids.shape[1], device=target_ids.device)
            target_emb = self.token_embed(target_ids) + self.decoder_pos_embed(positions)
            
            # Causal mask
            causal_mask = self._causal_mask(target_ids.shape[1], target_ids.device)
            
            output = self.decoder(target_emb, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output)
            
            return logits
        else:
            return self.generate(memory)
    
    def generate(self, memory, max_len=128, bos_token=0, eos_token=2):
        """Autoregressive generation"""
        B = memory.shape[0]
        device = memory.device
        
        # Start with BOS token
        generated = torch.full((B, 1), bos_token, dtype=torch.long, device=device)
        
        for step in range(max_len):
            positions = torch.arange(generated.shape[1], device=device)
            target_emb = self.token_embed(generated) + self.decoder_pos_embed(positions)
            
            causal_mask = self._causal_mask(generated.shape[1], device)
            
            output = self.decoder(target_emb, memory, tgt_mask=causal_mask)
            logits = self.output_proj(output[:, -1])  # Last position
            
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == eos_token).all():
                break
        
        return generated
    
    def _causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))


class CTCProblems:
    """Illustrate CTC limitations"""
    
    @staticmethod
    def repeated_characters():
        """CTC struggles with repeated characters like 'book' → 'bok'"""
        print("CTC Repeated Character Problem:")
        print("  Target: 'book' (4 chars)")
        print("  CTC path: b-o-o-k needs explicit blank between o's")
        print("  Issue: Model may predict b-o-k (merging repeated o's)")
        print("  Transformer: Autoregressive naturally handles 'o', 'o'")
    
    @staticmethod
    def variable_length():
        """CTC output length tied to input length"""
        print("\nCTC Length Constraint:")
        print("  Output length <= Input length / reduction_factor")
        print("  Very short words may not have enough frames")
        print("  Transformer: Output length independent of input")
    
    @staticmethod
    def no_language_model():
        """CTC has no built-in language modeling"""
        print("\nCTC Language Modeling:")
        print("  CTC predicts each character independently")
        print("  P(c_t) doesn't depend on P(c_1...c_{t-1})")
        print("  Needs external LM for beam search")
        print("  Transformer: Decoder is a language model!")


def pretraining_benefits():
    """Explain TrOCR pretraining advantages"""
    
    print("TrOCR Pretraining Strategy:")
    print("=" * 60)
    
    stages = [
        ("Stage 1: Encoder", "Pretrain ViT on ImageNet (visual understanding)"),
        ("Stage 2: Decoder", "Initialize from RoBERTa/GPT (language knowledge)"),
        ("Stage 3: Joint", "Finetune on OCR data (image-to-text mapping)"),
    ]
    
    for stage, description in stages:
        print(f"  {stage}: {description}")
    
    print("\nBenefits:")
    print("  - Encoder understands visual patterns")
    print("  - Decoder understands language/spelling")
    print("  - Joint training aligns modalities")
    print("  - Works with less OCR-specific data")


def attention_advantage():
    """Show how cross-attention helps alignment"""
    
    print("\nCross-Attention vs CTC Alignment:")
    print("=" * 60)
    print("CTC:")
    print("  - Forced monotonic left-to-right alignment")
    print("  - Cannot handle non-left-to-right scripts easily")
    print("  - Alignment is implicit in path")
    
    print("\nTransformer Cross-Attention:")
    print("  - Soft attention over all image patches")
    print("  - Can attend to any relevant region")
    print("  - Learns alignment during training")
    print("  - Handles arbitrary reading orders")
```

**Key TrOCR Improvements:**

| Feature | CNN-RNN-CTC | TrOCR |
|---------|-------------|-------|
| Context | Local (CNN) + sequential (RNN) | Global (self-attention) |
| Pretraining | ImageNet CNN | ViT + LM |
| Character dependencies | Independent (CTC) | Sequential (autoregressive) |
| Vocabulary | Character-level | Subword (BPE) |
| Language knowledge | External LM | Built-in |

**Interview Tip:** The key insight is that TrOCR treats OCR as image captioning—a well-studied encoder-decoder problem. The pretrained decoder brings language understanding (spelling patterns, word frequencies), while the ViT encoder provides global context. CTC's independence assumption limits accuracy on real-world text where context matters ("their" vs "there").

---

### Question 39
- [ ] Done
**What are the key challenges in handwritten vs. printed text recognition?**

**Answer:**

Handwritten text has high variability (different styles, slants, connections), inconsistent spacing, and ambiguous characters. Printed text has uniform fonts but may have noise, degradation, or unusual fonts. Handwritten recognition requires style-adaptive models and more training data; printed recognition focuses on robustness to image quality issues.

**Challenge Comparison:**

| Challenge | Handwritten | Printed |
|-----------|-------------|---------|
| Style variability | Very High (per person) | Low (per font) |
| Character shapes | Inconsistent | Consistent |
| Word spacing | Irregular | Regular |
| Connected letters | Common (cursive) | Rare |
| Slant/skew | Varies per writer | Usually upright |
| Training data | Hard to collect | Easier to synthesize |
| Character ambiguity | High (a/o, n/u, l/1) | Lower |
| Baseline | Wavy | Straight |

**Specific Handwriting Challenges:**

| Issue | Example | Impact |
|-------|---------|--------|
| Writer style | Same letter looks different | Need style normalization |
| Cursive | Letters connected | Segmentation hard |
| Slant | Italic personal style | Need deslanting |
| Pressure | Varying line thickness | Feature extraction |
| Abbreviations | Personal shortcuts | OOV handling |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class HandwritingPreprocessor:
    """Preprocessing specifically for handwritten text"""
    
    def __init__(self):
        self.slant_corrector = SlantCorrector()
        self.baseline_normalizer = BaselineNormalizer()
    
    def __call__(self, image):
        """
        image: (C, H, W) handwritten text line
        Returns: normalized image
        """
        # 1. Binarization (Otsu or adaptive)
        binary = self._adaptive_binarize(image)
        
        # 2. Deskew (remove rotation)
        deskewed = self._deskew(binary)
        
        # 3. Slant correction (personal writing angle)
        slant_corrected = self.slant_corrector(deskewed)
        
        # 4. Baseline normalization
        normalized = self.baseline_normalizer(slant_corrected)
        
        # 5. Size normalization
        resized = F.interpolate(normalized.unsqueeze(0), size=(64, None))
        
        return resized.squeeze(0)
    
    def _adaptive_binarize(self, image):
        """Adaptive thresholding for varying backgrounds"""
        # Convert to grayscale
        if image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image[0]
        
        # Local adaptive threshold (simplified)
        kernel_size = 31
        local_mean = F.avg_pool2d(gray.unsqueeze(0).unsqueeze(0), 
                                   kernel_size, stride=1, 
                                   padding=kernel_size//2)
        binary = (gray.unsqueeze(0) < local_mean.squeeze() - 0.1).float()
        
        return binary
    
    def _deskew(self, image):
        """Correct document rotation"""
        # Use Hough transform to find dominant angle
        # Apply rotation correction
        return image  # Placeholder


class SlantCorrector(nn.Module):
    """Correct personal writing slant"""
    
    def __init__(self, max_slant=45):
        super().__init__()
        self.max_slant = max_slant
    
    def forward(self, image):
        """
        Detect and correct slant using vertical projection profile
        """
        # Try different shear angles
        # Find angle that maximizes vertical projection variance
        best_angle = self._find_slant_angle(image)
        
        # Apply shear transformation
        corrected = self._shear_image(image, -best_angle)
        
        return corrected
    
    def _find_slant_angle(self, image):
        """Find dominant slant angle"""
        # Simplified: use vertical projection analysis
        return 0  # Placeholder
    
    def _shear_image(self, image, angle):
        """Apply horizontal shear"""
        return image  # Placeholder


class BaselineNormalizer(nn.Module):
    """Normalize text to horizontal baseline"""
    
    def forward(self, image):
        """
        Detect baseline and normalize
        """
        # Find lower contour of text
        # Fit polynomial/spline
        # Warp to straighten
        return image


class WriterAdaptiveRecognizer(nn.Module):
    """
    Adaptive recognition that adjusts to writer style
    Uses style embedding or adaptation layers
    """
    
    def __init__(self, base_model, num_writers=1000, style_dim=128):
        super().__init__()
        self.base_model = base_model
        
        # Writer style embedding
        self.writer_embed = nn.Embedding(num_writers, style_dim)
        
        # Style adaptation layer
        self.style_adapter = nn.Sequential(
            nn.Linear(style_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
    
    def forward(self, image, writer_id=None):
        """
        If writer_id known, use style embedding
        Otherwise, extract style from image
        """
        if writer_id is not None:
            style = self.writer_embed(writer_id)
        else:
            style = self._extract_style(image)
        
        # Adapt model features using style
        adapted_features = self._adapt(image, style)
        
        return self.base_model.decode(adapted_features)
    
    def _extract_style(self, image):
        """Extract style representation from image"""
        # Use style encoder trained with contrastive learning
        return torch.zeros(image.shape[0], 128)
    
    def _adapt(self, image, style):
        """Apply style-based feature adaptation"""
        # FiLM conditioning or similar
        return image


class CursiveSegmentation(nn.Module):
    """Handle connected/cursive writing"""
    
    def __init__(self):
        super().__init__()
        # Use implicit segmentation (attention) rather than explicit
        self.attention_ocr = AttentionOCR()
    
    def forward(self, image):
        """
        Attention-based decoding naturally handles
        cursive without explicit segmentation
        """
        return self.attention_ocr(image)


class AttentionOCR(nn.Module):
    """Attention-based OCR for cursive text"""
    
    def __init__(self, hidden_size=256, vocab_size=100):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
        )
        
        # Attention decoder
        self.attention = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
        self.decoder = nn.GRU(hidden_size + vocab_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, image):
        features = self.encoder(image)
        # Attention over spatial positions
        # Decoder generates sequence
        return features


class PrintedTextChallenges:
    """Specific challenges for printed text"""
    
    @staticmethod
    def quality_issues():
        """Common quality problems in printed documents"""
        
        issues = {
            'Low resolution': 'Scan/photo quality, need super-resolution',
            'Noise': 'Scanner artifacts, paper texture',
            'Blur': 'Motion blur, out of focus',
            'Compression artifacts': 'JPEG blocking, text edge degradation',
            'Faded text': 'Old documents, light ink',
            'Background clutter': 'Watermarks, stamps, graphics',
            'Non-standard fonts': 'Decorative, display fonts',
            'Damaged documents': 'Tears, stains, folds',
        }
        
        return issues
    
    @staticmethod
    def augmentation_strategies():
        """Data augmentation for printed text robustness"""
        
        augmentations = [
            ('Motion blur', 'Kernels in various directions'),
            ('Gaussian noise', 'Simulate low-light capture'),
            ('JPEG compression', 'Various quality levels'),
            ('Perspective warp', 'Non-flat documents'),
            ('Brightness/contrast', 'Varying lighting'),
            ('Erosion/dilation', 'Simulate ink spread/fade'),
        ]
        
        return augmentations


def create_augmented_dataset():
    """Augmentation pipeline for robust OCR training"""
    
    handwritten_aug = T.Compose([
        T.RandomAffine(degrees=5, shear=15, scale=(0.9, 1.1)),
        T.ElasticTransform(alpha=50.0),
        T.GaussianBlur(kernel_size=3),
        T.RandomInvert(p=0.1),
    ])
    
    printed_aug = T.Compose([
        T.RandomAffine(degrees=2, translate=(0.05, 0.05)),
        T.GaussianBlur(kernel_size=(1, 5)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        # Simulate JPEG artifacts would need custom transform
    ])
    
    return handwritten_aug, printed_aug


def compare_challenges():
    """Summary comparison"""
    
    print("Handwritten vs Printed OCR Challenges:")
    print("=" * 70)
    print(f"{'Aspect':<25} {'Handwritten':<25} {'Printed'}")
    print("-" * 70)
    
    comparisons = [
        ("Main difficulty", "Writer variability", "Quality/degradation"),
        ("Segmentation", "Cursive connections", "Usually easy"),
        ("Training data", "Expensive collection", "Easy synthesis"),
        ("Preprocessing", "Slant, baseline norm", "Deskew, denoise"),
        ("Model approach", "Writer adaptation", "Robust to noise"),
        ("Accuracy (clean)", "95-98%", "99%+"),
        ("Accuracy (degraded)", "80-90%", "90-95%"),
    ]
    
    for aspect, hw, pr in comparisons:
        print(f"{aspect:<25} {hw:<25} {pr}")
```

**Data Collection:**

| Type | Collection Method | Cost |
|------|------------------|------|
| Printed | Synthetic rendering | Low |
| Printed | Scanned documents | Medium |
| Handwritten | Writer enrollment | High |
| Handwritten | Crowdsourcing | High |
| Handwritten | Synthetic (GANs) | Medium |

**Interview Tip:** Handwritten OCR is fundamentally harder because each writer is essentially a different "font." The solution is either: (1) massive training data covering many writers, (2) writer adaptation techniques (fine-tuning, style embedding), or (3) attention-based models that learn implicit segmentation. For printed text, the main challenges are engineering problems (image quality, preprocessing) rather than fundamental modeling problems.

---

### Question 40
- [ ] Done
**How do you handle OCR for documents with complex layouts, tables, and mixed content?**

**Answer:**

Complex layouts require document understanding beyond OCR: (1) layout analysis to detect regions (text, tables, figures), (2) reading order detection to sequence regions logically, (3) table structure recognition for rows/columns, and (4) integration of OCR with layout. Models like LayoutLM combine text+layout+image features; Donut uses end-to-end vision-language approach.

**Pipeline for Complex Documents:**

```
Image → Layout Detection → Region Classification → Reading Order → OCR per region → Structure parsing → Output
```

**Key Components:**

| Component | Purpose | Methods |
|-----------|---------|---------|
| Layout detection | Find text/figure/table regions | Faster R-CNN, DETR |
| Region classification | Identify region types | CNN classifier |
| Reading order | Sequence regions logically | Graph-based, heuristic |
| Table structure | Parse rows/columns/cells | Graph neural networks |
| OCR | Recognize text in regions | CRNN, TrOCR |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DocumentRegion:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    region_type: str  # 'text', 'table', 'figure', 'title', etc.
    content: str = ""
    confidence: float = 0.0


class DocumentLayoutAnalysis(nn.Module):
    """
    Detect and classify document regions
    Similar to object detection but for document elements
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Region types: text, title, table, figure, list, header, footer, etc.
        self.region_types = ['text', 'title', 'table', 'figure', 'list', 
                            'header', 'footer', 'caption', 'equation', 'logo']
        
        # Backbone (ResNet or similar)
        self.backbone = self._build_backbone()
        
        # FPN for multi-scale features
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)
        
        # Detection heads (like Faster R-CNN)
        self.rpn = RegionProposalNetwork(256)
        self.roi_head = ROIHead(256, num_classes)
    
    def _build_backbone(self):
        # Use pretrained ResNet or DocVQA-specific model
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
    
    def forward(self, images):
        """
        images: (B, 3, H, W)
        Returns: list of detected regions per image
        """
        features = self.backbone(images)
        # ... detection pipeline
        return []


class ReadingOrderPredictor(nn.Module):
    """
    Determine logical reading order of detected regions
    Critical for multi-column layouts
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # Encode each region
        self.region_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),  # bbox (4) + size (2) + position (2)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Pairwise ordering
        self.order_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Region A before B?
        )
    
    def forward(self, regions: List[DocumentRegion]):
        """
        Predict reading order of regions
        """
        # Encode regions
        region_features = []
        for r in regions:
            # Features: normalized bbox, relative position, type embedding
            feat = self._encode_region(r)
            region_features.append(feat)
        
        # Pairwise comparison for ordering
        order_matrix = self._compute_order_matrix(region_features)
        
        # Sort by predicted order (topological sort)
        ordered_indices = self._topological_sort(order_matrix)
        
        return [regions[i] for i in ordered_indices]
    
    def _encode_region(self, region):
        x1, y1, x2, y2 = region.bbox
        features = torch.tensor([
            x1, y1, x2, y2,
            x2 - x1, y2 - y1,  # width, height
            (x1 + x2) / 2,  # center x
            (y1 + y2) / 2,  # center y
        ])
        return self.region_encoder(features)
    
    def _compute_order_matrix(self, features):
        n = len(features)
        matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    pair = torch.cat([features[i], features[j]])
                    matrix[i, j] = torch.sigmoid(self.order_predictor(pair))
        
        return matrix
    
    def _topological_sort(self, order_matrix):
        # Simple greedy sort by row sums
        scores = order_matrix.sum(dim=1)
        return torch.argsort(scores, descending=True).tolist()


class TableStructureRecognizer(nn.Module):
    """
    Parse table structure: rows, columns, cells, spanning
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # Visual encoder for table image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        
        # Predict row/column separators
        self.row_predictor = nn.Conv2d(256, 1, 1)  # Horizontal lines
        self.col_predictor = nn.Conv2d(256, 1, 1)  # Vertical lines
        
        # Cell detection
        self.cell_detector = nn.Conv2d(256, 1, 1)
    
    def forward(self, table_image):
        """
        table_image: (B, 3, H, W) cropped table region
        Returns: structured table representation
        """
        features = self.encoder(table_image)
        
        # Predict separators
        row_map = torch.sigmoid(self.row_predictor(features))
        col_map = torch.sigmoid(self.col_predictor(features))
        
        # Find separator positions
        row_positions = self._find_separators(row_map, axis='horizontal')
        col_positions = self._find_separators(col_map, axis='vertical')
        
        # Create cell grid
        cells = self._create_cell_grid(row_positions, col_positions)
        
        return cells
    
    def _find_separators(self, heatmap, axis):
        """Find row/column separator positions from heatmap"""
        # Project and find peaks
        if axis == 'horizontal':
            projection = heatmap.mean(dim=-1)  # Average across width
        else:
            projection = heatmap.mean(dim=-2)  # Average across height
        
        # Peak detection
        peaks = self._find_peaks(projection.squeeze())
        return peaks
    
    def _find_peaks(self, signal, threshold=0.5):
        """Simple peak detection"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        return peaks
    
    def _create_cell_grid(self, rows, cols):
        """Create cell bounding boxes from row/col positions"""
        cells = []
        for i in range(len(rows) - 1):
            row_cells = []
            for j in range(len(cols) - 1):
                cell = {
                    'row': i,
                    'col': j,
                    'bbox': (cols[j], rows[i], cols[j+1], rows[i+1])
                }
                row_cells.append(cell)
            cells.append(row_cells)
        return cells


class LayoutLMStyleModel(nn.Module):
    """
    LayoutLM-style: Combine text, layout, and image
    Jointly encodes all three modalities
    """
    
    def __init__(self, vocab_size=30522, hidden_dim=768, max_position=512):
        super().__init__()
        
        # Text embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # 2D position embedding (x, y, w, h)
        self.x_embed = nn.Embedding(1024, hidden_dim)
        self.y_embed = nn.Embedding(1024, hidden_dim)
        self.w_embed = nn.Embedding(1024, hidden_dim)
        self.h_embed = nn.Embedding(1024, hidden_dim)
        
        # 1D position embedding (token sequence)
        self.position_embed = nn.Embedding(max_position, hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 12, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 12)
        
        # Visual backbone (for LayoutLMv2+)
        self.visual_encoder = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
    
    def forward(self, token_ids, bbox, image=None):
        """
        token_ids: (B, T) OCR text tokens
        bbox: (B, T, 4) bounding box for each token [x1, y1, x2, y2] normalized
        image: (B, 3, H, W) optional document image
        """
        B, T = token_ids.shape
        
        # Text embedding
        text_emb = self.token_embed(token_ids)
        
        # Layout embedding (2D position)
        x1, y1, x2, y2 = bbox[:, :, 0], bbox[:, :, 1], bbox[:, :, 2], bbox[:, :, 3]
        layout_emb = (
            self.x_embed((x1 * 1000).long()) +
            self.y_embed((y1 * 1000).long()) +
            self.w_embed(((x2 - x1) * 1000).long()) +
            self.h_embed(((y2 - y1) * 1000).long())
        )
        
        # 1D position
        positions = torch.arange(T, device=token_ids.device)
        pos_emb = self.position_embed(positions)
        
        # Combine embeddings
        embeddings = text_emb + layout_emb + pos_emb
        
        # Add visual features if available (LayoutLMv2)
        if image is not None:
            visual_features = self.visual_encoder(image)
            # ... integrate visual features
        
        # Transformer encoding
        output = self.encoder(embeddings)
        
        return output


class DonutStyleModel(nn.Module):
    """
    Donut: End-to-end document understanding without OCR
    Direct image to structured output
    """
    
    def __init__(self, vocab_size=50000, d_model=768):
        super().__init__()
        
        # Swin Transformer encoder
        self.encoder = SwinTransformer()
        
        # BART-style decoder
        self.decoder = BARTDecoder(vocab_size, d_model)
    
    def forward(self, images, target_sequence=None):
        """
        Generates structured output directly from image
        Output can be JSON-like format for forms/tables
        """
        visual_features = self.encoder(images)
        
        if target_sequence is not None:
            # Training
            output = self.decoder(visual_features, target_sequence)
        else:
            # Inference: generate structured output
            output = self.decoder.generate(visual_features)
        
        return output


class SwinTransformer(nn.Module):
    """Placeholder for Swin Transformer encoder"""
    def forward(self, x):
        return x

class BARTDecoder(nn.Module):
    """Placeholder for BART decoder"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
    def forward(self, memory, target):
        return None
    def generate(self, memory):
        return None

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

class RegionProposalNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()

class ROIHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
```

**Modern Approaches:**

| Model | Approach | Strengths |
|-------|----------|-----------|
| LayoutLM v1 | Text + 2D position | Pretrained on documents |
| LayoutLM v2/v3 | + Visual features | Better visual understanding |
| Donut | Pure vision, no OCR | End-to-end, simpler pipeline |
| DocFormer | Multi-modal transformer | Strong on forms |
| Table Transformer | DETR for tables | Structure detection |

**Interview Tip:** The key insight is that complex documents need understanding beyond OCR—you need to know what type of content each region is and how they relate. LayoutLM family treats this as pretraining task (like BERT for documents). Donut eliminates OCR entirely by treating document understanding as image-to-text generation. For tables specifically, graph neural networks work well because cells have relational structure.

---

### Question 41
- [ ] Done
**Explain preprocessing steps (binarization, deskewing, denoising) for improving OCR accuracy.**

**Answer:**

Preprocessing normalizes document images to ideal conditions: binarization separates text from background, deskewing corrects rotation, and denoising removes artifacts. These steps significantly improve OCR accuracy, especially for scanned/photographed documents. Modern deep learning OCR is more robust, but preprocessing still helps for degraded inputs.

**Preprocessing Pipeline:**

```
Raw Image → Grayscale → Denoising → Binarization → Deskewing → Normalization → OCR
```

**Step Details:**

| Step | Purpose | Methods |
|------|---------|---------|
| Grayscale | Reduce complexity | Weighted RGB average |
| Denoising | Remove artifacts | Gaussian, bilateral, NLM |
| Binarization | Separate text/background | Otsu, Sauvola, adaptive |
| Deskewing | Correct rotation | Hough, projection profile |
| Normalization | Consistent size/contrast | Resize, histogram eq |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage import filters, morphology

class DocumentPreprocessor:
    """Complete preprocessing pipeline for OCR"""
    
    def __init__(self, target_dpi=300):
        self.target_dpi = target_dpi
    
    def __call__(self, image):
        """
        image: numpy array (H, W, C) or (H, W)
        Returns: preprocessed binary image
        """
        # 1. Convert to grayscale
        gray = self.to_grayscale(image)
        
        # 2. Denoise
        denoised = self.denoise(gray)
        
        # 3. Binarize
        binary = self.binarize(denoised)
        
        # 4. Deskew
        deskewed = self.deskew(binary)
        
        # 5. Clean up (morphological)
        cleaned = self.morphological_cleanup(deskewed)
        
        return cleaned
    
    def to_grayscale(self, image):
        """Convert to grayscale using luminosity method"""
        if len(image.shape) == 3:
            return 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        return image
    
    def denoise(self, image, method='bilateral'):
        """
        Remove noise while preserving edges
        
        Methods:
        - gaussian: Fast, may blur edges
        - bilateral: Edge-preserving, slower
        - nlm: Non-local means, best quality, slowest
        """
        if method == 'gaussian':
            return ndimage.gaussian_filter(image, sigma=1)
        
        elif method == 'bilateral':
            # Bilateral filter (edge-preserving)
            return self._bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
        
        elif method == 'nlm':
            # Non-local means (placeholder - use cv2.fastNlMeansDenoising)
            return image
        
        return image
    
    def _bilateral_filter(self, image, d, sigma_color, sigma_space):
        """Simple bilateral filter implementation"""
        # In practice, use cv2.bilateralFilter
        return ndimage.gaussian_filter(image, sigma=1)  # Simplified
    
    def binarize(self, image, method='sauvola'):
        """
        Separate text from background
        
        Methods:
        - otsu: Global threshold, fails on uneven lighting
        - adaptive: Local threshold, handles gradients
        - sauvola: Adaptive, good for documents
        """
        if method == 'otsu':
            # Global Otsu threshold
            threshold = filters.threshold_otsu(image)
            return image < threshold  # Invert for black text on white
        
        elif method == 'adaptive':
            # Adaptive mean threshold
            threshold = self._adaptive_threshold(image, block_size=31, C=10)
            return image < threshold
        
        elif method == 'sauvola':
            # Sauvola's method (best for documents)
            threshold = filters.threshold_sauvola(image, window_size=31)
            return image < threshold
        
        return image > 0.5
    
    def _adaptive_threshold(self, image, block_size, C):
        """Local adaptive thresholding"""
        # Compute local mean
        local_mean = ndimage.uniform_filter(image.astype(float), size=block_size)
        return local_mean - C
    
    def deskew(self, binary_image, max_angle=10):
        """
        Correct document rotation
        
        Uses projection profile or Hough transform
        """
        angle = self._detect_skew_angle(binary_image, max_angle)
        
        if abs(angle) > 0.1:
            return ndimage.rotate(binary_image, -angle, reshape=False, order=0)
        
        return binary_image
    
    def _detect_skew_angle(self, binary_image, max_angle):
        """
        Detect skew using projection profile
        Find angle that maximizes horizontal projection variance
        """
        best_angle = 0
        best_variance = 0
        
        for angle in np.linspace(-max_angle, max_angle, 41):
            rotated = ndimage.rotate(binary_image, angle, reshape=False, order=0)
            projection = rotated.sum(axis=1)
            variance = np.var(projection)
            
            if variance > best_variance:
                best_variance = variance
                best_angle = angle
        
        return best_angle
    
    def morphological_cleanup(self, binary_image):
        """
        Remove noise and fill holes using morphology
        """
        # Remove small noise (opening)
        kernel = morphology.disk(1)
        opened = morphology.opening(binary_image, kernel)
        
        # Fill small holes (closing)
        closed = morphology.closing(opened, kernel)
        
        # Remove small connected components
        cleaned = morphology.remove_small_objects(closed, min_size=20)
        
        return cleaned


class BinarizationMethods:
    """Compare different binarization methods"""
    
    @staticmethod
    def otsu(image):
        """
        Global threshold based on histogram
        Good for: clean, uniform lighting
        Bad for: shadows, uneven backgrounds
        """
        threshold = filters.threshold_otsu(image)
        return image < threshold
    
    @staticmethod
    def niblack(image, window_size=31, k=0.2):
        """
        Local threshold: T = mean + k * std
        Good for: local variations
        Bad for: background texture appears as noise
        """
        threshold = filters.threshold_niblack(image, window_size=window_size, k=k)
        return image < threshold
    
    @staticmethod
    def sauvola(image, window_size=31, k=0.2, r=128):
        """
        Modified Niblack: T = mean * (1 + k * (std/r - 1))
        Good for: documents, varying backgrounds
        Best general choice for OCR
        """
        threshold = filters.threshold_sauvola(image, window_size=window_size, k=k, r=r)
        return image < threshold
    
    @staticmethod
    def wolf(image, window_size=31, k=0.5):
        """
        Wolf-Jolion's method
        Good for: very low contrast documents
        """
        # Similar to Sauvola with different normalization
        pass


class NeuralBinarization(nn.Module):
    """
    Deep learning binarization
    Often better than traditional methods for degraded documents
    """
    
    def __init__(self):
        super().__init__()
        
        # U-Net style architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """
        x: (B, 1, H, W) grayscale image
        Returns: (B, 1, H, W) binary probability map
        """
        features = self.encoder(x)
        output = self.decoder(features)
        return output


class DeskewMethods:
    """Different deskewing approaches"""
    
    @staticmethod
    def projection_profile(binary_image):
        """
        Rotate and find angle with sharpest horizontal projection
        Best for: text documents with many lines
        """
        pass
    
    @staticmethod
    def hough_lines(binary_image):
        """
        Detect lines using Hough transform, find dominant angle
        Best for: documents with ruled lines or table borders
        """
        pass
    
    @staticmethod
    def connected_components(binary_image):
        """
        Analyze orientation of connected components
        Best for: sparse text, labels
        """
        pass


class DenoisingMethods:
    """Compare denoising approaches"""
    
    @staticmethod
    def comparison():
        methods = {
            'Gaussian': {
                'speed': 'Fast',
                'edge_preservation': 'Poor',
                'best_for': 'Light noise'
            },
            'Median': {
                'speed': 'Fast',
                'edge_preservation': 'Good',
                'best_for': 'Salt-and-pepper noise'
            },
            'Bilateral': {
                'speed': 'Medium',
                'edge_preservation': 'Very Good',
                'best_for': 'General purpose'
            },
            'Non-local Means': {
                'speed': 'Slow',
                'edge_preservation': 'Excellent',
                'best_for': 'High noise, quality critical'
            },
            'Deep Learning': {
                'speed': 'Medium (GPU)',
                'edge_preservation': 'Excellent',
                'best_for': 'Complex degradation'
            },
        }
        return methods


def preprocessing_pipeline_demo():
    """Example complete preprocessing pipeline"""
    
    pipeline = [
        ("1. Color → Grayscale", "Reduce from 3 to 1 channel"),
        ("2. DPI normalization", "Resize to consistent DPI (e.g., 300)"),
        ("3. Denoising", "Bilateral or NLM for quality preservation"),
        ("4. Contrast enhancement", "CLAHE for local contrast"),
        ("5. Binarization", "Sauvola for adaptive thresholding"),
        ("6. Deskewing", "Projection profile for angle detection"),
        ("7. Border removal", "Crop black borders"),
        ("8. Morphological cleanup", "Opening/closing for noise"),
    ]
    
    print("OCR Preprocessing Pipeline:")
    print("=" * 60)
    for step, description in pipeline:
        print(f"  {step}: {description}")


def when_to_skip():
    """When preprocessing may not be needed"""
    
    print("\nWhen to Reduce/Skip Preprocessing:")
    print("-" * 50)
    scenarios = [
        "Digital-born PDFs (already clean)",
        "Using robust deep learning OCR (TrOCR, etc.)",
        "Real-time applications (latency matters)",
        "Color information needed (forms, highlights)",
    ]
    
    for scenario in scenarios:
        print(f"  • {scenario}")
```

**When to Use Each Method:**

| Condition | Binarization | Denoising |
|-----------|-------------|-----------|
| Clean scan | Otsu | None/Light Gaussian |
| Uneven lighting | Sauvola | Bilateral |
| Old/degraded | Neural | Non-local means |
| Photo of document | Adaptive | Bilateral + neural |

**Interview Tip:** Modern OCR models (TrOCR, PaddleOCR) are trained on diverse data and are fairly robust to noise. Preprocessing is most important for: (1) severely degraded documents, (2) traditional OCR engines like Tesseract, or (3) when you need consistent output format. The most impactful step is usually deskewing—even small rotations hurt line detection and reading order.

---

### Question 42
- [ ] Done
**How do you implement OCR post-processing with language models to correct errors?**

**Answer:**

Post-processing uses language models to fix OCR errors: (1) spell checking with edit distance, (2) statistical language models (n-gram) for context, (3) neural LMs (BERT, GPT) for semantic correction, and (4) domain-specific dictionaries. The LM scores candidate corrections based on context probability, selecting the most likely intended text.

**Post-Processing Approaches:**

| Approach | Method | Best For |
|----------|--------|----------|
| Dictionary | Word lookup + edit distance | Isolated typos |
| N-gram LM | Character/word probabilities | Common substitutions |
| Neural LM | BERT/GPT masked prediction | Context-dependent |
| Seq2Seq | OCR error → correct text | Systematic errors |
| Hybrid | Rule + statistical + neural | Production systems |

**Common OCR Errors:**

| Error Type | Example | Cause |
|------------|---------|-------|
| Substitution | m → rn, O → 0 | Visual similarity |
| Insertion | hello → helllo | Segmentation |
| Deletion | running → runng | Merged chars |
| Split | word → w ord | Over-segmentation |
| Merge | a word → aword | Under-segmentation |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import re

class OCRPostProcessor:
    """Complete OCR post-processing pipeline"""
    
    def __init__(self, dictionary_path=None, use_neural=True):
        self.dictionary = self._load_dictionary(dictionary_path)
        self.spell_checker = SpellChecker(self.dictionary)
        self.ngram_model = NGramLanguageModel()
        
        if use_neural:
            self.neural_corrector = NeuralCorrector()
        else:
            self.neural_corrector = None
    
    def _load_dictionary(self, path):
        """Load word dictionary"""
        if path is None:
            # Default common words
            return {'the', 'a', 'is', 'are', 'was', 'were', 'be', 'been', 
                   'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                   'would', 'could', 'should', 'may', 'might', 'must'}
        
        with open(path) as f:
            return set(line.strip().lower() for line in f)
    
    def correct(self, ocr_text):
        """
        Apply post-processing corrections
        """
        # 1. Basic cleanup
        text = self._basic_cleanup(ocr_text)
        
        # 2. Dictionary-based spell check
        text = self.spell_checker.correct_text(text)
        
        # 3. N-gram based correction
        text = self.ngram_model.correct(text)
        
        # 4. Neural correction (if available)
        if self.neural_corrector:
            text = self.neural_corrector.correct(text)
        
        return text
    
    def _basic_cleanup(self, text):
        """Fix obvious OCR artifacts"""
        # Common substitutions
        replacements = {
            '|': 'l',  # Pipe to l
            '0': 'O',  # Context-dependent
            '1': 'l',  # Context-dependent
            '—': '-',  # Em dash to hyphen
            '"': '"',  # Curly to straight quotes
            ''': "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


class SpellChecker:
    """Edit distance based spell checking"""
    
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def correct_text(self, text):
        """Correct each word independently"""
        words = text.split()
        corrected = []
        
        for word in words:
            # Preserve punctuation
            prefix, core, suffix = self._extract_punctuation(word)
            
            if core.lower() in self.dictionary:
                corrected.append(word)
            else:
                correction = self._find_correction(core)
                corrected.append(prefix + correction + suffix)
        
        return ' '.join(corrected)
    
    def _extract_punctuation(self, word):
        """Separate leading/trailing punctuation"""
        prefix = ''
        suffix = ''
        
        while word and not word[0].isalnum():
            prefix += word[0]
            word = word[1:]
        
        while word and not word[-1].isalnum():
            suffix = word[-1] + suffix
            word = word[:-1]
        
        return prefix, word, suffix
    
    def _find_correction(self, word, max_distance=2):
        """Find closest dictionary word"""
        if not word:
            return word
        
        candidates = []
        
        for dict_word in self.dictionary:
            dist = self._edit_distance(word.lower(), dict_word.lower())
            if dist <= max_distance:
                candidates.append((dict_word, dist))
        
        if not candidates:
            return word  # No correction found
        
        # Return closest match (preserve original case if possible)
        best = min(candidates, key=lambda x: x[1])[0]
        
        if word[0].isupper():
            return best.capitalize()
        return best
    
    def _edit_distance(self, s1, s2):
        """Levenshtein distance"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]


class NGramLanguageModel:
    """Character n-gram for OCR error correction"""
    
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
    
    def train(self, text_corpus):
        """Train on clean text corpus"""
        for line in text_corpus:
            padded = '^' * (self.n - 1) + line + '$'
            
            for i in range(len(padded) - self.n + 1):
                ngram = padded[i:i+self.n]
                context = ngram[:-1]
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
    
    def probability(self, text):
        """Calculate probability of text"""
        padded = '^' * (self.n - 1) + text + '$'
        log_prob = 0
        
        for i in range(len(padded) - self.n + 1):
            ngram = padded[i:i+self.n]
            context = ngram[:-1]
            
            # Add-one smoothing
            prob = (self.ngram_counts[ngram] + 1) / (self.context_counts[context] + 256)
            log_prob += np.log(prob)
        
        return log_prob
    
    def correct(self, text):
        """Generate corrections and score them"""
        words = text.split()
        corrected = []
        
        for word in words:
            candidates = self._generate_candidates(word)
            
            # Score by n-gram probability
            best = max(candidates, key=lambda c: self.probability(c))
            corrected.append(best)
        
        return ' '.join(corrected)
    
    def _generate_candidates(self, word):
        """Generate edit-distance-1 candidates"""
        candidates = {word}  # Include original
        letters = 'abcdefghijklmnopqrstuvwxyz'
        
        # Deletions
        for i in range(len(word)):
            candidates.add(word[:i] + word[i+1:])
        
        # Insertions
        for i in range(len(word) + 1):
            for c in letters:
                candidates.add(word[:i] + c + word[i:])
        
        # Substitutions
        for i in range(len(word)):
            for c in letters:
                candidates.add(word[:i] + c + word[i+1:])
        
        # Transpositions
        for i in range(len(word) - 1):
            candidates.add(word[:i] + word[i+1] + word[i] + word[i+2:])
        
        return candidates


class NeuralCorrector(nn.Module):
    """BERT-based OCR error correction"""
    
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        # In practice: load from transformers
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model = None
        self.tokenizer = None
    
    def correct(self, text, confidence_threshold=0.7):
        """
        Use BERT masked language model to correct errors
        """
        if self.model is None:
            return text  # Fallback
        
        words = text.split()
        corrected = []
        
        for i, word in enumerate(words):
            # Check if word seems erroneous (not in vocab, low confidence)
            if self._is_suspicious(word):
                # Create masked sentence
                masked_text = ' '.join(words[:i] + ['[MASK]'] + words[i+1:])
                
                # Get BERT predictions
                prediction = self._predict_masked(masked_text)
                
                # Use prediction if confident enough
                if prediction[1] > confidence_threshold:
                    corrected.append(prediction[0])
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        
        return ' '.join(corrected)
    
    def _is_suspicious(self, word):
        """Check if word might be an OCR error"""
        # Heuristics:
        # - Not in vocabulary
        # - Contains unusual character combinations
        # - Very short/long
        return len(word) < 2 or len(word) > 20
    
    def _predict_masked(self, masked_text):
        """Get BERT's prediction for masked token"""
        # Placeholder - in practice use transformers
        return ('the', 0.9)


class Seq2SeqCorrector(nn.Module):
    """
    Sequence-to-sequence model trained on (OCR output, ground truth) pairs
    Can learn systematic error patterns
    """
    
    def __init__(self, vocab_size=256, hidden_size=256):
        super().__init__()
        
        # Character-level encoder-decoder
        self.encoder = nn.GRU(vocab_size, hidden_size, bidirectional=True, batch_first=True)
        self.decoder = nn.GRU(vocab_size + hidden_size * 2, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, ocr_chars, target_chars=None):
        """
        ocr_chars: (B, T1) OCR character sequence
        target_chars: (B, T2) correct character sequence (for training)
        """
        # Encode OCR text
        encoder_outputs, hidden = self.encoder(ocr_chars)
        
        # Decode to correct text
        if target_chars is not None:
            # Teacher forcing
            pass
        else:
            # Greedy decoding
            pass
        
        return None


class ConfusionMatrixCorrector:
    """
    Use OCR confusion matrix for targeted corrections
    """
    
    def __init__(self):
        # Common OCR confusions: (correct, error)
        self.confusions = {
            'rn': 'm',  # r + n often misread as m
            'cl': 'd',  # c + l often misread as d
            'li': 'h',  # l + i often looks like h
            '0': 'O',   # Zero vs O
            '1': 'l',   # One vs l
            '5': 'S',   # Five vs S
            '8': 'B',   # Eight vs B
        }
    
    def apply_confusion_rules(self, text):
        """Apply targeted confusion corrections"""
        # This requires context to know when to apply
        return text


import numpy as np

def post_processing_demo():
    """Example post-processing pipeline"""
    
    ocr_output = "Th1s is an exarnple of 0CR output w1th rnistakes"
    
    pipeline = [
        ("Original OCR", ocr_output),
        ("Basic cleanup", "This is an example of OCR output with mistakes"),
        ("Spell check", "This is an example of OCR output with mistakes"),
        ("Neural LM", "This is an example of OCR output with mistakes"),
    ]
    
    print("OCR Post-Processing Example:")
    print("=" * 60)
    for step, result in pipeline:
        print(f"{step}:")
        print(f"  {result}\n")
```

**Approach Selection:**

| Scenario | Best Approach |
|----------|--------------|
| Speed-critical | Dictionary + rules only |
| General documents | N-gram + neural hybrid |
| Domain-specific | Custom dictionary + fine-tuned LM |
| Systematic errors | Seq2Seq on error pairs |

**Interview Tip:** The key insight is that OCR errors are not random—they follow patterns based on visual similarity (m→rn, O→0). Language models can catch these because the erroneous text is improbable ("exarnple" is low probability, "example" is high). For production, combine fast dictionary lookup with neural LM for suspicious words only—running BERT on every word is too slow.

---

### Question 43
- [ ] Done
**What techniques work for multilingual OCR with different scripts and writing directions?**

**Answer:**

Multilingual OCR handles diverse scripts through: (1) script detection to identify language/script, (2) direction-aware models for RTL (Arabic, Hebrew) and vertical (CJK) text, (3) unified models trained on multiple scripts (mBART, XLM), and (4) script-specific preprocessing. Large character sets and mixed-script documents add complexity; transformer models handle this better than CNN-RNN.

**Script Categories and Challenges:**

| Script Type | Examples | Key Challenges |
|-------------|----------|----------------|
| Latin | English, French | Font variations |
| CJK | Chinese, Japanese, Korean | Large character set (5000+) |
| Arabic | Arabic, Urdu | RTL, connected cursive |
| Devanagari | Hindi, Sanskrit | Complex ligatures, matras |
| Thai | Thai | No word spacing |
| Mixed | Hindi-English, Japanese | Script switching |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class Script(Enum):
    LATIN = 'latin'
    CJK = 'cjk'
    ARABIC = 'arabic'
    DEVANAGARI = 'devanagari'
    CYRILLIC = 'cyrillic'
    THAI = 'thai'
    KOREAN = 'korean'


class ScriptDetector(nn.Module):
    """
    Detect script/language from image region
    First step in multilingual OCR pipeline
    """
    
    def __init__(self, num_scripts=10):
        super().__init__()
        
        # CNN classifier
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Linear(128, num_scripts)
    
    def forward(self, image):
        """
        image: (B, 1, H, W) text line image
        Returns: script probabilities
        """
        features = self.features(image).flatten(1)
        logits = self.classifier(features)
        return F.softmax(logits, dim=-1)


class DirectionAwareOCR(nn.Module):
    """
    Handle left-to-right, right-to-left, and vertical text
    """
    
    def __init__(self, vocab_size=50000):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        
        # Bidirectional LSTM for direction handling
        self.rnn = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        
        # Output for different directions
        self.output = nn.Linear(512, vocab_size)
    
    def forward(self, image, direction='ltr'):
        """
        direction: 'ltr', 'rtl', or 'vertical'
        """
        # Preprocess based on direction
        if direction == 'rtl':
            # Flip image horizontally for RTL
            image = torch.flip(image, dims=[-1])
        elif direction == 'vertical':
            # Rotate 90 degrees for vertical text
            image = image.transpose(-1, -2)
        
        # Encode
        features = self.encoder(image)
        
        # Reshape for RNN: (B, C, H, W) → (B, W, C*H)
        B, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2).reshape(B, W, C * H)
        
        # RNN
        rnn_out, _ = self.rnn(features)
        
        # Output
        logits = self.output(rnn_out)
        
        if direction == 'rtl':
            # Flip output back
            logits = torch.flip(logits, dims=[1])
        
        return logits


class UnifiedMultilingualOCR(nn.Module):
    """
    Single model supporting multiple scripts
    Like mTrOCR or PaddleOCR multilingual
    """
    
    def __init__(self, vocab_size=100000, d_model=768):
        super().__init__()
        
        # Unified vocabulary covering all scripts
        # Chinese: ~5000, Japanese: ~2000+hiragana+katakana
        # Korean: ~2000, Arabic: ~200, Latin: ~100
        
        # Vision encoder
        self.encoder = VisionTransformer(d_model)
        
        # Multilingual text decoder
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, 12, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6)
        
        self.output = nn.Linear(d_model, vocab_size)
        
        # Script embedding (optional: helps model adapt)
        self.script_embed = nn.Embedding(10, d_model)
    
    def forward(self, images, target_tokens=None, script_id=None):
        """
        images: (B, 3, H, W)
        target_tokens: (B, T) for training
        script_id: (B,) optional script hint
        """
        # Encode image
        memory = self.encoder(images)
        
        # Add script embedding if provided
        if script_id is not None:
            script_emb = self.script_embed(script_id)
            memory = memory + script_emb.unsqueeze(1)
        
        if target_tokens is not None:
            # Training with teacher forcing
            target_emb = self.token_embed(target_tokens)
            output = self.decoder(target_emb, memory)
            logits = self.output(output)
            return logits
        else:
            return self.generate(memory)
    
    def generate(self, memory, max_len=100):
        """Autoregressive generation"""
        B = memory.shape[0]
        generated = torch.zeros(B, 1, dtype=torch.long, device=memory.device)
        
        for _ in range(max_len):
            target_emb = self.token_embed(generated)
            output = self.decoder(target_emb, memory)
            logits = self.output(output[:, -1:])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == 2).all():  # EOS
                break
        
        return generated


class VisionTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Placeholder
        self.d_model = d_model
    
    def forward(self, x):
        return torch.randn(x.shape[0], 100, self.d_model, device=x.device)


class ScriptSpecificPreprocessing:
    """Preprocessing tailored to each script"""
    
    @staticmethod
    def arabic(image):
        """
        Arabic-specific preprocessing
        - RTL reading direction
        - Connected script (cursive)
        - Diacritics (optional markings)
        """
        # Flip for RTL if needed by model
        # Don't over-binarize (diacritics are thin)
        return image
    
    @staticmethod
    def cjk(image):
        """
        Chinese/Japanese/Korean preprocessing
        - Square aspect ratio per character
        - May be vertical or horizontal
        - Large character set
        """
        # Detect vertical vs horizontal
        # Segment by character grid if needed
        return image
    
    @staticmethod
    def devanagari(image):
        """
        Devanagari (Hindi) preprocessing
        - Shirorekha (headline) detection
        - Complex ligatures
        - Matras (vowel signs) above/below
        """
        # Detect headline for line segmentation
        # Don't split ligatures
        return image
    
    @staticmethod
    def thai(image):
        """
        Thai preprocessing
        - No word boundaries
        - Tonal marks above
        - Multi-level characters
        """
        # Word segmentation is post-processing concern
        # Preserve vertical structure
        return image


class MixedScriptHandler:
    """Handle documents with multiple scripts"""
    
    def __init__(self):
        self.script_detector = ScriptDetector()
        self.ocr_models = {
            Script.LATIN: None,  # Load appropriate models
            Script.CJK: None,
            Script.ARABIC: None,
        }
    
    def process(self, image, regions):
        """
        Process document with mixed scripts
        
        regions: list of text regions with bounding boxes
        """
        results = []
        
        for region in regions:
            # Crop region
            cropped = self._crop(image, region.bbox)
            
            # Detect script
            script = self._detect_script(cropped)
            
            # Apply appropriate OCR
            text = self._ocr(cropped, script)
            
            results.append({
                'bbox': region.bbox,
                'script': script,
                'text': text
            })
        
        return results
    
    def _detect_script(self, image):
        with torch.no_grad():
            probs = self.script_detector(image.unsqueeze(0))
            script_idx = probs.argmax().item()
            return list(Script)[script_idx]
    
    def _crop(self, image, bbox):
        x1, y1, x2, y2 = bbox
        return image[:, y1:y2, x1:x2]
    
    def _ocr(self, image, script):
        model = self.ocr_models.get(script)
        if model:
            return model(image)
        return ""


class CharacterSetManagement:
    """Handle large character vocabularies"""
    
    @staticmethod
    def get_charset_size():
        """Character set sizes by script"""
        return {
            'Latin (ASCII)': 95,
            'Latin Extended': 300,
            'Chinese Simplified': 6763,  # GB2312
            'Chinese Traditional': 13000,
            'Japanese': 3000,  # Kanji + kana
            'Korean': 2350,  # Common Hangul
            'Arabic': 200,
            'Devanagari': 150,
            'Thai': 87,
            'Combined (multilingual)': 20000,
        }
    
    @staticmethod
    def subword_vs_character():
        """Compare tokenization approaches"""
        
        approaches = {
            'Character-level': {
                'pros': ['Universal', 'No OOV'],
                'cons': ['Long sequences', 'Less semantic']
            },
            'Subword (BPE)': {
                'pros': ['Efficient', 'Semantic units'],
                'cons': ['Script-specific', 'May split wrong']
            },
            'Word-level': {
                'pros': ['Semantic', 'Short'],
                'cons': ['Large vocab', 'OOV issues']
            }
        }
        
        return approaches


def multilingual_training_strategy():
    """Training approaches for multilingual OCR"""
    
    strategies = [
        ("1. Joint training", 
         "Train single model on all scripts simultaneously",
         "Best quality, requires diverse data"),
        
        ("2. Transfer learning",
         "Pretrain on resource-rich, finetune on low-resource",
         "Good for rare scripts"),
        
        ("3. Multi-task",
         "Shared encoder, script-specific decoders",
         "Balance quality per script"),
        
        ("4. Curriculum",
         "Start with easy (Latin), add harder scripts",
         "Stable training"),
    ]
    
    print("Multilingual OCR Training Strategies:")
    print("=" * 60)
    for name, description, note in strategies:
        print(f"\n{name}:")
        print(f"  {description}")
        print(f"  Note: {note}")


def handling_code_switching():
    """Handle text that switches between scripts"""
    
    print("\nCode-Switching Examples:")
    print("-" * 50)
    examples = [
        "Hindi-English: यह एक example है",
        "Japanese-English: これはtestです",
        "Arabic-English: هذا test",
    ]
    
    for ex in examples:
        print(f"  {ex}")
    
    print("\nApproaches:")
    print("  1. Unified model with all scripts in vocab")
    print("  2. Segment by script, OCR each, merge")
    print("  3. Character-level model (script-agnostic)")
```

**Modern Multilingual OCR Models:**

| Model | Scripts | Approach |
|-------|---------|----------|
| PaddleOCR | 80+ languages | Unified model |
| EasyOCR | 80+ languages | Script-specific models |
| Tesseract | 100+ languages | Language packs |
| TrOCR (multilingual) | Many | Pretrained mBART decoder |
| Google Cloud Vision | 100+ | Unified cloud API |

**Interview Tip:** The key challenge in multilingual OCR is the vocabulary size—Chinese alone has 5000+ common characters. Solutions include: (1) unified subword tokenization (BPE across scripts), (2) character-level models with large output layers, or (3) hierarchical approaches (script detection → script-specific OCR). For production, use existing multilingual models (PaddleOCR, EasyOCR) rather than training from scratch.

---

## Super-Resolution

### Question 44
- [ ] Done
**Compare PSNR-oriented vs. perceptual quality-oriented super-resolution models.**

**Answer:**

PSNR-oriented models minimize pixel-wise error (MSE/MAE), producing accurate but often blurry results. Perceptual models use feature-space losses and adversarial training, generating sharper, more realistic images but with potential hallucinated details. The choice depends on whether fidelity or visual appeal matters more.

**Fundamental Trade-off:**

| Aspect | PSNR-Oriented | Perceptual-Oriented |
|--------|--------------|---------------------|
| Loss function | MSE/L1 | Perceptual + adversarial |
| Sharpness | Blurry | Sharp |
| PSNR score | Higher | Lower |
| Visual quality | Worse | Better |
| Artifacts | Smooth blur | Texture hallucination |
| Fidelity | High | May hallucinate |

**Mathematical Comparison:**

PSNR-oriented:
$$\mathcal{L}_{PSNR} = \|I_{SR} - I_{HR}\|_2^2$$

Perceptual-oriented:
$$\mathcal{L}_{perceptual} = \|\phi(I_{SR}) - \phi(I_{HR})\|_2^2 + \lambda_{adv}\mathcal{L}_{GAN}$$

Where $\phi$ is a pretrained feature extractor (VGG).

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PSNROrientedSR(nn.Module):
    """
    PSNR-oriented model (like EDSR, RCAN)
    Uses pixel-wise loss only
    """
    
    def __init__(self, scale=4, num_channels=64, num_blocks=16):
        super().__init__()
        
        # Feature extraction
        self.conv_first = nn.Conv2d(3, num_channels, 3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        
        # Upsampling
        self.upsampler = Upsampler(scale, num_channels)
        
        # Output
        self.conv_last = nn.Conv2d(num_channels, 3, 3, padding=1)
    
    def forward(self, x):
        feat = self.conv_first(x)
        res = self.residual_blocks(feat)
        feat = feat + res  # Global residual
        up = self.upsampler(feat)
        out = self.conv_last(up)
        return out


class PerceptualOrientedSR(nn.Module):
    """
    Perceptual-oriented model (like SRGAN, ESRGAN)
    Uses perceptual + adversarial losses
    """
    
    def __init__(self, scale=4, num_channels=64, num_rrdb=23):
        super().__init__()
        
        # Similar architecture but trained differently
        self.conv_first = nn.Conv2d(3, num_channels, 3, padding=1)
        
        # RRDB (Residual in Residual Dense Block)
        self.rrdbs = nn.Sequential(
            *[RRDB(num_channels) for _ in range(num_rrdb)]
        )
        
        self.conv_trunk = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.upsampler = Upsampler(scale, num_channels)
        self.conv_last = nn.Conv2d(num_channels, 3, 3, padding=1)
    
    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.conv_trunk(self.rrdbs(feat))
        feat = feat + trunk
        up = self.upsampler(feat)
        return self.conv_last(up)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class RRDB(nn.Module):
    """Residual in Residual Dense Block (ESRGAN)"""
    def __init__(self, channels, beta=0.2):
        super().__init__()
        self.beta = beta
        self.rdb1 = DenseBlock(channels)
        self.rdb2 = DenseBlock(channels)
        self.rdb3 = DenseBlock(channels)
    
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.beta * out


class DenseBlock(nn.Module):
    def __init__(self, channels, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2*growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3*growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4*growth, channels, 3, padding=1)
    
    def forward(self, x):
        c1 = F.leaky_relu(self.conv1(x), 0.2)
        c2 = F.leaky_relu(self.conv2(torch.cat([x, c1], 1)), 0.2)
        c3 = F.leaky_relu(self.conv3(torch.cat([x, c1, c2], 1)), 0.2)
        c4 = F.leaky_relu(self.conv4(torch.cat([x, c1, c2, c3], 1)), 0.2)
        c5 = self.conv5(torch.cat([x, c1, c2, c3, c4], 1))
        return c5 * 0.2 + x


class Upsampler(nn.Module):
    def __init__(self, scale, channels):
        super().__init__()
        layers = []
        for _ in range(int(scale / 2)):
            layers.extend([
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2)
            ])
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class LossComparison:
    """Compare different loss functions"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.vgg = VGGFeatures().to(device).eval()
    
    def mse_loss(self, sr, hr):
        """Pixel-wise MSE - leads to blurry results"""
        return F.mse_loss(sr, hr)
    
    def l1_loss(self, sr, hr):
        """Pixel-wise L1 - slightly better edges than MSE"""
        return F.l1_loss(sr, hr)
    
    def perceptual_loss(self, sr, hr):
        """
        Feature-space loss using pretrained VGG
        Captures high-level similarity, allows local differences
        """
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        
        loss = 0
        for sr_f, hr_f in zip(sr_features, hr_features):
            loss += F.l1_loss(sr_f, hr_f)
        
        return loss
    
    def style_loss(self, sr, hr):
        """Gram matrix loss for texture matching"""
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        
        loss = 0
        for sr_f, hr_f in zip(sr_features, hr_features):
            sr_gram = self._gram_matrix(sr_f)
            hr_gram = self._gram_matrix(hr_f)
            loss += F.l1_loss(sr_gram, hr_gram)
        
        return loss
    
    def _gram_matrix(self, feat):
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1)
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (C * H * W)


class VGGFeatures(nn.Module):
    """Extract VGG features for perceptual loss"""
    
    def __init__(self, layers=[2, 7, 12, 21, 30]):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        
        self.layers = layers
        self.slices = nn.ModuleList()
        
        prev = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:layer]))
            prev = layer
        
        # Freeze
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = []
        for slice in self.slices:
            x = slice(x)
            features.append(x)
        return features


def quality_comparison():
    """Explain the quality trade-off"""
    
    print("PSNR vs Perceptual Quality Trade-off:")
    print("=" * 60)
    
    comparison = [
        ("Metric", "PSNR-Oriented", "Perceptual-Oriented"),
        ("-" * 20, "-" * 20, "-" * 20),
        ("PSNR (dB)", "~32-34", "~28-30"),
        ("SSIM", "~0.92", "~0.85"),
        ("LPIPS (lower=better)", "~0.15", "~0.08"),
        ("Human preference", "30%", "70%"),
        ("Sharpness", "Blurry", "Sharp"),
        ("Artifacts", "None", "Possible hallucination"),
    ]
    
    for row in comparison:
        print(f"{row[0]:<20} {row[1]:<20} {row[2]}")
```

**When to Use Each:**

| Use Case | Recommended |
|----------|-------------|
| Medical imaging | PSNR (fidelity critical) |
| Satellite imagery | PSNR (accuracy matters) |
| Photo enhancement | Perceptual (visual quality) |
| Video streaming | Perceptual (viewer experience) |
| Document upscaling | PSNR (preserve text) |
| Face upscaling | Perceptual (natural appearance) |

**Interview Tip:** The perception-distortion trade-off is fundamental—you cannot maximize both PSNR and perceptual quality. MSE averages over multiple plausible solutions, creating blur. Perceptual losses pick one plausible solution (sharper but potentially wrong details). For evaluation, use both PSNR/SSIM (fidelity) and LPIPS/human studies (perceptual).

---

### Question 45
- [ ] Done
**Explain SRGAN and ESRGAN. How do perceptual and adversarial losses improve visual quality?**

**Answer:**

SRGAN introduced perceptual loss (VGG feature matching) and adversarial loss (GAN) for super-resolution. ESRGAN improved with RRDB architecture, VGG before activation, and Relativistic GAN. These losses shift optimization from pixel accuracy to perceptual realism—the generator learns to produce images that look real rather than just matching pixels.

**Evolution:**

| Model | Year | Key Innovations |
|-------|------|-----------------|
| SRGAN | 2017 | First GAN-based SR, VGG loss |
| ESRGAN | 2018 | RRDB, VGG before activation, RaGAN |
| Real-ESRGAN | 2021 | Real-world degradation, U-Net discriminator |

**Loss Components:**

$$\mathcal{L}_G = \mathcal{L}_{perceptual} + \lambda_1 \mathcal{L}_{adv} + \lambda_2 \mathcal{L}_{pixel}$$

| Loss | Formula | Purpose |
|------|---------|---------|
| Perceptual | $\|\phi(I_{SR}) - \phi(I_{HR})\|_1$ | High-level similarity |
| Adversarial | $-\log D(I_{SR})$ | Realism via discriminator |
| Pixel (optional) | $\|I_{SR} - I_{HR}\|_1$ | Stability, color accuracy |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SRGANGenerator(nn.Module):
    """
    SRGAN Generator: ResNet-based with 16 residual blocks
    """
    
    def __init__(self, scale=4, num_channels=64, num_blocks=16):
        super().__init__()
        
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, num_channels, 9, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlockSRGAN(num_channels) for _ in range(num_blocks)]
        )
        
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        
        # Upsampling (2x twice for 4x)
        self.upsampling = nn.Sequential(
            UpsampleBlock(num_channels, 2),
            UpsampleBlock(num_channels, 2),
        )
        
        self.conv_last = nn.Conv2d(num_channels, 3, 9, padding=4)
    
    def forward(self, x):
        feat = self.conv_first(x)
        res = self.res_blocks(feat)
        res = self.conv_mid(res)
        feat = feat + res  # Skip connection
        up = self.upsampling(feat)
        return self.conv_last(up)


class ResidualBlockSRGAN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.block(x)


class ESRGANGenerator(nn.Module):
    """
    ESRGAN Generator: RRDB (no batch norm) for better quality
    """
    
    def __init__(self, scale=4, num_channels=64, num_rrdb=23):
        super().__init__()
        
        self.conv_first = nn.Conv2d(3, num_channels, 3, padding=1)
        
        # RRDB blocks (no batch norm!)
        self.rrdbs = nn.Sequential(
            *[RRDB(num_channels) for _ in range(num_rrdb)]
        )
        
        self.conv_trunk = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        
        # Upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(num_channels, num_channels * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2),
        )
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, 3, 3, padding=1)
        )
    
    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.conv_trunk(self.rrdbs(feat))
        feat = feat + trunk
        up = self.upsampling(feat)
        return self.conv_last(up)


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, channels, beta=0.2):
        super().__init__()
        self.beta = beta
        self.rdb1 = DenseBlock(channels)
        self.rdb2 = DenseBlock(channels)
        self.rdb3 = DenseBlock(channels)
    
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.beta * out


class DenseBlock(nn.Module):
    def __init__(self, channels, growth=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.layers = nn.ModuleList([
            nn.Conv2d(channels + i * growth, growth, 3, padding=1)
            for i in range(4)
        ])
        self.final = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = F.leaky_relu(layer(torch.cat(features, 1)), 0.2)
            features.append(out)
        return x + self.beta * self.final(torch.cat(features, 1))


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    ESRGAN uses features BEFORE activation (more detailed)
    """
    
    def __init__(self, layer_idx=35, before_activation=True):
        super().__init__()
        
        vgg = models.vgg19(pretrained=True).features
        
        if before_activation:
            # ESRGAN improvement: use features before ReLU
            # Conv5_4 before ReLU is layer 34
            self.vgg = nn.Sequential(*list(vgg.children())[:layer_idx])
        else:
            # SRGAN: after activation
            self.vgg = nn.Sequential(*list(vgg.children())[:layer_idx + 1])
        
        # Freeze
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, sr, hr):
        # Normalize
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        
        # Extract features
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        
        return F.l1_loss(sr_features, hr_features)


class Discriminator(nn.Module):
    """VGG-style discriminator for SRGAN"""
    
    def __init__(self, input_size=128):
        super().__init__()
        
        self.features = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            # 64 -> 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            # 32 -> 16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            # 16 -> 8
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class RelativisticGANLoss(nn.Module):
    """
    ESRGAN uses Relativistic GAN (RaGAN)
    D predicts whether real is more realistic than fake
    """
    
    def __init__(self, discriminator):
        super().__init__()
        self.D = discriminator
    
    def d_loss(self, real, fake):
        """Discriminator loss"""
        d_real = self.D(real)
        d_fake = self.D(fake.detach())
        
        # Relativistic: real should be more real than fake
        loss_real = F.binary_cross_entropy_with_logits(
            d_real - d_fake.mean(), torch.ones_like(d_real)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            d_fake - d_real.mean(), torch.zeros_like(d_fake)
        )
        
        return (loss_real + loss_fake) / 2
    
    def g_loss(self, real, fake):
        """Generator loss"""
        d_real = self.D(real)
        d_fake = self.D(fake)
        
        # Generator wants fake to be more real than real
        loss_real = F.binary_cross_entropy_with_logits(
            d_real - d_fake.mean(), torch.zeros_like(d_real)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            d_fake - d_real.mean(), torch.ones_like(d_fake)
        )
        
        return (loss_real + loss_fake) / 2


class ESRGANTrainer:
    """Training loop for ESRGAN"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        self.generator = ESRGANGenerator().to(device)
        self.discriminator = Discriminator().to(device)
        
        self.perceptual_loss = VGGPerceptualLoss(before_activation=True).to(device)
        self.ragan_loss = RelativisticGANLoss(self.discriminator)
        
        # Loss weights
        self.lambda_perceptual = 1.0
        self.lambda_gan = 0.1
        self.lambda_pixel = 0.01  # Small pixel loss for stability
    
    def train_step(self, lr, hr):
        """One training iteration"""
        sr = self.generator(lr)
        
        # Generator loss
        loss_perceptual = self.perceptual_loss(sr, hr)
        loss_gan = self.ragan_loss.g_loss(hr, sr)
        loss_pixel = F.l1_loss(sr, hr)
        
        loss_g = (self.lambda_perceptual * loss_perceptual +
                  self.lambda_gan * loss_gan +
                  self.lambda_pixel * loss_pixel)
        
        # Discriminator loss
        loss_d = self.ragan_loss.d_loss(hr, sr)
        
        return loss_g, loss_d


def esrgan_improvements():
    """Key ESRGAN improvements over SRGAN"""
    
    improvements = {
        'No Batch Norm': 'BN introduces artifacts, RRDB removes it',
        'RRDB': 'Deeper, denser connections than ResBlock',
        'VGG before ReLU': 'Preserves more detail information',
        'Relativistic GAN': 'More stable training, better gradients',
        'Network interpolation': 'Blend PSNR and perceptual models',
    }
    
    print("ESRGAN Improvements over SRGAN:")
    print("=" * 60)
    for improvement, explanation in improvements.items():
        print(f"  {improvement}: {explanation}")
```

**Why These Losses Help:**

| Loss | Problem Solved | Visual Effect |
|------|---------------|---------------|
| Perceptual | MSE blurs edges | Sharper textures |
| Adversarial | Not realistic | Natural appearance |
| Before activation VGG | Lose high-freq info | More detail |
| Relativistic GAN | Training instability | Consistent quality |

**Interview Tip:** The key insight is that MSE is the "wrong" objective for visual quality—it's convex and averages over all plausible outputs, creating blur. Perceptual loss measures high-level similarity, allowing local variations. Adversarial loss pushes toward the manifold of natural images. Together they generate sharp, realistic outputs at the cost of perfect pixel accuracy.

---

### Question 46
- [ ] Done
**How do you handle real-world degradation (blur, noise, compression) vs. simple bicubic downsampling?**

**Answer:**

Real-world images have complex degradation: blur (motion, defocus), noise (sensor, ISO), compression (JPEG), and combinations. Models trained on bicubic downsampling fail on real data. Solutions include: (1) realistic degradation simulation during training, (2) blind SR that estimates degradation, and (3) degradation-aware models. Real-ESRGAN uses high-order degradation modeling.

**Degradation Gap:**

| Training Data | Real-World Reality | Result |
|--------------|-------------------|--------|
| Bicubic only | Blur + noise + JPEG | Model fails |
| Synthetic mix | Close to reality | Works better |
| Real pairs | Best match | Best performance |

**Real-World Degradation Pipeline:**

```
HR → Blur → Downsample → Noise → Compression → LR
      ↓         ↓           ↓          ↓
   (motion,   (various    (Gaussian,  (JPEG,
   Gaussian,   scales)     Poisson)   artifacts)
   defocus)
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io

class RealWorldDegradation:
    """
    Simulate real-world image degradation
    Based on Real-ESRGAN degradation model
    """
    
    def __init__(self, scale=4):
        self.scale = scale
        
        # Blur kernels
        self.blur_kernels = {
            'gaussian': self._gaussian_blur,
            'motion': self._motion_blur,
            'iso': self._isotropic_blur,
            'aniso': self._anisotropic_blur,
        }
    
    def __call__(self, hr_image):
        """
        Apply realistic degradation chain
        
        hr_image: (C, H, W) tensor, clean HR image
        Returns: (C, H', W') degraded LR image
        """
        # First degradation (simulate camera capture)
        degraded = self._first_degradation(hr_image)
        
        # Second degradation (post-processing, sharing)
        degraded = self._second_degradation(degraded)
        
        return degraded
    
    def _first_degradation(self, img):
        """First degradation: blur + resize + noise"""
        
        # 1. Blur (random kernel type)
        kernel_type = np.random.choice(['gaussian', 'iso', 'aniso'])
        kernel = self._get_blur_kernel(kernel_type)
        blurred = self._apply_blur(img, kernel)
        
        # 2. Resize (random mode)
        resize_mode = np.random.choice(['bilinear', 'bicubic', 'area'])
        scale = np.random.uniform(1, self.scale)
        resized = self._resize(blurred, 1/scale, mode=resize_mode)
        
        # 3. Noise
        noise_type = np.random.choice(['gaussian', 'poisson'])
        noisy = self._add_noise(resized, noise_type)
        
        return noisy
    
    def _second_degradation(self, img):
        """Second degradation: blur + resize + noise + JPEG"""
        
        # 1. Light blur
        if np.random.random() < 0.5:
            kernel = self._gaussian_blur(sigma=np.random.uniform(0.1, 1.5))
            img = self._apply_blur(img, kernel)
        
        # 2. Resize to target
        current_size = img.shape[-1]
        target_size = current_size // (self.scale // 2) if current_size > 64 else current_size
        img = self._resize(img, target_size / current_size)
        
        # 3. Light noise
        if np.random.random() < 0.5:
            img = self._add_noise(img, 'gaussian', sigma=np.random.uniform(0, 5))
        
        # 4. JPEG compression (very common in real images)
        if np.random.random() < 0.7:
            quality = np.random.randint(30, 95)
            img = self._jpeg_compress(img, quality)
        
        return img
    
    def _get_blur_kernel(self, kernel_type):
        """Generate blur kernel"""
        if kernel_type == 'gaussian':
            return self._gaussian_blur(sigma=np.random.uniform(0.2, 3.0))
        elif kernel_type == 'iso':
            return self._isotropic_blur(size=np.random.choice([7, 9, 11, 13, 15, 17, 19, 21]))
        elif kernel_type == 'aniso':
            return self._anisotropic_blur()
        else:
            return self._gaussian_blur()
    
    def _gaussian_blur(self, sigma=1.0, size=21):
        """2D Gaussian kernel"""
        x = torch.arange(size) - size // 2
        kernel_1d = torch.exp(-x.float() ** 2 / (2 * sigma ** 2))
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        return kernel_2d / kernel_2d.sum()
    
    def _motion_blur(self, length=15, angle=0):
        """Motion blur kernel"""
        kernel = torch.zeros(length, length)
        center = length // 2
        
        for i in range(length):
            kernel[center, i] = 1
        
        # Rotate
        if angle != 0:
            # Apply rotation (simplified)
            pass
        
        return kernel / kernel.sum()
    
    def _isotropic_blur(self, size=15):
        """Isotropic blur with random sigma"""
        sigma = np.random.uniform(0.1, 2.5) * size / 21
        return self._gaussian_blur(sigma, size)
    
    def _anisotropic_blur(self, size=21):
        """Anisotropic Gaussian with random orientation"""
        sigma_x = np.random.uniform(0.5, 4)
        sigma_y = np.random.uniform(0.5, 4)
        theta = np.random.uniform(0, np.pi)
        
        x = torch.arange(size) - size // 2
        y = torch.arange(size) - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Rotate coordinates
        xx_rot = xx * np.cos(theta) + yy * np.sin(theta)
        yy_rot = -xx * np.sin(theta) + yy * np.cos(theta)
        
        kernel = torch.exp(-(xx_rot.float() ** 2 / (2 * sigma_x ** 2) + 
                            yy_rot.float() ** 2 / (2 * sigma_y ** 2)))
        
        return kernel / kernel.sum()
    
    def _apply_blur(self, img, kernel):
        """Apply blur kernel to image"""
        kernel = kernel.view(1, 1, *kernel.shape).to(img.device)
        kernel = kernel.expand(img.shape[0], 1, -1, -1)
        
        padding = kernel.shape[-1] // 2
        
        # Apply per channel
        blurred = []
        for c in range(img.shape[0]):
            ch = img[c:c+1].unsqueeze(0)
            ch = F.pad(ch, [padding] * 4, mode='reflect')
            ch = F.conv2d(ch, kernel[c:c+1])
            blurred.append(ch.squeeze(0))
        
        return torch.cat(blurred, dim=0)
    
    def _resize(self, img, scale, mode='bicubic'):
        """Resize image"""
        h, w = img.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        
        img = img.unsqueeze(0)
        resized = F.interpolate(img, size=(new_h, new_w), mode=mode, 
                               align_corners=False if mode != 'area' else None)
        return resized.squeeze(0)
    
    def _add_noise(self, img, noise_type='gaussian', sigma=10):
        """Add noise to image"""
        if noise_type == 'gaussian':
            noise = torch.randn_like(img) * (sigma / 255.0)
            return (img + noise).clamp(0, 1)
        
        elif noise_type == 'poisson':
            # Poisson noise (signal-dependent)
            scaled = img * 255
            noisy = torch.poisson(scaled) / 255.0
            return noisy.clamp(0, 1)
        
        return img
    
    def _jpeg_compress(self, img, quality=70):
        """Simulate JPEG compression artifacts"""
        # Convert to PIL, compress, convert back
        # This is a simplified version
        
        # In practice, use actual JPEG encoding:
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        compressed = Image.open(buffer)
        compressed_np = np.array(compressed).astype(np.float32) / 255.0
        
        return torch.from_numpy(compressed_np).permute(2, 0, 1).to(img.device)


class BlindSR(nn.Module):
    """
    Blind super-resolution that estimates degradation
    Combines degradation prediction with restoration
    """
    
    def __init__(self):
        super().__init__()
        
        # Degradation predictor
        self.degradation_encoder = DegradationEncoder()
        
        # Restoration network (conditioned on degradation)
        self.restoration = ConditionalRestoration()
    
    def forward(self, lr):
        """
        Estimate degradation and restore accordingly
        """
        # Estimate degradation parameters
        degradation_features = self.degradation_encoder(lr)
        
        # Restore conditioned on degradation
        sr = self.restoration(lr, degradation_features)
        
        return sr


class DegradationEncoder(nn.Module):
    """Estimate degradation type and parameters"""
    
    def __init__(self, feature_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
        )
    
    def forward(self, x):
        return self.encoder(x)


class ConditionalRestoration(nn.Module):
    """Restoration network conditioned on degradation"""
    
    def __init__(self, deg_dim=256):
        super().__init__()
        
        # Modulation layers based on degradation
        self.mod1 = nn.Linear(deg_dim, 64 * 2)
        self.mod2 = nn.Linear(deg_dim, 128 * 2)
        
        # Main network
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        # ... more layers
    
    def forward(self, x, degradation):
        # FiLM conditioning
        gamma1, beta1 = self.mod1(degradation).chunk(2, dim=-1)
        gamma2, beta2 = self.mod2(degradation).chunk(2, dim=-1)
        
        # Apply with modulation
        h = self.conv1(x)
        h = h * gamma1.view(-1, 64, 1, 1) + beta1.view(-1, 64, 1, 1)
        h = F.relu(h)
        
        h = self.conv2(h)
        h = h * gamma2.view(-1, 128, 1, 1) + beta2.view(-1, 128, 1, 1)
        h = F.relu(h)
        
        # ... upsampling
        return h


class RealESRGAN(nn.Module):
    """
    Real-ESRGAN: Uses high-order degradation
    """
    
    def __init__(self):
        super().__init__()
        # Same generator as ESRGAN
        self.generator = ESRGANGenerator()
        
        # U-Net discriminator (better for real images)
        self.discriminator = UNetDiscriminator()
    
    def forward(self, lr):
        return self.generator(lr)


class ESRGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder
        self.net = nn.Conv2d(3, 3, 3, padding=1)
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode='bilinear')


class UNetDiscriminator(nn.Module):
    """U-Net discriminator for better real image discrimination"""
    def __init__(self):
        super().__init__()
        # Skip connections help distinguish real degradation


def degradation_comparison():
    """Compare degradation models"""
    
    print("Degradation Model Comparison:")
    print("=" * 60)
    
    models = [
        ("Bicubic", "Simple downsampling only", "Fails on real images"),
        ("BSRGAN", "Diverse blur + noise + JPEG", "Better generalization"),
        ("Real-ESRGAN", "High-order (blur→down→noise→JPEG)×2", "Best on wild images"),
        ("Blind SR", "Estimate degradation first", "Adaptive to image"),
    ]
    
    for name, method, result in models:
        print(f"{name:15} | {method:35} | {result}")
```

**Degradation Components:**

| Component | Types | Real-World Source |
|-----------|-------|-------------------|
| Blur | Gaussian, motion, defocus | Camera shake, focus |
| Downsampling | Bilinear, bicubic, area | Resolution reduction |
| Noise | Gaussian, Poisson, speckle | Sensor, low light |
| Compression | JPEG, video codec | Storage, transmission |

**Interview Tip:** The key insight is that "bicubic-trained" models learn the inverse of bicubic—they fail when given real degradation. Real-ESRGAN's high-order degradation (apply the pipeline twice) better simulates the real distribution. For production, always test on real low-quality images, not just synthetic ones.

---

### Question 47
- [ ] Done
**What techniques preserve fine details and textures during upscaling?**

**Answer:**

Preserving fine details requires: (1) perceptual/texture losses instead of pixel-wise, (2) attention mechanisms to focus on texture regions, (3) reference-based SR using high-res exemplars, (4) frequency separation to process high and low frequencies differently, and (5) GAN training for realistic texture synthesis. Key is avoiding blur from MSE averaging.

**Why Details Get Lost:**

| Cause | Effect | Solution |
|-------|--------|----------|
| MSE loss | Averages solutions → blur | Perceptual/adversarial loss |
| Convolution | Local receptive field | Attention, non-local |
| Pooling | Destroys high-freq | Skip connections |
| Downsampling | Aliasing | Anti-aliased operations |

**Techniques Summary:**

| Technique | How It Helps |
|-----------|-------------|
| Perceptual loss | Matches high-level features |
| Style/Gram loss | Matches texture statistics |
| GAN | Generates realistic textures |
| Attention | Focus on texture regions |
| Reference SR | Copy textures from HR examples |
| Frequency separation | Process bands separately |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TextureAwareSR(nn.Module):
    """
    Super-resolution with explicit texture preservation
    """
    
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        
        # Texture attention module
        self.texture_attention = TextureAttention()
        
        # Main SR network
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Residual blocks with attention
        self.res_blocks = nn.ModuleList([
            AttentiveResBlock(64) for _ in range(16)
        ])
        
        # Upsampling
        self.upsampler = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2),
        )
        
        self.output = nn.Conv2d(64, 3, 3, padding=1)
    
    def forward(self, x):
        # Compute texture attention map
        texture_map = self.texture_attention(x)
        
        # Feature extraction
        feat = self.feature_extract(x)
        
        # Residual blocks with texture guidance
        for block in self.res_blocks:
            feat = block(feat, texture_map)
        
        # Upsample and output
        up = self.upsampler(feat)
        out = self.output(up)
        
        return out


class TextureAttention(nn.Module):
    """Identify texture-rich regions that need special care"""
    
    def __init__(self):
        super().__init__()
        
        # Gradient-based texture detection
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        # Initialize Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x.weight.data = sobel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_y.view(1, 1, 3, 3)
        
        # Freeze
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Compute gradients
        grad_x = self.sobel_x(gray)
        grad_y = self.sobel_y(gray)
        
        # Gradient magnitude = texture indication
        texture_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Normalize
        texture_map = texture_map / (texture_map.max() + 1e-8)
        
        return texture_map


class AttentiveResBlock(nn.Module):
    """Residual block with texture attention"""
    
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # Attention modulation
        self.attention_fc = nn.Conv2d(1, channels, 1)
    
    def forward(self, x, texture_map):
        # Compute attention from texture map
        attention = torch.sigmoid(self.attention_fc(texture_map))
        
        # Forward with attention
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        
        # Apply more processing to texture regions
        h = h * (1 + attention)  # Scale by texture importance
        
        return x + h


class TextureLoss(nn.Module):
    """
    Texture/style loss using Gram matrix
    Encourages matching texture statistics
    """
    
    def __init__(self):
        super().__init__()
        
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:21])  # Up to relu4_1
        
        for param in self.layers.parameters():
            param.requires_grad = False
    
    def forward(self, sr, hr):
        sr_feat = self.layers(sr)
        hr_feat = self.layers(hr)
        
        sr_gram = self._gram_matrix(sr_feat)
        hr_gram = self._gram_matrix(hr_feat)
        
        return F.mse_loss(sr_gram, hr_gram)
    
    def _gram_matrix(self, feat):
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1)
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (C * H * W)


class NonLocalBlock(nn.Module):
    """
    Non-local attention for capturing long-range texture correlations
    """
    
    def __init__(self, channels):
        super().__init__()
        
        self.inter_channels = channels // 2
        
        self.theta = nn.Conv2d(channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(channels, self.inter_channels, 1)
        self.g = nn.Conv2d(channels, self.inter_channels, 1)
        self.out = nn.Conv2d(self.inter_channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute attention
        theta = self.theta(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        phi = self.phi(x).view(B, -1, H * W)  # (B, C', HW)
        g = self.g(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        
        # Attention map
        attention = F.softmax(torch.bmm(theta, phi), dim=-1)  # (B, HW, HW)
        
        # Aggregate
        out = torch.bmm(attention, g).permute(0, 2, 1).view(B, -1, H, W)
        out = self.out(out)
        
        return x + out


class ReferenceSR(nn.Module):
    """
    Reference-based SR: Use high-res reference for texture transfer
    Like SRNTT, TTSR
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extractor (shared for LR and Ref)
        self.feature_extractor = VGGFeatureExtractor()
        
        # Correspondence matching
        self.correspondence = CorrespondenceModule()
        
        # Texture transfer
        self.texture_transfer = TextureTransfer()
        
        # SR network
        self.sr_network = SRNetwork()
    
    def forward(self, lr, reference):
        """
        lr: Low-resolution input
        reference: High-resolution reference image (similar content)
        """
        # Extract features
        lr_feat = self.feature_extractor(lr)
        ref_feat = self.feature_extractor(reference)
        
        # Find correspondences
        correspondence_map = self.correspondence(lr_feat, ref_feat)
        
        # Transfer textures
        transferred = self.texture_transfer(reference, correspondence_map)
        
        # SR with texture guidance
        sr = self.sr_network(lr, transferred)
        
        return sr


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:35])
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        return self.layers(x)


class CorrespondenceModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, lr_feat, ref_feat):
        # Compute cosine similarity between patches
        B, C, H, W = lr_feat.shape
        _, _, H_ref, W_ref = ref_feat.shape
        
        lr_flat = lr_feat.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        ref_flat = ref_feat.view(B, C, -1)  # (B, C, H_ref*W_ref)
        
        # Normalize
        lr_norm = F.normalize(lr_flat, dim=-1)
        ref_norm = F.normalize(ref_flat, dim=1)
        
        # Similarity
        similarity = torch.bmm(lr_norm, ref_norm)  # (B, HW, HW_ref)
        
        # Get best matches
        correspondence = similarity.argmax(dim=-1)  # (B, HW)
        
        return correspondence.view(B, H, W)


class TextureTransfer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, reference, correspondence):
        # Warp reference according to correspondence
        B, C, H, W = reference.shape
        
        # Create sampling grid from correspondence
        # ... (simplified)
        
        return reference  # Placeholder


class SRNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(6, 3, 3, padding=1)
    
    def forward(self, lr, texture):
        # Concatenate and process
        lr_up = F.interpolate(lr, scale_factor=4, mode='bilinear')
        combined = torch.cat([lr_up, texture], dim=1)
        return self.net(combined)


class FrequencySeparation(nn.Module):
    """
    Process high and low frequencies separately
    High-freq needs more care for detail preservation
    """
    
    def __init__(self):
        super().__init__()
        
        # Low-frequency path (structure)
        self.low_path = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        
        # High-frequency path (detail)
        self.high_path = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def forward(self, x):
        # Separate frequencies
        low = self._low_pass(x)
        high = x - low
        
        # Process separately
        low_sr = self.low_path(low)
        high_sr = self.high_path(high)
        
        # Combine
        return low_sr + high_sr
    
    def _low_pass(self, x, sigma=2):
        """Gaussian low-pass filter"""
        kernel_size = int(6 * sigma) | 1
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size) - kernel_size // 2
        kernel = torch.exp(-coords.float() ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        # 2D kernel
        kernel_2d = kernel.view(1, 1, -1, 1) * kernel.view(1, 1, 1, -1)
        kernel_2d = kernel_2d.expand(3, 1, -1, -1).to(x.device)
        
        # Apply
        padding = kernel_size // 2
        x_pad = F.pad(x, [padding] * 4, mode='reflect')
        return F.conv2d(x_pad, kernel_2d, groups=3)
```

**Loss Functions for Detail:**

| Loss | Focus | Effect |
|------|-------|--------|
| L1/MSE | Pixel accuracy | Blur |
| VGG/Perceptual | Features | Sharp edges |
| Gram/Style | Textures | Realistic patterns |
| GAN | Realism | Natural textures |
| Contextual | Patches | Local detail |

**Interview Tip:** The core insight is that pixel-wise losses average over uncertainty, creating blur. Perceptual losses in feature space allow local variations while maintaining semantic content. GAN losses push toward the natural image manifold. Reference-based SR goes further by copying actual textures from similar high-res images, ideal when reference is available.

---

### Question 48
- [ ] Done
**How do you implement efficient super-resolution for real-time video streaming?**

**Answer:**

Real-time video SR requires: (1) lightweight architectures (fewer parameters, efficient ops), (2) temporal consistency using previous frames, (3) hardware optimization (TensorRT, quantization), (4) adaptive quality based on bandwidth/compute, and (5) recurrent designs that reuse computation across frames. Target is >30fps for 720p→4K on consumer GPUs.

**Speed vs Quality Trade-offs:**

| Model | Parameters | PSNR | FPS (4K) | Use Case |
|-------|------------|------|----------|----------|
| ESRGAN | 16.7M | 30.5 | 2-5 | Offline |
| Real-Time SR | 1-3M | 28-29 | 30-60 | Gaming |
| Neural scaler | <1M | 27 | 60-120 | Streaming |

**Efficiency Techniques:**

| Technique | Speedup | Quality Impact |
|-----------|---------|----------------|
| Smaller backbone | 5-10x | -1-2 dB PSNR |
| Depthwise separable | 2-3x | Minimal |
| Quantization (INT8) | 2-4x | -0.5 dB |
| Pruning | 2-3x | -0.5 dB |
| Temporal reuse | 1.5-2x | Improves consistency |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientVideoSR(nn.Module):
    """
    Lightweight real-time video super-resolution
    Uses recurrent design for temporal consistency
    """
    
    def __init__(self, scale=4, channels=32):
        super().__init__()
        self.scale = scale
        
        # Shallow feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Recurrent aggregation (uses previous frame features)
        self.recurrent = ConvLSTMCell(channels, channels)
        
        # Lightweight residual blocks
        self.res_blocks = nn.Sequential(
            *[EfficientResBlock(channels) for _ in range(4)]
        )
        
        # Efficient upsampling
        self.upsampler = EfficientUpsampler(scale, channels)
        
        # Output
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        
        # Hidden state
        self.hidden = None
    
    def forward(self, x, reset_hidden=False):
        """
        x: (B, C, H, W) current frame
        reset_hidden: True at start of new video
        """
        if reset_hidden:
            self.hidden = None
        
        # Feature extraction
        feat = self.feature_extract(x)
        
        # Recurrent aggregation
        if self.hidden is None:
            B, C, H, W = feat.shape
            self.hidden = (torch.zeros_like(feat), torch.zeros_like(feat))
        
        feat, self.hidden = self.recurrent(feat, self.hidden)
        
        # Residual processing
        feat = self.res_blocks(feat)
        
        # Upsample
        up = self.upsampler(feat)
        
        # Output with residual from upsampled input
        out = self.output(up)
        out = out + F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        return out


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM for temporal processing"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Combined gates
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 3, padding=1)
    
    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        return h, (h, c)


class EfficientResBlock(nn.Module):
    """Lightweight residual block with depthwise separable conv"""
    
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            # Pointwise
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Depthwise
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            # Pointwise
            nn.Conv2d(channels, channels, 1),
        )
    
    def forward(self, x):
        return x + self.block(x)


class EfficientUpsampler(nn.Module):
    """Memory-efficient upsampling"""
    
    def __init__(self, scale, channels):
        super().__init__()
        
        if scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
    
    def forward(self, x):
        return self.up(x)


class AdaptiveQualitySR(nn.Module):
    """
    Adaptive quality based on available compute/bandwidth
    Multiple exit points for different quality levels
    """
    
    def __init__(self, scale=4, base_channels=32):
        super().__init__()
        
        # Shared feature extraction
        self.shared_features = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            EfficientResBlock(base_channels),
        )
        
        # Fast path (2 blocks) - lowest quality
        self.fast_blocks = nn.Sequential(
            EfficientResBlock(base_channels),
            EfficientResBlock(base_channels),
        )
        
        # Medium path (+2 blocks)
        self.medium_blocks = nn.Sequential(
            EfficientResBlock(base_channels),
            EfficientResBlock(base_channels),
        )
        
        # High quality path (+4 blocks)
        self.high_blocks = nn.Sequential(
            EfficientResBlock(base_channels),
            EfficientResBlock(base_channels),
            EfficientResBlock(base_channels),
            EfficientResBlock(base_channels),
        )
        
        # Shared upsampler
        self.upsampler = EfficientUpsampler(scale, base_channels)
        self.output = nn.Conv2d(base_channels, 3, 3, padding=1)
    
    def forward(self, x, quality='high'):
        """
        quality: 'fast', 'medium', 'high'
        """
        feat = self.shared_features(x)
        feat = self.fast_blocks(feat)
        
        if quality == 'fast':
            pass  # Use current features
        elif quality == 'medium':
            feat = self.medium_blocks(feat)
        elif quality == 'high':
            feat = self.medium_blocks(feat)
            feat = self.high_blocks(feat)
        
        up = self.upsampler(feat)
        out = self.output(up)
        
        # Global residual
        out = out + F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out


class TemporalConsistency:
    """
    Techniques for temporal consistency in video SR
    """
    
    @staticmethod
    def flow_warp(prev_frame, flow):
        """Warp previous frame using optical flow"""
        B, C, H, W = prev_frame.shape
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=prev_frame.device),
            torch.arange(W, device=prev_frame.device),
            indexing='ij'
        )
        
        # Add flow
        grid_x = grid_x.float() + flow[:, 0]
        grid_y = grid_y.float() + flow[:, 1]
        
        # Normalize to [-1, 1]
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        return F.grid_sample(prev_frame, grid, mode='bilinear', align_corners=False)
    
    @staticmethod
    def temporal_loss(current_sr, prev_sr_warped):
        """Penalize temporal inconsistency"""
        return F.l1_loss(current_sr, prev_sr_warped)


class HardwareOptimization:
    """Techniques for hardware deployment"""
    
    @staticmethod
    def quantize_model(model, calibration_data):
        """
        INT8 quantization for faster inference
        """
        # PyTorch quantization
        model.eval()
        model_fp32 = model
        
        # Fuse modules
        model_fused = torch.quantization.fuse_modules(model, [['conv', 'relu']])
        
        # Prepare for quantization
        model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model_fused)
        
        # Calibrate
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data)
        
        # Convert
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized
    
    @staticmethod
    def export_tensorrt(model, example_input):
        """Export to TensorRT for NVIDIA GPUs"""
        # torch.onnx.export(model, example_input, 'model.onnx')
        # Then use trtexec to convert ONNX to TensorRT
        pass
    
    @staticmethod
    def optimize_memory():
        """Memory optimization tips"""
        tips = [
            "Use inplace operations (nn.ReLU(inplace=True))",
            "Gradient checkpointing for training",
            "Process in tiles for large images",
            "Use half precision (FP16)",
            "Fuse BatchNorm with Conv for inference",
        ]
        return tips


class RealTimeVideoSRPipeline:
    """Complete real-time video SR pipeline"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
        
        # Use CUDA streams for async processing
        self.stream = torch.cuda.Stream()
    
    def process_video(self, video_frames, batch_size=4):
        """
        Process video with batching for efficiency
        """
        results = []
        
        with torch.no_grad():
            for i in range(0, len(video_frames), batch_size):
                batch = torch.stack(video_frames[i:i+batch_size]).to(self.device)
                
                # Reset hidden state at video start
                reset = (i == 0)
                
                with torch.cuda.stream(self.stream):
                    sr_batch = self.model(batch, reset_hidden=reset)
                
                self.stream.synchronize()
                results.extend(sr_batch.cpu())
        
        return results


def benchmark_comparison():
    """Compare different real-time SR approaches"""
    
    print("Real-Time Video SR Comparison:")
    print("=" * 70)
    
    models = [
        ("ESRGAN", "16.7M", "2", "30.5", "Offline processing"),
        ("Real-ESRGAN (fast)", "5M", "15", "29.5", "Near real-time"),
        ("EDVR", "20M", "5", "31.0", "High quality offline"),
        ("BasicVSR++", "8M", "10", "30.8", "Quality-speed balance"),
        ("Real-Time SR (ours)", "1M", "60", "28.0", "Real-time streaming"),
    ]
    
    print(f"{'Model':<20} {'Params':<10} {'FPS':<8} {'PSNR':<8} {'Use Case'}")
    print("-" * 70)
    for name, params, fps, psnr, use in models:
        print(f"{name:<20} {params:<10} {fps:<8} {psnr:<8} {use}")
```

**Architecture Choices:**

| Component | Fast Option | Quality Impact |
|-----------|-------------|----------------|
| Backbone | MobileNet-style | -1 dB |
| Residual blocks | 4-8 instead of 16-23 | -0.5-1 dB |
| Channels | 32-48 instead of 64 | -0.5 dB |
| Convolutions | Depthwise separable | Minimal |
| Upsampling | PixelShuffle (efficient) | None |

**Interview Tip:** Real-time video SR is about smart trade-offs. Key insights: (1) recurrent architectures amortize cost over frames, (2) temporal consistency is both quality feature and efficiency tool (reuse prev frame info), (3) adaptive quality lets you scale with available compute, (4) hardware optimization (TensorRT, quantization) often gives 2-4x speedup with minimal quality loss.

---

## Facial Recognition

### Question 49
- [ ] Done
**Explain face embedding networks (FaceNet, ArcFace, CosFace) and metric learning losses.**

**Answer:**

Face embedding networks map face images to compact vectors where similar faces are close and different faces are far apart. FaceNet uses triplet loss, ArcFace adds angular margin to softmax, and CosFace uses cosine margin. These margin-based losses create more discriminative embeddings than standard softmax by enforcing geometric separation.

**Evolution of Loss Functions:**

| Method | Year | Key Idea | Formula |
|--------|------|----------|---------|
| Softmax | - | Classification | $-\log\frac{e^{W_y^T f}}{∑_j e^{W_j^T f}}$ |
| Triplet Loss | 2015 | Anchor-pos-neg | $\|f_a - f_p\|^2 - \|f_a - f_n\|^2 + \alpha$ |
| SphereFace | 2017 | Multiplicative angular | $-\log\frac{e^{\|f\|\cos(m\theta_y)}}{...}$ |
| CosFace | 2018 | Additive cosine | $-\log\frac{e^{s(\cos\theta_y - m)}}{...}$ |
| ArcFace | 2019 | Additive angular | $-\log\frac{e^{s\cos(\theta_y + m)}}{...}$ |

**Geometric Interpretation:**

- **Softmax**: No explicit margin, embeddings may be close
- **Triplet**: Direct distance constraint, but hard mining needed
- **ArcFace**: Arc (geodesic) distance margin on hypersphere
- **CosFace**: Cosine similarity margin

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FaceEmbeddingNetwork(nn.Module):
    """
    Face embedding with backbone + embedding head
    """
    
    def __init__(self, backbone='resnet50', embedding_dim=512):
        super().__init__()
        
        # Backbone (ResNet, MobileNet, etc.)
        self.backbone = self._build_backbone(backbone)
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def _build_backbone(self, name):
        from torchvision import models
        if name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC
            return backbone
        # Add other backbones
        return nn.Identity()
    
    def forward(self, x):
        """
        x: (B, 3, 112, 112) face image
        Returns: (B, embedding_dim) normalized embedding
        """
        features = self.backbone(x).flatten(1)
        embedding = self.embedding(features)
        
        # L2 normalize for angular losses
        embedding = F.normalize(embedding, dim=1)
        
        return embedding


class TripletLoss(nn.Module):
    """
    FaceNet triplet loss
    L = max(0, ||f_a - f_p||^2 - ||f_a - f_n||^2 + margin)
    """
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: (B, D) embeddings
        """
        dist_ap = (anchor - positive).pow(2).sum(dim=1)
        dist_an = (anchor - negative).pow(2).sum(dim=1)
        
        loss = F.relu(dist_ap - dist_an + self.margin)
        
        return loss.mean()
    
    @staticmethod
    def hard_mining(embeddings, labels):
        """
        Select hardest triplets within batch
        - Hardest positive: same class, max distance
        - Hardest negative: different class, min distance
        """
        B = embeddings.shape[0]
        dist_matrix = torch.cdist(embeddings, embeddings)
        
        triplets = []
        for i in range(B):
            pos_mask = labels == labels[i]
            neg_mask = labels != labels[i]
            
            if pos_mask.sum() > 1 and neg_mask.sum() > 0:
                # Hardest positive (excluding self)
                pos_dists = dist_matrix[i][pos_mask]
                pos_dists[pos_dists == 0] = -1
                hardest_pos = pos_mask.nonzero()[pos_dists.argmax()].item()
                
                # Hardest negative
                neg_dists = dist_matrix[i][neg_mask]
                hardest_neg = neg_mask.nonzero()[neg_dists.argmin()].item()
                
                triplets.append((i, hardest_pos, hardest_neg))
        
        return triplets


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    Most popular choice for face recognition
    
    L = -log(e^{s*cos(θ_y + m)} / (e^{s*cos(θ_y + m)} + Σ_j≠y e^{s*cos(θ_j)}))
    """
    
    def __init__(self, num_classes, embedding_dim=512, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
        # Class weight matrix W
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos/sin margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, labels):
        """
        embeddings: (B, D) L2-normalized face embeddings
        labels: (B,) class labels
        """
        # Normalize weights
        W = F.normalize(self.weight, dim=1)
        
        # Cosine similarity (since embeddings are normalized)
        cosine = F.linear(embeddings, W)  # (B, num_classes)
        
        # Get cos(θ + m) using cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)
        
        # Numerical stability: when cos(θ) < cos(π - m), use cos(θ) - m*sin(m)
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to ground truth class
        output = one_hot * phi + (1 - one_hot) * cosine
        
        # Scale
        output = output * self.scale
        
        # Cross-entropy
        loss = F.cross_entropy(output, labels)
        
        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace: Large Margin Cosine Loss
    
    L = -log(e^{s*(cos(θ_y) - m)} / (e^{s*(cos(θ_y) - m)} + Σ_j≠y e^{s*cos(θ_j)}))
    """
    
    def __init__(self, num_classes, embedding_dim=512, scale=64.0, margin=0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        W = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, W)
        
        # Subtract margin from ground truth class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = cosine - one_hot * self.margin
        output = output * self.scale
        
        return F.cross_entropy(output, labels)


class CombinedMarginLoss(nn.Module):
    """
    Combined margin: m1 * angle + m2 * cosine + m3
    Unifies SphereFace, CosFace, ArcFace
    """
    
    def __init__(self, num_classes, embedding_dim=512, scale=64.0,
                 m1=1.0, m2=0.0, m3=0.4):
        super().__init__()
        self.scale = scale
        self.m1 = m1  # Angular multiplier (SphereFace)
        self.m2 = m2  # Cosine margin (CosFace)
        self.m3 = m3  # Angular margin (ArcFace)
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        W = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, W)
        
        # Get angle
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        
        # Apply margins: m1 * θ + m3, then take cos
        theta_margin = self.m1 * theta + self.m3
        phi = torch.cos(theta_margin) - self.m2
        
        # Apply to ground truth only
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = one_hot * phi + (1 - one_hot) * cosine
        output = output * self.scale
        
        return F.cross_entropy(output, labels)


class FaceRecognitionInference:
    """Complete face recognition pipeline"""
    
    def __init__(self, model, threshold=0.6):
        self.model = model
        self.threshold = threshold
        self.gallery = {}  # name -> embedding
    
    def enroll(self, name, face_images):
        """Enroll a person with multiple images"""
        embeddings = []
        for img in face_images:
            emb = self.model(img.unsqueeze(0))
            embeddings.append(emb)
        
        # Average embedding
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        avg_embedding = F.normalize(avg_embedding, dim=1)
        
        self.gallery[name] = avg_embedding
    
    def identify(self, face_image):
        """Identify a face against gallery"""
        query_emb = self.model(face_image.unsqueeze(0))
        
        best_match = None
        best_score = -1
        
        for name, gallery_emb in self.gallery.items():
            # Cosine similarity
            score = F.cosine_similarity(query_emb, gallery_emb).item()
            
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score > self.threshold:
            return best_match, best_score
        else:
            return "Unknown", best_score
    
    def verify(self, face1, face2):
        """1:1 verification - are these the same person?"""
        emb1 = self.model(face1.unsqueeze(0))
        emb2 = self.model(face2.unsqueeze(0))
        
        similarity = F.cosine_similarity(emb1, emb2).item()
        
        return similarity > self.threshold, similarity


def loss_comparison():
    """Compare different loss functions"""
    
    print("Face Recognition Loss Comparison:")
    print("=" * 70)
    
    losses = [
        ("Softmax", "None", "Baseline, not discriminative enough"),
        ("Triplet", "Euclidean distance", "Hard mining required, slow convergence"),
        ("SphereFace", "m * θ (multiplicative)", "First angular margin, unstable"),
        ("CosFace", "cos(θ) - m (cosine)", "Stable, good performance"),
        ("ArcFace", "θ + m (angular)", "Best geometric interpretation, SOTA"),
    ]
    
    print(f"{'Loss':<15} {'Margin Type':<25} {'Notes'}")
    print("-" * 70)
    for name, margin, notes in losses:
        print(f"{name:<15} {margin:<25} {notes}")
```

**Key Hyperparameters:**

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| Scale (s) | 30-64 | Larger = sharper softmax |
| Margin (m) | 0.3-0.5 | Larger = harder, more discriminative |
| Embedding dim | 128-512 | Larger = more capacity |

**Interview Tip:** ArcFace is the current standard because its angular margin has clear geometric meaning—embeddings must be at least $m$ radians apart on the hypersphere. The scale parameter $s$ controls softmax temperature. Triplet loss is mostly replaced by margin losses because margin losses are easier to train (no mining) and scale to millions of classes.

---

### Question 50
- [ ] Done
**How do you handle face recognition across different ethnicities with fairness considerations?**

**Answer:**

Face recognition shows performance disparities across demographics due to imbalanced training data and feature biases. Solutions include: (1) balanced datasets across demographics, (2) fairness-aware training with demographic parity constraints, (3) bias evaluation on diverse test sets, and (4) adaptive thresholds per demographic. Addressing fairness is both ethical imperative and legal requirement.

**Common Bias Sources:**

| Source | Effect | Mitigation |
|--------|--------|------------|
| Training data imbalance | Poor accuracy on minorities | Balanced sampling |
| Feature extraction | Favors majority appearance | Fair representation learning |
| Threshold setting | Different FPR across groups | Per-group calibration |
| Image quality | Lighting bias against dark skin | Better imaging, normalization |

**Fairness Metrics:**

| Metric | Definition | Target |
|--------|------------|--------|
| Demographic Parity | Same acceptance rate across groups | Business fairness |
| Equal Opportunity | Same TPR across groups | Equal benefit |
| Equalized Odds | Same TPR and FPR across groups | Error rate parity |
| Calibration | Same precision across groups | Consistent confidence |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class FairFaceRecognition:
    """Face recognition with fairness monitoring and mitigation"""
    
    def __init__(self, model, demographic_groups):
        self.model = model
        self.groups = demographic_groups  # ['Asian', 'Black', 'White', etc.]
        
        # Per-group thresholds (calibrated)
        self.thresholds = {g: 0.5 for g in demographic_groups}
        
        # Performance tracking
        self.metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0})
    
    def calibrate_thresholds(self, val_data, target_fpr=0.01):
        """
        Set per-group thresholds to achieve equal FPR
        """
        for group in self.groups:
            group_data = [(emb, label) for emb, label, demo in val_data if demo == group]
            
            # Find threshold that gives target FPR for this group
            self.thresholds[group] = self._find_threshold_for_fpr(
                group_data, target_fpr
            )
        
        return self.thresholds
    
    def _find_threshold_for_fpr(self, data, target_fpr):
        """Binary search for threshold"""
        low, high = 0.0, 1.0
        
        while high - low > 0.001:
            mid = (low + high) / 2
            fpr = self._compute_fpr(data, mid)
            
            if fpr > target_fpr:
                low = mid
            else:
                high = mid
        
        return mid
    
    def _compute_fpr(self, data, threshold):
        """Compute FPR at given threshold"""
        fp, tn = 0, 0
        for emb, is_match in data:
            pred = emb > threshold
            if not is_match:
                if pred:
                    fp += 1
                else:
                    tn += 1
        return fp / (fp + tn + 1e-10)
    
    def verify_fair(self, face1, face2, demographic):
        """
        Verification with demographic-aware threshold
        """
        emb1 = self.model(face1)
        emb2 = self.model(face2)
        
        similarity = F.cosine_similarity(emb1, emb2).item()
        threshold = self.thresholds.get(demographic, 0.5)
        
        return similarity > threshold, similarity
    
    def compute_fairness_metrics(self):
        """Compute fairness metrics across groups"""
        results = {}
        
        for group in self.groups:
            m = self.metrics[group]
            tpr = m['TP'] / (m['TP'] + m['FN'] + 1e-10)
            fpr = m['FP'] / (m['FP'] + m['TN'] + 1e-10)
            
            results[group] = {'TPR': tpr, 'FPR': fpr}
        
        # Compute disparities
        tprs = [results[g]['TPR'] for g in self.groups]
        fprs = [results[g]['FPR'] for g in self.groups]
        
        results['max_tpr_gap'] = max(tprs) - min(tprs)
        results['max_fpr_gap'] = max(fprs) - min(fprs)
        
        return results


class FairDataLoader:
    """Balanced sampling across demographics"""
    
    def __init__(self, dataset, demographic_labels, batch_size=64):
        self.dataset = dataset
        self.demographics = demographic_labels
        self.batch_size = batch_size
        
        # Group indices
        self.group_indices = defaultdict(list)
        for i, demo in enumerate(demographic_labels):
            self.group_indices[demo].append(i)
    
    def __iter__(self):
        """Yield balanced batches"""
        groups = list(self.group_indices.keys())
        samples_per_group = self.batch_size // len(groups)
        
        # Shuffle each group
        for g in groups:
            np.random.shuffle(self.group_indices[g])
        
        # Create batches
        group_cursors = {g: 0 for g in groups}
        
        while True:
            batch_indices = []
            
            for g in groups:
                indices = self.group_indices[g]
                cursor = group_cursors[g]
                
                if cursor + samples_per_group > len(indices):
                    np.random.shuffle(indices)
                    cursor = 0
                
                batch_indices.extend(indices[cursor:cursor + samples_per_group])
                group_cursors[g] = cursor + samples_per_group
            
            yield [self.dataset[i] for i in batch_indices]


class FairnessAwareLoss(nn.Module):
    """
    Loss function with fairness regularization
    """
    
    def __init__(self, base_loss, fairness_weight=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.fairness_weight = fairness_weight
    
    def forward(self, embeddings, labels, demographics):
        """
        Add fairness penalty to encourage similar performance across groups
        """
        # Base classification loss
        base = self.base_loss(embeddings, labels)
        
        # Fairness regularization
        fairness = self._compute_fairness_penalty(embeddings, labels, demographics)
        
        return base + self.fairness_weight * fairness
    
    def _compute_fairness_penalty(self, embeddings, labels, demographics):
        """
        Penalize if embedding statistics differ across demographics
        """
        unique_demos = torch.unique(demographics)
        
        means = []
        stds = []
        
        for demo in unique_demos:
            mask = demographics == demo
            demo_embeddings = embeddings[mask]
            
            means.append(demo_embeddings.mean(dim=0))
            stds.append(demo_embeddings.std(dim=0))
        
        # Penalize difference in means (representation alignment)
        mean_penalty = 0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                mean_penalty += (means[i] - means[j]).pow(2).mean()
        
        return mean_penalty


class AdversarialDebiasing(nn.Module):
    """
    Learn embeddings that don't encode demographic information
    """
    
    def __init__(self, embedding_model, num_demographics):
        super().__init__()
        self.embedding_model = embedding_model
        
        # Adversarial demographic predictor
        self.demographic_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_demographics)
        )
        
        # Gradient reversal layer
        self.grad_reversal = GradientReversalLayer()
    
    def forward(self, x, predict_demographic=False):
        embeddings = self.embedding_model(x)
        
        if predict_demographic:
            # Apply gradient reversal for adversarial training
            reversed_emb = self.grad_reversal(embeddings)
            demo_pred = self.demographic_predictor(reversed_emb)
            return embeddings, demo_pred
        
        return embeddings


class GradientReversalLayer(torch.autograd.Function):
    """Reverses gradients during backward pass"""
    
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class BiasEvaluation:
    """Evaluate model bias across demographics"""
    
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
    
    def evaluate(self):
        """
        Compute performance metrics per demographic group
        """
        results = defaultdict(list)
        
        for face1, face2, is_same, demo1, demo2 in self.test_data:
            emb1 = self.model(face1)
            emb2 = self.model(face2)
            
            sim = F.cosine_similarity(emb1, emb2).item()
            
            # Track by demographic pairs
            demo_pair = f"{demo1}-{demo2}"
            results[demo_pair].append((sim, is_same))
        
        # Compute metrics per group
        metrics = {}
        for demo_pair, data in results.items():
            metrics[demo_pair] = self._compute_metrics(data)
        
        return metrics
    
    def _compute_metrics(self, data, threshold=0.5):
        tp = fp = tn = fn = 0
        
        for sim, is_same in data:
            pred = sim > threshold
            if is_same:
                if pred:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred:
                    fp += 1
                else:
                    tn += 1
        
        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn + 1e-10),
            'tpr': tp / (tp + fn + 1e-10),
            'fpr': fp / (fp + tn + 1e-10),
        }


def fairness_report():
    """Example fairness evaluation report"""
    
    print("Face Recognition Fairness Report")
    print("=" * 60)
    
    # Example disparities (based on NIST studies)
    demographics = ['White', 'Black', 'Asian', 'Hispanic']
    
    print("\nVerification Error Rates (@ FAR=0.01):")
    print(f"{'Demographic':<15} {'FRR (%)':<12} {'Gap from best'}")
    print("-" * 45)
    
    # Hypothetical but realistic disparities
    frrs = {'White': 2.0, 'Black': 10.0, 'Asian': 3.0, 'Hispanic': 5.0}
    best_frr = min(frrs.values())
    
    for demo in demographics:
        gap = frrs[demo] - best_frr
        print(f"{demo:<15} {frrs[demo]:<12.1f} {gap:+.1f}%")
    
    print("\nRecommendations:")
    print("  1. Increase training data for underrepresented groups")
    print("  2. Use demographic-aware threshold calibration")
    print("  3. Apply adversarial debiasing during training")
    print("  4. Regular bias auditing with diverse test sets")
```

**Key Strategies:**

| Strategy | Implementation | Effectiveness |
|----------|---------------|---------------|
| Data balancing | Equal samples per group | Medium-High |
| Adaptive thresholds | Per-group calibration | High |
| Adversarial debiasing | GRL for demographic prediction | Medium |
| Fairness loss | Regularize group statistics | Medium |
| Diverse test sets | RFW, BFW benchmarks | Monitoring |

**Interview Tip:** Fairness in face recognition is not just ethical but often legally required (GDPR, CCPA). The NIST FRVT studies show 10-100x error rate differences between demographics in commercial systems. Key insight: a single global threshold cannot be fair across groups if the embedding distributions differ. Solutions include per-group calibration or learning embeddings where demographics cannot be predicted.

---

### Question 51
- [ ] Done
**What techniques work for face recognition with masks, glasses, or partial occlusion?**

**Answer:**

Partial occlusion requires: (1) occlusion-aware training with augmented data, (2) attention mechanisms focusing on visible regions, (3) part-based models that use available face parts, and (4) mask-specific architectures. Post-COVID, masked face recognition became critical—periocular (eye region) features become primary, requiring specialized training.

**Occlusion Types and Solutions:**

| Occlusion | Visible Region | Primary Features |
|-----------|---------------|------------------|
| Surgical mask | Eyes, forehead | Periocular |
| Sunglasses | Lower face, forehead | Mouth, jaw |
| Hat/cap | Full face | Standard features |
| Hair occlusion | Partial face | Adaptive |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class OcclusionAwareFaceNet(nn.Module):
    """
    Face recognition robust to partial occlusion
    Uses attention to focus on visible regions
    """
    
    def __init__(self, backbone, embedding_dim=512):
        super().__init__()
        self.backbone = backbone
        
        # Spatial attention to weight visible regions
        self.attention = SpatialAttention()
        
        # Part-based feature aggregation
        self.part_features = PartBasedFeatures()
        
        # Final embedding
        self.embedding = nn.Linear(2048, embedding_dim)
    
    def forward(self, x, occlusion_mask=None):
        """
        x: (B, 3, 112, 112) face image
        occlusion_mask: (B, 1, 112, 112) optional mask (1=occluded)
        """
        # Extract features
        features = self.backbone(x)  # (B, 2048, 7, 7)
        
        # Compute attention (or use provided occlusion mask)
        if occlusion_mask is not None:
            attention = 1 - F.interpolate(occlusion_mask, features.shape[-2:])
        else:
            attention = self.attention(features)
        
        # Weighted feature aggregation
        weighted_features = features * attention
        pooled = weighted_features.mean(dim=(2, 3))
        
        # Embedding
        embedding = self.embedding(pooled)
        embedding = F.normalize(embedding, dim=1)
        
        return embedding


class SpatialAttention(nn.Module):
    """Learn to attend to unoccluded regions"""
    
    def __init__(self, channels=2048):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        return self.conv(features)


class PartBasedFeatures(nn.Module):
    """
    Extract features from face parts independently
    More robust when some parts are occluded
    """
    
    def __init__(self, num_parts=4, feature_dim=512):
        super().__init__()
        
        # Part regions: forehead, eyes, nose, mouth
        self.num_parts = num_parts
        
        # Separate embeddings per part
        self.part_embeddings = nn.ModuleList([
            nn.Linear(512, feature_dim // num_parts) for _ in range(num_parts)
        ])
        
        # Part quality estimator (predict if part is visible)
        self.quality_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(num_parts)
        ])
    
    def forward(self, features, spatial_shape):
        """
        features: (B, C, H, W) feature map
        Returns weighted combination of part features
        """
        B, C, H, W = features.shape
        
        # Define part regions (vertical strips for simplicity)
        part_height = H // self.num_parts
        
        embeddings = []
        qualities = []
        
        for i in range(self.num_parts):
            # Extract part region
            start = i * part_height
            end = start + part_height
            part_feat = features[:, :, start:end, :].mean(dim=(2, 3))
            
            # Part embedding
            emb = self.part_embeddings[i](part_feat)
            embeddings.append(emb)
            
            # Quality score
            qual = self.quality_estimator[i](part_feat)
            qualities.append(qual)
        
        # Weighted aggregation by quality
        embeddings = torch.stack(embeddings, dim=1)  # (B, num_parts, D)
        qualities = torch.cat(qualities, dim=1)  # (B, num_parts)
        qualities = F.softmax(qualities, dim=1).unsqueeze(-1)
        
        final_embedding = (embeddings * qualities).sum(dim=1)
        
        return final_embedding


class MaskedFaceAugmentation:
    """Data augmentation with synthetic occlusions"""
    
    def __init__(self, mask_prob=0.5):
        self.mask_prob = mask_prob
        
        # Mask templates
        self.mask_types = ['surgical', 'n95', 'cloth', 'sunglasses', 'random']
    
    def __call__(self, image):
        """
        image: (C, H, W) face image
        Returns: augmented image with synthetic occlusion
        """
        if random.random() > self.mask_prob:
            return image
        
        mask_type = random.choice(self.mask_types)
        
        if mask_type in ['surgical', 'n95', 'cloth']:
            return self._apply_face_mask(image)
        elif mask_type == 'sunglasses':
            return self._apply_sunglasses(image)
        else:
            return self._apply_random_occlusion(image)
    
    def _apply_face_mask(self, image):
        """Simulate face mask covering lower face"""
        C, H, W = image.shape
        
        # Mask lower 40-50% of face
        mask_start = int(H * random.uniform(0.4, 0.5))
        
        # Random mask color
        mask_color = random.choice([
            torch.tensor([0.9, 0.9, 0.9]),  # White
            torch.tensor([0.1, 0.3, 0.6]),  # Blue
            torch.tensor([0.1, 0.1, 0.1]),  # Black
        ])
        
        masked = image.clone()
        for c in range(C):
            masked[c, mask_start:, :] = mask_color[c]
        
        return masked
    
    def _apply_sunglasses(self, image):
        """Simulate sunglasses covering eye region"""
        C, H, W = image.shape
        
        # Cover 20-35% from top
        start = int(H * random.uniform(0.15, 0.25))
        end = int(H * random.uniform(0.35, 0.45))
        
        masked = image.clone()
        masked[:, start:end, :] = 0.1  # Dark glasses
        
        return masked
    
    def _apply_random_occlusion(self, image):
        """Random rectangular occlusion"""
        C, H, W = image.shape
        
        # Random rectangle
        h_size = int(H * random.uniform(0.2, 0.4))
        w_size = int(W * random.uniform(0.2, 0.4))
        
        h_start = random.randint(0, H - h_size)
        w_start = random.randint(0, W - w_size)
        
        masked = image.clone()
        masked[:, h_start:h_start+h_size, w_start:w_start+w_size] = 0.5
        
        return masked


class PeriocularRecognition(nn.Module):
    """
    Specialized model for masked face recognition
    Focus on eye/periocular region
    """
    
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # Eye region feature extractor
        self.eye_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.embedding = nn.Linear(512, embedding_dim)
    
    def forward(self, eye_region):
        """
        eye_region: (B, 3, H, W) cropped eye region
        """
        features = self.eye_backbone(eye_region).flatten(1)
        embedding = self.embedding(features)
        return F.normalize(embedding, dim=1)


class OcclusionDetector(nn.Module):
    """Detect type and extent of occlusion"""
    
    def __init__(self):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Occlusion type classifier
        self.type_classifier = nn.Linear(128, 5)  # none, mask, glasses, hand, other
        
        # Occlusion map predictor
        self.map_predictor = nn.ConvTranspose2d(128, 1, 4, stride=4)
    
    def forward(self, x):
        features = self.backbone(x)
        
        occlusion_type = self.type_classifier(features.flatten(1))
        occlusion_map = torch.sigmoid(self.map_predictor(features))
        
        return occlusion_type, occlusion_map


class HybridMaskedRecognition(nn.Module):
    """
    Combine full-face and periocular models
    Use periocular when mask detected
    """
    
    def __init__(self):
        super().__init__()
        
        self.occlusion_detector = OcclusionDetector()
        self.full_face_model = FaceEmbeddingNetwork()
        self.periocular_model = PeriocularRecognition()
    
    def forward(self, face_image, eye_region):
        # Detect occlusion
        occ_type, occ_map = self.occlusion_detector(face_image)
        is_masked = occ_type.argmax(dim=1) == 1  # Assuming 1 = mask
        
        # Use appropriate model
        full_emb = self.full_face_model(face_image)
        peri_emb = self.periocular_model(eye_region)
        
        # Weighted combination based on occlusion
        mask_weight = is_masked.float().unsqueeze(1)
        embedding = (1 - mask_weight) * full_emb + mask_weight * peri_emb
        
        return F.normalize(embedding, dim=1)


class FaceEmbeddingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(112*112*3, 512)
    def forward(self, x):
        return F.normalize(self.net(x.flatten(1)), dim=1)


def masked_face_strategies():
    """Summary of masked face recognition strategies"""
    
    print("Masked Face Recognition Strategies:")
    print("=" * 60)
    
    strategies = [
        ("Data augmentation", "Train with synthetic masks", "Simple, effective"),
        ("Periocular focus", "Specialized eye region model", "Best for masks"),
        ("Attention mechanism", "Weight visible regions", "Generalizes well"),
        ("Part-based", "Independent part features", "Handles any occlusion"),
        ("Multi-model fusion", "Full + periocular ensemble", "Highest accuracy"),
    ]
    
    print(f"{'Strategy':<20} {'Method':<30} {'Notes'}")
    print("-" * 60)
    for name, method, notes in strategies:
        print(f"{name:<20} {method:<30} {notes}")
```

**Performance Impact:**

| Condition | Standard Model | Occlusion-Aware |
|-----------|---------------|-----------------|
| No occlusion | 99.5% | 99.3% |
| Mask | 70-80% | 90-95% |
| Sunglasses | 85-90% | 95%+ |

**Interview Tip:** Post-COVID, masked face recognition became a major research area. The key insight is that the periocular region (eyes + surrounding area) carries 50-60% of discriminative information. Solutions include: training with augmented masked data, attention mechanisms that ignore occluded areas, and specialized periocular models. Commercial systems now often run both full-face and periocular models and fuse results.

---

### Question 52
- [ ] Done
**Explain liveness detection and anti-spoofing techniques for face recognition systems.**

**Answer:**

Liveness detection distinguishes real faces from spoofing attacks (photos, videos, masks). Attacks include: (1) print attacks (photo), (2) replay attacks (video on screen), (3) 3D mask attacks. Defenses use: texture analysis (detect printing artifacts), depth sensing (2D vs 3D), motion analysis (blinking, head movement), and multi-modal sensing (IR, depth cameras).

**Attack Types and Defenses:**

| Attack Type | Example | Defense |
|-------------|---------|---------|
| Print | Printed photo | Texture, depth |
| Replay | Video on screen | Moiré pattern, reflection |
| 3D Mask | Silicone mask | IR imaging, texture |
| Deepfake | AI-generated video | Temporal inconsistency |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LivenessDetector(nn.Module):
    """
    Multi-cue liveness detection combining:
    - Texture analysis (LBP-like features)
    - Depth estimation
    - Temporal cues (blinking, motion)
    """
    
    def __init__(self):
        super().__init__()
        
        # Texture branch - detects printing artifacts
        self.texture_branch = TextureBranch()
        
        # Depth branch - distinguishes 2D from 3D
        self.depth_branch = DepthBranch()
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # [spoof, live]
        )
    
    def forward(self, rgb_image, depth_map=None):
        """
        rgb_image: (B, 3, 224, 224)
        depth_map: (B, 1, 224, 224) optional
        Returns: liveness probability
        """
        texture_feat = self.texture_branch(rgb_image)
        depth_feat = self.depth_branch(rgb_image, depth_map)
        
        combined = torch.cat([texture_feat, depth_feat], dim=1)
        logits = self.classifier(combined)
        
        return logits


class TextureBranch(nn.Module):
    """
    Detect texture differences between real faces and spoofs
    Spoofs often show printing artifacts, moiré patterns
    """
    
    def __init__(self):
        super().__init__()
        
        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 256)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.pool(x).flatten(1)
        return self.fc(x)


class DepthBranch(nn.Module):
    """
    Estimate pseudo-depth to distinguish 2D from 3D
    Real faces have depth variation, photos are flat
    """
    
    def __init__(self):
        super().__init__()
        
        # Depth estimation network
        self.depth_estimator = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 1, 1),  # Depth map
        )
        
        # Depth feature encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.fc = nn.Linear(64, 256)
    
    def forward(self, rgb, depth=None):
        if depth is None:
            depth = self.depth_estimator(rgb)
        
        depth_feat = self.depth_encoder(depth).flatten(1)
        return self.fc(depth_feat)


class TemporalLivenessDetector(nn.Module):
    """
    Detect liveness from video using temporal cues
    - Eye blinking
    - Head movement
    - Lip motion during speech
    """
    
    def __init__(self, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        
        # Per-frame feature extractor
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Temporal modeling
        self.temporal = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        
        # Classifier
        self.classifier = nn.Linear(256, 2)
    
    def forward(self, video_frames):
        """
        video_frames: (B, T, 3, H, W)
        Returns: liveness probability
        """
        B, T, C, H, W = video_frames.shape
        
        # Extract per-frame features
        frames = video_frames.view(B * T, C, H, W)
        frame_features = self.frame_encoder(frames).view(B, T, -1)
        
        # Temporal modeling
        lstm_out, _ = self.temporal(frame_features)
        
        # Use final state for classification
        final_state = lstm_out[:, -1, :]
        
        return self.classifier(final_state)


class BlinkDetector(nn.Module):
    """
    Detect eye blinking as liveness cue
    Eye Aspect Ratio (EAR) changes during blink
    """
    
    def __init__(self):
        super().__init__()
        
        # Eye region feature extractor
        self.eye_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()  # Eye openness score
        )
        
        self.blink_threshold = 0.2  # EAR threshold for blink
    
    def forward(self, eye_sequence):
        """
        eye_sequence: (B, T, 3, H, W) sequence of eye crops
        Returns: blink detection result
        """
        B, T = eye_sequence.shape[:2]
        
        openness_scores = []
        for t in range(T):
            eye_frame = eye_sequence[:, t]
            score = self.eye_encoder(eye_frame).mean(dim=(1, 2, 3))
            openness_scores.append(score)
        
        openness = torch.stack(openness_scores, dim=1)  # (B, T)
        
        # Detect blink pattern (dip in openness)
        blink_detected = (openness.min(dim=1)[0] < self.blink_threshold)
        
        return blink_detected


class ChallengeResponseDetector(nn.Module):
    """
    Active liveness with challenge-response
    Ask user to perform actions (turn head, blink, smile)
    """
    
    def __init__(self):
        super().__init__()
        
        self.pose_estimator = PoseEstimator()
        self.expression_classifier = ExpressionClassifier()
        
        # Challenge types
        self.challenges = [
            'turn_left', 'turn_right', 'look_up', 'look_down',
            'blink', 'smile', 'open_mouth'
        ]
    
    def verify_challenge(self, video_sequence, challenge_type):
        """
        Verify if user performed requested challenge
        """
        if challenge_type in ['turn_left', 'turn_right', 'look_up', 'look_down']:
            return self._verify_pose_challenge(video_sequence, challenge_type)
        else:
            return self._verify_expression_challenge(video_sequence, challenge_type)
    
    def _verify_pose_challenge(self, sequence, target_pose):
        """Check head movement"""
        poses = [self.pose_estimator(frame) for frame in sequence]
        
        # Check if pose changed in expected direction
        if target_pose == 'turn_left':
            return poses[-1]['yaw'] < poses[0]['yaw'] - 15
        elif target_pose == 'turn_right':
            return poses[-1]['yaw'] > poses[0]['yaw'] + 15
        # Similar for other poses...
        return False
    
    def _verify_expression_challenge(self, sequence, target_expr):
        """Check expression change"""
        expressions = [self.expression_classifier(frame) for frame in sequence]
        
        # Check if target expression detected in sequence
        return any(expr == target_expr for expr in expressions)


class PoseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(224*224*3, 3)
    def forward(self, x):
        angles = self.net(x.flatten())
        return {'yaw': angles[0], 'pitch': angles[1], 'roll': angles[2]}


class ExpressionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(224*224*3, 7)
    def forward(self, x):
        return self.net(x.flatten()).argmax()


class MultiModalLiveness(nn.Module):
    """
    Combine RGB, Depth, and IR for robust liveness
    iPhone FaceID uses similar approach
    """
    
    def __init__(self):
        super().__init__()
        
        # Modality-specific encoders
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.ir_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, rgb, depth, ir):
        """
        rgb: (B, 3, H, W) visible light image
        depth: (B, 1, H, W) depth map from structured light/ToF
        ir: (B, 1, H, W) near-infrared image
        """
        rgb_feat = self.rgb_encoder(rgb).flatten(1)
        depth_feat = self.depth_encoder(depth).flatten(1)
        ir_feat = self.ir_encoder(ir).flatten(1)
        
        combined = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
        return self.classifier(combined)


def anti_spoofing_strategies():
    """Summary of anti-spoofing methods"""
    
    print("Anti-Spoofing Strategies:")
    print("=" * 70)
    
    methods = [
        ("Texture analysis", "LBP, color histogram", "Detects printing artifacts"),
        ("Depth estimation", "Mono/stereo depth", "2D vs 3D distinction"),
        ("Motion analysis", "Optical flow, blinking", "Static spoof detection"),
        ("Multi-spectral", "RGB + IR + Depth", "Most robust, needs hardware"),
        ("Challenge-response", "User actions", "Active verification"),
    ]
    
    print(f"{'Method':<20} {'Technique':<25} {'Effectiveness'}")
    print("-" * 70)
    for method, technique, effectiveness in methods:
        print(f"{method:<20} {technique:<25} {effectiveness}")
```

**Evaluation Metrics:**

| Metric | Description |
|--------|-------------|
| APCER | Attack Presentation Classification Error Rate (spoof misclassified as live) |
| BPCER | Bona fide Presentation Classification Error Rate (live misclassified as spoof) |
| ACER | Average of APCER and BPCER |

**Interview Tip:** Liveness detection is critical for security. Modern smartphones (FaceID) use structured light depth + IR cameras. For software-only solutions, temporal cues (blinking, head movement) are most reliable. Key challenge: generalization to unseen attack types—models often overfit to training attacks. Cross-dataset evaluation is essential.

---

### Question 53
- [ ] Done
**How do you design face recognition systems robust to aging and appearance changes?**

**Answer:**

Age-invariant face recognition handles: (1) facial structure changes (bone, fat distribution), (2) texture changes (wrinkles, skin), and (3) cosmetic changes (hair, makeup). Solutions include: age-invariant feature learning (disentangle identity from age), age progression/regression synthesis for training, and attention to stable features (bone structure, eye region).

**Aging Effects on Face:**

| Age Change | Effect | Stability |
|------------|--------|-----------|
| Bone structure | Subtle changes | Most stable |
| Soft tissue | Fat redistribution | Moderate |
| Skin texture | Wrinkles, spots | Changes significantly |
| Hair | Color, density changes | Least stable |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeInvariantFaceNet(nn.Module):
    """
    Learn features invariant to age while preserving identity
    Uses disentanglement and adversarial training
    """
    
    def __init__(self, backbone, identity_dim=512, age_dim=64):
        super().__init__()
        
        self.backbone = backbone
        
        # Disentangle identity and age
        self.identity_encoder = nn.Linear(2048, identity_dim)
        self.age_encoder = nn.Linear(2048, age_dim)
        
        # Adversarial age predictor (to be fooled)
        self.age_predictor = AgePredictor(identity_dim)
        
        # Gradient reversal for adversarial training
        self.grl = GradientReversalLayer()
    
    def forward(self, x, return_age=False):
        """
        x: (B, 3, 112, 112) face image
        Returns age-invariant identity embedding
        """
        features = self.backbone(x)  # (B, 2048)
        
        # Extract identity and age features
        identity_feat = self.identity_encoder(features)
        identity_feat = F.normalize(identity_feat, dim=1)
        
        if return_age:
            age_feat = self.age_encoder(features)
            return identity_feat, age_feat
        
        return identity_feat
    
    def adversarial_forward(self, x):
        """For adversarial training to remove age from identity"""
        features = self.backbone(x)
        identity_feat = self.identity_encoder(features)
        identity_feat = F.normalize(identity_feat, dim=1)
        
        # Gradient reversal before age prediction
        reversed_feat = self.grl(identity_feat)
        age_pred = self.age_predictor(reversed_feat)
        
        return identity_feat, age_pred


class GradientReversalLayer(torch.autograd.Function):
    """Reverses gradient during backprop for adversarial training"""
    
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalModule(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)


class AgePredictor(nn.Module):
    """Predict age from features (adversarially trained)"""
    
    def __init__(self, feature_dim):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 101)  # Age 0-100
        )
    
    def forward(self, x):
        return self.predictor(x)


class OrthogonalDecomposition(nn.Module):
    """
    Decompose features into identity and age-specific components
    Enforce orthogonality between them
    """
    
    def __init__(self, feature_dim=2048, identity_dim=512, age_dim=128):
        super().__init__()
        
        # Identity projection
        self.W_identity = nn.Parameter(torch.randn(feature_dim, identity_dim))
        nn.init.orthogonal_(self.W_identity)
        
        # Age projection
        self.W_age = nn.Parameter(torch.randn(feature_dim, age_dim))
        nn.init.orthogonal_(self.W_age)
        
    def forward(self, features):
        # Project to identity and age subspaces
        identity = features @ self.W_identity
        age = features @ self.W_age
        
        return F.normalize(identity, dim=1), age
    
    def orthogonality_loss(self):
        """Encourage orthogonality between subspaces"""
        # W_identity^T @ W_age should be zero
        overlap = self.W_identity.T @ self.W_age
        return torch.norm(overlap, p='fro')


class AgeProgressionAugmentation:
    """
    Synthesize aged/young versions for data augmentation
    Increases robustness to age variations
    """
    
    def __init__(self, age_model):
        self.age_model = age_model  # Pre-trained age progression model
    
    def augment_with_age(self, face_image, current_age, target_ages):
        """
        Generate face at different ages
        
        Args:
            face_image: Input face
            current_age: Estimated current age
            target_ages: List of target ages to synthesize
        
        Returns:
            List of age-transformed faces
        """
        augmented = []
        
        for target_age in target_ages:
            if target_age == current_age:
                augmented.append(face_image)
            else:
                age_diff = target_age - current_age
                aged_face = self.age_model.transform(face_image, age_diff)
                augmented.append(aged_face)
        
        return augmented


class AttentionToStableFeatures(nn.Module):
    """
    Attention mechanism focusing on age-stable face regions
    Eye region and bone structure are most stable with age
    """
    
    def __init__(self):
        super().__init__()
        
        # Define stable regions (learned or predefined)
        self.region_importance = nn.Parameter(torch.ones(7, 7))  # Feature map size
        
        # Initialize with prior: upper face more stable
        with torch.no_grad():
            for i in range(7):
                # Higher weight to upper regions (eyes, forehead)
                self.region_importance[i, :] = 1.0 + 0.5 * (3 - i) / 3
    
    def forward(self, feature_map):
        """
        feature_map: (B, C, H, W)
        Returns: weighted feature map
        """
        attention = torch.sigmoid(self.region_importance)
        return feature_map * attention.unsqueeze(0).unsqueeze(0)


class CrossAgeContrastiveLoss(nn.Module):
    """
    Contrastive loss that pulls same-identity different-age pairs together
    Requires paired data of same person at different ages
    """
    
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels, ages):
        """
        embeddings: (B, D) feature embeddings
        labels: (B,) identity labels
        ages: (B,) age values
        """
        B = embeddings.shape[0]
        
        loss = 0.0
        num_pairs = 0
        
        for i in range(B):
            for j in range(i + 1, B):
                same_identity = (labels[i] == labels[j])
                different_age = (ages[i] != ages[j])
                
                dist = F.pairwise_distance(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                )
                
                if same_identity and different_age:
                    # Pull together same person at different ages
                    loss += dist ** 2
                    num_pairs += 1
                elif not same_identity:
                    # Push apart different people
                    loss += F.relu(self.margin - dist) ** 2
                    num_pairs += 1
        
        return loss / max(num_pairs, 1)


class TemporalConsistencyNetwork(nn.Module):
    """
    Use temporal information when available
    e.g., multiple photos from same person over years
    """
    
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # Per-image embedding
        self.image_encoder = nn.Linear(2048, embedding_dim)
        
        # Temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(embedding_dim, 8)
        
        # Final identity embedding
        self.identity_layer = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, image_features, timestamps=None):
        """
        image_features: (B, T, 2048) features from multiple images
        timestamps: (B, T) relative timestamps/ages
        Returns: aggregated identity embedding
        """
        B, T, _ = image_features.shape
        
        # Encode each image
        embeddings = self.image_encoder(image_features)  # (B, T, D)
        
        # Add temporal encoding if available
        if timestamps is not None:
            temporal_enc = self._get_temporal_encoding(timestamps)
            embeddings = embeddings + temporal_enc
        
        # Self-attention over temporal sequence
        embeddings = embeddings.permute(1, 0, 2)  # (T, B, D)
        attended, _ = self.temporal_attention(embeddings, embeddings, embeddings)
        
        # Aggregate (mean pooling over time)
        aggregated = attended.mean(dim=0)  # (B, D)
        
        identity = self.identity_layer(aggregated)
        return F.normalize(identity, dim=1)
    
    def _get_temporal_encoding(self, timestamps):
        """Sinusoidal temporal encoding"""
        # Simplified encoding
        return torch.sin(timestamps.unsqueeze(-1) * 0.01)


def age_invariant_training_strategy():
    """Complete training strategy for age-robust recognition"""
    
    print("Age-Invariant Training Strategy:")
    print("=" * 60)
    
    components = [
        ("Identity loss", "ArcFace/CosFace on identity", "Core recognition"),
        ("Age adversarial", "Fool age predictor", "Remove age from features"),
        ("Cross-age pairs", "Same person, different age", "Explicit age invariance"),
        ("Age augmentation", "Synthetic aging", "More training data"),
        ("Stable region attention", "Focus on eyes, bones", "Implicit invariance"),
    ]
    
    print(f"{'Component':<20} {'Method':<30} {'Purpose'}")
    print("-" * 60)
    for comp, method, purpose in components:
        print(f"{comp:<20} {method:<30} {purpose}")
```

**Evaluation Datasets:**

| Dataset | Description |
|---------|-------------|
| MORPH | 55k images, multiple ages per person |
| CACD | 163k images, celebrities 16-62 years |
| FG-NET | 1002 images, 82 subjects, 0-69 years |

**Interview Tip:** Age-invariant recognition is challenging because it requires learning what changes with age (wrinkles, fat distribution) vs. what stays constant (bone structure, eye shape). Two main approaches: (1) adversarial training to remove age information from identity features, (2) explicit disentanglement with orthogonal subspaces. Data augmentation with synthetic aging (using GANs like IPCGAN) significantly helps.

---

### Question 54
- [ ] Done
**What are the privacy considerations when deploying facial recognition in public spaces?**

**Answer:**

Privacy concerns include: (1) consent—people don't consent to public surveillance, (2) bias—systems perform worse on minorities, (3) mass surveillance enabling authoritarian control, (4) data security—face databases can be breached, and (5) function creep—systems deployed for one purpose used for another. Solutions include: privacy-preserving recognition, on-device processing, strict data retention policies, and regulatory compliance (GDPR, CCPA).

**Key Privacy Issues:**

| Issue | Concern | Mitigation |
|-------|---------|------------|
| Consent | No opt-out in public | Clear signage, opt-out zones |
| Bias | Unequal accuracy | Diverse training, audits |
| Surveillance | Tracking without warrant | Legal restrictions |
| Data breach | Face templates stolen | Encryption, on-device |
| Function creep | Mission expansion | Strict usage policies |

**Technical Solutions:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib

class PrivacyPreservingFaceRecognition:
    """
    Techniques to reduce privacy risks in face recognition
    """
    
    def __init__(self):
        self.encryption_key = self._generate_key()
    
    def _generate_key(self):
        return np.random.bytes(32)
    
    # ========== On-Device Processing ==========
    def on_device_embedding(self, face_image, model):
        """
        Compute embeddings on device, never send raw images
        Only encrypted embeddings leave the device
        """
        with torch.no_grad():
            embedding = model(face_image)
        
        # Encrypt before transmission
        encrypted = self._encrypt_embedding(embedding)
        
        return encrypted
    
    def _encrypt_embedding(self, embedding):
        """Simple encryption (use proper encryption in production)"""
        embedding_bytes = embedding.numpy().tobytes()
        # In practice, use AES or similar
        return hashlib.sha256(embedding_bytes + self.encryption_key).hexdigest()
    
    # ========== Template Protection ==========
    def cancelable_biometrics(self, embedding, transform_key):
        """
        Transform embedding so it can be revoked if compromised
        Different keys produce different templates from same face
        """
        # Random projection matrix from key
        np.random.seed(int(hashlib.sha256(transform_key.encode()).hexdigest()[:8], 16))
        projection = np.random.randn(512, 512)
        
        # Transform embedding
        transformed = embedding @ projection
        
        # Non-linear transformation
        transformed = np.tanh(transformed)
        
        return transformed / np.linalg.norm(transformed)
    
    def revoke_template(self, user_id):
        """Revoke compromised template by changing transform key"""
        new_key = f"{user_id}_{np.random.randint(1000000)}"
        return new_key  # User re-enrolls with new key
    
    # ========== Differential Privacy ==========
    def add_differential_privacy(self, embedding, epsilon=1.0):
        """
        Add noise for differential privacy
        Prevents exact template matching while preserving utility
        """
        # Sensitivity of embedding (L2 norm bound)
        sensitivity = 1.0  # Assuming normalized embeddings
        
        # Laplacian noise scale
        scale = sensitivity / epsilon
        
        noise = np.random.laplace(0, scale, embedding.shape)
        noisy_embedding = embedding + noise
        
        return noisy_embedding / np.linalg.norm(noisy_embedding)
    
    # ========== Federated Learning ==========
    def federated_enrollment(self, local_images, local_model):
        """
        Train/update model without centralizing data
        Each device trains locally, only gradients shared
        """
        # Compute local embedding
        local_embedding = local_model(local_images).mean(dim=0)
        
        # Compute gradient update locally
        # Only send encrypted gradients to server
        # Never send raw images or raw embeddings
        
        return "gradients_encrypted"


class BiasAuditFramework:
    """
    Audit face recognition system for demographic bias
    Required for ethical deployment
    """
    
    def __init__(self, model):
        self.model = model
    
    def compute_accuracy_by_group(self, test_data, groups):
        """
        Compute accuracy for each demographic group
        
        Args:
            test_data: [(image, label, group), ...]
            groups: List of demographic groups
        
        Returns:
            Accuracy per group
        """
        group_results = {g: {'correct': 0, 'total': 0} for g in groups}
        
        for image, label, group in test_data:
            prediction = self.model.predict(image)
            is_correct = (prediction == label)
            
            group_results[group]['total'] += 1
            if is_correct:
                group_results[group]['correct'] += 1
        
        accuracies = {
            g: r['correct'] / max(r['total'], 1) 
            for g, r in group_results.items()
        }
        
        return accuracies
    
    def compute_fairness_metrics(self, accuracies):
        """
        Compute fairness metrics across groups
        """
        values = list(accuracies.values())
        
        metrics = {
            'max_accuracy': max(values),
            'min_accuracy': min(values),
            'accuracy_gap': max(values) - min(values),
            'fairness_ratio': min(values) / max(values) if max(values) > 0 else 0,
        }
        
        return metrics
    
    def generate_bias_report(self, test_data, groups):
        """Generate comprehensive bias audit report"""
        accuracies = self.compute_accuracy_by_group(test_data, groups)
        fairness = self.compute_fairness_metrics(accuracies)
        
        report = {
            'group_accuracies': accuracies,
            'fairness_metrics': fairness,
            'recommendation': self._get_recommendation(fairness)
        }
        
        return report
    
    def _get_recommendation(self, fairness):
        if fairness['accuracy_gap'] < 0.02:
            return "PASS: Minimal demographic disparity"
        elif fairness['accuracy_gap'] < 0.05:
            return "WARNING: Some disparity detected, consider retraining"
        else:
            return "FAIL: Significant disparity, deployment not recommended"


class DataRetentionPolicy:
    """
    Implement data retention and deletion policies
    Required for GDPR compliance
    """
    
    def __init__(self, retention_days=30):
        self.retention_days = retention_days
        self.data_store = {}  # In practice, use secure database
    
    def store_with_expiry(self, face_id, embedding, purpose):
        """Store embedding with automatic expiry"""
        import time
        
        self.data_store[face_id] = {
            'embedding': embedding,
            'purpose': purpose,
            'timestamp': time.time(),
            'expiry': time.time() + (self.retention_days * 86400)
        }
    
    def delete_expired(self):
        """Delete all expired data"""
        import time
        current_time = time.time()
        
        expired = [
            fid for fid, data in self.data_store.items()
            if data['expiry'] < current_time
        ]
        
        for fid in expired:
            del self.data_store[fid]
        
        return len(expired)
    
    def user_deletion_request(self, user_id):
        """GDPR right to deletion"""
        if user_id in self.data_store:
            del self.data_store[user_id]
            return True
        return False
    
    def user_data_export(self, user_id):
        """GDPR right to data portability"""
        if user_id in self.data_store:
            return {
                'embedding_hash': hashlib.sha256(
                    str(self.data_store[user_id]['embedding']).encode()
                ).hexdigest(),
                'purpose': self.data_store[user_id]['purpose'],
                'stored_date': self.data_store[user_id]['timestamp'],
            }
        return None


class ConsentManagement:
    """
    Manage user consent for face recognition
    """
    
    def __init__(self):
        self.consent_records = {}
    
    def request_consent(self, user_id, purposes):
        """
        Request explicit consent with clear purposes
        
        Args:
            user_id: User identifier
            purposes: List of intended uses
        
        Returns:
            Consent request ID
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        self.consent_records[request_id] = {
            'user_id': user_id,
            'purposes': purposes,
            'status': 'pending',
            'timestamp': None
        }
        
        return request_id
    
    def record_consent(self, request_id, granted, purposes_accepted):
        """Record user's consent decision"""
        import time
        
        if request_id in self.consent_records:
            self.consent_records[request_id].update({
                'status': 'granted' if granted else 'denied',
                'purposes_accepted': purposes_accepted if granted else [],
                'timestamp': time.time()
            })
    
    def check_consent(self, user_id, purpose):
        """Check if user consented to specific purpose"""
        for record in self.consent_records.values():
            if (record['user_id'] == user_id and 
                record['status'] == 'granted' and
                purpose in record.get('purposes_accepted', [])):
                return True
        return False


def privacy_deployment_checklist():
    """Checklist for ethical deployment"""
    
    print("Privacy-Conscious Deployment Checklist:")
    print("=" * 70)
    
    items = [
        ("Legal review", "Ensure compliance with GDPR, CCPA, local laws"),
        ("Bias audit", "Test accuracy across demographics"),
        ("Consent mechanism", "Clear opt-in/opt-out process"),
        ("Data minimization", "Collect only necessary data"),
        ("Retention policy", "Automatic deletion after purpose fulfilled"),
        ("Security audit", "Encryption, access controls, breach response"),
        ("Transparency", "Public disclosure of system capabilities"),
        ("Human oversight", "Human review for high-stakes decisions"),
    ]
    
    for item, description in items:
        print(f"[ ] {item}: {description}")
```

**Regulatory Landscape:**

| Region | Regulation | Key Requirements |
|--------|------------|------------------|
| EU | GDPR | Consent, data minimization, right to deletion |
| California | CCPA | Consumer rights, disclosure |
| Illinois | BIPA | Explicit written consent required |
| China | PIPL | Consent, data localization |

**Interview Tip:** Privacy in face recognition is both a technical and ethical issue. Technical solutions (on-device processing, cancelable biometrics, differential privacy) address some concerns but not mass surveillance issues. Key points: (1) accuracy disparities across demographics are a legal liability, (2) GDPR requires consent and right to deletion, (3) some jurisdictions (Illinois BIPA) require explicit written consent. Companies have faced lawsuits over unauthorized face recognition.

---

## Video Tracking

### Question 55
- [ ] Done
**Compare detection-based (tracking-by-detection) vs. correlation-based tracking approaches.**

**Answer:**

Detection-based (tracking-by-detection) runs a detector every frame and associates detections across time—robust to occlusion but slower. Correlation-based tracking initializes a template and finds it in subsequent frames via correlation/regression—faster but struggles with appearance changes and occlusions. Modern systems often combine both: detection provides reliability, correlation provides speed between detections.

**Comparison:**

| Aspect | Detection-Based | Correlation-Based |
|--------|-----------------|-------------------|
| Speed | Slower (detector per frame) | Faster (template matching) |
| Appearance change | Handles well | Struggles (model drift) |
| Occlusion recovery | Good (re-detect) | Poor (loses track) |
| Initialization | Automatic (detector) | Manual (bounding box) |
| Examples | SORT, DeepSORT | KCF, MOSSE, SiamFC |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

# ========== DETECTION-BASED TRACKING ==========

class TrackingByDetection:
    """
    Tracking-by-detection paradigm
    1. Run detector on each frame
    2. Associate detections with existing tracks
    """
    
    def __init__(self, detector, iou_threshold=0.3):
        self.detector = detector
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_track_id = 0
    
    def update(self, frame):
        """
        Process new frame
        Returns: List of (track_id, bbox)
        """
        # Step 1: Detect objects in current frame
        detections = self.detector(frame)  # List of [x1, y1, x2, y2, conf]
        
        if len(self.tracks) == 0:
            # Initialize tracks from detections
            for det in detections:
                self._create_track(det)
            return [(t['id'], t['bbox']) for t in self.tracks]
        
        # Step 2: Compute cost matrix (1 - IoU)
        cost_matrix = self._compute_iou_matrix(detections)
        
        # Step 3: Hungarian algorithm for assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Step 4: Update matched tracks
        matched_tracks = set()
        matched_dets = set()
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] < (1 - self.iou_threshold):
                self.tracks[t_idx]['bbox'] = detections[d_idx][:4]
                self.tracks[t_idx]['age'] = 0
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
        
        # Step 5: Handle unmatched tracks and detections
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['age'] += 1
        
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det)
        
        # Step 6: Remove old tracks
        self.tracks = [t for t in self.tracks if t['age'] < 10]
        
        return [(t['id'], t['bbox']) for t in self.tracks]
    
    def _create_track(self, detection):
        """Create new track from detection"""
        self.tracks.append({
            'id': self.next_track_id,
            'bbox': detection[:4],
            'age': 0
        })
        self.next_track_id += 1
    
    def _compute_iou_matrix(self, detections):
        """Compute IoU cost matrix between tracks and detections"""
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        
        cost_matrix = np.ones((n_tracks, n_dets))
        
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track['bbox'], det[:4])
                cost_matrix[i, j] = 1 - iou  # Cost = 1 - IoU
        
        return cost_matrix
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)


# ========== CORRELATION-BASED TRACKING ==========

class KCFTracker:
    """
    Kernelized Correlation Filter (KCF) tracker
    - Fast template matching using FFT
    - Learns discriminative correlation filter
    """
    
    def __init__(self, lambda_=0.01, sigma=0.5):
        self.lambda_ = lambda_  # Regularization
        self.sigma = sigma  # Gaussian kernel bandwidth
        self.alpha = None  # Learned filter
        self.template = None
        self.pos = None
        self.size = None
    
    def init(self, frame, bbox):
        """
        Initialize tracker with first frame and bounding box
        bbox: (x, y, w, h)
        """
        x, y, w, h = bbox
        self.pos = (x + w/2, y + h/2)  # Center position
        self.size = (w, h)
        
        # Extract template patch
        self.template = self._get_patch(frame, self.pos, self.size)
        
        # Create Gaussian label (peak at center)
        self.y = self._create_gaussian_label(self.template.shape[:2])
        
        # Learn initial correlation filter
        self._train(self.template)
    
    def update(self, frame):
        """
        Track object in new frame
        Returns: (x, y, w, h) bounding box
        """
        # Extract search patch at predicted position
        patch = self._get_patch(frame, self.pos, self.size)
        
        # Compute correlation response
        response = self._detect(patch)
        
        # Find peak in response
        peak_y, peak_x = np.unravel_index(response.argmax(), response.shape)
        
        # Update position
        cy, cx = response.shape[0] // 2, response.shape[1] // 2
        dy, dx = peak_y - cy, peak_x - cx
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        
        # Update model (online learning)
        new_patch = self._get_patch(frame, self.pos, self.size)
        self._train(new_patch, update=True)
        
        # Return bounding box
        x = self.pos[0] - self.size[0] / 2
        y = self.pos[1] - self.size[1] / 2
        return (x, y, self.size[0], self.size[1])
    
    def _get_patch(self, frame, center, size):
        """Extract image patch centered at position"""
        x1 = int(center[0] - size[0] / 2)
        y1 = int(center[1] - size[1] / 2)
        x2 = int(x1 + size[0])
        y2 = int(y1 + size[1])
        
        # Handle boundary conditions
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        
        return frame[y1:y2, x1:x2].astype(np.float32)
    
    def _create_gaussian_label(self, size):
        """Create 2D Gaussian centered at origin"""
        h, w = size
        y, x = np.mgrid[:h, :w] - np.array([[h//2], [w//2]])
        sigma = 0.1 * min(h, w)
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return gaussian
    
    def _gaussian_kernel(self, x1, x2):
        """Compute Gaussian kernel between patches"""
        # Simplified kernel computation
        c = np.exp(-self.sigma * np.sum((x1 - x2)**2) / x1.size)
        return c
    
    def _train(self, patch, update=False):
        """Train or update correlation filter"""
        # Simplified training (actual KCF uses FFT)
        k = self._gaussian_kernel(patch, patch)
        
        alpha_new = self.y / (k + self.lambda_)
        
        if update and self.alpha is not None:
            # Incremental update
            self.alpha = 0.9 * self.alpha + 0.1 * alpha_new
            self.template = 0.9 * self.template + 0.1 * patch
        else:
            self.alpha = alpha_new
            self.template = patch
    
    def _detect(self, patch):
        """Compute detection response"""
        k = self._gaussian_kernel(patch, self.template)
        response = self.alpha * k
        return response.reshape(patch.shape[:2])


class SiameseTracker(nn.Module):
    """
    Siamese network-based tracker (SiamFC style)
    - Learn similarity between template and search region
    - No online update needed (fully offline trained)
    """
    
    def __init__(self):
        super().__init__()
        
        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3), nn.ReLU(),
            nn.Conv2d(384, 384, 3), nn.ReLU(),
            nn.Conv2d(384, 256, 3),
        )
        
        self.template_feat = None
    
    def init(self, template):
        """Initialize with template image"""
        self.template_feat = self.backbone(template)
    
    def forward(self, search_region):
        """
        Find template in search region
        Returns response map with peak at target location
        """
        search_feat = self.backbone(search_region)
        
        # Cross-correlation
        response = F.conv2d(search_feat, self.template_feat)
        
        return response


class HybridTracker:
    """
    Combine detection and correlation tracking
    - Use correlation for fast tracking between frames
    - Use detection to correct and recover from failures
    """
    
    def __init__(self, detector, correlation_tracker):
        self.detector = detector
        self.tracker = correlation_tracker
        self.detection_interval = 10  # Run detector every N frames
        self.frame_count = 0
        self.track_id = None
    
    def update(self, frame):
        """Process frame with hybrid approach"""
        self.frame_count += 1
        
        # Run correlation tracker
        if self.track_id is not None:
            bbox = self.tracker.update(frame)
        else:
            bbox = None
        
        # Periodic detection for correction
        if self.frame_count % self.detection_interval == 0:
            detections = self.detector(frame)
            
            if len(detections) > 0:
                # Find detection closest to current track
                if bbox is not None:
                    best_det = self._find_best_match(bbox, detections)
                else:
                    best_det = detections[0]  # Take highest confidence
                
                # Reinitialize tracker with detection
                self.tracker.init(frame, best_det[:4])
                bbox = best_det[:4]
                self.track_id = 0
        
        return bbox
    
    def _find_best_match(self, bbox, detections):
        """Find detection closest to current track"""
        best_iou = 0
        best_det = detections[0]
        
        for det in detections:
            iou = self._compute_iou(bbox, det[:4])
            if iou > best_iou:
                best_iou = iou
                best_det = det
        
        return best_det
    
    def _compute_iou(self, box1, box2):
        # Same IoU computation as above
        pass


def compare_tracking_paradigms():
    """Summary comparison of tracking approaches"""
    
    print("Tracking Paradigm Comparison:")
    print("=" * 70)
    
    paradigms = [
        ("Detection-based", "SORT, DeepSORT, ByteTrack", "Robust, handles occlusion"),
        ("Correlation-based", "KCF, MOSSE, DCF", "Fast, real-time capable"),
        ("Siamese", "SiamFC, SiamRPN", "End-to-end, no online update"),
        ("Hybrid", "Detection + correlation", "Best of both worlds"),
    ]
    
    print(f"{'Paradigm':<20} {'Examples':<25} {'Strengths'}")
    print("-" * 70)
    for paradigm, examples, strengths in paradigms:
        print(f"{paradigm:<20} {examples:<25} {strengths}")
```

**Interview Tip:** Detection-based tracking dominates benchmarks (MOT Challenge) because it naturally handles occlusions and appearance changes. However, correlation-based trackers are still used when speed is critical (embedded systems) or when the detector is too slow. Modern state-of-the-art (ByteTrack, OC-SORT) are detection-based but incorporate motion models (Kalman filters) to interpolate between detections.

---

### Question 56
- [ ] Done
**Explain multi-object tracking (MOT) and the data association problem (Hungarian algorithm, DeepSORT).**

**Answer:**

Multi-Object Tracking (MOT) tracks multiple objects simultaneously across video frames. The core challenge is data association: matching detections in the current frame to existing tracks. Solutions include: Hungarian algorithm (optimal bipartite matching based on IoU/distance), DeepSORT (adds appearance features for better matching), and graph-based methods (global optimization across multiple frames).

**MOT Pipeline:**
```
Frame → Detection → Feature Extraction → Data Association → Track Management
                                                ↓
                                    Cost Matrix (IoU + Appearance)
                                                ↓
                                    Hungarian Algorithm / Greedy
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# ========== HUNGARIAN ALGORITHM FOR DATA ASSOCIATION ==========

def hungarian_matching(cost_matrix, threshold=0.7):
    """
    Optimal bipartite matching using Hungarian algorithm
    
    Args:
        cost_matrix: (N_tracks, M_detections) - lower is better
        threshold: Maximum cost for valid assignment
    
    Returns:
        matches: List of (track_idx, detection_idx) pairs
        unmatched_tracks: List of track indices without matches
        unmatched_detections: List of detection indices without matches
    """
    # Handle empty cases
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    # Hungarian algorithm (Kuhn-Munkres)
    track_indices, det_indices = linear_sum_assignment(cost_matrix)
    
    # Filter by threshold
    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_dets = list(range(cost_matrix.shape[1]))
    
    for t_idx, d_idx in zip(track_indices, det_indices):
        if cost_matrix[t_idx, d_idx] <= threshold:
            matches.append((t_idx, d_idx))
            unmatched_tracks.remove(t_idx)
            unmatched_dets.remove(d_idx)
    
    return matches, unmatched_tracks, unmatched_dets


# ========== SORT: Simple Online Realtime Tracking ==========

class KalmanFilter:
    """
    Kalman filter for constant velocity motion model
    State: [x, y, a, h, vx, vy, va, vh]
    x, y: center position
    a: aspect ratio
    h: height
    """
    
    def __init__(self):
        # State transition matrix (constant velocity)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1  # Position += velocity * dt
        
        # Measurement matrix (observe position only)
        self.H = np.eye(4, 8)
        
        # Process noise
        self.Q = np.eye(8) * 0.01
        
        # Measurement noise
        self.R = np.eye(4) * 0.1
    
    def init(self, measurement):
        """Initialize state from first measurement"""
        self.x = np.zeros(8)
        self.x[:4] = measurement  # [x, y, a, h]
        self.P = np.eye(8)  # State covariance
        return self.x[:4]
    
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]
    
    def update(self, measurement):
        """Update state with new measurement"""
        y = measurement - self.H @ self.x  # Residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        
        return self.x[:4]


class Track:
    """Single object track"""
    
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.kalman = KalmanFilter()
        
        # Convert bbox to [x, y, aspect_ratio, height]
        measurement = self._bbox_to_xyah(detection)
        self.kalman.init(measurement)
        
        self.hits = 1  # Consecutive detections
        self.age = 0  # Frames since creation
        self.time_since_update = 0  # Frames since last detection
        
        # For DeepSORT: appearance feature
        self.features = []
    
    def predict(self):
        """Predict next position"""
        self.age += 1
        self.time_since_update += 1
        return self.kalman.predict()
    
    def update(self, detection, feature=None):
        """Update with new detection"""
        measurement = self._bbox_to_xyah(detection)
        self.kalman.update(measurement)
        
        self.hits += 1
        self.time_since_update = 0
        
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 100:  # Keep last 100 features
                self.features = self.features[-100:]
    
    def get_bbox(self):
        """Get current bounding box"""
        return self._xyah_to_bbox(self.kalman.x[:4])
    
    def _bbox_to_xyah(self, bbox):
        """Convert [x1, y1, x2, y2] to [cx, cy, aspect, height]"""
        x1, y1, x2, y2 = bbox[:4]
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w/2, y1 + h/2, w/h, h])
    
    def _xyah_to_bbox(self, xyah):
        """Convert [cx, cy, aspect, height] to [x1, y1, x2, y2]"""
        cx, cy, a, h = xyah
        w = a * h
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


class SORT:
    """
    Simple Online Realtime Tracker
    Uses Kalman filter + Hungarian algorithm
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # Delete track after N frames without detection
        self.min_hits = min_hits  # Minimum hits to output track
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.next_id = 0
    
    def update(self, detections):
        """
        Update tracks with new detections
        
        Args:
            detections: (N, 5) array of [x1, y1, x2, y2, conf]
        
        Returns:
            (M, 5) array of [x1, y1, x2, y2, track_id]
        """
        # Step 1: Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Step 2: Compute IoU cost matrix
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = self._iou_cost_matrix(detections)
            matches, unmatched_tracks, unmatched_dets = hungarian_matching(
                cost_matrix, threshold=1 - self.iou_threshold
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))
        
        # Step 3: Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks.append(Track(detections[det_idx], self.next_id))
            self.next_id += 1
        
        # Step 5: Delete old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Step 6: Return confirmed tracks
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                bbox = track.get_bbox()
                results.append([*bbox, track.track_id])
        
        return np.array(results) if results else np.empty((0, 5))
    
    def _iou_cost_matrix(self, detections):
        """Compute IoU-based cost matrix"""
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        
        cost_matrix = np.zeros((n_tracks, n_dets))
        
        for i, track in enumerate(self.tracks):
            track_bbox = track.get_bbox()
            for j, det in enumerate(detections):
                iou = self._compute_iou(track_bbox, det[:4])
                cost_matrix[i, j] = 1 - iou
        
        return cost_matrix
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)


# ========== DeepSORT: SORT + Appearance Features ==========

class AppearanceExtractor(nn.Module):
    """Extract appearance features for re-identification"""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.fc = nn.Linear(128, embedding_dim)
    
    def forward(self, x):
        """
        x: (B, 3, 128, 64) cropped person image
        Returns: (B, embedding_dim) normalized features
        """
        features = self.backbone(x).flatten(1)
        embeddings = self.fc(features)
        return F.normalize(embeddings, dim=1)


class DeepSORT(SORT):
    """
    Deep SORT: SORT + deep appearance descriptors
    Handles occlusions better by using appearance similarity
    """
    
    def __init__(self, feature_extractor, max_age=30, min_hits=3, 
                 iou_threshold=0.3, appearance_threshold=0.4):
        super().__init__(max_age, min_hits, iou_threshold)
        
        self.feature_extractor = feature_extractor
        self.appearance_threshold = appearance_threshold
        self.lambda_iou = 0.5  # Weight for IoU vs appearance
    
    def update(self, detections, frame):
        """
        Update with detections and full frame (for feature extraction)
        """
        # Extract appearance features for each detection
        features = self._extract_features(detections, frame)
        
        # Step 1: Predict
        for track in self.tracks:
            track.predict()
        
        # Step 2: Compute combined cost matrix
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_cost = self._iou_cost_matrix(detections)
            appearance_cost = self._appearance_cost_matrix(features)
            
            # Combine costs
            cost_matrix = self.lambda_iou * iou_cost + (1 - self.lambda_iou) * appearance_cost
            
            # Two-stage matching: first confirmed tracks, then tentative
            matches, unmatched_tracks, unmatched_dets = self._cascaded_matching(
                cost_matrix, detections
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))
        
        # Step 3: Update matched tracks with features
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], features[det_idx])
        
        # Step 4: Create new tracks
        for det_idx in unmatched_dets:
            new_track = Track(detections[det_idx], self.next_id)
            new_track.features.append(features[det_idx])
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Step 5: Delete old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Step 6: Return results
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                bbox = track.get_bbox()
                results.append([*bbox, track.track_id])
        
        return np.array(results) if results else np.empty((0, 5))
    
    def _extract_features(self, detections, frame):
        """Extract appearance features for detections"""
        features = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            crop = frame[y1:y2, x1:x2]
            
            # Resize and normalize
            crop = torch.tensor(crop).permute(2, 0, 1).float() / 255.0
            crop = F.interpolate(crop.unsqueeze(0), size=(128, 64))
            
            with torch.no_grad():
                feat = self.feature_extractor(crop)
            
            features.append(feat.squeeze().numpy())
        
        return features
    
    def _appearance_cost_matrix(self, detection_features):
        """Compute appearance-based cost matrix"""
        n_tracks = len(self.tracks)
        n_dets = len(detection_features)
        
        cost_matrix = np.ones((n_tracks, n_dets))
        
        for i, track in enumerate(self.tracks):
            if len(track.features) == 0:
                continue
            
            # Compare with track's appearance history
            track_features = np.array(track.features[-30:])  # Last 30 features
            
            for j, det_feat in enumerate(detection_features):
                # Cosine similarity (converted to cost)
                similarities = track_features @ det_feat
                cost_matrix[i, j] = 1 - similarities.max()
        
        return cost_matrix
    
    def _cascaded_matching(self, cost_matrix, detections):
        """
        Cascaded matching: prioritize recently seen tracks
        """
        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.hits >= 3]
        tentative_indices = [i for i, t in enumerate(self.tracks) if t.hits < 3]
        
        # First match confirmed tracks
        if confirmed_indices:
            confirmed_cost = cost_matrix[confirmed_indices]
            matches1, unmatched_t, unmatched_d = hungarian_matching(
                confirmed_cost, threshold=0.7
            )
            matches1 = [(confirmed_indices[t], d) for t, d in matches1]
        else:
            matches1 = []
            unmatched_d = list(range(len(detections)))
        
        # Then match tentative tracks with remaining detections
        if tentative_indices and unmatched_d:
            tentative_cost = cost_matrix[np.ix_(tentative_indices, unmatched_d)]
            matches2, _, still_unmatched = hungarian_matching(
                tentative_cost, threshold=0.7
            )
            matches2 = [(tentative_indices[t], unmatched_d[d]) for t, d in matches2]
            unmatched_d = [unmatched_d[i] for i in still_unmatched]
        else:
            matches2 = []
        
        all_matches = matches1 + matches2
        matched_tracks = {m[0] for m in all_matches}
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        
        return all_matches, unmatched_tracks, unmatched_d
```

**Comparison:**

| Method | Association | Strengths |
|--------|-------------|-----------|
| SORT | IoU only | Simple, fast |
| DeepSORT | IoU + Appearance | Handles occlusion |
| ByteTrack | IoU + low-conf detections | Uses all detections |
| OC-SORT | IoU + velocity | Better motion handling |

**Interview Tip:** Hungarian algorithm provides optimal matching in O(n³) time. DeepSORT's key insight: appearance features help re-identify objects after occlusion, when IoU fails (no overlap). Recent methods like ByteTrack achieve better results by also using low-confidence detections in a second matching stage. The trend is moving toward simpler methods with better motion models (OC-SORT).

---

### Question 57
- [ ] Done
**How do you handle tracking through occlusions and temporary object disappearances?**

**Answer:**

Occlusion handling requires: (1) motion prediction (Kalman filter continues trajectory during occlusion), (2) appearance memory (store features for re-identification after occlusion), (3) track management (keep track "alive" for N frames without detection), and (4) re-identification (match reappearing objects to suspended tracks). Key insight: separate short-term (motion-based) from long-term (appearance-based) matching.

**Occlusion Strategies:**

| Strategy | Mechanism | Use Case |
|----------|-----------|----------|
| Motion prediction | Kalman filter extrapolation | Brief occlusion (1-5 frames) |
| Track buffer | Keep track alive N frames | Medium occlusion |
| ReID matching | Appearance features | Long occlusion / reappearance |
| Scene context | Use other objects' motion | Crowd scenarios |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class OcclusionAwareTrack:
    """
    Track with occlusion handling capabilities
    """
    
    def __init__(self, detection, track_id, feature=None):
        self.track_id = track_id
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.state = self._init_state(detection)
        
        # Track lifecycle
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Appearance memory for re-identification
        self.feature_buffer = deque(maxlen=50)
        if feature is not None:
            self.feature_buffer.append(feature)
        
        # Trajectory history for motion prediction
        self.trajectory = deque(maxlen=30)
        self.trajectory.append(detection[:4])
        
        # Occlusion status
        self.is_occluded = False
        self.occlusion_start = None
    
    def predict(self):
        """
        Predict next position using constant velocity model
        Critical for maintaining track during occlusion
        """
        self.age += 1
        self.time_since_update += 1
        
        # Update position with velocity
        self.state[:4] += self.state[4:]
        
        # Mark as occluded if not updated for several frames
        if self.time_since_update > 3 and not self.is_occluded:
            self.is_occluded = True
            self.occlusion_start = self.age
        
        return self.get_bbox()
    
    def update(self, detection, feature=None):
        """Update track with new detection"""
        old_pos = self.state[:4].copy()
        new_pos = self._detection_to_state(detection)
        
        # Update velocity estimate
        self.state[4:] = 0.9 * self.state[4:] + 0.1 * (new_pos[:4] - old_pos)
        self.state[:4] = new_pos[:4]
        
        self.hits += 1
        self.time_since_update = 0
        
        # Store feature for re-identification
        if feature is not None:
            self.feature_buffer.append(feature)
        
        # Store trajectory
        self.trajectory.append(detection[:4])
        
        # Clear occlusion status
        if self.is_occluded:
            self.is_occluded = False
            self.occlusion_start = None
    
    def get_bbox(self):
        """Get current bounding box"""
        x, y, w, h = self.state[:4]
        return np.array([x, y, x + w, y + h])
    
    def get_average_feature(self):
        """Get mean appearance feature for matching"""
        if len(self.feature_buffer) == 0:
            return None
        features = np.array(list(self.feature_buffer))
        return features.mean(axis=0)
    
    def get_recent_features(self, n=10):
        """Get recent features for matching"""
        features = list(self.feature_buffer)[-n:]
        return np.array(features) if features else None
    
    def _init_state(self, detection):
        x1, y1, x2, y2 = detection[:4]
        return np.array([x1, y1, x2-x1, y2-y1, 0, 0, 0, 0], dtype=np.float32)
    
    def _detection_to_state(self, detection):
        x1, y1, x2, y2 = detection[:4]
        return np.array([x1, y1, x2-x1, y2-y1])


class OcclusionAwareTracker:
    """
    Multi-object tracker with explicit occlusion handling
    """
    
    def __init__(self, feature_extractor, max_age=70, max_occlusion=30):
        self.feature_extractor = feature_extractor
        self.max_age = max_age  # Total max frames to keep track
        self.max_occlusion = max_occlusion  # Max frames of occlusion
        
        self.active_tracks = []
        self.suspended_tracks = []  # Tracks under occlusion
        self.next_id = 0
    
    def update(self, detections, frame):
        """
        Update with new detections
        """
        # Extract features
        features = self._extract_features(detections, frame)
        
        # Step 1: Predict all tracks
        for track in self.active_tracks + self.suspended_tracks:
            track.predict()
        
        # Step 2: Match active tracks (IoU + appearance)
        matched_active, unmatched_active, unmatched_dets = self._match_active(
            detections, features
        )
        
        # Step 3: Try to match suspended tracks with remaining detections
        matched_suspended, still_suspended, still_unmatched = self._match_suspended(
            unmatched_dets, detections, features
        )
        
        # Step 4: Update matched tracks
        for track_idx, det_idx in matched_active:
            self.active_tracks[track_idx].update(detections[det_idx], features[det_idx])
        
        for track_idx, det_idx in matched_suspended:
            track = self.suspended_tracks[track_idx]
            track.update(detections[det_idx], features[det_idx])
            # Reactivate track
            self.active_tracks.append(track)
        
        # Step 5: Suspend unmatched active tracks
        for idx in sorted(unmatched_active, reverse=True):
            track = self.active_tracks[idx]
            if track.time_since_update > 5:
                self.suspended_tracks.append(track)
                self.active_tracks.pop(idx)
        
        # Step 6: Remove matched suspended tracks from suspended list
        for idx in sorted([m[0] for m in matched_suspended], reverse=True):
            self.suspended_tracks.pop(idx)
        
        # Step 7: Create new tracks
        for det_idx in still_unmatched:
            self.active_tracks.append(
                OcclusionAwareTrack(detections[det_idx], self.next_id, features[det_idx])
            )
            self.next_id += 1
        
        # Step 8: Delete expired tracks
        self.active_tracks = [
            t for t in self.active_tracks if t.time_since_update <= self.max_age
        ]
        self.suspended_tracks = [
            t for t in self.suspended_tracks 
            if t.time_since_update <= self.max_occlusion
        ]
        
        return self._get_results()
    
    def _match_active(self, detections, features):
        """Match detections to active tracks using IoU"""
        if not self.active_tracks or len(detections) == 0:
            return [], list(range(len(self.active_tracks))), list(range(len(detections)))
        
        # Compute IoU cost matrix
        cost_matrix = np.zeros((len(self.active_tracks), len(detections)))
        for i, track in enumerate(self.active_tracks):
            track_bbox = track.get_bbox()
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self._iou(track_bbox, det[:4])
        
        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        track_idx, det_idx = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(self.active_tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for t, d in zip(track_idx, det_idx):
            if cost_matrix[t, d] < 0.7:  # IoU > 0.3
                matches.append((t, d))
                unmatched_tracks.remove(t)
                unmatched_dets.remove(d)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _match_suspended(self, det_indices, detections, features):
        """
        Match remaining detections to suspended tracks using appearance
        IoU won't work for occluded tracks (prediction may have drifted)
        """
        if not self.suspended_tracks or len(det_indices) == 0:
            return [], list(range(len(self.suspended_tracks))), det_indices
        
        # Compute appearance cost matrix
        cost_matrix = np.ones((len(self.suspended_tracks), len(det_indices)))
        
        for i, track in enumerate(self.suspended_tracks):
            track_features = track.get_recent_features()
            if track_features is None:
                continue
            
            for j, det_idx in enumerate(det_indices):
                det_feat = features[det_idx]
                
                # Cosine similarity
                similarities = track_features @ det_feat
                cost_matrix[i, j] = 1 - similarities.max()
        
        # Hungarian matching with strict threshold
        from scipy.optimize import linear_sum_assignment
        track_idx, det_rel_idx = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(self.suspended_tracks)))
        unmatched_dets = list(det_indices)
        
        for t, d_rel in zip(track_idx, det_rel_idx):
            if cost_matrix[t, d_rel] < 0.3:  # High appearance similarity required
                d_abs = det_indices[d_rel]
                matches.append((t, d_abs))
                unmatched_tracks.remove(t)
                unmatched_dets.remove(d_abs)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _extract_features(self, detections, frame):
        """Extract ReID features"""
        features = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(128))
                continue
            
            crop = frame[y1:y2, x1:x2]
            feat = self.feature_extractor.extract(crop)
            features.append(feat)
        
        return features
    
    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / max(area1 + area2 - inter, 1e-6)
    
    def _get_results(self):
        results = []
        for track in self.active_tracks:
            if track.hits >= 3 and track.time_since_update == 0:
                bbox = track.get_bbox()
                results.append([*bbox, track.track_id])
        return results


class InterpolationRecovery:
    """
    Recover missing detections through interpolation
    For post-processing offline videos
    """
    
    @staticmethod
    def interpolate_gaps(track_history, max_gap=30):
        """
        Fill gaps in track with linear interpolation
        
        Args:
            track_history: Dict[frame_id -> bbox]
            max_gap: Maximum gap to interpolate
        
        Returns:
            Filled track history
        """
        frames = sorted(track_history.keys())
        filled = dict(track_history)
        
        for i in range(len(frames) - 1):
            start_frame = frames[i]
            end_frame = frames[i + 1]
            gap = end_frame - start_frame - 1
            
            if gap > 0 and gap <= max_gap:
                start_bbox = track_history[start_frame]
                end_bbox = track_history[end_frame]
                
                for j in range(1, gap + 1):
                    alpha = j / (gap + 1)
                    interp_bbox = (1 - alpha) * np.array(start_bbox) + alpha * np.array(end_bbox)
                    filled[start_frame + j] = interp_bbox
        
        return filled
```

**Occlusion Handling Stages:**

```
Detection Missing
      ↓
Motion Prediction (Kalman filter continues trajectory)
      ↓
Track Suspension (move to suspended pool)
      ↓
Appearance Matching (match reappearing objects)
      ↓
Track Reactivation or Deletion
```

**Interview Tip:** Occlusion handling is the key differentiator between tracking algorithms. SORT fails quickly under occlusion (IoU drops to 0). DeepSORT improves by using appearance features to re-identify after occlusion. Key design decisions: how long to keep occluded tracks alive (too short loses identity, too long causes ID switches), and how to weight appearance vs motion. ByteTrack's insight: use low-confidence detections which often occur during partial occlusion.

---

### Question 58
- [ ] Done
**What techniques maintain identity consistency across long video sequences?**

**Answer:**

Long-term identity consistency requires: (1) robust appearance models (features invariant to pose/lighting changes), (2) temporal feature aggregation (accumulate appearance over time), (3) global trajectory optimization (consider full video, not just adjacent frames), and (4) re-identification mechanisms (match across gaps). Offline methods can optimize globally; online methods use appearance memory and track confidence scores.

**Key Techniques:**

| Technique | Approach | Online/Offline |
|-----------|----------|----------------|
| Feature averaging | Rolling mean of appearance | Online |
| Trajectory smoothing | Global optimization | Offline |
| Graph-based association | Multi-frame matching | Both |
| Appearance clustering | Group consistent appearances | Offline |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class LongTermIdentityManager:
    """
    Maintain consistent identities across long videos
    """
    
    def __init__(self, feature_dim=512, memory_size=200):
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # Identity database: id -> feature history
        self.identity_database = defaultdict(list)
        
        # Track-to-identity mapping
        self.track_to_identity = {}
        
        # Identity appearance model (EMA of features)
        self.identity_features = {}
        
        self.next_identity_id = 0
    
    def update_identity(self, track_id, feature, confidence=1.0):
        """
        Update identity model with new observation
        
        Args:
            track_id: Current track ID
            feature: Appearance feature (normalized)
            confidence: Detection/track confidence
        """
        # Get or create identity for this track
        if track_id not in self.track_to_identity:
            identity_id = self._find_or_create_identity(feature)
            self.track_to_identity[track_id] = identity_id
        else:
            identity_id = self.track_to_identity[track_id]
        
        # Update feature history
        self.identity_database[identity_id].append({
            'feature': feature,
            'confidence': confidence
        })
        
        # Limit memory size
        if len(self.identity_database[identity_id]) > self.memory_size:
            self.identity_database[identity_id] = self.identity_database[identity_id][-self.memory_size:]
        
        # Update EMA feature
        self._update_ema_feature(identity_id, feature, confidence)
    
    def _find_or_create_identity(self, feature):
        """Match feature to existing identity or create new"""
        if not self.identity_features:
            return self._create_new_identity(feature)
        
        # Compare with all existing identities
        best_match_id = None
        best_similarity = 0
        
        for identity_id, ema_feature in self.identity_features.items():
            similarity = np.dot(feature, ema_feature)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = identity_id
        
        # Threshold for matching
        if best_similarity > 0.7:
            return best_match_id
        else:
            return self._create_new_identity(feature)
    
    def _create_new_identity(self, feature):
        """Create new identity"""
        identity_id = self.next_identity_id
        self.next_identity_id += 1
        self.identity_features[identity_id] = feature.copy()
        return identity_id
    
    def _update_ema_feature(self, identity_id, feature, confidence, alpha=0.1):
        """Exponential moving average of features"""
        effective_alpha = alpha * confidence
        
        if identity_id in self.identity_features:
            old_feat = self.identity_features[identity_id]
            self.identity_features[identity_id] = (
                (1 - effective_alpha) * old_feat + effective_alpha * feature
            )
            # Re-normalize
            self.identity_features[identity_id] /= np.linalg.norm(
                self.identity_features[identity_id]
            )
        else:
            self.identity_features[identity_id] = feature.copy()
    
    def get_identity_feature(self, identity_id):
        """Get stable identity feature"""
        return self.identity_features.get(identity_id)
    
    def merge_identities(self, id1, id2):
        """Merge two identities that are the same person"""
        # Keep id1, merge id2 into it
        if id2 in self.identity_database:
            self.identity_database[id1].extend(self.identity_database[id2])
            del self.identity_database[id2]
        
        # Update track mappings
        for track_id, identity_id in self.track_to_identity.items():
            if identity_id == id2:
                self.track_to_identity[track_id] = id1
        
        # Recalculate EMA feature
        features = [item['feature'] for item in self.identity_database[id1]]
        self.identity_features[id1] = np.mean(features, axis=0)
        self.identity_features[id1] /= np.linalg.norm(self.identity_features[id1])
        
        if id2 in self.identity_features:
            del self.identity_features[id2]


class GlobalTrajectoryOptimization:
    """
    Offline optimization of trajectories across entire video
    Resolves ID switches and fragments by global optimization
    """
    
    def __init__(self, appearance_weight=0.5, motion_weight=0.3, length_weight=0.2):
        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight
        self.length_weight = length_weight
    
    def optimize(self, tracklets, features_per_tracklet):
        """
        Merge tracklets into consistent trajectories
        
        Args:
            tracklets: List of tracklets, each with (start_frame, end_frame, bboxes)
            features_per_tracklet: Appearance features for each tracklet
        
        Returns:
            Merged trajectories
        """
        n = len(tracklets)
        if n <= 1:
            return tracklets
        
        # Build affinity matrix
        affinity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check temporal compatibility (j starts after i ends)
                if tracklets[j]['start'] > tracklets[i]['end']:
                    gap = tracklets[j]['start'] - tracklets[i]['end']
                    
                    if gap < 50:  # Max gap to consider
                        score = self._compute_affinity(
                            tracklets[i], tracklets[j],
                            features_per_tracklet[i], features_per_tracklet[j],
                            gap
                        )
                        affinity[i, j] = score
        
        # Greedy merging based on affinity
        merged = self._greedy_merge(tracklets, affinity)
        
        return merged
    
    def _compute_affinity(self, track1, track2, feat1, feat2, gap):
        """Compute affinity between two tracklets"""
        
        # Appearance affinity (cosine similarity)
        app_score = np.dot(feat1.mean(axis=0), feat2.mean(axis=0))
        
        # Motion affinity (velocity consistency)
        end_pos = track1['bboxes'][-1][:2]  # x, y at end
        start_pos = track2['bboxes'][0][:2]  # x, y at start
        
        velocity = (start_pos - end_pos) / gap
        expected_velocity = self._estimate_velocity(track1['bboxes'])
        
        motion_score = np.exp(-np.linalg.norm(velocity - expected_velocity))
        
        # Length bonus (prefer merging short fragments)
        length_score = 1.0 / (len(track1['bboxes']) + len(track2['bboxes']))
        
        total = (self.appearance_weight * app_score + 
                 self.motion_weight * motion_score +
                 self.length_weight * length_score)
        
        return total
    
    def _estimate_velocity(self, bboxes):
        """Estimate average velocity from bounding boxes"""
        if len(bboxes) < 2:
            return np.array([0, 0])
        
        positions = np.array([b[:2] for b in bboxes])
        velocities = np.diff(positions, axis=0)
        return velocities.mean(axis=0)
    
    def _greedy_merge(self, tracklets, affinity, threshold=0.6):
        """Greedy merging based on affinity scores"""
        n = len(tracklets)
        merged = list(tracklets)
        merged_indices = list(range(n))
        
        while True:
            # Find best pair
            best_score = 0
            best_pair = None
            
            for i in range(len(merged_indices)):
                for j in range(i + 1, len(merged_indices)):
                    orig_i, orig_j = merged_indices[i], merged_indices[j]
                    if affinity[orig_i, orig_j] > best_score:
                        best_score = affinity[orig_i, orig_j]
                        best_pair = (i, j)
            
            if best_score < threshold or best_pair is None:
                break
            
            # Merge pair
            i, j = best_pair
            merged[i] = self._merge_tracklets(merged[i], merged[j])
            merged.pop(j)
            merged_indices.pop(j)
        
        return merged
    
    def _merge_tracklets(self, t1, t2):
        """Merge two tracklets into one"""
        return {
            'start': t1['start'],
            'end': t2['end'],
            'bboxes': t1['bboxes'] + t2['bboxes']
        }


class TemporalFeatureAggregation(nn.Module):
    """
    Aggregate appearance features over time for stable identity
    """
    
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        
        # Attention-based aggregation
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.scale = hidden_dim ** -0.5
    
    def forward(self, features, mask=None):
        """
        Aggregate sequence of features into single identity descriptor
        
        Args:
            features: (B, T, D) sequence of appearance features
            mask: (B, T) optional mask for valid features
        
        Returns:
            (B, D) aggregated identity feature
        """
        B, T, D = features.shape
        
        # Self-attention
        q = self.query(features)  # (B, T, H)
        k = self.key(features)    # (B, T, H)
        v = self.value(features)  # (B, T, D)
        
        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, T, T)
        
        if mask is not None:
            attention = attention.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        attention = torch.softmax(attention, dim=-1)
        
        # Weighted aggregation
        aggregated = torch.matmul(attention, v)  # (B, T, D)
        
        # Final pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            aggregated = (aggregated * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            aggregated = aggregated.mean(dim=1)
        
        return aggregated


class TrackConfidenceScoring:
    """
    Score track reliability for identity consistency
    High-confidence tracks used as anchors
    """
    
    def __init__(self):
        self.weights = {
            'length': 0.3,        # Longer tracks more reliable
            'detection_conf': 0.3, # Higher detection confidence
            'motion_smooth': 0.2,  # Smooth motion trajectory
            'appearance_var': 0.2  # Consistent appearance
        }
    
    def compute_score(self, track):
        """Compute overall track confidence"""
        scores = {}
        
        # Length score (normalized)
        scores['length'] = min(len(track.trajectory) / 100, 1.0)
        
        # Detection confidence (average)
        if hasattr(track, 'detection_confidences'):
            scores['detection_conf'] = np.mean(track.detection_confidences)
        else:
            scores['detection_conf'] = 0.5
        
        # Motion smoothness (low acceleration)
        scores['motion_smooth'] = self._motion_smoothness(track.trajectory)
        
        # Appearance consistency (low variance in features)
        scores['appearance_var'] = self._appearance_consistency(track.feature_buffer)
        
        # Weighted sum
        total = sum(self.weights[k] * scores[k] for k in self.weights)
        
        return total
    
    def _motion_smoothness(self, trajectory):
        """Compute motion smoothness score"""
        if len(trajectory) < 3:
            return 0.5
        
        positions = np.array([t[:2] for t in trajectory])
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Low acceleration = smooth motion
        smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(accelerations, axis=1)))
        return smoothness
    
    def _appearance_consistency(self, features):
        """Compute appearance consistency score"""
        if len(features) < 2:
            return 0.5
        
        features = np.array(list(features))
        mean_feat = features.mean(axis=0)
        
        # Similarity to mean
        similarities = features @ mean_feat
        consistency = similarities.mean()
        
        return max(0, consistency)
```

**Interview Tip:** Long-term identity consistency is where offline methods shine—they can look at the entire video and perform global optimization. Online methods must rely on robust appearance models and careful track management. Key insight: accumulate appearance features over time (EMA or attention-based aggregation) to build stable identity representations robust to single-frame variations. Track confidence scoring helps identify reliable tracks to use as anchors.

---

### Question 59
- [ ] Done
**Explain re-identification (ReID) features and their role in multi-camera tracking.**

**Answer:**

Re-identification (ReID) learns appearance embeddings that identify the same person across different cameras, viewpoints, and times. Unlike face recognition, ReID uses whole-body appearance (clothing, body shape). Features must be robust to viewpoint changes, illumination, and pose variations. Multi-camera tracking uses ReID to associate trajectories when the same person appears in different camera views.

**ReID Challenges:**

| Challenge | Cause | Solution |
|-----------|-------|----------|
| Viewpoint change | Front vs back view | Multi-view training |
| Illumination | Indoor vs outdoor | Color normalization |
| Occlusion | Partial visibility | Part-based models |
| Clothing change | Long-term tracking | Body shape features |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PersonReID(nn.Module):
    """
    Person Re-identification network
    Learns discriminative embeddings for person matching
    """
    
    def __init__(self, backbone, embedding_dim=512, num_classes=None):
        super().__init__()
        
        self.backbone = backbone  # e.g., ResNet50
        
        # Global feature
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.fc = nn.Linear(2048, embedding_dim)
        
        # Classification head (for training)
        if num_classes:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x):
        """
        x: (B, 3, 256, 128) person crop
        Returns: (B, embedding_dim) normalized embedding
        """
        features = self.backbone(x)  # (B, 2048, 8, 4)
        
        # Global average pooling
        global_feat = self.global_pool(features).flatten(1)  # (B, 2048)
        global_feat = self.bottleneck(global_feat)
        
        embedding = self.fc(global_feat)
        embedding = F.normalize(embedding, dim=1)
        
        if self.training and self.classifier:
            logits = self.classifier(embedding)
            return embedding, logits
        
        return embedding


class PartBasedReID(nn.Module):
    """
    Part-based ReID for handling partial occlusion
    Divide body into horizontal stripes
    """
    
    def __init__(self, backbone, num_parts=6, embedding_dim=256):
        super().__init__()
        
        self.backbone = backbone
        self.num_parts = num_parts
        
        # Part-specific embeddings
        self.part_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)) for _ in range(num_parts)
        ])
        
        self.part_fc = nn.ModuleList([
            nn.Linear(2048, embedding_dim) for _ in range(num_parts)
        ])
        
        self.part_bn = nn.ModuleList([
            nn.BatchNorm1d(embedding_dim) for _ in range(num_parts)
        ])
    
    def forward(self, x):
        """
        Returns list of part embeddings
        """
        features = self.backbone(x)  # (B, 2048, H, W)
        B, C, H, W = features.shape
        
        part_height = H // self.num_parts
        embeddings = []
        
        for i in range(self.num_parts):
            start = i * part_height
            end = start + part_height if i < self.num_parts - 1 else H
            
            part_feat = features[:, :, start:end, :]
            pooled = self.part_pools[i](part_feat).flatten(1)
            
            emb = self.part_fc[i](pooled)
            emb = self.part_bn[i](emb)
            emb = F.normalize(emb, dim=1)
            
            embeddings.append(emb)
        
        return embeddings
    
    def compute_distance(self, feat1_list, feat2_list):
        """
        Compute part-based distance
        Allow partial matching when some parts are occluded
        """
        distances = []
        for f1, f2 in zip(feat1_list, feat2_list):
            dist = 1 - torch.sum(f1 * f2, dim=1)
            distances.append(dist)
        
        # Average distance across parts
        return torch.stack(distances).mean(dim=0)


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning
    Pull same-person embeddings together, push different apart
    """
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        embeddings: (B, D) normalized embeddings
        labels: (B,) person IDs
        """
        B = embeddings.shape[0]
        
        # Pairwise distances
        dist_matrix = 1 - embeddings @ embeddings.T
        
        loss = 0
        count = 0
        
        for i in range(B):
            # Same person (positive)
            pos_mask = (labels == labels[i]) & (torch.arange(B) != i)
            # Different person (negative)
            neg_mask = (labels != labels[i])
            
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            
            # Hard positive: furthest same person
            pos_dist = dist_matrix[i][pos_mask].max()
            # Hard negative: closest different person
            neg_dist = dist_matrix[i][neg_mask].min()
            
            loss += F.relu(pos_dist - neg_dist + self.margin)
            count += 1
        
        return loss / max(count, 1)


class CenterLoss(nn.Module):
    """
    Center loss: pull embeddings toward class centers
    Complements softmax loss
    """
    
    def __init__(self, num_classes, feature_dim, lambda_c=0.005):
        super().__init__()
        
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.lambda_c = lambda_c
    
    def forward(self, embeddings, labels):
        """
        Minimize distance to class centers
        """
        batch_centers = self.centers[labels]
        loss = ((embeddings - batch_centers) ** 2).sum(dim=1).mean()
        return self.lambda_c * loss


# ========== MULTI-CAMERA TRACKING ==========

class MultiCameraTracker:
    """
    Track persons across multiple non-overlapping cameras
    Uses ReID to associate trajectories
    """
    
    def __init__(self, reid_model, cameras, matching_threshold=0.5):
        self.reid_model = reid_model
        self.cameras = cameras
        self.matching_threshold = matching_threshold
        
        # Per-camera trackers
        self.camera_trackers = {cam: SingleCameraTracker() for cam in cameras}
        
        # Global identity mapping
        self.global_identities = {}  # global_id -> {camera -> local_track_id}
        self.identity_features = {}  # global_id -> aggregated features
        
        self.next_global_id = 0
    
    def update(self, camera_id, frame, detections):
        """
        Update tracking for one camera
        """
        # Get local tracks from single-camera tracker
        local_tracks = self.camera_trackers[camera_id].update(frame, detections)
        
        # Extract ReID features
        track_features = {}
        for track_id, bbox in local_tracks:
            crop = self._crop_person(frame, bbox)
            with torch.no_grad():
                feature = self.reid_model(crop).squeeze().cpu().numpy()
            track_features[track_id] = feature
        
        # Match to global identities
        results = []
        for track_id, bbox in local_tracks:
            global_id = self._match_to_global(
                camera_id, track_id, track_features[track_id]
            )
            results.append((global_id, bbox))
        
        return results
    
    def _match_to_global(self, camera_id, local_track_id, feature):
        """
        Match local track to global identity
        """
        # Check if already matched
        for global_id, camera_tracks in self.global_identities.items():
            if camera_id in camera_tracks:
                if camera_tracks[camera_id] == local_track_id:
                    # Update feature
                    self._update_identity_feature(global_id, feature)
                    return global_id
        
        # Try to match by appearance
        best_match = None
        best_similarity = 0
        
        for global_id, global_feature in self.identity_features.items():
            similarity = np.dot(feature, global_feature)
            
            # Must not already have a track in this camera
            if camera_id not in self.global_identities.get(global_id, {}):
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = global_id
        
        if best_similarity > self.matching_threshold:
            self._assign_to_global(best_match, camera_id, local_track_id, feature)
            return best_match
        else:
            return self._create_global_identity(camera_id, local_track_id, feature)
    
    def _create_global_identity(self, camera_id, local_track_id, feature):
        """Create new global identity"""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        self.global_identities[global_id] = {camera_id: local_track_id}
        self.identity_features[global_id] = feature
        
        return global_id
    
    def _assign_to_global(self, global_id, camera_id, local_track_id, feature):
        """Assign local track to existing global identity"""
        if global_id not in self.global_identities:
            self.global_identities[global_id] = {}
        
        self.global_identities[global_id][camera_id] = local_track_id
        self._update_identity_feature(global_id, feature)
    
    def _update_identity_feature(self, global_id, feature, alpha=0.1):
        """EMA update of identity feature"""
        old = self.identity_features[global_id]
        self.identity_features[global_id] = (1 - alpha) * old + alpha * feature
        self.identity_features[global_id] /= np.linalg.norm(
            self.identity_features[global_id]
        )
    
    def _crop_person(self, frame, bbox):
        """Crop and preprocess person image"""
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        
        # Convert to tensor and resize
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        crop = F.interpolate(crop.unsqueeze(0), size=(256, 128))
        
        return crop


class SingleCameraTracker:
    """Placeholder for single-camera tracker"""
    def update(self, frame, detections):
        return [(i, det[:4]) for i, det in enumerate(detections)]


class CameraTopologyLearning:
    """
    Learn camera network topology for multi-camera tracking
    Which cameras are connected and transition times
    """
    
    def __init__(self, camera_ids):
        self.camera_ids = camera_ids
        n = len(camera_ids)
        
        # Transition probability matrix
        self.transition_prob = np.ones((n, n)) / n
        
        # Typical transition time (mean, std)
        self.transition_time = np.ones((n, n, 2)) * 60  # Default 60 seconds
    
    def update_from_matches(self, camera1, camera2, time_diff):
        """Update topology from observed matches"""
        i, j = self.camera_ids.index(camera1), self.camera_ids.index(camera2)
        
        # Update transition probability (EMA)
        self.transition_prob[i, :] *= 0.99
        self.transition_prob[i, j] += 0.01
        self.transition_prob[i, :] /= self.transition_prob[i, :].sum()
        
        # Update transition time (running mean/std)
        old_mean, old_std = self.transition_time[i, j]
        alpha = 0.1
        new_mean = (1 - alpha) * old_mean + alpha * time_diff
        new_std = (1 - alpha) * old_std + alpha * abs(time_diff - old_mean)
        self.transition_time[i, j] = [new_mean, new_std]
    
    def get_matching_probability(self, camera1, camera2, time_diff):
        """
        Get probability of transition given time difference
        Used to weight appearance matching
        """
        i, j = self.camera_ids.index(camera1), self.camera_ids.index(camera2)
        
        mean, std = self.transition_time[i, j]
        
        # Gaussian probability based on expected transition time
        z = (time_diff - mean) / max(std, 1)
        prob = np.exp(-0.5 * z ** 2) * self.transition_prob[i, j]
        
        return prob
```

**ReID Training Data:**

| Dataset | Size | Cameras | Use |
|---------|------|---------|-----|
| Market-1501 | 32k images | 6 | Benchmark |
| DukeMTMC | 36k images | 8 | Cross-domain |
| CUHK03 | 14k images | 2 | Small-scale |

**Interview Tip:** ReID is different from face recognition—it relies on clothing, body shape, and gait rather than biometrics. Major challenge: person wearing same clothes as another person (similar appearance). Part-based models help with occlusion. For multi-camera tracking, learning camera topology (which cameras connect, typical transition times) significantly improves matching by constraining the search space.

---

### Question 60
- [ ] Done
**How do you implement real-time tracking with computational efficiency constraints?**

**Answer:**

Real-time tracking requires: (1) efficient detection (lightweight detectors like YOLOv8-nano), (2) simple association (IoU-based like SORT, skip expensive ReID), (3) sparse detection (detect every N frames, track between), and (4) hardware optimization (TensorRT, ONNX, GPU batching). Key trade-off: accuracy vs latency—simpler methods achieve 100+ FPS while complex methods (DeepSORT) run at 20-30 FPS.

**Speed vs Accuracy Trade-offs:**

| Method | Speed | MOTA | Approach |
|--------|-------|------|----------|
| SORT | 260 FPS | 59.8 | IoU only |
| DeepSORT | 20 FPS | 61.4 | IoU + ReID |
| ByteTrack | 30 FPS | 77.8 | Two-stage association |
| CenterTrack | 22 FPS | 67.3 | Joint detection + tracking |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import time

class RealTimeTracker:
    """
    Efficient real-time multi-object tracker
    Optimized for speed while maintaining reasonable accuracy
    """
    
    def __init__(self, detector, max_age=30, min_hits=3):
        self.detector = detector
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks = []
        self.next_id = 0
        
        # Skip detection frames for speed
        self.detection_interval = 3
        self.frame_count = 0
        
        # Timing statistics
        self.timing = {'detection': [], 'association': [], 'total': []}
    
    def update(self, frame):
        """
        Process frame with speed optimizations
        """
        start_time = time.time()
        
        # Step 1: Detection (expensive - do less frequently)
        if self.frame_count % self.detection_interval == 0:
            det_start = time.time()
            detections = self.detector(frame)
            self.timing['detection'].append(time.time() - det_start)
        else:
            detections = []  # Use predicted positions
        
        # Step 2: Predict all tracks (fast - Kalman prediction)
        for track in self.tracks:
            track['predicted'] = self._predict_next_position(track)
        
        # Step 3: Association (IoU-based, no ReID for speed)
        assoc_start = time.time()
        if len(detections) > 0:
            matches, unmatched_tracks, unmatched_dets = self._fast_associate(
                detections
            )
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                self._update_track(self.tracks[track_idx], detections[det_idx])
            
            # Create new tracks
            for det_idx in unmatched_dets:
                self._create_track(detections[det_idx])
        
        self.timing['association'].append(time.time() - assoc_start)
        
        # Step 4: Track management
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        
        self.frame_count += 1
        self.timing['total'].append(time.time() - start_time)
        
        # Return confirmed tracks
        return self._get_output_tracks()
    
    def _predict_next_position(self, track):
        """Simple constant velocity prediction"""
        pos = track['bbox']
        vel = track.get('velocity', np.zeros(4))
        return pos + vel
    
    def _fast_associate(self, detections):
        """
        Fast IoU-based association using vectorized operations
        """
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))
        
        # Vectorized IoU computation
        track_boxes = np.array([t['predicted'] for t in self.tracks])
        det_boxes = np.array([d[:4] for d in detections])
        
        iou_matrix = self._vectorized_iou(track_boxes, det_boxes)
        
        # Greedy matching (faster than Hungarian for small N)
        matches = []
        used_tracks = set()
        used_dets = set()
        
        # Sort by IoU descending
        indices = np.dstack(np.unravel_index(
            np.argsort(-iou_matrix.ravel()), iou_matrix.shape
        ))[0]
        
        for t_idx, d_idx in indices:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            if iou_matrix[t_idx, d_idx] < 0.3:
                break
            
            matches.append((t_idx, d_idx))
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
        
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _vectorized_iou(self, boxes1, boxes2):
        """Vectorized IoU computation"""
        # boxes: [x1, y1, x2, y2]
        n, m = len(boxes1), len(boxes2)
        
        boxes1 = boxes1.reshape(n, 1, 4)
        boxes2 = boxes2.reshape(1, m, 4)
        
        x1 = np.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
        y1 = np.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
        x2 = np.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
        y2 = np.minimum(boxes1[:, :, 3], boxes2[:, :, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
        area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
        
        iou = intersection / (area1 + area2 - intersection + 1e-6)
        
        return iou
    
    def _update_track(self, track, detection):
        """Update track with detection"""
        old_pos = track['bbox']
        new_pos = detection[:4]
        
        track['velocity'] = 0.7 * track.get('velocity', np.zeros(4)) + 0.3 * (new_pos - old_pos)
        track['bbox'] = new_pos
        track['hits'] += 1
        track['age'] = 0
    
    def _create_track(self, detection):
        """Create new track"""
        self.tracks.append({
            'id': self.next_id,
            'bbox': detection[:4],
            'velocity': np.zeros(4),
            'hits': 1,
            'age': 0
        })
        self.next_id += 1
    
    def _get_output_tracks(self):
        """Return confirmed tracks"""
        results = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits and track['age'] == 0:
                results.append((track['id'], track['bbox']))
        return results
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.timing['total']) > 0:
            avg_time = np.mean(self.timing['total'][-100:])
            return 1.0 / max(avg_time, 1e-6)
        return 0


class LightweightDetector(nn.Module):
    """
    Lightweight object detector for real-time tracking
    Uses depthwise separable convolutions
    """
    
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Depthwise separable backbone
        self.features = nn.Sequential(
            self._depthwise_sep_conv(3, 32, stride=2),
            self._depthwise_sep_conv(32, 64, stride=2),
            self._depthwise_sep_conv(64, 128, stride=2),
            self._depthwise_sep_conv(128, 256, stride=2),
            self._depthwise_sep_conv(256, 512, stride=2),
        )
        
        # Detection head
        self.cls_head = nn.Conv2d(512, num_classes, 1)
        self.reg_head = nn.Conv2d(512, 4, 1)
    
    def _depthwise_sep_conv(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
    
    def forward(self, x):
        features = self.features(x)
        cls_output = self.cls_head(features)
        reg_output = self.reg_head(features)
        return cls_output, reg_output


class TensorRTOptimizedTracker:
    """
    Tracker with TensorRT-optimized detection
    """
    
    def __init__(self, trt_engine_path):
        # Load TensorRT engine
        import tensorrt as trt
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        """Pre-allocate GPU buffers for zero-copy inference"""
        import pycuda.driver as cuda
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = np.prod(shape)
            dtype = np.float32
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def detect(self, frame):
        """Run TensorRT detection"""
        import pycuda.driver as cuda
        
        # Preprocess
        processed = self._preprocess(frame)
        np.copyto(self.inputs[0]['host'], processed.ravel())
        
        # Copy to GPU
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Run inference
        self.context.execute_v2(self.bindings)
        
        # Copy results back
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        
        # Postprocess
        detections = self._postprocess(self.outputs[0]['host'])
        
        return detections
    
    def _preprocess(self, frame):
        """Preprocess frame for TensorRT"""
        # Resize, normalize, transpose
        return frame.astype(np.float32) / 255.0
    
    def _postprocess(self, output):
        """Convert TensorRT output to detections"""
        # Decode boxes, apply NMS
        return []


class AdaptiveFrameSkipping:
    """
    Dynamically adjust detection frequency based on scene complexity
    """
    
    def __init__(self, target_fps=30, min_interval=1, max_interval=10):
        self.target_fps = target_fps
        self.min_interval = min_interval
        self.max_interval = max_interval
        
        self.current_interval = 3
        self.frame_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=10)
    
    def should_detect(self, frame_id, scene_motion):
        """
        Decide whether to run detection on this frame
        
        Args:
            frame_id: Current frame number
            scene_motion: Estimated scene motion (0-1)
        
        Returns:
            bool: Whether to run detection
        """
        # High motion = more frequent detection
        if scene_motion > 0.7:
            interval = self.min_interval
        elif scene_motion < 0.2:
            interval = self.max_interval
        else:
            interval = int(self.min_interval + 
                          (self.max_interval - self.min_interval) * (1 - scene_motion))
        
        # Adjust based on current FPS
        current_fps = self._get_current_fps()
        if current_fps < self.target_fps * 0.8:
            interval = min(interval + 1, self.max_interval)
        elif current_fps > self.target_fps * 1.2:
            interval = max(interval - 1, self.min_interval)
        
        self.current_interval = interval
        return frame_id % interval == 0
    
    def _get_current_fps(self):
        if len(self.frame_times) < 2:
            return self.target_fps
        return len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0] + 1e-6)
    
    def record_frame_time(self, timestamp):
        self.frame_times.append(timestamp)


class BatchedDetection:
    """
    Batch multiple frames for efficient GPU utilization
    """
    
    def __init__(self, detector, batch_size=4):
        self.detector = detector
        self.batch_size = batch_size
        self.frame_buffer = []
    
    def add_frame(self, frame):
        """Add frame to batch buffer"""
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) >= self.batch_size:
            return self._process_batch()
        
        return None
    
    def _process_batch(self):
        """Process accumulated batch"""
        batch = np.stack(self.frame_buffer)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
        
        with torch.no_grad():
            detections = self.detector(batch)
        
        self.frame_buffer = []
        return detections
    
    def flush(self):
        """Process remaining frames in buffer"""
        if len(self.frame_buffer) > 0:
            return self._process_batch()
        return None


def optimization_checklist():
    """Speed optimization checklist for real-time tracking"""
    
    print("Real-Time Tracking Optimization Checklist:")
    print("=" * 60)
    
    optimizations = [
        ("Lightweight detector", "YOLO-nano, MobileNet backbone", "2-4x speedup"),
        ("TensorRT/ONNX", "Optimize inference engine", "2-3x speedup"),
        ("Frame skipping", "Detect every N frames", "N x speedup"),
        ("Greedy matching", "Skip Hungarian algorithm", "Minor speedup"),
        ("Skip ReID", "IoU-only association", "10x+ speedup"),
        ("GPU batching", "Process multiple frames", "2x throughput"),
        ("Half precision", "FP16 inference", "2x speedup"),
    ]
    
    print(f"{'Optimization':<20} {'Technique':<30} {'Impact'}")
    print("-" * 60)
    for opt, technique, impact in optimizations:
        print(f"{opt:<20} {technique:<30} {impact}")
```

**Performance Tips:**

| Technique | FPS Impact | Accuracy Impact |
|-----------|-----------|-----------------|
| FP16 inference | +2x | Minimal |
| Frame skipping (3x) | +3x | Slight decrease |
| Greedy vs Hungarian | +1.2x | Minimal for small N |
| No ReID | +10x | Handles occlusion worse |

**Interview Tip:** Real-time tracking is all about trade-offs. The biggest wins: (1) use lightweight detectors (YOLO-nano, 6ms vs 30ms for full YOLO), (2) skip ReID when not needed (most speed loss in DeepSORT is feature extraction), (3) detect every N frames and interpolate. For embedded systems, TensorRT optimization can give 2-3x speedup. ByteTrack achieves good accuracy with IoU-only association by cleverly using low-confidence detections.

---

## Diffusion Models (Bonus)

### Question 61
- [ ] Done
**Explain the forward and reverse diffusion process. How do diffusion models generate images?**

**Answer:**

Diffusion models work in two phases: (1) forward process progressively adds Gaussian noise to data over T timesteps until it becomes pure noise, (2) reverse process learns to denoise step-by-step, recovering the original data. Generation starts from random noise and iteratively denoises to produce realistic images. The neural network learns to predict the noise added at each step.

**Forward Process (Fixed):**
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

**Reverse Process (Learned):**
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiffusionModel:
    """
    Denoising Diffusion Probabilistic Model (DDPM)
    """
    
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model  # Noise prediction network (U-Net)
        self.timesteps = timesteps
        
        # Noise schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    # ========== FORWARD PROCESS ==========
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: add noise to x_0 to get x_t
        
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_0: (B, C, H, W) original clean images
            t: (B,) timesteps
            noise: Optional pre-generated noise
        
        Returns:
            x_t: Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
        return x_t
    
    # ========== REVERSE PROCESS ==========
    
    def p_sample(self, x_t, t):
        """
        Reverse process: one denoising step from x_t to x_{t-1}
        
        Args:
            x_t: (B, C, H, W) noisy image at timestep t
            t: (B,) timesteps
        
        Returns:
            x_{t-1}: Less noisy image
        """
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Compute mean of p(x_{t-1} | x_t)
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        # Mean: mu = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * predicted_noise)
        mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
        )
        
        # Add noise (except for t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self._extract(self.posterior_variance, t, x_t.shape)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, shape, device='cuda'):
        """
        Generate images by iterative denoising
        
        Args:
            shape: (B, C, H, W) output shape
            device: Device to use
        
        Returns:
            Generated images
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        
        return x
    
    # ========== TRAINING ==========
    
    def training_loss(self, x_0):
        """
        Training objective: predict the noise added to x_0
        
        Loss = ||noise - predicted_noise||^2
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device)
        
        # Add noise
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Simple L2 loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def _extract(self, a, t, shape):
        """Extract values from a at indices t and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


class UNet(nn.Module):
    """
    U-Net architecture for noise prediction
    Takes noisy image and timestep, predicts noise
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        
        # Encoder
        self.down1 = DownBlock(in_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = ResBlock(base_channels * 4, base_channels * 4)
        
        # Decoder
        self.up3 = UpBlock(base_channels * 4, base_channels * 2)
        self.up2 = UpBlock(base_channels * 2, base_channels)
        self.up1 = UpBlock(base_channels, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, x, t):
        """
        x: (B, C, H, W) noisy image
        t: (B,) timestep
        Returns: (B, C, H, W) predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        h1 = self.down1(x, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.down3(h2, t_emb)
        
        # Bottleneck
        h = self.bottleneck(h3, t_emb)
        
        # Decoder with skip connections
        h = self.up3(h, h3, t_emb)
        h = self.up2(h, h2, t_emb)
        h = self.up1(h, h1, t_emb)
        
        return self.out(h)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time conditioning"""
    
    def __init__(self, in_ch, out_ch, time_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(F.silu(self.conv2(h)))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x, t_emb):
        return self.pool(self.res(x, t_emb))


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res = ResBlock(in_ch * 2, out_ch)
    
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


def diffusion_overview():
    """Summary of diffusion process"""
    
    print("Diffusion Model Overview:")
    print("=" * 60)
    
    print("\nForward Process (Noising):")
    print("  x_0 → x_1 → ... → x_T ≈ N(0, I)")
    print("  Add small Gaussian noise at each step")
    
    print("\nReverse Process (Denoising):")
    print("  x_T → x_{T-1} → ... → x_0")
    print("  Neural network predicts noise to remove")
    
    print("\nTraining Objective:")
    print("  Minimize ||ε - ε_θ(x_t, t)||²")
    print("  Where ε is true noise, ε_θ is predicted noise")
    
    print("\nGeneration:")
    print("  1. Sample x_T ~ N(0, I)")
    print("  2. For t = T to 1:")
    print("     x_{t-1} = denoise(x_t, t)")
    print("  3. Return x_0")
```

**Key Concepts:**

| Component | Role |
|-----------|------|
| β schedule | Controls noise addition rate |
| α̅ (alpha_bar) | Cumulative noise level |
| ε_θ (model) | Predicts noise to remove |
| U-Net | Architecture for noise prediction |

**Interview Tip:** Diffusion models' key insight: instead of learning p(x) directly (hard), learn to gradually denoise (easy local operations). The forward process is fixed (just adding Gaussian noise), only the reverse process is learned. Training is simple (predict noise), but sampling is slow (requires 1000 steps). Understanding the reparameterization (predict noise vs. predict x_0) is crucial—both are equivalent mathematically.

---

### Question 62
- [ ] Done
**Compare diffusion models vs. GANs for image generation quality, diversity, and training stability.**

**Answer:**

Diffusion models produce higher quality and diversity than GANs but are slower to sample. GANs suffer from mode collapse and training instability but generate images in one forward pass. Diffusion models have stable training (simple MSE loss) and better coverage of the data distribution. Current state-of-the-art (Stable Diffusion, DALL-E 3) uses diffusion; GANs remain relevant for real-time applications.

**Comparison Table:**

| Aspect | Diffusion Models | GANs |
|--------|-----------------|------|
| Image Quality | Excellent (FID ~2) | Very Good (FID ~4) |
| Diversity | High (full distribution) | Mode collapse risk |
| Training | Stable (MSE loss) | Unstable (adversarial) |
| Sampling Speed | Slow (1000 steps) | Fast (one forward pass) |
| Mode Coverage | Complete | May miss modes |
| Controllability | Excellent (guidance) | Limited |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ========== QUALITY COMPARISON ==========

class QualityMetrics:
    """Compare generation quality between models"""
    
    def __init__(self, inception_model):
        self.inception = inception_model
    
    def compute_fid(self, real_images, generated_images):
        """
        Fréchet Inception Distance (FID)
        Lower = better, measures quality and diversity
        """
        # Get Inception features
        real_features = self._get_features(real_images)
        gen_features = self._get_features(generated_images)
        
        # Compute statistics
        mu_real, sigma_real = real_features.mean(0), np.cov(real_features.T)
        mu_gen, sigma_gen = gen_features.mean(0), np.cov(gen_features.T)
        
        # FID formula
        diff = mu_real - mu_gen
        covmean = self._matrix_sqrt(sigma_real @ sigma_gen)
        
        fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        
        return fid
    
    def compute_inception_score(self, generated_images):
        """
        Inception Score (IS)
        Higher = better quality and diversity
        """
        # Get class predictions
        preds = self._get_predictions(generated_images)
        
        # p(y|x) should be peaked (quality)
        # p(y) should be uniform (diversity)
        p_y = preds.mean(axis=0)
        
        kl_divs = []
        for p_y_given_x in preds:
            kl = np.sum(p_y_given_x * (np.log(p_y_given_x + 1e-10) - np.log(p_y + 1e-10)))
            kl_divs.append(kl)
        
        is_score = np.exp(np.mean(kl_divs))
        return is_score
    
    def _get_features(self, images):
        with torch.no_grad():
            features = self.inception.extract_features(images)
        return features.cpu().numpy()
    
    def _get_predictions(self, images):
        with torch.no_grad():
            logits = self.inception(images)
            preds = F.softmax(logits, dim=1)
        return preds.cpu().numpy()
    
    def _matrix_sqrt(self, matrix):
        """Compute matrix square root"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        sqrt_diag = np.sqrt(np.maximum(eigenvalues, 0))
        return eigenvectors @ np.diag(sqrt_diag) @ np.linalg.inv(eigenvectors)


# ========== DIVERSITY COMPARISON ==========

class DiversityAnalysis:
    """Analyze mode coverage and diversity"""
    
    def measure_mode_coverage(self, generated_samples, mode_classifier):
        """
        Check how many modes are covered by generated samples
        
        Args:
            generated_samples: Generated images
            mode_classifier: Classifier trained on modes (e.g., classes)
        
        Returns:
            Fraction of modes covered
        """
        # Classify generated samples
        with torch.no_grad():
            predictions = mode_classifier(generated_samples).argmax(dim=1)
        
        unique_modes = len(torch.unique(predictions))
        total_modes = mode_classifier.num_classes
        
        return unique_modes / total_modes
    
    def measure_lpips_diversity(self, generated_samples, lpips_model):
        """
        Measure perceptual diversity using LPIPS
        Higher = more diverse
        """
        n = len(generated_samples)
        distances = []
        
        for i in range(min(n, 100)):
            for j in range(i + 1, min(n, 100)):
                dist = lpips_model(generated_samples[i], generated_samples[j])
                distances.append(dist.item())
        
        return np.mean(distances)
    
    def detect_mode_collapse(self, generator, num_samples=1000, threshold=0.9):
        """
        Detect if GAN has mode collapse
        Check if many samples are too similar
        """
        samples = []
        for _ in range(num_samples // 16):
            z = torch.randn(16, 512)
            with torch.no_grad():
                fake = generator(z)
            samples.append(fake)
        
        samples = torch.cat(samples)
        
        # Compute pairwise similarities
        flat = samples.view(len(samples), -1)
        flat = F.normalize(flat, dim=1)
        similarity = flat @ flat.T
        
        # Remove diagonal
        mask = ~torch.eye(len(samples), dtype=bool)
        similarities = similarity[mask]
        
        # High similarity fraction indicates mode collapse
        collapse_fraction = (similarities > threshold).float().mean()
        
        return collapse_fraction > 0.1


# ========== TRAINING STABILITY ==========

class TrainingStabilityAnalysis:
    """Analyze training stability differences"""
    
    def __init__(self):
        self.gan_history = {'d_loss': [], 'g_loss': [], 'gradient_norms': []}
        self.diffusion_history = {'loss': [], 'gradient_norms': []}
    
    def log_gan_step(self, d_loss, g_loss, d_grad_norm, g_grad_norm):
        """Log GAN training metrics"""
        self.gan_history['d_loss'].append(d_loss)
        self.gan_history['g_loss'].append(g_loss)
        self.gan_history['gradient_norms'].append((d_grad_norm, g_grad_norm))
    
    def log_diffusion_step(self, loss, grad_norm):
        """Log diffusion training metrics"""
        self.diffusion_history['loss'].append(loss)
        self.diffusion_history['gradient_norms'].append(grad_norm)
    
    def analyze_stability(self):
        """Compare training stability"""
        results = {}
        
        # GAN loss variance (high variance = unstable)
        if len(self.gan_history['d_loss']) > 100:
            g_variance = np.var(self.gan_history['g_loss'][-100:])
            d_variance = np.var(self.gan_history['d_loss'][-100:])
            results['gan_loss_variance'] = (g_variance + d_variance) / 2
        
        # Diffusion loss variance
        if len(self.diffusion_history['loss']) > 100:
            results['diffusion_loss_variance'] = np.var(
                self.diffusion_history['loss'][-100:]
            )
        
        # Gradient explosion/vanishing
        if len(self.gan_history['gradient_norms']) > 0:
            d_norms, g_norms = zip(*self.gan_history['gradient_norms'])
            results['gan_grad_explosion'] = np.max(g_norms) > 100
            results['gan_grad_vanishing'] = np.min(g_norms) < 1e-6
        
        return results


# ========== SPEED COMPARISON ==========

class SpeedComparison:
    """Compare sampling speed"""
    
    @staticmethod
    def benchmark_gan(generator, latent_dim=512, batch_size=16, num_batches=100):
        """Benchmark GAN generation speed"""
        import time
        
        times = []
        for _ in range(num_batches):
            z = torch.randn(batch_size, latent_dim).cuda()
            
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = generator(z)
            
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        return {
            'mean_time': np.mean(times),
            'images_per_second': batch_size / np.mean(times)
        }
    
    @staticmethod
    def benchmark_diffusion(diffusion, shape, num_steps=1000, num_samples=10):
        """Benchmark diffusion generation speed"""
        import time
        
        times = []
        for _ in range(num_samples):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = diffusion.sample(shape)
            
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        return {
            'mean_time': np.mean(times),
            'images_per_second': shape[0] / np.mean(times),
            'steps': num_steps
        }


# ========== COMPREHENSIVE COMPARISON ==========

def comprehensive_comparison():
    """
    Full comparison of diffusion vs GAN approaches
    """
    
    comparison = {
        'Image Quality': {
            'Diffusion': 'State-of-the-art (FID ~2 on ImageNet)',
            'GAN': 'Excellent but slightly lower (FID ~4-5)',
            'Winner': 'Diffusion'
        },
        'Sample Diversity': {
            'Diffusion': 'Full distribution coverage',
            'GAN': 'Risk of mode collapse',
            'Winner': 'Diffusion'
        },
        'Training Stability': {
            'Diffusion': 'Very stable (simple MSE loss)',
            'GAN': 'Unstable (min-max game)',
            'Winner': 'Diffusion'
        },
        'Sampling Speed': {
            'Diffusion': 'Slow (1000 steps, ~10-60s/image)',
            'GAN': 'Fast (1 forward pass, ~0.01s/image)',
            'Winner': 'GAN'
        },
        'Controllability': {
            'Diffusion': 'Excellent (classifier-free guidance, ControlNet)',
            'GAN': 'Limited (StyleGAN latent manipulation)',
            'Winner': 'Diffusion'
        },
        'Memory': {
            'Diffusion': 'High (store activations for backprop)',
            'GAN': 'Lower',
            'Winner': 'GAN'
        },
        'Ease of Training': {
            'Diffusion': 'Simple (one loss function)',
            'GAN': 'Complex (hyperparameter sensitive)',
            'Winner': 'Diffusion'
        },
        'Real-time Applications': {
            'Diffusion': 'Not suitable without acceleration',
            'GAN': 'Suitable',
            'Winner': 'GAN'
        }
    }
    
    print("Diffusion vs GAN Comparison:")
    print("=" * 70)
    
    for aspect, data in comparison.items():
        print(f"\n{aspect}:")
        print(f"  Diffusion: {data['Diffusion']}")
        print(f"  GAN: {data['GAN']}")
        print(f"  → Winner: {data['Winner']}")


def when_to_use_which():
    """Recommendations for choosing between diffusion and GANs"""
    
    print("\nWhen to Use Which:")
    print("=" * 60)
    
    recommendations = [
        ("Diffusion", "Highest quality required", "Text-to-image, art generation"),
        ("Diffusion", "Controllable generation", "ControlNet, inpainting"),
        ("Diffusion", "Training stability matters", "Limited compute/expertise"),
        ("GAN", "Real-time generation", "Video games, interactive apps"),
        ("GAN", "Low latency required", "Live video processing"),
        ("GAN", "Edge deployment", "Mobile, embedded devices"),
    ]
    
    print(f"{'Model':<12} {'Scenario':<30} {'Example'}")
    print("-" * 60)
    for model, scenario, example in recommendations:
        print(f"{model:<12} {scenario:<30} {example}")
```

**FID Scores Comparison (ImageNet 256x256):**

| Model | FID ↓ | Type |
|-------|-------|------|
| ADM-G (diffusion) | 2.97 | Diffusion |
| BigGAN-deep | 4.06 | GAN |
| StyleGAN-XL | 2.30 | GAN |
| DiT-XL/2 (diffusion) | 2.27 | Diffusion |

**Interview Tip:** The landscape has shifted—diffusion models now dominate high-quality image generation (DALL-E, Stable Diffusion, Midjourney). GANs remain relevant for real-time applications (NVIDIA's face generation, game engines). Key trade-off: quality/diversity (diffusion wins) vs. speed (GANs win 1000x faster). Hybrid approaches are emerging—use GAN to accelerate diffusion sampling.

---

### Question 63
- [ ] Done
**How do classifier-free guidance and text conditioning work in Stable Diffusion?**

**Answer:**

Classifier-free guidance (CFG) improves conditional generation without a separate classifier. During training, randomly drop the condition (text) and train both conditional and unconditional. During inference, extrapolate away from the unconditional prediction toward the conditional one. Text conditioning uses CLIP text encoder to convert prompts to embeddings, which are injected into U-Net via cross-attention layers.

**Classifier-Free Guidance Formula:**
$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

Where $s$ is the guidance scale (typically 7-15), $c$ is the condition, and $\emptyset$ is the null condition.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

class StableDiffusionPipeline:
    """
    Stable Diffusion with classifier-free guidance
    """
    
    def __init__(self, unet, vae, text_encoder, tokenizer, scheduler):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
    
    @torch.no_grad()
    def generate(self, prompt, negative_prompt="", 
                 guidance_scale=7.5, num_steps=50, 
                 height=512, width=512):
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description
            negative_prompt: What to avoid
            guidance_scale: CFG strength (higher = more adherence to prompt)
            num_steps: Denoising steps
        
        Returns:
            Generated image
        """
        device = self.unet.device
        
        # Step 1: Encode text prompts
        prompt_embeds = self._encode_prompt(prompt)
        negative_embeds = self._encode_prompt(negative_prompt)
        
        # Concatenate for batch processing
        text_embeddings = torch.cat([negative_embeds, prompt_embeds])
        
        # Step 2: Initialize latents (in VAE latent space, 8x smaller)
        latent_height = height // 8
        latent_width = width // 8
        latents = torch.randn(1, 4, latent_height, latent_width, device=device)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Step 3: Denoising loop with CFG
        self.scheduler.set_timesteps(num_steps)
        
        for t in self.scheduler.timesteps:
            # Duplicate latents for CFG (unconditional + conditional)
            latent_model_input = torch.cat([latents, latents])
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise (both unconditional and conditional)
            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            
            # Denoise step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Step 4: Decode latents to image
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        
        # Normalize to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        return image
    
    def _encode_prompt(self, prompt):
        """Encode text prompt using CLIP"""
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.unet.device)
        
        embeddings = self.text_encoder(tokens).last_hidden_state
        
        return embeddings


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention for text conditioning
    Image features attend to text embeddings
    """
    
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = query_dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Projections
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
    
    def forward(self, x, context):
        """
        x: (B, N, D) image features (query)
        context: (B, M, D_context) text embeddings (key, value)
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, -1).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Aggregate
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        
        return self.to_out(out)


class ConditionalUNet(nn.Module):
    """
    U-Net with cross-attention for text conditioning
    """
    
    def __init__(self, base_channels=320, context_dim=768):
        super().__init__()
        
        # Time embedding
        self.time_embed = TimeEmbedding(base_channels * 4)
        
        # Encoder blocks with cross-attention
        self.down_blocks = nn.ModuleList([
            CrossAttnDownBlock(base_channels, base_channels, context_dim),
            CrossAttnDownBlock(base_channels, base_channels * 2, context_dim),
            CrossAttnDownBlock(base_channels * 2, base_channels * 4, context_dim),
        ])
        
        # Bottleneck
        self.mid_block = MidBlock(base_channels * 4, context_dim)
        
        # Decoder blocks
        self.up_blocks = nn.ModuleList([
            CrossAttnUpBlock(base_channels * 4, base_channels * 2, context_dim),
            CrossAttnUpBlock(base_channels * 2, base_channels, context_dim),
            CrossAttnUpBlock(base_channels, base_channels, context_dim),
        ])
        
        # Output
        self.out = nn.Conv2d(base_channels, 4, 3, padding=1)
    
    def forward(self, x, t, encoder_hidden_states):
        """
        x: (B, 4, H, W) noisy latent
        t: (B,) timestep
        encoder_hidden_states: (B, 77, 768) text embeddings
        """
        t_emb = self.time_embed(t)
        
        # Encoder
        skip_connections = []
        h = x
        for block in self.down_blocks:
            h = block(h, t_emb, encoder_hidden_states)
            skip_connections.append(h)
        
        # Bottleneck
        h = self.mid_block(h, t_emb, encoder_hidden_states)
        
        # Decoder
        for block in self.up_blocks:
            skip = skip_connections.pop()
            h = block(h, skip, t_emb, encoder_hidden_states)
        
        return self.out(h)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        # Sinusoidal embedding
        half_dim = self.mlp[0].in_features
        emb = torch.exp(-np.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class CrossAttnDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, context_dim):
        super().__init__()
        self.res = ResnetBlock(in_ch, out_ch)
        self.attn = CrossAttentionBlock(out_ch, context_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    
    def forward(self, x, t_emb, context):
        h = self.res(x, t_emb)
        B, C, H, W = h.shape
        h_flat = h.flatten(2).transpose(1, 2)  # (B, H*W, C)
        h_flat = h_flat + self.attn(h_flat, context)
        h = h_flat.transpose(1, 2).view(B, C, H, W)
        return self.down(h)


class CrossAttnUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, context_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.res = ResnetBlock(in_ch * 2, out_ch)
        self.attn = CrossAttentionBlock(out_ch, context_dim)
    
    def forward(self, x, skip, t_emb, context):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        h = self.res(x, t_emb)
        B, C, H, W = h.shape
        h_flat = h.flatten(2).transpose(1, 2)
        h_flat = h_flat + self.attn(h_flat, context)
        h = h_flat.transpose(1, 2).view(B, C, H, W)
        return h


class MidBlock(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__()
        self.res1 = ResnetBlock(channels, channels)
        self.attn = CrossAttentionBlock(channels, context_dim)
        self.res2 = ResnetBlock(channels, channels)
    
    def forward(self, x, t_emb, context):
        h = self.res1(x, t_emb)
        B, C, H, W = h.shape
        h_flat = h.flatten(2).transpose(1, 2)
        h_flat = h_flat + self.attn(h_flat, context)
        h = h_flat.transpose(1, 2).view(B, C, H, W)
        return self.res2(h, t_emb)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=1280):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class CFGTraining:
    """
    Training with classifier-free guidance
    Randomly drop conditioning during training
    """
    
    def __init__(self, model, null_token_embedding, dropout_prob=0.1):
        self.model = model
        self.null_embedding = null_token_embedding  # Embedding for ""
        self.dropout_prob = dropout_prob
    
    def training_step(self, images, text_embeddings, noise, timesteps):
        """
        Training step with random condition dropout
        """
        B = images.shape[0]
        
        # Randomly drop conditioning
        mask = torch.rand(B) < self.dropout_prob
        
        # Replace dropped conditions with null embedding
        cond_embeddings = text_embeddings.clone()
        cond_embeddings[mask] = self.null_embedding.expand(mask.sum(), -1, -1)
        
        # Add noise
        noisy_images = self.add_noise(images, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_images, timesteps, cond_embeddings)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def add_noise(self, x, noise, t):
        # Standard diffusion noising
        alpha_cumprod = self.get_alpha_cumprod(t)
        return alpha_cumprod.sqrt() * x + (1 - alpha_cumprod).sqrt() * noise
```

**Guidance Scale Effects:**

| Scale | Effect |
|-------|--------|
| 1.0 | No guidance (diverse but may ignore prompt) |
| 3-5 | Balanced quality and diversity |
| 7-8 | Standard (good prompt adherence) |
| 10-15 | Strong adherence (may reduce diversity) |
| >20 | Oversaturated, artifacts |

**Interview Tip:** CFG is the key innovation enabling controllable text-to-image generation. The trick: train one model for both conditional and unconditional generation (10% dropout). At inference, compute both predictions and extrapolate. Higher guidance scale = stronger prompt following but less diversity. Cross-attention is how text conditions the U-Net—each image patch attends to relevant text tokens.

---

### Question 64
- [ ] Done
**Explain ControlNet and how it adds spatial control to diffusion models.**

**Answer:**

ControlNet adds spatial conditioning (edges, pose, depth) to pre-trained diffusion models without retraining them. It creates a trainable copy of the U-Net encoder blocks, connected via "zero convolutions" (initialized to zero). The conditioning image (e.g., Canny edges) is processed by this copy and added to the original model's features. This allows precise spatial control while preserving the original model's quality.

**ControlNet Architecture:**
```
Input Condition (edges/pose/depth)
         ↓
   [Trainable Copy of U-Net Encoder]
         ↓
   Zero Convolution (output = 0 initially)
         ↓
   Add to Original U-Net Features
         ↓
   [Frozen Original U-Net]
         ↓
      Output Image
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class ControlNet(nn.Module):
    """
    ControlNet: Adding spatial control to diffusion models
    Creates a trainable copy connected via zero convolutions
    """
    
    def __init__(self, pretrained_unet, hint_channels=3):
        super().__init__()
        
        # Freeze original model
        self.pretrained_unet = pretrained_unet
        for param in self.pretrained_unet.parameters():
            param.requires_grad = False
        
        # Create trainable copy of encoder
        self.control_encoder = deepcopy(pretrained_unet.encoder)
        for param in self.control_encoder.parameters():
            param.requires_grad = True
        
        # Input hint encoder (condition image to latent space)
        self.hint_encoder = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 4, 1),  # Match latent channels
        )
        
        # Zero convolutions for each skip connection
        self.zero_convs = nn.ModuleList([
            self._make_zero_conv(ch) for ch in [320, 640, 1280, 1280]
        ])
        
        # Zero conv for middle block
        self.middle_zero_conv = self._make_zero_conv(1280)
    
    def _make_zero_conv(self, channels):
        """Create zero-initialized convolution"""
        conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv
    
    def forward(self, latents, timestep, text_embeddings, control_hint):
        """
        Args:
            latents: (B, 4, H, W) noisy latents
            timestep: (B,) diffusion timestep
            text_embeddings: (B, 77, 768) text condition
            control_hint: (B, 3, H*8, W*8) spatial condition (e.g., Canny edges)
        
        Returns:
            Noise prediction with spatial control
        """
        # Encode hint to latent space size
        hint_latent = self.hint_encoder(control_hint)
        
        # Add hint to input latents
        controlled_latents = latents + hint_latent
        
        # Run through trainable control encoder
        control_features = self.control_encoder(
            controlled_latents, timestep, text_embeddings
        )
        
        # Apply zero convolutions
        control_outputs = []
        for feat, zero_conv in zip(control_features['skip_connections'], self.zero_convs):
            control_outputs.append(zero_conv(feat))
        
        middle_control = self.middle_zero_conv(control_features['middle'])
        
        # Run original U-Net with added control features
        noise_pred = self.pretrained_unet.forward_with_control(
            latents, timestep, text_embeddings,
            control_skip_connections=control_outputs,
            control_middle=middle_control
        )
        
        return noise_pred


class ZeroConvolution(nn.Module):
    """
    Zero-initialized convolution
    Key innovation: starts outputting zeros, gradually learns control
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              padding=kernel_size // 2)
        
        # Initialize weights and bias to zero
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)


class ControlNetUNet(nn.Module):
    """
    Modified U-Net that accepts control signals
    """
    
    def __init__(self, base_unet):
        super().__init__()
        self.encoder = base_unet.encoder
        self.middle = base_unet.middle
        self.decoder = base_unet.decoder
        self.out = base_unet.out
    
    def forward_with_control(self, x, t, context, 
                             control_skip_connections=None,
                             control_middle=None):
        """
        Forward with optional control signal injection
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Encoder
        skip_connections = []
        h = x
        for i, block in enumerate(self.encoder.blocks):
            h = block(h, t_emb, context)
            
            # Add control signal
            if control_skip_connections and i < len(control_skip_connections):
                h = h + control_skip_connections[i]
            
            skip_connections.append(h)
        
        # Middle block
        h = self.middle(h, t_emb, context)
        if control_middle is not None:
            h = h + control_middle
        
        # Decoder
        for block in self.decoder.blocks:
            skip = skip_connections.pop()
            h = block(h, skip, t_emb, context)
        
        return self.out(h)


class MultiControlNet(nn.Module):
    """
    Combine multiple ControlNets for multi-condition control
    e.g., pose + depth + edges simultaneously
    """
    
    def __init__(self, pretrained_unet, control_types):
        super().__init__()
        
        self.pretrained_unet = pretrained_unet
        
        # Create ControlNet for each condition type
        self.controlnets = nn.ModuleDict({
            ctype: ControlNet(pretrained_unet, hint_channels=channels)
            for ctype, channels in control_types.items()
        })
    
    def forward(self, latents, timestep, text_embeddings, conditions, weights=None):
        """
        Args:
            conditions: Dict[str, Tensor] mapping condition type to hint image
            weights: Dict[str, float] weights for each condition
        """
        if weights is None:
            weights = {k: 1.0 for k in conditions}
        
        # Collect control signals from each ControlNet
        all_skip_controls = [[] for _ in range(4)]  # 4 skip connection levels
        all_middle_controls = []
        
        for ctype, hint in conditions.items():
            if ctype not in self.controlnets:
                continue
            
            weight = weights.get(ctype, 1.0)
            controlnet = self.controlnets[ctype]
            
            # Get control signals
            skip_controls, middle_control = controlnet.get_control_signals(
                latents, timestep, text_embeddings, hint
            )
            
            # Accumulate with weight
            for i, sc in enumerate(skip_controls):
                all_skip_controls[i].append(weight * sc)
            all_middle_controls.append(weight * middle_control)
        
        # Sum control signals
        combined_skips = [sum(scs) for scs in all_skip_controls]
        combined_middle = sum(all_middle_controls)
        
        # Run through U-Net with combined control
        noise_pred = self.pretrained_unet.forward_with_control(
            latents, timestep, text_embeddings,
            control_skip_connections=combined_skips,
            control_middle=combined_middle
        )
        
        return noise_pred


class ControlNetConditionPreprocessors:
    """
    Preprocessors for different control conditions
    """
    
    @staticmethod
    def canny_edge(image, low_threshold=100, high_threshold=200):
        """Extract Canny edges"""
        import cv2
        
        if isinstance(image, torch.Tensor):
            image = (image.permute(1, 2, 0).numpy() * 255).astype('uint8')
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Stack to 3 channels
        edges = np.stack([edges] * 3, axis=-1)
        
        return torch.from_numpy(edges).permute(2, 0, 1).float() / 255.0
    
    @staticmethod
    def openpose_skeleton(image, pose_model):
        """Extract human pose skeleton"""
        with torch.no_grad():
            keypoints = pose_model(image)
        
        # Render skeleton as image
        skeleton_image = render_skeleton(keypoints, image.shape[-2:])
        
        return skeleton_image
    
    @staticmethod
    def depth_map(image, depth_model):
        """Extract depth using MiDaS or similar"""
        with torch.no_grad():
            depth = depth_model(image)
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth.expand(3, -1, -1)
    
    @staticmethod
    def segmentation_map(image, seg_model):
        """Extract semantic segmentation"""
        with torch.no_grad():
            seg = seg_model(image)
        
        # Convert to color-coded map
        seg_colored = colorize_segmentation(seg)
        
        return seg_colored


class ControlNetTraining:
    """
    Training procedure for ControlNet
    """
    
    def __init__(self, controlnet, frozen_unet, optimizer):
        self.controlnet = controlnet
        self.frozen_unet = frozen_unet
        self.optimizer = optimizer
    
    def training_step(self, images, conditions, text_embeddings, noise, timesteps):
        """
        Training step
        Only ControlNet parameters are updated
        """
        self.optimizer.zero_grad()
        
        # Add noise to get noisy latents
        noisy_latents = self.add_noise(images, noise, timesteps)
        
        # Get control-influenced prediction
        noise_pred = self.controlnet(
            noisy_latents, timesteps, text_embeddings, conditions
        )
        
        # Standard diffusion loss
        loss = F.mse_loss(noise_pred, noise)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def controlnet_conditions():
    """Available ControlNet condition types"""
    
    print("ControlNet Condition Types:")
    print("=" * 60)
    
    conditions = [
        ("Canny Edge", "Edge detection map", "Outline preservation"),
        ("Depth (MiDaS)", "Depth estimation", "3D structure control"),
        ("OpenPose", "Human skeleton", "Pose control"),
        ("Segmentation", "Semantic segments", "Composition control"),
        ("Normal Map", "Surface normals", "Lighting/shape"),
        ("Scribble", "User drawings", "Creative sketches"),
        ("HED", "Soft edge detection", "Style transfer"),
        ("Line Art", "Line drawing", "Illustration style"),
    ]
    
    print(f"{'Condition':<15} {'Input':<25} {'Use Case'}")
    print("-" * 60)
    for cond, input_type, use_case in conditions:
        print(f"{cond:<15} {input_type:<25} {use_case}")
```

**Key Design Choices:**

| Choice | Reason |
|--------|--------|
| Trainable copy | Preserve original model quality |
| Zero convolution | Start with no effect, learn gradually |
| Skip connections | Control at multiple resolutions |
| Frozen base model | No catastrophic forgetting |

**Interview Tip:** ControlNet's genius is the zero convolution—by initializing to zero, the model starts outputting exactly what the original model would, then gradually learns to add control. This prevents training instability. Multiple ControlNets can be combined with different weights for multi-condition control (e.g., pose + depth). The architecture is now standard for controllable generation.

---

### Question 65
- [ ] Done
**What techniques speed up diffusion model inference (DDIM, DPM-Solver, distillation)?**

**Answer:**

Diffusion inference is slow (1000 steps). Acceleration techniques: (1) DDIM—deterministic sampling allowing larger step sizes (50 steps vs 1000), (2) DPM-Solver—higher-order ODE solvers for fewer steps (20-25 steps), (3) distillation—train a student model to predict multiple steps at once (1-4 steps), (4) latent diffusion—operate in compressed latent space (8x smaller), and (5) model pruning/quantization.

**Speed Comparison:**

| Method | Steps | Time | Quality |
|--------|-------|------|---------|
| DDPM (original) | 1000 | ~60s | Baseline |
| DDIM | 50 | ~3s | ~Same |
| DPM-Solver++ | 20 | ~1.2s | ~Same |
| LCM (distilled) | 4 | ~0.3s | Slight decrease |
| SDXL Turbo | 1 | ~0.1s | Lower |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ========== DDIM: Denoising Diffusion Implicit Models ==========

class DDIM:
    """
    DDIM: Deterministic sampling with fewer steps
    Key insight: DDPM's stochastic term is optional
    """
    
    def __init__(self, model, alphas_cumprod, timesteps=1000):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps
    
    @torch.no_grad()
    def sample(self, shape, num_inference_steps=50, eta=0.0):
        """
        DDIM sampling with configurable steps
        
        Args:
            shape: Output shape (B, C, H, W)
            num_inference_steps: Number of denoising steps (can be much less than 1000)
            eta: Stochasticity (0 = deterministic, 1 = DDPM)
        
        Returns:
            Generated samples
        """
        device = next(self.model.parameters()).device
        
        # Create subset of timesteps
        step_ratio = self.timesteps // num_inference_steps
        timesteps = torch.arange(0, self.timesteps, step_ratio).flip(0)
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t_batch)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else 1.0
            
            # DDIM update (deterministic when eta=0)
            x = self._ddim_step(x, noise_pred, t, alpha_t, alpha_prev, eta)
        
        return x
    
    def _ddim_step(self, x, noise_pred, t, alpha_t, alpha_prev, eta):
        """
        DDIM update step
        
        x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + 
                  sqrt(1 - alpha_{t-1} - sigma^2) * noise_pred +
                  sigma * noise
        """
        # Predict x_0
        x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = x0_pred.clamp(-1, 1)  # Clip for stability
        
        # Compute sigma (stochasticity)
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * \
                torch.sqrt(1 - alpha_t / alpha_prev)
        
        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt
        
        if sigma > 0:
            noise = torch.randn_like(x)
            x_prev = x_prev + sigma * noise
        
        return x_prev


# ========== DPM-SOLVER: Fast ODE Solver ==========

class DPMSolver:
    """
    DPM-Solver++: Higher-order ODE solver for faster sampling
    Can achieve good quality in 15-25 steps
    """
    
    def __init__(self, model, alphas_cumprod, order=2):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.order = order  # 1, 2, or 3
        
        # Precompute lambda (log-SNR)
        self.lambdas = torch.log(alphas_cumprod / (1 - alphas_cumprod)) / 2
    
    @torch.no_grad()
    def sample(self, shape, num_steps=20):
        """
        DPM-Solver++ sampling
        
        Args:
            shape: Output shape
            num_steps: Number of steps (typically 15-25)
        """
        device = next(self.model.parameters()).device
        
        # Create timestep schedule
        timesteps = self._get_timestep_schedule(num_steps)
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Store previous model outputs for multi-step methods
        model_outputs = []
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Get model output (noise prediction)
            noise_pred = self.model(x, t_batch)
            model_outputs.append(noise_pred)
            
            # Apply appropriate order solver
            if i == 0 or self.order == 1:
                x = self._first_order_update(x, noise_pred, t, timesteps[i + 1] if i < len(timesteps) - 1 else 0)
            elif self.order >= 2 and len(model_outputs) >= 2:
                x = self._second_order_update(x, model_outputs[-2:], timesteps[i-1:i+2])
            
            # Keep only last few outputs
            if len(model_outputs) > self.order:
                model_outputs.pop(0)
        
        return x
    
    def _first_order_update(self, x, noise_pred, t, t_next):
        """First-order update (equivalent to DDIM)"""
        alpha_t = self.alphas_cumprod[t]
        alpha_next = self.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0)
        
        x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
        
        return x_next
    
    def _second_order_update(self, x, noise_preds, timesteps_window):
        """
        Second-order update using linear multistep
        More accurate than first-order
        """
        # Use previous noise prediction for better estimate
        t_prev, t_curr, t_next = timesteps_window
        noise_prev, noise_curr = noise_preds
        
        # Linear extrapolation of noise
        lambda_prev = self.lambdas[t_prev]
        lambda_curr = self.lambdas[t_curr]
        lambda_next = self.lambdas[t_next] if t_next > 0 else torch.tensor(float('inf'))
        
        h = lambda_curr - lambda_prev
        h_next = lambda_next - lambda_curr
        
        # Second-order correction
        r = h_next / h
        noise_corrected = (1 + 0.5 * r) * noise_curr - 0.5 * r * noise_prev
        
        return self._first_order_update(x, noise_corrected, t_curr, t_next)
    
    def _get_timestep_schedule(self, num_steps):
        """Create timestep schedule (uniform in lambda space)"""
        lambda_max = self.lambdas.max()
        lambda_min = self.lambdas.min()
        
        lambdas = torch.linspace(lambda_max, lambda_min, num_steps + 1)
        
        timesteps = []
        for l in lambdas:
            idx = (self.lambdas - l).abs().argmin()
            timesteps.append(idx.item())
        
        return timesteps


# ========== DISTILLATION: Progressive & Consistency Distillation ==========

class ConsistencyDistillation:
    """
    Consistency Models: Train to map any x_t directly to x_0
    Enables 1-4 step generation
    """
    
    def __init__(self, teacher_model, student_model, alphas_cumprod):
        self.teacher = teacher_model
        self.student = student_model
        self.alphas_cumprod = alphas_cumprod
        
        # EMA of student for consistency target
        self.ema_student = self._create_ema(student_model)
    
    def training_step(self, x_0):
        """
        Consistency distillation training step
        
        Key idea: f(x_t, t) should equal f(x_{t-1}, t-1) for any t
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(1, 1000, (B,), device=device)
        t_next = t - 1  # Adjacent timestep
        
        # Add noise to get x_t
        noise = torch.randn_like(x_0)
        x_t = self._add_noise(x_0, noise, t)
        
        # Get teacher's one-step prediction: x_t -> x_{t-1}
        with torch.no_grad():
            teacher_noise_pred = self.teacher(x_t, t)
            x_t_minus_1 = self._one_step_denoise(x_t, teacher_noise_pred, t, t_next)
        
        # Student should map both to same prediction
        student_pred_t = self.student(x_t, t)
        
        with torch.no_grad():
            # Use EMA student for target (stability)
            target_pred = self.ema_student(x_t_minus_1, t_next)
        
        # Consistency loss
        loss = F.mse_loss(student_pred_t, target_pred)
        
        return loss
    
    def _add_noise(self, x, noise, t):
        alpha = self.alphas_cumprod[t][:, None, None, None]
        return torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
    def _one_step_denoise(self, x_t, noise_pred, t, t_next):
        alpha_t = self.alphas_cumprod[t][:, None, None, None]
        alpha_next = self.alphas_cumprod[t_next][:, None, None, None]
        
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
        
        return x_next
    
    def _create_ema(self, model, decay=0.999):
        ema = deepcopy(model)
        for param in ema.parameters():
            param.requires_grad = False
        return ema


class ProgressiveDistillation:
    """
    Progressive distillation: Halve steps iteratively
    1000 -> 500 -> 250 -> ... -> 4 steps
    """
    
    def __init__(self, model, alphas_cumprod):
        self.current_model = model
        self.alphas_cumprod = alphas_cumprod
        self.current_steps = 1000
    
    def distill_to_half_steps(self, dataloader, num_epochs=10):
        """
        Train student to match teacher with half the steps
        """
        target_steps = self.current_steps // 2
        student = deepcopy(self.current_model)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        
        for epoch in range(num_epochs):
            for x_0 in dataloader:
                loss = self._distillation_step(student, x_0, target_steps)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Update for next round
        self.current_model = student
        self.current_steps = target_steps
        
        return student
    
    def _distillation_step(self, student, x_0, target_steps):
        """
        Student does one step to match teacher's two steps
        """
        device = x_0.device
        B = x_0.shape[0]
        
        # Sample timestep in student's schedule
        t_student = torch.randint(0, target_steps, (B,), device=device)
        
        # Map to teacher's timesteps
        t_teacher = t_student * 2
        t_teacher_mid = t_teacher - 1
        t_teacher_end = t_teacher - 2
        
        # Add noise
        noise = torch.randn_like(x_0)
        x_t = self._add_noise(x_0, noise, t_teacher)
        
        # Teacher: two steps
        with torch.no_grad():
            teacher_pred_1 = self.current_model(x_t, t_teacher)
            x_mid = self._denoise_step(x_t, teacher_pred_1, t_teacher, t_teacher_mid)
            
            teacher_pred_2 = self.current_model(x_mid, t_teacher_mid)
            x_end_teacher = self._denoise_step(x_mid, teacher_pred_2, t_teacher_mid, t_teacher_end)
        
        # Student: one step
        student_pred = student(x_t, t_student)
        x_end_student = self._denoise_step(x_t, student_pred, t_teacher, t_teacher_end)
        
        # Match outputs
        loss = F.mse_loss(x_end_student, x_end_teacher)
        
        return loss


# ========== LATENT DIFFUSION (LDM) ==========

class LatentDiffusion:
    """
    Latent Diffusion: Operate in compressed latent space
    8x spatial compression = 64x fewer pixels
    """
    
    def __init__(self, vae, diffusion_model):
        self.vae = vae  # Encoder-decoder
        self.diffusion = diffusion_model
    
    def encode(self, images):
        """Compress images to latent space"""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def decode(self, latents):
        """Decode latents back to images"""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return images
    
    def sample(self, shape, num_steps=50):
        """
        Generate in latent space (8x smaller)
        """
        # Latent shape is 8x smaller spatially
        latent_shape = (shape[0], 4, shape[2] // 8, shape[3] // 8)
        
        # Run diffusion in latent space (much faster!)
        latents = self.diffusion.sample(latent_shape, num_steps)
        
        # Decode to image space
        images = self.decode(latents)
        
        return images


def speedup_summary():
    """Summary of speedup techniques"""
    
    print("Diffusion Speedup Techniques:")
    print("=" * 70)
    
    techniques = [
        ("DDIM", "Deterministic sampling", "1000→50 steps", "~20x"),
        ("DPM-Solver++", "Higher-order ODE solver", "1000→20 steps", "~50x"),
        ("LCM", "Consistency distillation", "1000→4 steps", "~250x"),
        ("SDXL Turbo", "Adversarial distillation", "1000→1 step", "~1000x"),
        ("Latent Diffusion", "Compressed space", "64x fewer pixels", "~8x"),
        ("FP16/TensorRT", "Optimized inference", "Same steps", "~2-3x"),
    ]
    
    print(f"{'Technique':<18} {'Method':<25} {'Reduction':<18} {'Speedup'}")
    print("-" * 70)
    for tech, method, reduction, speedup in techniques:
        print(f"{tech:<18} {method:<25} {reduction:<18} {speedup}")
```

**Interview Tip:** Know the trade-offs: DDIM is simple and works well at 50 steps; DPM-Solver++ is state-of-the-art for quality at 20-25 steps; distillation (LCM, Turbo) enables 1-4 steps but requires additional training and may lose some quality. Latent diffusion (used by Stable Diffusion) is orthogonal—it reduces spatial dimensions, making each step faster. For production, combine latent diffusion + DPM-Solver++ + TensorRT optimization.

---


---

# --- StyleGAN Questions (from 47_stylegan) ---

# StyleGAN / StyleGAN2 / StyleGAN3 Interview Questions - Theory Questions

## Question 1
- [ ] Done

**Explain generator architecture of original StyleGAN.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Define style mapping network and AdaIN.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Explain progressive growing in StyleGAN.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Describe path-length regularisation in StyleGAN2.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Explain weight demodulation and its purpose.**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Discuss removal of blob-shaped artifacts in StyleGAN2-ADA.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Explain Adaptive Discriminator Augmentation (ADA).**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Compare StyleGAN vs. ProGAN.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**Describe style mixing regularisation.**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Explain noise injection and stochastic variation.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Discuss truncation trick and ψ parameter.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**Describe interface-GAN and latent editing.**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**Explain StyleGAN inversion methods (e.g., e4e, pSp).**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Compare pixel-wise vs. channel-wise noise.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**Discuss StyleGAN3's alias-free design.**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**Explain 2-D fourier features in StyleGAN3.**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Describe continuous depth translation invariance.**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**Discuss out-of-distribution generalisation and datasets.**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**Compare FID vs. Inception Score for generative quality.**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**Explain projector for identity-preserving face editing.**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**Discuss attribute manipulations with GANSpace.**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Explain GAN dissection and unit visualisation.**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**Describe style generalisation across datasets.**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**Explain fine-tuning StyleGAN for few-shot generation.**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**Discuss GAN-based data augmentation for downstream tasks.**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**Explain CLIP-guided latent editing.**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**Describe training StyleGAN on non-aligned data.**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**Explain hyperspherical latent space and spherical embeddings.**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**Discuss generative prior for image restoration.**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**Explain StyleGAN for text-to-image via multi-modal alignment.**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**Describe Co-mod-GAN bridging StyleGAN with segmentation.**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**Explain 3-D aware StyleGAN (StyleNeRF).**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**Discuss hyper-style for high-quality editing.**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**Explain adversarial robustness of StyleGAN.**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**Describe model compression and distillation for mobile GANs.**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**Discuss fairness and demographic bias in face GANs.**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**Explain integrating diffusion loss into StyleGAN training.**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**Describe watermarking generated images.**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**Explain infinite zoom and out-painting with StyleGAN.**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**Discuss licensing and ethical concerns of dataset usage.**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**Explain style mixing latent interpolation.**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**Describe transferring StyleGAN generator between domains.**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**Explain segmentation-guided GANs for part mixing.**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**Discuss real-time inference optimisation.**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**Explain zero-shot generative domain adaptation.**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**Discuss adversarial detection of GAN-generated images.**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**Explain quality tuning via per-layer noise magnitude.**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**Predict future of StyleGAN-like architectures.**

**Answer:** _[To be filled]_

---


---

# --- CycleGAN Questions (from 48_cyclegan) ---

# CycleGAN Interview Questions - Theory Questions

## Question 1
- [ ] Done

**Describe adversarial loss and cycle-consistency loss.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Discuss identity loss and when needed.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Explain role of PatchGAN discriminator.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Describe generator architecture in CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Explain training stability challenges in CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Discuss mode collapse and mitigation.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Explain mapping ambiguity problem.**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Describe data augmentation strategies.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**Explain CycleGAN for style transfer vs. domain transfer.**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Discuss applications in medical imaging.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Explain CycleGAN in video translation and temporal consistency.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**Discuss evaluation with FID and LPIPS.**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**Explain spectral normalization in CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Describe attention-guided CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**Explain multi-domain CycleGAN (StarGAN).**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**Discuss geometry-consistent CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Explain cut (contrastive unpaired translation) vs. CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**Describe cyclic perceptual loss.**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**Discuss memory use with high-res images.**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**Explain one-sided label smoothing in discriminators.**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**Discuss CubeGAN for 3-D domain translation.**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Explain partial weight sharing across generators.**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**Describe training with mixed precision for CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**Explain patch-based training vs. full-image.**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**Discuss CycleGAN for style untransferability issues.**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**Explain domain adaptation using CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**Describe unsupervised depth transfer via CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**Explain CycleGAN failure cases.**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**Discuss cyclic consistency with contrastive loss.**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**Explain multi-cycle synergy (double cycle).**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**Discuss GAN inversion for CycleGAN editing.**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**Describe face ageing with CycleGAN variations.**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**Explain dual learning relationship.**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**Discuss training time reduction via teacher-student.**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**Explain zero-shot translation with pre-trained CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**Describe patch-swap enhancements.**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**Explain dynamical cropping for training.**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**Discuss CycleGAN for audio domain.**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**Explain progressive growing for CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**Describe multi-scale discriminators.**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**Discuss regularization techniques specific to CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**Explain semantic consistency losses.**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**Describe cross-domain feature alignment.**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**Discuss conditional CycleGAN variants.**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**Explain deployment considerations for CycleGAN.**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**Describe quality assessment metrics for unpaired translation.**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**Discuss ethical considerations in domain transfer.**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**Explain recent improvements to CycleGAN architecture.**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**Predict future directions for unpaired image translation.**

**Answer:** _[To be filled]_

---


---

# --- Pix2Pix Questions (from 49_pix2pix) ---

# Pix2Pix Interview Questions - Theory Questions

## Question 1
- [ ] Done

**Explain conditional GAN and its paired training data.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Describe U-Net generator in Pix2Pix.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Explain PatchGAN discriminator role.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Discuss L1 reconstruction loss vs. adversarial loss balance.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Explain importance of paired datasets (e.g., edges-to-photo).**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Discuss limitations when pairs unavailable.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Explain multi-scale discriminator variant.**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Describe Pix2PixHD improvements.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**Explain instance normalization benefits.**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Discuss pix2pix for style transfer.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Explain semantic label to photo translation.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**Discuss photorealism and tiling for high-res.**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**Explain total variation loss addition.**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Describe domain adaptation from synthetic to real.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**Explain attention mechanism integration.**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**Discuss paired depth-to-RGB translation.**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Explain training schedule and progressive growing.**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**Discuss evaluation metrics for Pix2Pix.**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**Explain pix2pix for medical segmentation masks translation.**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**Describe interactive editing with pix2pix.**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**Explain zero-shot pix2pix adaptation (pix2pix-Zero).**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Discuss use in face frontalization.**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**Explain energy-based adversarial training modifications.**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**Describe env-to-map translation in autonomous driving.**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**Explain identity preservation constraints.**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**Discuss GAN illness and artifact removal.**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**Explain pix2pix for anime line-art colorization.**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**Describe differentiable rendering supervisory signals.**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**Explain cross-domain escape.**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**Discuss conditional batch norm.**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**Explain interactive scribble-to-image generation.**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**Describe UNet++ deep supervision variant.**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**Discuss mixup strategies for conditional GAN.**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**Explain quantisation for mobile deployment.**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**Describe cross-modal (sketch-to-sound) pix2pix.**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**Discuss training paired dataset size effects.**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**Explain curriculum learning in pix2pix.**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**Describe noise injection for improved diversity.**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**Explain spectral norm in generator.**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**Discuss pix2pix in satellite imagery translation.**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**Explain adversarial perceptual loss.**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**Describe use in data augmentation for segmentation.**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**Discuss semi-supervised pix2pix with limited pairs.**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**Explain mask-guided pix2pix.**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**Describe safety filtering of generated images.**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**Discuss dual discriminators for local and global realism.**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**Explain diffusion-based pix2pix adaptation.**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**Describe physics-guided pix2pix for scientific images.**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**Discuss future directions in conditional GANs.**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**Predict pix2pix's role amid diffusion models.**

**Answer:** _[To be filled]_

---


---

# --- Diffusion Models Questions (from 52_diffusion_models) ---

# Diffusion Models - Theory Questions

## Question 1
- [ ] Done
**Explain forward diffusion process and noise scheduling.**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**Describe reverse diffusion and denoising process.**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**Explain DDPM (Denoising Diffusion Probabilistic Models).**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**Describe DDIM (Denoising Diffusion Implicit Models) advantages.**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**Explain U-Net architecture in diffusion models.**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**Describe classifier guidance vs. classifier-free guidance.**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**Explain latent diffusion models (Stable Diffusion).**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**Describe VAE encoder-decoder in latent space.**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**Explain CLIP text encoding for conditioning.**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**Describe noise prediction vs. score matching.**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**Explain timestep embedding and sinusoidal encoding.**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**Describe attention mechanisms in diffusion U-Net.**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**Explain sampling strategies (DDPM, DDIM, DPM-Solver).**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**Compare deterministic vs. stochastic sampling.**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**Describe inpainting with diffusion models.**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**Explain outpainting and image extension.**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**Describe ControlNet for spatial conditioning.**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**Explain IP-Adapter for image prompting.**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**Compare different noise schedules (linear, cosine).**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**Describe multi-scale diffusion architectures.**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**Explain video diffusion models.**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**Describe 3D diffusion for shape generation.**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**Explain audio diffusion models.**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**Compare continuous vs. discrete timesteps.**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**Describe training objective and loss functions.**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**Explain gradient accumulation strategies.**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**Describe mixed precision training benefits.**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**Explain EMA (Exponential Moving Average) in training.**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**Compare memory requirements during training/inference.**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**Describe progressive distillation techniques.**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**Explain consistency models for fast sampling.**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**Describe edit-friendly inversions (DDIM inversion).**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**Explain self-attention guidance.**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**Describe cascading diffusion models.**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**Discuss spectrogram diffusion for audio generation.**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**Explain safe completions via policy guidance.**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**Describe hardware acceleration (FP8) for diffusion.**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**Explain mixture of experts diffusion.**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**Discuss evaluation metrics (CLIP-FID).**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**Explain diffusion vs. GANs advantages.**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**Describe computational cost mitigation.**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**Discuss legal considerations of dataset copyright.**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**Explain multi-modal diffusion (image+depth).**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**Describe timeline of diffusion research.**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**Explain diffusion for super-resolution.**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**Discuss slot diffusion for object compositionality.**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**Explain zero-shot human motion diffusion.**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**Describe guided diffusion in RL policy sampling.**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**Predict future of diffusion in content creation.**

**Answer:** _To be filled_

---


---

# --- NeRF Questions (from 53_nerf) ---

# Neural Radiance Fields (NeRF) - Theory Questions

## Question 1
- [ ] Done
**Explain volumetric rendering equation used in NeRF.**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**Describe positional encoding and high-frequency mapping.**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**Explain hierarchical sampling (coarse and fine networks).**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**Discuss overfitting to a single scene.**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**Explain view synthesis from posed images.**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**Describe inverse rendering scenario.**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**Explain NeRF-in-the-wild for unknown cameras.**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**Discuss accelerating NeRF via mip-NeRF.**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**Explain Instant-NGP's hash grid encoding.**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**Describe Plenoxels (sparse voxels).**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**Discuss PlenOctrees for real-time view generation.**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**Explain dynamic NeRF for time-varying scenes.**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**Describe implicit vs. explicit representations.**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**Explain integrating normals estimation.**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**Discuss NeRF for relighting.**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**Explain depth supervision integration.**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**Describe pose estimation with NeRF (iNeRF).**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**Discuss neural scene graphs.**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**Explain anti-aliasing in mip-NeRF-360.**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**Describe oriented NeRF for novel view extrapolation.**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**Explain semantic NeRF with multi-task loss.**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**Discuss Radiance Fields for humans (NeRFies).**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**Explain Gaussian Splatting acceleration idea.**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**Describe importance sampling strategies.**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**Explain memory footprint challenges.**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**Discuss VR/AR application pipelines.**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**Explain physically-based NeRF to model BRDF.**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**Describe editing NeRF with local rigging.**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**Explain multiview supervision number needed.**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**Discuss combining NeRF and LiDAR.**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**Explain 3-DGS vs. NeRF differences (see next section).**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**Describe compression of NeRF models.**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**Explain CODEC avatars from NeRF.**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**Discuss generative NeRF (GenerNeRF).**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**Explain distilling NeRF into mesh.**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**Describe neural point light field.**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**Explain path guiding in NeRF rendering.**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**Discuss training acceleration with forward-facing scenes.**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**Explain semantic editing brush for NeRF.**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**Describe uncertainty estimation in NeRF.**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**Explain gradient scaling of hash encodings.**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**Discuss NeRF limitations outdoors.**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**Explain neural reflectance fields.**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**Describe spectral NeRF for wavelength-dependent scenes.**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**Discuss privacy considerations of capturing scenes.**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**Explain differentiable SLAM with NeRF integration.**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**Describe knowledge distillation to Gaussian Splatting.**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**Discuss future hardware (RTX, tensor cores).**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**Explain NeRF for microscopy.**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**Predict future real-time NeRF breakthroughs.**

**Answer:** _To be filled_

---


---

# --- 3D Gaussian Splatting Questions (from 54_3d_gaussian_splatting) ---

# 3D Gaussian Splatting - Theory Questions

## Question 1
- [ ] Done
**Explain concept of representing a scene as millions of 3-D Gaussians.**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**Describe ellipsoidal Gaussian parameters (means, covariances, opacities).**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**Explain rasterisation vs. volumetric integration for splats.**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**Discuss differentiable splatting render pipeline.**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**Explain data capture and camera calibration requirements.**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**Compare rendering speed of 3-DGS to NeRF.**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**Describe memory footprint differences.**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**Explain training objective for opacity and color.**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**Discuss adaptive density pruning of Gaussians.**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**Explain hierarchical Gaussians.**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**Compare visual quality metrics to Instant-NGP.**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**Explain fine-to-coarse training schedule.**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**Discuss handling of thin structures.**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**Explain foveated rendering possibilities.**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**Compare novel view extrapolation accuracy.**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**Discuss scalability to city-scale scenes.**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**Explain interaction (editing) ease vs. NeRF.**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**Describe progressive streaming of Gaussian data.**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**Explain integrating dynamic motion in 3-DGS.**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**Discuss GPU pipeline implementation specifics.**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**Explain alias-free splatting filter.**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**Compare radiance vs. surface representation.**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**Explain BRDF modelling with 3-DGS.**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**Discuss integration into game engines (Unity/UE).**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**Explain photogrammetry vs. 3-DGS pipelines.**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**Describe compression (quantisation of parameters).**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**Explain silhouette supervision improvements.**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**Discuss limitations in extreme lighting changes.**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**Explain combining Gaussians with mesh proxy.**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**Compare training time to NeRF techniques.**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**Explain memory bandwidth vs. compute trade-off.**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**Discuss gradient flow stability in splatting.**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**Explain usage for real-time telepresence.**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**Describe model-based vs. model-free editing.**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**Explain hyper-parameter for Gaussian scale adaptation.**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**Discuss depth-sensor fused Gaussian models.**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**Explain zero-shot generalisation across scenes.**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**Compare transparency handling in Gaussians vs. NeRF.**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**Explain multi-resolution octree of Gaussians.**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**Discuss research on neural Gaussian fields.**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**Explain plug-and-play relighting.**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**Describe evaluation datasets used in papers.**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**Explain limitations of current GPU rasterisers.**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**Discuss alignment of Gaussians using ICP.**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**Explain anti-aliasing by covariance scaling.**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**Compare gradient memory of NeRF vs. 3-DGS.**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**Discuss potential hybrid models combining both.**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**Explain open-source implementations (e.g., graphics-knt).**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**Describe future hardware acceleration possibilities.**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**Summarise key advantages and open challenges relative to NeRF.**

**Answer:** _To be filled_

---
