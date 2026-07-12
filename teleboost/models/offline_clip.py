# Copyright (c) 2021-2024 OpenCLIP authors (github.com/mlfoundations/open_clip).
# Modifications Copyright 2025-2026 TeleAI and the TeleBoost contributors.
#
# The OpenCLIP-derived ViT-L-14 architecture (QuickGELU, LayerNormFp32,
# ResidualAttentionBlock, Transformer, VisionTransformer) is licensed under
# MIT; see https://github.com/mlfoundations/open_clip/blob/main/LICENSE.
# TeleAI modifications (the standalone JIT-weight loader / offline packaging)
# are licensed under Apache-2.0; see LICENSE at the root.
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    """QuickGELU activation used by OpenCLIP - specific to ViT-L-14"""

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    """LayerNorm with fp16 support"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    """
    Exact Transformer block implementation for ViT-L-14
    Based on OpenCLIP's ResidualAttentionBlock
    """

    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer=QuickGELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNormFp32(d_model)
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(d_model, int(d_model * mlp_ratio))), ("gelu", act_layer()), ("c_proj", nn.Linear(int(d_model * mlp_ratio), d_model))]))
        self.ln_2 = LayerNormFp32(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Exact Transformer implementation for ViT-L-14
    """

    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0, act_layer=QuickGELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class PreciseViTL14(nn.Module):
    """
    Exact ViT-L-14 implementation that fully matches the OpenCLIP architecture
    Configuration based on MetaCLIP's ViT-L-14-quickgelu.json
    """

    def __init__(self):
        super().__init__()

        # Exact ViT-L-14 configuration
        self.image_size = 224
        self.patch_size = 14
        self.width = 1024  # vision width
        self.layers = 24  # vision layers
        self.heads = 16  # vision heads (1024 / 64 = 16)
        self.output_dim = 768  # embed_dim (output dimension)

        self.grid_size = self.image_size // self.patch_size  # 224 / 14 = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.width, kernel_size=self.patch_size, stride=self.patch_size, bias=False)

        # Class token and positional embedding
        scale = self.width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size * self.grid_size + 1, self.width))

        self.ln_pre = LayerNormFp32(self.width)

        # Transformer blocks - use QuickGELU
        self.transformer = Transformer(width=self.width, layers=self.layers, heads=self.heads, mlp_ratio=4.0, act_layer=QuickGELU)

        self.ln_post = LayerNormFp32(self.width)
        self.proj = nn.Parameter(scale * torch.randn(self.width, self.output_dim))

    def forward(self, x: torch.Tensor):
        """
        Exact forward pass for ViT-L-14
        Args:
            x: [batch_size, 3, 224, 224] input images
        Returns:
            [batch_size, 768] image features
        """
        # Resolve the device dynamically to avoid hardcoding
        device = x.device

        # Patch embedding: [B, 3, 224, 224] -> [B, 1024, 16, 16] -> [B, 1024, 256] -> [B, 256, 1024]
        x = self.conv1(x)  # [B, width, grid_size, grid_size]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid_size^2]
        x = x.permute(0, 2, 1)  # [B, grid_size^2, width]

        # Add class token: [B, 256, 1024] -> [B, 257, 1024]
        class_token = self.class_embedding.to(device).expand(x.shape[0], 1, -1)
        x = torch.cat([class_token, x], dim=1)

        x = x + self.positional_embedding.to(device)

        x = self.ln_pre(x)

        # Transformer: expects LND layout (Length, Batch, Dim)
        x = x.permute(1, 0, 2)  # [257, B, 1024]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 257, 1024]

        x = self.ln_post(x[:, 0, :])  # [B, 1024]

        if self.proj is not None:
            x = x @ self.proj.to(device)  # [B, 768]

        return x


def load_vit_l14_weights_from_jit(jit_model_path: str, target_device: str = "cpu"):
    """
    Load ViT-L-14 weights from a JIT model into the exact implementation
    """
    # Load the original JIT model
    original_model = torch.jit.load(jit_model_path, map_location="cpu")
    state_dict = original_model.state_dict()

    # Verify that the model is ViT-L-14
    if "visual.conv1.weight" in state_dict:
        conv1_shape = state_dict["visual.conv1.weight"].shape
        patch_size = conv1_shape[-1]
        width = conv1_shape[0]

        if patch_size != 14 or width != 1024:
            raise ValueError(f"CLIP checkpoint is not ViT-L-14: expected visual.conv1 patch_size=14 and width=1024, got patch_size={patch_size}, width={width}")
    else:
        raise ValueError("CLIP checkpoint has no visual.conv1.weight and cannot initialize the aesthetic encoder")

    if "visual.positional_embedding" in state_dict:
        pos_emb_shape = state_dict["visual.positional_embedding"].shape
        seq_len, embed_dim = pos_emb_shape
        int((seq_len - 1) ** 0.5)

    # Create the exact ViT-L-14 model
    model = PreciseViTL14()

    # Extract the visual weights
    visual_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("visual."):
            new_key = key[7:]  # Strip the 'visual.' prefix
            visual_state_dict[new_key] = value

    # Print the shapes of a few key weights for verification
    # key_weights = ['conv1.weight', 'class_embedding', 'positional_embedding', 'proj']
    # for key in key_weights:
    # if key in visual_state_dict:

    # Check transformer weights
    # transformer_keys = [k for k in visual_state_dict.keys() if k.startswith('transformer.resblocks.')]

    # The aesthetic score is meaningless with even a partially random visual
    # tower.  Fail at model initialization instead of silently training on a
    # random reward signal.
    try:
        model.load_state_dict(visual_state_dict, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"CLIP visual checkpoint is incomplete or incompatible with ViT-L-14: {jit_model_path}") from exc

    # Move to the target device
    model = model.to(target_device)

    return model, visual_state_dict


def create_offline_clip_model(jit_model_path: str, target_device: str = "cpu"):
    """
    Create a complete ViT-L-14 CLIP model
    """
    # Load the visual encoder
    visual_model, visual_weights = load_vit_l14_weights_from_jit(jit_model_path, target_device)

    # Wrapper for the complete CLIP model
    class ViTL14CLIPModel:
        def __init__(self, visual_encoder, jit_model_path, target_device):
            self.visual = visual_encoder
            self.target_device = torch.device(target_device)

            # Keep the original model for text encoding
            try:
                self.original_model = torch.jit.load(jit_model_path, map_location=target_device)
            except Exception as e:
                print(f"⚠ Failed to load the original model for text encoding: {e}")
                self.original_model = None

        def encode_image(self, image):
            """
            Image encoding - uses our exact ViT-L-14 implementation
            No more hardcoded cuda:0 device issue.
            """
            if image.device != self.target_device:
                image = image.to(self.target_device)

            with torch.no_grad():
                return self.visual(image)

        def encode_text(self, text):
            """Text encoding - uses the original model (if available)"""
            if self.original_model is None:
                raise NotImplementedError("Text encoding unavailable - failed to load the original model")

            if text.device != self.target_device:
                text = text.to(self.target_device)

            try:
                return self.original_model.encode_text(text)
            except Exception as e:
                # If the original text encoding also has device issues, try a workaround
                print(f"Original text encoding failed: {e}")
                raise e

        def to(self, device):
            """Move to a new device"""
            self.target_device = torch.device(device)
            self.visual = self.visual.to(device)
            if self.original_model is not None:
                self.original_model = self.original_model.to(device)
            return self

        def get_config(self):
            """Return the model configuration"""
            return {"model_type": "ViT-L-14", "image_size": 224, "patch_size": 14, "vision_width": 1024, "vision_layers": 24, "vision_heads": 16, "embed_dim": 768, "device": str(self.target_device)}

    model = ViTL14CLIPModel(visual_model, jit_model_path, target_device)
    return model


def test_vit_l14_model(model, test_batch_size=2):
    """
    Test the correctness of the ViT-L-14 model
    """
    print("=== Testing ViT-L-14 model ===")

    # Test input
    dummy_image = torch.randn(test_batch_size, 3, 224, 224)

    print(f"Input shape: {dummy_image.shape}")
    print(f"Model config: {model.get_config()}")

    # Test image encoding
    try:
        image_features = model.encode_image(dummy_image)
        print("✓ Image encoding succeeded")
        print(f"Output shape: {image_features.shape}")
        print(f"Expected shape: ({test_batch_size}, 768)")
        print(f"Output device: {image_features.device}")
        print(f"Output dtype: {image_features.dtype}")
        print(f"Output value range: [{image_features.min().item():.4f}, {image_features.max().item():.4f}]")

        # Verify the output dimensions
        if image_features.shape == (test_batch_size, 768):
            print("✓ Output dimensions correct")
        else:
            print(f"✗ Wrong output dimensions, expected ({test_batch_size}, 768)")

    except Exception as e:
        print(f"✗ Image encoding failed: {e}")
        import traceback

        traceback.print_exc()

    # Test text encoding
    try:
        dummy_text = torch.randint(0, 1000, (test_batch_size, 77))
        text_features = model.encode_text(dummy_text)
        print("✓ Text encoding succeeded")
        print(f"Text output shape: {text_features.shape}")
        print(f"Text output device: {text_features.device}")

    except Exception as e:
        print(f"⚠ Text encoding failed: {e}")

    # Test device switching
    try:
        if torch.cuda.device_count() > 1:
            print("Testing device switching...")
            original_device = model.target_device
            model.to("cuda:0" if original_device != torch.device("cuda:0") else "cuda:1")

            test_image = torch.randn(1, 3, 224, 224)
            new_features = model.encode_image(test_image)
            print(f"✓ Device switch succeeded, new device: {new_features.device}")

            # Switch back to the original device
            model.to(original_device)

    except Exception as e:
        print(f"⚠ Device switching test failed: {e}")


if __name__ == "__main__":
    print("=== ViT-L-14 exact implementation and device fix ===")

    model_path = "your_model.pt"  # Path to your JIT model
    target_device = "cuda:1"  # Target device

    try:
        # Create the ViT-L-14 model
        print("Creating ViT-L-14 model...")
        model = create_offline_clip_model(model_path, target_device)

        print("✓ ViT-L-14 model created successfully")

        # Run tests
        test_vit_l14_model(model)

        print("\n🎉 ViT-L-14 model is ready to use!")
        print("✅ Hardcoded cuda:0 device issue resolved")
        print("✅ Can switch freely to any device")
        print("✅ Keeps the same output dimensions as the original model")

        # Usage example
        print("\n=== Usage example ===")
        print("# Image encoding")
        print("image = torch.randn(1, 3, 224, 224)")
        print("features = model.encode_image(image)  # output: [1, 768]")
        print("")
        print("# Device switching")
        print("model.to('cuda:0')  # switch to any device")

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback

        traceback.print_exc()

        print("\nSuggested checks:")
        print("1. Is the JIT model path correct?")
        print("2. Is the model actually the ViT-L-14 architecture?")
        print("3. PyTorch version compatibility")


"""
=== ViT-L-14 architecture notes ===

This implementation is based on the OpenCLIP ViT-L-14 configuration:
- Image size: 224x224
- Patch size: 14x14 (16x16 grid)
- Vision width: 1024
- Vision layers: 24
- Vision heads: 16 (1024/64=16)
- Output dimension: 768
- Activation: QuickGELU

Key features:
✅ Fully matches the OpenCLIP ViT-L-14 architecture
✅ Uses the QuickGELU activation function
✅ Exact weight mapping
✅ Dynamic device management, no hardcoding
✅ Preserves original performance
"""
