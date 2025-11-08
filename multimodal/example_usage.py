"""
Example usage of the MultiModalModel.

This script demonstrates how to:
1. Initialize the model with different configurations
2. Perform forward passes
3. Use the model with different learning rates
4. Extract embeddings
"""
import torch
from multimodal import MultiModalModel


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create model with default configuration
    model = MultiModalModel()
    
    # Example input data
    batch_size = 4
    image_size = 120
    s1_data = torch.randn(batch_size, 2, image_size, image_size)  # S1: 2 channels
    s2_data = torch.randn(batch_size, 12, image_size, image_size)  # S2: 12 channels
    
    # Forward pass
    logits = model(s1_data, s2_data)
    print(f"Input shapes:")
    print(f"  S1: {s1_data.shape}")
    print(f"  S2: {s2_data.shape}")
    print(f"Output logits shape: {logits.shape}")
    print()


def example_custom_config():
    """Example with custom configuration."""
    print("=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    config = {
        "backbones": {
            "dinov3": {
                "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",  # Base model
                "pretrained": True,
                "freeze": False,
                "lr": 1e-4,
            },
            "resnet101": {
                "pretrained": True,
                "freeze": False,
                "lr": 1e-4,
            },
        },
        "fusion": {
            "type": "weighted",  # Use weighted fusion
        },
        "classifier": {
            "type": "mlp",  # Use MLP classifier
            "hidden_dim": 512,
            "num_classes": 19,
            "drop_rate": 0.15,
        },
        "image_size": 120,
    }
    
    model = MultiModalModel(config=config)
    
    # Test forward pass
    batch_size = 2
    image_size = 120
    s1_data = torch.randn(batch_size, 2, image_size, image_size)
    s2_data = torch.randn(batch_size, 14, image_size, image_size)
    
    logits = model(s1_data, s2_data)
    print(f"Model configuration:")
    print(f"  Fusion: {config['fusion']['type']}")
    print(f"  Classifier: {config['classifier']['type']}")
    print(f"  Output logits shape: {logits.shape}")
    print()


def example_with_embeddings():
    """Example with embedding extraction."""
    print("=" * 60)
    print("Example 3: Extracting Embeddings")
    print("=" * 60)
    
    model = MultiModalModel()
    
    batch_size = 2
    image_size = 120
    s1_data = torch.randn(batch_size, 2, image_size, image_size)
    s2_data = torch.randn(batch_size, 14, image_size, image_size)
    
    # Get logits and embeddings
    logits, embeddings = model(s1_data, s2_data, return_embeddings=True)
    
    print(f"Logits shape: {logits.shape}")
    print(f"DINOv3 embeddings shape: {embeddings['dinov3'].shape}")
    print(f"ResNet embeddings shape: {embeddings['resnet'].shape}")
    print(f"Fused embeddings shape: {embeddings['fused'].shape}")
    print()


def example_parameter_groups():
    """Example of using parameter groups for different learning rates."""
    print("=" * 60)
    print("Example 4: Parameter Groups for Training")
    print("=" * 60)
    
    config = {
        "backbones": {
            "dinov3": {"pretrained": True, "freeze": False, "lr": 1e-4},
            "resnet101": {"pretrained": True, "freeze": False, "lr": 1e-4},
        },
        "fusion": {"type": "concat"},
        "classifier": {"type": "linear", "num_classes": 19, "drop_rate": 0.15},
        "image_size": 120,
    }
    
    model = MultiModalModel(config=config)
    
    # Get parameter groups
    param_groups = model.get_parameter_groups()
    
    print("Parameter groups for optimizer:")
    for group in param_groups:
        print(f"  {group['name']}: lr={group['lr']}, params={sum(p.numel() for p in group['params'])}")
    print()
    
    # Create optimizer
    import torch.optim as optim
    optimizer = optim.AdamW(param_groups)
    print(f"Optimizer created with {len(param_groups)} parameter groups")
    print()


def example_frozen_backbones():
    """Example with frozen backbones (linear probing)."""
    print("=" * 60)
    print("Example 5: Frozen Backbones (Linear Probing)")
    print("=" * 60)
    
    config = {
        "backbones": {
            "dinov3": {"pretrained": True, "freeze": True, "lr": 1e-4},
            "resnet101": {"pretrained": True, "freeze": True, "lr": 1e-4},
        },
        "fusion": {"type": "linear_projection", "output_dim": 512},
        "classifier": {"type": "mlp", "hidden_dim": 256, "num_classes": 19, "drop_rate": 0.2},
        "image_size": 120,
    }
    
    model = MultiModalModel(config=config)
    
    # Check which parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"  (Only fusion and classifier are trainable)")
    print()


def example_all_fusion_types():
    """Example showing all fusion types."""
    print("=" * 60)
    print("Example 6: All Fusion Types")
    print("=" * 60)
    
    batch_size = 2
    image_size = 120
    s1_data = torch.randn(batch_size, 2, image_size, image_size)
    s2_data = torch.randn(batch_size, 14, image_size, image_size)
    
    fusion_types = ["concat", "weighted", "linear_projection"]
    
    for fusion_type in fusion_types:
        config = {
            "backbones": {
                "dinov3": {"pretrained": False, "freeze": False, "lr": 1e-4},
                "resnet101": {"pretrained": False, "freeze": False, "lr": 1e-4},
            },
            "fusion": {
                "type": fusion_type,
                "output_dim": 512 if fusion_type == "linear_projection" else None,
            },
            "classifier": {"type": "linear", "num_classes": 19, "drop_rate": 0.15},
            "image_size": 120,
        }
        
        model = MultiModalModel(config=config)
        logits = model(s1_data, s2_data)
        print(f"  {fusion_type:20s}: output shape = {logits.shape}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MultiModalModel Usage Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    example_basic_usage()
    example_custom_config()
    example_with_embeddings()
    example_parameter_groups()
    example_frozen_backbones()
    example_all_fusion_types()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

