# resnet50_imagenet_pruning.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
import argparse
import os

# ----------------------------
# Prune Similar Filters (Full-precision L2 similarity)
# ----------------------------
def prune_similar_filters(model, threshold_c=0.1, threshold_b=1.0):
    total_pruned = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            param = module.weight
            weights = param.data
            num_filters = weights.size(0)
            if num_filters <= 1:
                continue

            filter_vecs = weights.view(num_filters, -1)
            non_zero_mask = filter_vecs.abs().sum(dim=1) > 1e-8
            if non_zero_mask.sum().item() <= 1:
                continue

            indices = torch.where(non_zero_mask)[0].tolist()
            distances = []
            pairs = []

            for i_idx in range(len(indices)):
                for j_idx in range(i_idx + 1, len(indices)):
                    i, j = indices[i_idx], indices[j_idx]
                    dist = torch.norm(filter_vecs[i] - filter_vecs[j], p=2).item()
                    distances.append(dist)
                    pairs.append((i, j, dist))

            if not distances:
                continue

            mean_dist = np.mean(distances)
            std_dist = np.sqrt(np.var(distances))
            threshold_dist = mean_dist - threshold_b * std_dist
            threshold_dist = max(threshold_dist, 0.0)

            filters_to_prune = []
            for i, j, dist in pairs:
                if dist < threshold_dist:
                    filters_to_prune.append(i)
                    filters_to_prune.append(j)

            filter_counts = Counter(filters_to_prune)
            min_count = threshold_c * (len(indices) - 1)
            filters_to_prune_final = [f for f, cnt in filter_counts.items() if cnt > min_count]

            if filters_to_prune_final:
                print(f"  🔪 Pruning {len(filters_to_prune_final)} filters in layer: {name}")

            with torch.no_grad():
                for f in filters_to_prune_final:
                    param.data[f] = 0.0
            total_pruned += len(filters_to_prune_final)

    if total_pruned == 0:
        print("  ⚠️  No filters pruned. Try: --threshold_c 0.01 --threshold_b 0.0")
    else:
        print(f"  ✅ Total filters pruned this epoch: {total_pruned}")


# ----------------------------
# Freezer for Pruned Filters
# ----------------------------
class PrunedFilterFreezer:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.masks = {}
        self._init_masks_and_hooks()

    def _init_masks_and_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                param = module.weight
                mask = torch.ones_like(param.data)
                self.masks[param] = mask
                hook = param.register_hook(
                    lambda grad, p=param: grad * self.masks[p]
                )
                self.hooks.append(hook)

    def update_masks_after_pruning(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                param = module.weight
                with torch.no_grad():
                    is_nonzero = (param.data.abs().sum(dim=(1, 2, 3), keepdim=True) > 1e-8).float()
                    channel_mask = is_nonzero.unsqueeze(-1).unsqueeze(-1)
                    if param in self.masks:
                        self.masks[param] = self.masks[param] * channel_mask
                    else:
                        self.masks[param] = channel_mask

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ----------------------------
# Evaluation Function
# ----------------------------
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    acc = 100. * correct / total
    return total_loss / len(dataloader), acc


# ----------------------------
# Structured Pruning Ratio
# ----------------------------
def count_pruned_ratio(model, input_size=(3, 224, 224)):
    model.eval()
    total_params_original = 0
    total_params_pruned = 0
    current_in_channels = input_size[0]

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            out_channels_orig = weight.shape[0]
            in_channels_orig = weight.shape[1]
            kH, kW = weight.shape[2], weight.shape[3]

            is_filter_zero = (weight.abs().sum(dim=(1, 2, 3)) == 0)
            num_pruned_filters = is_filter_zero.sum().item()
            num_remaining_filters = out_channels_orig - num_pruned_filters
            effective_in_channels = current_in_channels

            params_orig = out_channels_orig * effective_in_channels * kH * kW
            params_pruned = num_remaining_filters * effective_in_channels * kH * kW

            total_params_original += params_orig
            total_params_pruned += params_pruned

            current_in_channels = num_remaining_filters

    param_reduction = (1 - total_params_pruned / total_params_original) * 100 if total_params_original > 0 else 0
    return {'param_reduction_percent': param_reduction}


# ----------------------------
# Zero Weight Counter
# ----------------------------
def count_zero_weights_in_conv_layers(model):
    zero_count = 0
    total_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            param = module.weight
            zero_count += torch.sum(param == 0).item()
            total_count += param.numel()
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")
    return zero_count, total_count, zero_percentage


# ----------------------------
# Data Loaders for ImageNet
# ----------------------------
def get_imagenet_loaders(data_dir, batch_size=64, num_workers=8):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.ImageNet(root=data_dir, split='train', transform=train_transform)
    val_dataset = torchvision.datasets.ImageNet(root=data_dir, split='val', transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader


# ----------------------------
# Main Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='ResNet-50 Pruning on ImageNet')
    parser.add_argument('--data', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to custom pretrained .pth (if None, use torchvision)')
    parser.add_argument('--prune_epochs', type=int, default=30, help='Number of pruning epochs')
    parser.add_argument('--finetune_epochs', type=int, default=30, help='Number of fine-tuning epochs')
    parser.add_argument('--threshold_c', type=float, default=0.1, help='Filter frequency threshold')
    parser.add_argument('--threshold_b', type=float, default=1.0, help='Distance threshold (in std)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--lr_prune', type=float, default=1e-4, help='Learning rate for pruning phase')
    parser.add_argument('--lr_finetune', type=float, default=1e-5, help='Learning rate for fine-tuning phase')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("=> Loading ImageNet dataset...")
    train_loader, val_loader = get_imagenet_loaders(args.data, batch_size=args.batch_size, num_workers=args.workers)

    # Model
    if args.ckpt_path:
        print(f"=> Loading checkpoint '{args.ckpt_path}'")
        model = torchvision.models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    else:
        print("=> Using torchvision pretrained ResNet-50")
        model = torchvision.models.resnet50(pretrained=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0.0

    # Initial evaluation
    print("=> Evaluating initial model...")
    _, init_acc = evaluate(model, val_loader, criterion, device)
    print(f"Initial Val Accuracy: {init_acc:.2f}%")

    # Initialize freezer
    freezer = PrunedFilterFreezer(model)

    # === Phase 2: Pruning ===
    print(f"\n=> Phase 2: Pruning ({args.prune_epochs} epochs)")
    optimizer = optim.SGD(model.parameters(), lr=args.lr_prune, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.prune_epochs)

    for epoch in range(args.prune_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"  Epoch [{epoch+1}/{args.prune_epochs}], Iter [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()

        # Prune and freeze
        prune_similar_filters(model, threshold_c=args.threshold_c, threshold_b=args.threshold_b)
        freezer.update_masks_after_pruning()

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        stats = count_pruned_ratio(model, input_size=(3, 224, 224))
        prune_rate = stats['param_reduction_percent']
        zero_count, total_count, zero_pct = count_zero_weights_in_conv_layers(model)

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = 'resnet50_imagenet_pruned_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  → New best! Saved to {save_path}")

        print(f"[Pruning] Epoch {epoch+1}/{args.prune_epochs}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Structural Prune: {prune_rate:.2f}%, "
              f"Weight Zero %: {zero_pct:.2f}%")

    # Load best pruned model
    best_pruned_path = 'resnet50_imagenet_pruned_best.pth'
    if os.path.exists(best_pruned_path):
        model.load_state_dict(torch.load(best_pruned_path, map_location=device))
        print(f"\n=> Loaded best pruned model from {best_pruned_path}")

    # === Phase 3: Fine-tuning ===
    print(f"\n=> Phase 3: Fine-tuning ({args.finetune_epochs} epochs)")
    optimizer = optim.SGD(model.parameters(), lr=args.lr_finetune, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

    for epoch in range(args.finetune_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = 'resnet50_imagenet_finetuned_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  → New best! Saved to {save_path}")

        print(f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs}, Val Acc: {val_acc:.2f}%")

    # Final statistics
    print(f"\n✅ Final Best Accuracy: {best_acc:.2f}%")
    finetuned_path = 'resnet50_imagenet_finetuned_best.pth'
    if os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(finetuned_path, map_location=device))
    
    print("\n📊 Final Structured Pruning Statistics:")
    stats = count_pruned_ratio(model, input_size=(3, 224, 224))
    print(f"  Param Reduction: {stats['param_reduction_percent']:.2f}%")

    print("\n🔍 Final Zero Weight Statistics:")
    count_zero_weights_in_conv_layers(model)

    # Cleanup
    freezer.remove_hooks()
    print("\n✅ Pruning pipeline completed!")


if __name__ == '__main__':
    main()