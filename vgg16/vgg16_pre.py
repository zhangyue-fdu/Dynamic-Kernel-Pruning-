# vgg16_pruning_correct_conv_layers.py
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
# 1. VGG16 Model (with nn.Sequential)
# ----------------------------
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ----------------------------
# 2. Prune Similar Filters (Correct Conv Detection)
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
        print("  ⚠️  No filters pruned. Consider lowering thresholds (e.g., --threshold_c 0.01 --threshold_b 0.0).")
    else:
        print(f"  ✅ Total filters pruned this epoch: {total_pruned}")


# ----------------------------
# 3. Freezer for Pruned Filters
# ----------------------------
# class PrunedFilterFreezer:
#     def __init__(self, model):
#         self.model = model
#         self.hooks = []
#         self.masks = {}
#         self._init_masks_and_hooks()

#     def _init_masks_and_hooks(self):
#         for name, module in self.model.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 param = module.weight
#                 mask = torch.ones_like(param.data)
#                 self.masks[param] = mask
#                 hook = param.register_hook(
#                     lambda grad, p=param: grad * self.masks[p]
#                 )
#                 self.hooks.append(hook)

#     def update_masks_after_pruning(self):
#         for name, module in self.model.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 param = module.weight
#                 with torch.no_grad():
#                     is_nonzero = (param.data.abs().sum(dim=(1, 2, 3), keepdim=True) > 1e-8).float()
#                     channel_mask = is_nonzero.unsqueeze(-1).unsqueeze(-1)
#                     if param in self.masks:
#                         self.masks[param] = self.masks[param] * channel_mask
#                     else:
#                         self.masks[param] = channel_mask

#     def remove_hooks(self):
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks = []

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
                    is_nonzero = (param.data.abs().sum(dim=(1, 2, 3)) > 1e-8).float()
                    channel_mask = is_nonzero.view(-1, 1, 1, 1)
                    full_mask = channel_mask.expand_as(param.data)
                    if param in self.masks:
                        self.masks[param] = self.masks[param] * full_mask
                    else:
                        self.masks[param] = full_mask

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ----------------------------
# 4. Evaluation
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
# 5. Structured Pruning Ratio (by filter removal)
# ----------------------------
def count_pruned_ratio(model, input_size=(3, 32, 32)):
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
# 6. Zero Weight Counter (Conv layers only)
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
# 7. Data Loaders
# ----------------------------
def get_dataloaders(dataset='cifar10', batch_size=128):
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader, num_classes


# ----------------------------
# 8. Main: Resume from pretrained
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to pretrained .pth file')
    parser.add_argument('--prune_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=50)
    parser.add_argument('--threshold_c', type=float, default=0.1)
    parser.add_argument('--threshold_b', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    trainloader, testloader, num_classes = get_dataloaders(args.dataset, args.batch_size)

    # Load model
    model = VGG16(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    print(f"✅ Loaded pretrained model from: {args.ckpt_path}")

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    # Initial accuracy
    _, init_acc = evaluate(model, testloader, criterion, device)
    print(f"Initial Test Accuracy: {init_acc:.2f}%")

    # Initialize freezer
    freezer = PrunedFilterFreezer(model)

    # === Phase 2: Pruning ===
    print(f"\n=== Phase 2: Pruning ({args.prune_epochs} epochs) ===")
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.prune_epochs)

    for epoch in range(args.prune_epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Prune + freeze
        prune_similar_filters(model, threshold_c=args.threshold_c, threshold_b=args.threshold_b)
        freezer.update_masks_after_pruning()

        # Evaluate
        _, test_acc = evaluate(model, testloader, criterion, device)
        stats = count_pruned_ratio(model, input_size=(3, 32, 32))
        prune_rate = stats['param_reduction_percent']
        zero_count, total_count, zero_pct = count_zero_weights_in_conv_layers(model)

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = f'vgg16_{args.dataset}_pruned_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  → New best! Saved to {save_path}")

        print(f"[Pruning] Epoch {epoch+1}/{args.prune_epochs}, "
              f"Acc: {test_acc:.2f}%, "
              f"Structural Prune: {prune_rate:.2f}%, "
              f"Weight Zero %: {zero_pct:.2f}%")

    # Load best pruned model
    best_pruned_path = f'vgg16_{args.dataset}_pruned_best.pth'
    if os.path.exists(best_pruned_path):
        model.load_state_dict(torch.load(best_pruned_path, map_location=device))
        print(f"\nLoaded best pruned model from {best_pruned_path}")

    # === Phase 3: Fine-tuning ===
    print(f"\n=== Phase 3: Fine-tuning ({args.finetune_epochs} epochs) ===")
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

    for epoch in range(args.finetune_epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        _, test_acc = evaluate(model, testloader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = f'vgg16_{args.dataset}_finetuned_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  → New best! Saved to {save_path}")

        print(f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs}, Test Acc: {test_acc:.2f}%")

    # Final stats
    print(f"\n✅ Final Best Accuracy: {best_acc:.2f}%")
    model.load_state_dict(torch.load(f'vgg16_{args.dataset}_finetuned_best.pth', map_location=device))
    
    print("\n📊 Final Structured Pruning:")
    stats = count_pruned_ratio(model, input_size=(3, 32, 32))
    print(f"  Param Reduction: {stats['param_reduction_percent']:.2f}%")

    print("\n🔍 Final Zero Weight Stats:")
    count_zero_weights_in_conv_layers(model)

    freezer.remove_hooks()


if __name__ == '__main__':
    main()