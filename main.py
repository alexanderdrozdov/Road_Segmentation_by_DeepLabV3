import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import gc
import time


class RoadDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        image_dir = os.path.join(root, split, 'img')
        mask_dir = os.path.join(root, split, 'masks')
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        assert len(self.images) == len(self.masks), "Mismatched images/masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.split, 'img', self.images[idx])
        mask_path = os.path.join(self.root, self.split, 'masks', self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return {
            'pixel_values': image,
            'mask_labels': mask.long()
        }


def initialize_model(num_classes=1, use_lite=True):
    if use_lite:
        print("Using MobileNetV3-Large backbone")
        model = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    else:
        print("Using ResNet50 backbone")
        model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def combined_loss(pred, target, bce_weight=0.5):

    # Binary Cross Entropy
    bce_loss = nn.BCEWithLogitsLoss()(pred, target.float())

    # Dice Loss
    smooth = 1.0
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum()
    dice = (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
    dice_loss = 1 - dice

    total_loss = bce_weight * bce_loss + (1 - bce_weight) * dice_loss
    return total_loss, bce_loss.item(), dice_loss.item()


def calculate_iou(pred, target):

    smooth = 1e-6
    pred_bin = (torch.sigmoid(pred) > 0.5).float()


    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_accuracy(pred, target):

    smooth = 1e-6
    pred_bin = (torch.sigmoid(pred) > 0.5).float()

    correct = (pred_bin == target).float().sum()
    total = target.numel()

    accuracy = (correct + smooth) / (total + smooth)
    return accuracy.item()


def train_model(model, dataloaders, optimizer, device, num_epochs=25, patience=5, accumulation_steps=2):
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_bce': [], 'val_bce': [], 'test_bce': [],
        'train_dice': [], 'val_dice': [], 'test_dice': [],
        'train_iou': [], 'val_iou': [], 'test_iou': [],
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'lr': []
    }

    best_iou = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        start_time = time.time()

        # Фазы: обучение, валидация
        phases = ['train', 'val'] + (['test'] if epoch == num_epochs - 1 else [])

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_bce = 0.0
            running_dice = 0.0
            running_iou = 0.0
            running_acc = 0.0
            total_samples = 0
            optimizer.zero_grad()

            loader = dataloaders[phase]
            for batch_idx, batch in enumerate(tqdm(loader, desc=f'{phase.capitalize()} Epoch {epoch + 1}')):
                inputs = batch['pixel_values'].to(device, non_blocking=True)
                masks = batch['mask_labels'].to(device, non_blocking=True).unsqueeze(1)
                batch_size = inputs.size(0)
                total_samples += batch_size

                with torch.set_grad_enabled(phase == 'train'), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    outputs = model(inputs)['out']
                    loss, bce_loss, dice_loss = combined_loss(outputs, masks)
                    loss = loss / accumulation_steps  # Нормализация потерь для накопления градиентов

                # Вычисление метрик качества
                iou = calculate_iou(outputs, masks)
                accuracy = calculate_accuracy(outputs, masks)

                running_iou += iou * batch_size
                running_acc += accuracy * batch_size

                if phase == 'train':
                    scaler.scale(loss).backward()

                    # Обновление весов
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()

                    running_loss += loss.item() * batch_size * accumulation_steps
                    running_bce += bce_loss * batch_size
                    running_dice += dice_loss * batch_size
                else:
                    running_loss += loss.item() * batch_size * accumulation_steps
                    running_bce += bce_loss * batch_size
                    running_dice += dice_loss * batch_size


            if phase == 'train':
                epoch_loss = running_loss / total_samples
                epoch_bce = running_bce / total_samples
                epoch_dice = running_dice / total_samples
                epoch_iou = running_iou / total_samples
                epoch_acc = running_acc / total_samples

                history['train_loss'].append(epoch_loss)
                history['train_bce'].append(epoch_bce)
                history['train_dice'].append(epoch_dice)
                history['train_iou'].append(epoch_iou)
                history['train_acc'].append(epoch_acc)

                print(
                    f'TRAIN - Loss: {epoch_loss:.4f} | BCE: {epoch_bce:.4f} | Dice: {epoch_dice:.4f} | IoU: {epoch_iou:.4f} | Acc: {epoch_acc:.4f}')

            elif phase == 'val':
                epoch_loss = running_loss / total_samples
                epoch_bce = running_bce / total_samples
                epoch_dice = running_dice / total_samples
                epoch_iou = running_iou / total_samples
                epoch_acc = running_acc / total_samples

                history['val_loss'].append(epoch_loss)
                history['val_bce'].append(epoch_bce)
                history['val_dice'].append(epoch_dice)
                history['val_iou'].append(epoch_iou)
                history['val_acc'].append(epoch_acc)

                print(
                    f'VAL - Loss: {epoch_loss:.4f} | BCE: {epoch_bce:.4f} | Dice: {epoch_dice:.4f} | IoU: {epoch_iou:.4f} | Acc: {epoch_acc:.4f}')

                scheduler.step(epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                history['lr'].append(current_lr)
                print(f'Learning Rate: {current_lr:.2e}')


                if epoch_iou > best_iou + 0.001:
                    best_iou = epoch_iou
                    epochs_no_improve = 0
                    #torch.save(model.state_dict(), 'best_model.pth')
                    print(f'Validation IoU improved to {best_iou:.4f}. Model saved.')
                else:
                    epochs_no_improve += 1
                    print(f'No improvement in IoU for {epochs_no_improve}/{patience} epochs')

                    if epochs_no_improve >= patience:
                        print('Early stopping!')
                        model.load_state_dict(torch.load('best_model.pth'))
                        return model, history

        torch.save(model.state_dict(), 'best_model.pth')
        epoch_time = time.time() - start_time
        print(f'Epoch time: {epoch_time:.2f} seconds')
        torch.cuda.empty_cache()

    return model, history


def plot_results(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 12))

    # График потерь
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # График IoU
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_iou'], 'g-', label='Training IoU')
    plt.plot(epochs, history['val_iou'], 'm-', label='Validation IoU')
    plt.title('IoU Metric')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.ylim(0, 1)  # Ограничение для IoU [0, 1]
    plt.legend()
    plt.grid(True)

    # График точности
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_acc'], 'c-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'y-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Ограничение для точности [0, 1]
    plt.legend()
    plt.grid(True)

    # График Dice Loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_dice'], 'k-', label='Training Dice Loss')
    plt.plot(epochs, history['val_dice'], 'orange', label='Validation Dice Loss')
    plt.title('Dice Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

    # Дополнительный график для BCE и LR
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_bce'], 'b-', label='Training BCE')
    plt.plot(epochs, history['val_bce'], 'r-', label='Validation BCE')
    plt.title('Binary Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['lr'], 'g-')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('bce_and_lr.png')
    plt.close()


def visualize_predictions(model, dataset, device, num_samples=3):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['pixel_values'].unsqueeze(0).to(device)
        true_mask = sample['mask_labels'].numpy().squeeze()

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(image)['out']
            pred_probs = torch.sigmoid(output).cpu().numpy().squeeze()
            pred_mask = (pred_probs > 0.5).astype(np.uint8)

        # Денормализация изображения
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = image.cpu().numpy().squeeze().transpose(1, 2, 0)
        img_vis = img_vis * std + mean
        img_vis = np.clip(img_vis, 0, 1)

        sample_iou = calculate_iou(output, torch.tensor(true_mask).unsqueeze(0).unsqueeze(0).float().to(device))

        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title(f'Sample {i + 1} - Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title(f'Sample {i + 1} - True Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f'Sample {i + 1} - Pred Mask (IoU: {sample_iou:.3f})')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()


def check_gpu_memory():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        print(f"GPU Memory: Total: {total:.2f}GB, Free: {free:.2f}GB")
        return free
    return 0


def limit_gpu_memory_usage(limit_gb=4):
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        limit = int(limit_gb * 1024 ** 3)
        torch.cuda.set_per_process_memory_fraction(limit / total_memory)
        print(f"Limited GPU memory usage to {limit_gb}GB")


if __name__ == '__main__':

    DATA_ROOT = r"D:\dataset"
    BATCH_SIZE = 6
    NUM_EPOCHS = 25
    ACCUMULATION_STEPS = 2
    USE_LITE_MODEL = True

    torch.multiprocessing.freeze_support()


    limit_gpu_memory_usage(limit_gb=8)

    free_mem = check_gpu_memory()
    IMG_SIZE = 512
    print(f"Selected image size: {IMG_SIZE}x{IMG_SIZE}")

    # Трансформации
    base_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Создание датасетов
    train_dataset = RoadDataset(DATA_ROOT, 'train', base_transform)
    val_dataset = RoadDataset(DATA_ROOT, 'val', base_transform)
    test_dataset = RoadDataset(DATA_ROOT, 'test', base_transform)


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=5,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        prefetch_factor=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = initialize_model(num_classes=1, use_lite=USE_LITE_MODEL)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model, history = train_model(
        model,
        dataloaders,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS
    )

    model.eval()
    test_loss = 0.0
    test_bce = 0.0
    test_dice = 0.0
    test_iou = 0.0
    test_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Final Testing'):
            inputs = batch['pixel_values'].to(device)
            masks = batch['mask_labels'].to(device).unsqueeze(1)
            batch_size = inputs.size(0)
            total_samples += batch_size

            outputs = model(inputs)['out']
            loss, bce_loss, dice_loss = combined_loss(outputs, masks)

            # Вычисление метрик
            iou = calculate_iou(outputs, masks)
            acc = calculate_accuracy(outputs, masks)

            test_loss += loss.item() * batch_size
            test_bce += bce_loss * batch_size
            test_dice += dice_loss * batch_size
            test_iou += iou * batch_size
            test_acc += acc * batch_size

    # Сохранение финальных метрик теста
    history['test_loss'].append(test_loss / total_samples)
    history['test_bce'].append(test_bce / total_samples)
    history['test_dice'].append(test_dice / total_samples)
    history['test_iou'].append(test_iou / total_samples)
    history['test_acc'].append(test_acc / total_samples)

    print("\nFinal Test Results:")
    print(f"Test Loss: {history['test_loss'][0]:.4f}")
    print(f"Test BCE: {history['test_bce'][0]:.4f}")
    print(f"Test Dice: {history['test_dice'][0]:.4f}")
    print(f"Test IoU: {history['test_iou'][0]:.4f}")
    print(f"Test Accuracy: {history['test_acc'][0]:.4f}")

    plot_results(history)
    visualize_predictions(model, test_dataset, device, num_samples=5)

    torch.save(history, 'training_history.pth')
    print("Training completed and results saved!")