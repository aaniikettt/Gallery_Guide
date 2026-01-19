import torch
import torchvision
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def main():
    # -------------------------
    # Device (M1-safe)
    # -------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # -------------------------
    # Transforms
    # -------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------------------------
    # Datasets
    # -------------------------
    trainset = ImageFolder(
        root="wiki_art_dataset/train",
        transform=train_transform
    )
    valset = ImageFolder(
        root="wiki_art_dataset/valid",
        transform=val_transform
    )

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(valset, batch_size=32, shuffle=False, num_workers=0)

    num_classes = len(trainset.classes)

    print("TRAIN classes:")
    print(trainset.class_to_idx)

    print("\nVAL classes:")
    print(valset.class_to_idx)

    # -------------------------
    # Model (Pretrained ResNet18)
    # -------------------------
    model = models.resnet18(pretrained=True)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # -------------------------
    # Loss & Optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.layer3.parameters(), "lr": 5e-5},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 3e-4}
    ])

    # -------------------------
    # Training Loop
    # -------------------------
    num_epochs = 10 

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )
        best_val_acc = 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": trainset.class_to_idx
            }, "wikiart_resnet18_best.pth")

        patience = 3
        counter = 0
        if val_acc <= best_val_acc:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
        else:
            counter = 0


    # -------------------------
    # Save Model
    # -------------------------
    print("Model saved as wikiart_resnet18_best.pth")


if __name__ == "__main__":
    main()
