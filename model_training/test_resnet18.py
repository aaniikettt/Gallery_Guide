import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict
from collections import Counter
from PIL import Image


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load("model/wikiart_resnet18_best.pth", map_location=device)

class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# Rebuild model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
# -------------------------

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

testset = ImageFolder(
    root="wiki_art_dataset/test",
    transform=test_transform
)

testloader = DataLoader(
    testset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)
# -------------------------
assert testset.class_to_idx == class_to_idx, "Class index mismatch!"

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

class_correct = defaultdict(int)
class_total = defaultdict(int)

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for label, pred in zip(labels, preds):
            class_name = idx_to_class[label.item()]
            class_total[class_name] += 1
            if label == pred:
                class_correct[class_name] += 1

print("\nPer-class accuracy:")
for cls in sorted(class_total.keys()):
    acc = 100 * class_correct[cls] / class_total[cls]
    print(f"{cls:30s}: {acc:5.2f}%")

results = []

with torch.no_grad():
    for paths, labels in testset.samples:
        # load single image
        image = test_transform(Image.open(paths).convert("RGB")).unsqueeze(0).to(device)
        output = model(image)
        _, pred = torch.max(output, 1)

        results.append({
            "image": paths,
            "true": idx_to_class[labels],
            "pred": idx_to_class[pred.item()]
        })

print(Counter(testset.targets))

