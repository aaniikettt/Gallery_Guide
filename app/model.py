import torch
import torchvision.models as models
import torch.nn as nn

MODEL_PATH = "model/wikiart_resnet18_best.pth"  # ✅ RELATIVE TO /app

def load_model(device="cpu"):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, idx_to_class
