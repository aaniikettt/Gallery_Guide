from torchvision import transforms
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image: Image.Image, device):
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def postprocess(output, idx_to_class, top_k=3):
    probs = torch.softmax(output, dim=1)
    values, indices = torch.topk(probs, top_k)

    results = []
    for v, i in zip(values[0], indices[0]):
        results.append({
            "class": idx_to_class[i.item()],
            "confidence": round(v.item(), 4)
        })

    return results
