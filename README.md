# Gallery_Guide
An end-to-end **art style classification system** trained on the **WikiArt dataset** (26 art styles), using **ResNet-18 with transfer learning**, and deployed as a **Dockerized FastAPI inference service** (Apple Silicon / ARM compatible).

This project covers the **full machine learning lifecycle**:
> data loading → training → evaluation → debugging → model saving → API deployment → Dockerization

---

## 📌 Project Highlights

- 🧠 **Model**: ResNet-18 (transfer learning) (wikiart_resnet18_best.pth on my dropbox)
- 🎨 **Dataset**: WikiArt (26 art styles) (Download Dataset and wikiart_resnet18_best.pth from my Dropbox) (🔗 (https://www.dropbox.com/scl/fo/wr7cellnqqf2rg0vojv3z/AIamvUT7JKoTKrXDXw7bi1s?rlkey=7qdey75eldu6ohqza9nweu8kq&st=4wtx1nhi&dl=0))
- 📊 **Performance**:
  - **Validation Accuracy**: ~45%
  - **Test Accuracy**: **41.18%**
- 📈 **Per-class evaluation** (style-wise accuracy analysis)
- 🚀 **Inference API** built with **FastAPI**
- 🐳 **Dockerized deployment** (Mac M1 / ARM-safe)
- 📦 Accepts image uploads and returns **Top-K predictions with confidence**

---

## 🗂️ Repository Structure
```text
├── app/
│   ├── main.py            # FastAPI inference service
│   ├── model.py           # Model loading logic
│   └── utils.py           # Preprocessing & postprocessing
│
├── model/
│   └── wikiart_resnet18_best.pth   # Trained model weights
│
├── model_training/
│   ├── train_resnet.py     # Training script
│   └── test_resnet.py      # Test & evaluation script
│
├── Dockerfile
├── requirements.txt
└── README.md
```
(Download model and dataset from the dropbox link I have provided and organize them in the structure mentioned above)

---

## 🧠 Model Details

- **Backbone**: ResNet-18 (ImageNet pretrained)
- **Fine-tuning**:
  - Frozen backbone initially
  - Partial unfreezing for better generalization
- **Loss**: Cross-Entropy
- **Optimizer**: Adam
- **Input size**: `224 × 224`
- **Output**: 26 art style classes

---

## 📊 Test Performance (Per-Class Accuracy)

| Style | Accuracy |
|------|---------|
| Color Field Painting | 83.33% |
| Minimalism | 80.00% |
| Ukiyo-e | 75.00% |
| Impressionism | 67.33% |
| Cubism | 53.33% |
| Realism | 45.12% |
| **Overall Test Accuracy** | **41.18%** |

> Some styles show lower accuracy due to **class imbalance and visual overlap**, which is common in WikiArt.

---

## 🚀 Running the Inference API with Docker

### 1️⃣ Build the Docker Image

```bash
docker build --platform=linux/arm64 -t wikiart-api .
```
### 2️⃣ Run the Container

```bash
docker run -p 8000:8000 wikiart-api
```
🧪 API Usage 

🔍 Health Check (in a new Terminal)
```bash
curl http://localhost:8000
```

#### Response:
```bash
{ "status": "ok" }
```
🎨 Predict Art Style
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@/path/to/image.jpg"
```

##### Example Response
```bash
{
  "predictions": [
    { "class": "Impressionism", "confidence": 0.63 },
    { "class": "Post_Impressionism", "confidence": 0.21 },
    { "class": "Expressionism", "confidence": 0.08 }
  ]
}
```

🧩 Tech Stack
1. Python
2. PyTorch
3. Torchvision
4. FastAPI
5. Docker
6. Pillow
7. NumPy

🧠 Key Learnings & Engineering Challenges

1. Handling class index mismatches across train/val/test splits
2. Debugging severe overfitting vs label inconsistency
3. Adapting PyTorch training for Apple M1 (MPS)
4. Resolving Docker path & dependency issues
5. Building a production-style inference API

📌 Future Improvements
1. 🔝 Use larger backbones (ResNet-50 / ViT)
2. 📊 Add confusion matrix & Grad-CAM visualizations
3. ⚖️ Handle class imbalance (weighted loss)
4. ☁️ Cloud deployment (AWS / Fly.io)
5. 🌐 Frontend UI for image upload

⭐ If you find this project useful
Give it a ⭐ on GitHub — it really helps!

