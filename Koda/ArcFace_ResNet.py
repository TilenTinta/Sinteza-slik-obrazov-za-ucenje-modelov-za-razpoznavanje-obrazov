######################################################################################################
# Univerza v Ljubljani, Fakulteta za elektrotehniko - Avtomatika in informatika (BMA)
# Predmet: Seminar iz biometričnih sistemov
# Naloga: Seminar - Sinteza slik obrazov za učenje modelov za razpoznavanje obrazov
# Program: ArcFace_LFW.py - ArcFace z ResNet50 razpoznavalnik obrazov
# Autor: Tilen Tinta
# Datum: januar 2025
#######################################################################################################


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Priprava podatkov - poravnani obrazi
#dataset_path = "./LFW/aligned_img"
#weights = "arcface_LFW_weights.pth"

dataset_path = "./Dataset"
weights = "arcface_Dataset_weights.pth"

min_img_num = 5  # Prag za minimalno število slik na identiteto

# Funkcija za prilagoditev collate_fn
def collate_fn_imagefolder(batch, transform):
    images = [transform(Image.open(path).convert("RGB")) for path, _ in batch]
    labels = [label for _, label in batch]
    return torch.stack(images), torch.tensor(labels)

# Definiraj transformacije z augmentacijo
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Naloži podatke
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
label_counts = Counter([label for _, label in dataset.samples])

# Filtriranje identitet
filtered_indices = [i for i, (_, label) in enumerate(dataset.samples) if label_counts[label] > min_img_num]
filtered_samples = [dataset.samples[i] for i in filtered_indices]
filtered_samples_by_label = defaultdict(list)
for path, label in filtered_samples:
    filtered_samples_by_label[label].append((path, label))

# Ustvari nov mapping za label
label_mapping = {old_label: new_label for new_label, old_label in enumerate(filtered_samples_by_label.keys())}

# Delitev na učno in testno množico
train_samples, remaining_samples = [], []
for label, samples in filtered_samples_by_label.items():
    train, remaining = train_test_split(samples, test_size=0.3, random_state=42)
    train_samples.extend(train)
    remaining_samples.extend(remaining)

# Delitev preostale množice na validacijsko in testno
validation_samples, test_samples = train_test_split(remaining_samples, test_size=0.5, random_state=42)

# Priprava končnih množic z uporabo label_mapping
train_dataset = [(path, label_mapping[label]) for path, label in train_samples]
validation_dataset = [(path, label_mapping[label]) for path, label in validation_samples]
test_dataset = [(path, label_mapping[label]) for path, label in test_samples]

# DataLoaderji
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=lambda x: collate_fn_imagefolder(x, transform))
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False, collate_fn=lambda x: collate_fn_imagefolder(x, transform))
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=lambda x: collate_fn_imagefolder(x, transform))


# Število razredov
num_classes = len(label_mapping)
print(f"Število identitet po filtriranju: {len(filtered_samples_by_label)}")
print(f"Število učnih vzorcev: {len(train_samples)}")
print(f"Število validacijskih vzorcev: {len(validation_samples)}")
print(f"Število testnih vzorcev: {len(test_samples)}")

# ArcFace Loss
class ArcFaceLoss(nn.Module):
    def __init__(self, scale=128.0, margin=0.40):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(3.14159265 - margin))
        self.mm = torch.sin(torch.tensor(3.14159265 - margin)) * margin

    def forward(self, logits, labels):
        cosine = logits / torch.norm(logits, dim=1, keepdim=True)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return nn.CrossEntropyLoss()(output, labels)

# Model ResNet-50
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        # 1) RezNet50 brez zadnje FC, ki jo zamenjamo z Linear(2048 -> 512)
        self.base_model = models.resnet18(weights="IMAGENET1K_V1")
        # Namesto originalne končne plasti:
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 512)

        # 2) Parametri za "ArcFace glavo" - teža, ki jo bomo ročno normalizirali
        #    Oblika: (num_classes, 512). Vsaka vrstica ustreza eni identiteti.
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, 512))
        nn.init.xavier_uniform_(self.weight)  # Začetna inicializacija

    def forward(self, x):
        # 3) Dobimo 512-dim vektor vdelave iz ResNet50
        embeddings = self.base_model(x)                       # [B, 512]
        embeddings = nn.functional.normalize(embeddings, dim=1)  # L2-normalizacija po dim=1

        # 4) Normaliziramo še matriko uteži -> vsaka vrstica postane vektor dolžine 1
        w = nn.functional.normalize(self.weight, dim=1)       # [num_classes, 512]

        # 5) Skalarni produkt vektorjev => dejanski cos(θ), saj sta oba normalizirana.
        cosines = torch.matmul(embeddings, w.t())             # [B, num_classes]

        # Za združljivost z vašo strukturo še vedno vrnemo "logits" in "embeddings",
        # a tokrat so "logits" dejansko = cosines.
        return cosines, embeddings


# Inicializacija modela in parametrov
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = FaceRecognitionModel(num_classes).to(device)
criterion = ArcFaceLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Učenje modela
num_epochs = 300
early_stopping_patience = 8
train_loss_history, val_loss_history, val_accuracy_history = [], [], []
best_val_accuracy = 0.0
stopping_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss_history.append(running_loss / len(train_loader))

    # Validacija
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss_history.append(val_loss / len(test_loader))
    val_accuracy = 100 * correct / total
    val_accuracy_history.append(val_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_history[-1]:.4f}, Val Loss: {val_loss_history[-1]:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Early stopping
    if val_accuracy > best_val_accuracy :
        best_val_accuracy = val_accuracy
        stopping_counter = 0
        torch.save(model.state_dict(), weights)
        print("New best score! Saving weights...")
    else:
        stopping_counter += 1

    if stopping_counter >= early_stopping_patience or val_accuracy == 100:
        print("Early stopping triggered. Best validation accuracy: {:.2f}%".format(best_val_accuracy))
        break

# Testiranje modela
model.load_state_dict(torch.load(weights))
model.eval()
test_accuracy = 0.0
all_labels = []
all_preds = []

with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"Testna točnost: {test_accuracy:.2f}%")

all_labels_np = np.array(all_labels)
all_preds_np = np.array(all_preds)

# 1) Natančnost (top-1 accuracy)
test_accuracy = accuracy_score(all_labels_np, all_preds_np)
print(f"Končna točnost (accuracy): {100.0*test_accuracy:.2f}%")

# 2) Confusion matrix
cm = confusion_matrix(all_labels_np, all_preds_np)
print("Matrika zmede (confusion matrix):")
print(cm)

# 3) Precision, Recall, F1-score (makro povprečje)
report = classification_report(all_labels_np, all_preds_np)
print("Poročilo klasifikacije (precision, recall, f1-score):")
print(report)

# Vizualizacija rezultatov
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()

