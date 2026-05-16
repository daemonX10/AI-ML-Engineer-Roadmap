# Transfer Learning Interview Questions - Coding Questions

---

## Question 1: Write a Python script to fine-tune a pre-trained CNN on a new dataset using Keras
- [ ] Done

### Complete Implementation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_transfer_model(num_classes, input_shape=(224, 224, 3)):
    """
    Create a transfer learning model using ResNet50.
    
    Args:
        num_classes: Number of target classes
        input_shape: Input image dimensions
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50 without top layer
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build new model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_data_generators(train_dir, val_dir, batch_size=32):
    """Create training and validation data generators with augmentation."""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator


def fine_tune_model(model, train_gen, val_gen, epochs_frozen=10, epochs_unfrozen=10):
    """
    Two-stage fine-tuning:
    1. Train only new layers (frozen backbone)
    2. Unfreeze and fine-tune entire model
    """
    
    # Stage 1: Train with frozen backbone
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Stage 1: Training with frozen backbone...")
    history1 = model.fit(
        train_gen,
        epochs=epochs_frozen,
        validation_data=val_gen,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )
    
    # Stage 2: Unfreeze and fine-tune
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze early layers, fine-tune later layers
    for layer in base_model.layers[:140]:  # Freeze first 140 layers
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Stage 2: Fine-tuning entire model...")
    history2 = model.fit(
        train_gen,
        epochs=epochs_unfrozen,
        validation_data=val_gen,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )
    
    return model, (history1, history2)


# Main execution
if __name__ == "__main__":
    # Configuration
    NUM_CLASSES = 10
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    
    # Create model
    model = create_transfer_model(num_classes=NUM_CLASSES)
    model.summary()
    
    # Create data generators
    train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)
    
    # Fine-tune
    model, history = fine_tune_model(model, train_gen, val_gen)
    
    # Evaluate
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model.save('fine_tuned_model.h5')
```

---

## Question 2: Implement a transfer learning model with PyTorch using a pre-trained BERT model for text classification
- [ ] Done

### Complete Implementation

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3, freeze_bert=False):
        super().__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Optionally freeze BERT layers
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def train_bert_classifier(texts, labels, num_classes, epochs=4, batch_size=16):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = BertClassifier(num_classes=num_classes, freeze_bert=False)
    model.to(device)
    
    # Optimizer with different learning rates
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bert_classifier.pt')
    
    return model


# Example usage
if __name__ == "__main__":
    # Sample data
    texts = ["This movie is great!", "Terrible waste of time", "Amazing performance"]
    labels = [1, 0, 1]  # 1=positive, 0=negative
    
    model = train_bert_classifier(texts, labels, num_classes=2)
```

---

## Question 3: Fine-tune a pre-trained image recognition network to classify a new set of images
- [ ] Done

### Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm

class ImageClassifier:
    def __init__(self, num_classes, model_name='resnet50', pretrained=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )
        
        self.model.to(self.device)
        
    def get_transforms(self, is_training=True):
        if is_training:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def freeze_backbone(self, freeze=True):
        """Freeze all layers except the classifier"""
        for name, param in self.model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = not freeze
    
    def train_step(self, dataloader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(dataloader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(dataloader), 100. * correct / total
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(dataloader), 100. * correct / total
    
    def fine_tune(self, train_dir, val_dir, epochs_frozen=5, epochs_unfrozen=10, batch_size=32):
        # Create data loaders
        train_dataset = datasets.ImageFolder(train_dir, self.get_transforms(is_training=True))
        val_dataset = datasets.ImageFolder(val_dir, self.get_transforms(is_training=False))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        criterion = nn.CrossEntropyLoss()
        
        # Phase 1: Train with frozen backbone
        print("Phase 1: Training with frozen backbone")
        self.freeze_backbone(freeze=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
        
        for epoch in range(epochs_frozen):
            train_loss, train_acc = self.train_step(train_loader, criterion, optimizer)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Phase 2: Fine-tune entire model
        print("\nPhase 2: Fine-tuning entire model")
        self.freeze_backbone(freeze=False)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        best_acc = 0
        for epoch in range(epochs_unfrozen):
            train_loss, train_acc = self.train_step(train_loader, criterion, optimizer)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pt')
        
        return best_acc


# Usage
if __name__ == "__main__":
    classifier = ImageClassifier(num_classes=10, model_name='resnet50')
    best_accuracy = classifier.fine_tune(
        train_dir='data/train',
        val_dir='data/val',
        epochs_frozen=5,
        epochs_unfrozen=10
    )
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
```

---

## Question 4: Code an example that demonstrates transfer of learning from a source model trained on MNIST to a target dataset of hand-written letters
- [ ] Done

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

class FeatureExtractor(nn.Module):
    """CNN feature extractor for handwritten character recognition"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.features(x)


class DigitClassifier(nn.Module):
    """Complete model for MNIST digit classification"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)  # 10 digits
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class LetterClassifier(nn.Module):
    """Model for EMNIST letter classification using transferred features"""
    def __init__(self, pretrained_extractor, num_classes=26, freeze_features=True):
        super().__init__()
        self.feature_extractor = pretrained_extractor
        
        # Freeze feature extractor
        if freeze_features:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # New classifier for letters
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  # 26 letters
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


def train_on_mnist(epochs=5, batch_size=64):
    """Train feature extractor on MNIST digits"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Model
    model = DigitClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)
        
        print(f"Epoch {epoch+1}: Train Acc: {100*correct/total:.2f}%, Test Acc: {100*test_correct/test_total:.2f}%")
    
    return model


def transfer_to_letters(pretrained_model, epochs=10, batch_size=64, freeze=True):
    """Transfer learned features to EMNIST letters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data (EMNIST Letters)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # EMNIST images need rotation/flip to match MNIST orientation
        transforms.Lambda(lambda x: x.transpose(1, 2).flip(2))
    ])
    
    train_data = datasets.EMNIST('data', split='letters', train=True, download=True, transform=transform)
    test_data = datasets.EMNIST('data', split='letters', train=False, transform=transform)
    
    # EMNIST letters are 1-indexed (A=1, B=2, ...), adjust to 0-indexed
    train_data.targets = train_data.targets - 1
    test_data.targets = test_data.targets - 1
    
    # Use subset for demonstration (simulating limited data)
    indices = np.random.choice(len(train_data), 5000, replace=False)
    train_subset = Subset(train_data, indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Create transfer model
    letter_model = LetterClassifier(
        pretrained_extractor=pretrained_model.feature_extractor,
        num_classes=26,
        freeze_features=freeze
    ).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, letter_model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print(f"\nTransfer Learning ({'frozen' if freeze else 'fine-tuned'} features)")
    for epoch in range(epochs):
        letter_model.train()
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = letter_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        # Evaluate
        letter_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = letter_model(images)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)
        
        print(f"Epoch {epoch+1}: Train Acc: {100*correct/total:.2f}%, Test Acc: {100*test_correct/test_total:.2f}%")
    
    return letter_model


if __name__ == "__main__":
    print("Step 1: Training on MNIST digits")
    digit_model = train_on_mnist(epochs=5)
    
    print("\nStep 2: Transfer to EMNIST letters (frozen features)")
    letter_model_frozen = transfer_to_letters(digit_model, epochs=10, freeze=True)
    
    print("\nStep 3: Transfer to EMNIST letters (fine-tuned)")
    letter_model_finetuned = transfer_to_letters(digit_model, epochs=10, freeze=False)
```

---

## Question 5: Using TensorFlow, extract feature vectors from a pre-trained model and use them to train a new classifier
- [ ] Done

### Complete Implementation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os

class FeatureExtractor:
    """Extract features from pre-trained CNN models"""
    
    def __init__(self, model_name='resnet50'):
        if model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        elif model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = base_model
        self.input_shape = (224, 224)
    
    def preprocess(self, images):
        """Preprocess images for the model"""
        if self.model.name.startswith('resnet'):
            return tf.keras.applications.resnet50.preprocess_input(images)
        elif self.model.name.startswith('vgg'):
            return tf.keras.applications.vgg16.preprocess_input(images)
    
    def extract_features(self, image_dir, batch_size=32):
        """Extract features from all images in a directory"""
        
        datagen = ImageDataGenerator(preprocessing_function=self.preprocess)
        
        generator = datagen.flow_from_directory(
            image_dir,
            target_size=self.input_shape,
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        # Extract features in batches
        features = []
        labels = []
        
        num_batches = len(generator)
        for i in range(num_batches):
            batch_images, batch_labels = next(generator)
            batch_features = self.model.predict(batch_images, verbose=0)
            features.append(batch_features)
            labels.append(batch_labels)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_batches} batches")
        
        features = np.vstack(features)
        labels = np.concatenate(labels)
        
        return features, labels, generator.class_indices
    
    def extract_single_image(self, image_path):
        """Extract features from a single image"""
        img = keras.preprocessing.image.load_img(image_path, target_size=self.input_shape)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = self.preprocess(img_array)
        
        features = self.model.predict(img_array, verbose=0)
        return features.flatten()


def train_classifiers(X_train, y_train, X_val, y_val):
    """Train multiple classifiers on extracted features"""
    
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        
        results[name] = {
            'model': clf,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
    
    return results


def build_neural_classifier(input_dim, num_classes):
    """Build a simple neural network classifier for the features"""
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Main execution
if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    MODEL_NAME = 'resnet50'
    
    # Initialize feature extractor
    print(f"Loading {MODEL_NAME} for feature extraction...")
    extractor = FeatureExtractor(model_name=MODEL_NAME)
    
    # Extract features
    print("\nExtracting training features...")
    X_train, y_train, class_indices = extractor.extract_features(TRAIN_DIR)
    print(f"Training features shape: {X_train.shape}")
    
    print("\nExtracting validation features...")
    X_val, y_val, _ = extractor.extract_features(VAL_DIR)
    print(f"Validation features shape: {X_val.shape}")
    
    # Save features for future use
    np.save('train_features.npy', X_train)
    np.save('train_labels.npy', y_train)
    np.save('val_features.npy', X_val)
    np.save('val_labels.npy', y_val)
    
    # Train traditional ML classifiers
    print("\n" + "="*50)
    print("Training Traditional ML Classifiers")
    print("="*50)
    results = train_classifiers(X_train, y_train, X_val, y_val)
    
    # Train neural network classifier
    print("\n" + "="*50)
    print("Training Neural Network Classifier")
    print("="*50)
    
    num_classes = len(class_indices)
    nn_model = build_neural_classifier(X_train.shape[1], num_classes)
    
    history = nn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=1
    )
    
    final_val_acc = nn_model.evaluate(X_val, y_val)[1]
    print(f"\nNeural Network Validation Accuracy: {final_val_acc:.4f}")
```

---

## Question 6: How would you implement transfer learning for enhancing a model trained to recognize car models to also recognize trucks?
- [ ] Done

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm

class VehicleClassifier:
    """Transfer learning from cars to trucks"""
    
    def __init__(self, num_car_classes, num_truck_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_car_classes = num_car_classes
        self.num_truck_classes = num_truck_classes
        
        # Load pre-trained backbone
        backbone = models.resnet50(pretrained=True)
        self.feature_dim = backbone.fc.in_features
        
        # Remove original classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Car classifier (trained first)
        self.car_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_car_classes)
        )
        
        # Truck classifier (transfer learned)
        self.truck_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_truck_classes)
        )
        
        self.backbone.to(self.device)
        self.car_classifier.to(self.device)
        self.truck_classifier.to(self.device)
    
    def get_transforms(self, is_training=True):
        if is_training:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def train_on_cars(self, car_train_dir, car_val_dir, epochs=10, batch_size=32):
        """Phase 1: Train on car dataset"""
        print("Phase 1: Training on car dataset")
        
        train_data = datasets.ImageFolder(car_train_dir, self.get_transforms(True))
        val_data = datasets.ImageFolder(car_val_dir, self.get_transforms(False))
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        # Train backbone + car classifier
        params = list(self.backbone.parameters()) + list(self.car_classifier.parameters())
        optimizer = optim.Adam(params, lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(epochs):
            # Training
            self.backbone.train()
            self.car_classifier.train()
            
            correct, total = 0, 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                features = self.backbone(images)
                outputs = self.car_classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            train_acc = 100 * correct / total
            
            # Validation
            val_acc = self._evaluate(val_loader, self.car_classifier)
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'backbone': self.backbone.state_dict(),
                    'car_classifier': self.car_classifier.state_dict()
                }, 'car_model.pt')
        
        return best_acc
    
    def transfer_to_trucks(self, truck_train_dir, truck_val_dir, epochs=10, 
                           batch_size=32, freeze_backbone=True):
        """Phase 2: Transfer to truck classification"""
        print(f"\nPhase 2: Transfer to truck classification (backbone {'frozen' if freeze_backbone else 'unfrozen'})")
        
        train_data = datasets.ImageFolder(truck_train_dir, self.get_transforms(True))
        val_data = datasets.ImageFolder(truck_val_dir, self.get_transforms(False))
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        # Freeze backbone if specified
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
        
        # Initialize truck classifier from car classifier weights
        # Transfer learned representations
        with torch.no_grad():
            self.truck_classifier[1].weight.copy_(self.car_classifier[1].weight)
            self.truck_classifier[1].bias.copy_(self.car_classifier[1].bias)
            self.truck_classifier[3].weight.copy_(self.car_classifier[3].weight)
            self.truck_classifier[3].bias.copy_(self.car_classifier[3].bias)
            # Reinitialize final layer for different number of classes
            nn.init.xavier_uniform_(self.truck_classifier[5].weight)
            nn.init.zeros_(self.truck_classifier[5].bias)
        
        # Setup optimizer
        if freeze_backbone:
            params = self.truck_classifier.parameters()
            lr = 1e-3
        else:
            params = [
                {'params': self.backbone.parameters(), 'lr': 1e-5},
                {'params': self.truck_classifier.parameters(), 'lr': 1e-3}
            ]
            lr = None  # Using param groups
        
        optimizer = optim.Adam(params, lr=lr) if lr else optim.Adam(params)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(epochs):
            # Training
            self.backbone.train() if not freeze_backbone else self.backbone.eval()
            self.truck_classifier.train()
            
            correct, total = 0, 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                if freeze_backbone:
                    with torch.no_grad():
                        features = self.backbone(images)
                else:
                    features = self.backbone(images)
                
                outputs = self.truck_classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            train_acc = 100 * correct / total
            
            # Validation
            val_acc = self._evaluate(val_loader, self.truck_classifier)
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'backbone': self.backbone.state_dict(),
                    'truck_classifier': self.truck_classifier.state_dict()
                }, 'truck_model.pt')
        
        return best_acc
    
    def _evaluate(self, dataloader, classifier):
        self.backbone.eval()
        classifier.eval()
        
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.backbone(images)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return 100 * correct / total
    
    def joint_training(self, car_train_dir, truck_train_dir, val_dirs, epochs=10, batch_size=32):
        """Optional: Joint multi-task learning"""
        print("\nJoint multi-task training (cars + trucks)")
        
        car_train = datasets.ImageFolder(car_train_dir, self.get_transforms(True))
        truck_train = datasets.ImageFolder(truck_train_dir, self.get_transforms(True))
        
        # Combine datasets with task labels
        # ... (implementation for multi-task learning)
        pass


# Usage
if __name__ == "__main__":
    NUM_CAR_CLASSES = 196  # Stanford Cars dataset
    NUM_TRUCK_CLASSES = 50  # Custom truck classes
    
    classifier = VehicleClassifier(NUM_CAR_CLASSES, NUM_TRUCK_CLASSES)
    
    # Phase 1: Train on cars
    car_acc = classifier.train_on_cars(
        car_train_dir='data/cars/train',
        car_val_dir='data/cars/val',
        epochs=10
    )
    print(f"Best car classification accuracy: {car_acc:.2f}%")
    
    # Phase 2: Transfer to trucks (frozen)
    truck_acc_frozen = classifier.transfer_to_trucks(
        truck_train_dir='data/trucks/train',
        truck_val_dir='data/trucks/val',
        epochs=10,
        freeze_backbone=True
    )
    print(f"Truck accuracy (frozen backbone): {truck_acc_frozen:.2f}%")
    
    # Phase 3: Fine-tune (optional)
    truck_acc_finetuned = classifier.transfer_to_trucks(
        truck_train_dir='data/trucks/train',
        truck_val_dir='data/trucks/val',
        epochs=5,
        freeze_backbone=False
    )
    print(f"Truck accuracy (fine-tuned): {truck_acc_finetuned:.2f}%")
```

---
