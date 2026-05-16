# Computer Vision Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a Python program to perform edge detection using the Canny filter.**

**Answer:**

The Canny edge detection algorithm is one of the most popular and effective edge detection techniques in computer vision. It uses a multi-stage algorithm to detect a wide range of edges in images with good localization and minimal noise.

### Complete Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature
import argparse
import os

class**Implement a simple image classifier using a pre-trained CNN with TensorFlow.**

**Answer:**

Image classification using pre-trained CNNs is a fundamental task in computer vision that leverages transfer learning to achieve high accuracy with minimal training time. This approach uses models pre-trained on large datasets like ImageNet.

### Complete Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, EfficientNetB0, 
    InceptionV3, DenseNet121
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from PIL import Image
import json
import time

class PretrainedImageClassifier:
    """
    Comprehensive image classifier using pre-trained CNNs
    """
    
    def __init__(self, model_name='ResNet50', num_classes=10, input_shape=(224, 224, 3)):
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.base_model = None
        self.history = None
        self.class_names = None
        
        # Model configurations
        self.model_configs = {
            'VGG16': {'model': VGG16, 'input_size': (224, 224), 'preprocess': tf.keras.applications.vgg16.preprocess_input},
            'ResNet50': {'model': ResNet50, 'input_size': (224, 224), 'preprocess': tf.keras.applications.resnet50.preprocess_input},
            'MobileNetV2': {'model': MobileNetV2, 'input_size': (224, 224), 'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input},
            'EfficientNetB0': {'model': EfficientNetB0, 'input_size': (224, 224), 'preprocess': tf.keras.applications.efficientnet.preprocess_input},
            'InceptionV3': {'model': InceptionV3, 'input_size': (299, 299), 'preprocess': tf.keras.applications.inception_v3.preprocess_input},
            'DenseNet121': {'model': DenseNet121, 'input_size': (224, 224), 'preprocess': tf.keras.applications.densenet.preprocess_input}
        }
        
    def build_model(self, fine_tune=False, fine_tune_layers=None):
        """
        Build the complete model with pre-trained base and custom head
        """
        config = self.model_configs[self.model_name]
        
        # Update input shape based on model requirements
        if self.model_name == 'InceptionV3':
            self.input_shape = (299, 299, 3)
        else:
            self.input_shape = (224, 224, 3)
        
        # Load pre-trained base model
        self.base_model = config['model'](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        self.base_model.trainable = False
        
        # Build complete model
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Preprocessing
        x = config['preprocess'](inputs)
        
        # Base model
        x = self.base_model(x, training=False)
        
        # Custom head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model = tf.keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        print(f"Model built with {self.model_name} backbone")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        return self.model
    
    def setup_fine_tuning(self, fine_tune_from_layer=None, learning_rate=0.0001):
        """
        Setup fine-tuning by unfreezing some layers of the base model
        """
        if self.base_model is None:
            raise ValueError("Model must be built before fine-tuning setup")
        
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Fine-tune from specific layer
        if fine_tune_from_layer is not None:
            for layer in self.base_model.layers[:fine_tune_from_layer]:
                layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        print(f"Fine-tuning enabled from layer {fine_tune_from_layer}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
    
    def create_data_generators(self, train_dir, val_dir, test_dir=None, batch_size=32):
        """
        Create data generators with appropriate preprocessing
        """
        config = self.model_configs[self.model_name]
        target_size = config['input_size']
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            rescale=1./255
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse' if self.num_classes > 2 else 'binary'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse' if self.num_classes > 2 else 'binary',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        generators = {
            'train': train_generator,
            'validation': val_generator
        }
        
        # Test generator if provided
        if test_dir:
            test_generator = val_datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='sparse' if self.num_classes > 2 else 'binary',
                shuffle=False
            )
            generators['test'] = test_generator
        
        return generators
    
    def train_model(self, train_generator, val_generator, epochs=20, 
                   fine_tune_epochs=10, callbacks=None):
        """
        Train the model with initial training and optional fine-tuning
        """
        # Callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
        
        print("Starting initial training...")
        
        # Initial training with frozen base
        history1 = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        if fine_tune_epochs > 0:
            print("\nStarting fine-tuning...")
            
            # Setup fine-tuning
            fine_tune_from = len(self.base_model.layers) // 2  # Unfreeze last half
            self.setup_fine_tuning(fine_tune_from, learning_rate=0.0001/10)
            
            # Continue training
            history2 = self.model.fit(
                train_generator,
                initial_epoch=len(history1.history['loss']),
                epochs=epochs + fine_tune_epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            self.history = {
                key: history1.history[key] + history2.history[key]
                for key in history1.history.keys()
            }
        else:
            self.history = history1.history
        
        return self.history
    
    def predict_single_image(self, image_path, return_probabilities=False):
        """
        Predict class for a single image
        """
        config = self.model_configs[self.model_name]
        target_size = config['input_size']
        
        # Load and preprocess image
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0
        
        # Predict
        predictions = self.model.predict(image_array, verbose=0)
        
        if self.num_classes == 2:
            predicted_class = int(predictions[0] > 0.5)
            confidence = float(predictions[0]) if predicted_class == 1 else float(1 - predictions[0])
        else:
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
        
        result = {
            'class_index': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence
        }
        
        if return_probabilities:
            if self.num_classes == 2:
                result['probabilities'] = {
                    self.class_names[0]: float(1 - predictions[0]),
                    self.class_names[1]: float(predictions[0])
                }
            else:
                result['probabilities'] = {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(self.num_classes)
                }
        
        return result
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict classes for multiple images
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for path in batch_paths:
                try:
                    result = self.predict_single_image(path, return_probabilities=True)
                    result['image_path'] = path
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    batch_results.append({
                        'image_path': path,
                        'error': str(e)
                    })
            
            results.extend(batch_results)
        
        return results
    
    def evaluate_model(self, test_generator, save_results=False):
        """
        Comprehensive model evaluation
        """
        print("Evaluating model...")
        
        # Predict on test set
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=1)
        
        if self.num_classes == 2:
            y_pred = (predictions > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(predictions, axis=1)
        
        y_true = test_generator.classes
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Visualize results
        self.plot_evaluation_results(report, cm, save_results)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy']
        }
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0,0].plot(self.history['loss'], label='Training Loss')
        axes[0,0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Accuracy
        axes[0,1].plot(self.history['accuracy'], label='Training Accuracy')
        axes[0,1].plot(self.history['val_accuracy'], label='Validation Accuracy')
        axes[0,1].set_title('Model Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history:
            axes[1,0].plot(self.history['lr'])
            axes[1,0].set_title('Learning Rate')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].set_yscale('log')
            axes[1,0].grid(True)
        
        # Model architecture summary
        axes[1,1].text(0.1, 0.5, f'Model: {self.model_name}\n'
                                 f'Classes: {self.num_classes}\n'
                                 f'Parameters: {self.model.count_params():,}\n'
                                 f'Input Shape: {self.input_shape}',
                      transform=axes[1,1].transAxes, fontsize=12,
                      verticalalignment='center')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_evaluation_results(self, report, cm, save_results=False):
        """
        Plot evaluation results
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Classification Report
        metrics_df = {
            'Precision': [report[cls]['precision'] for cls in self.class_names],
            'Recall': [report[cls]['recall'] for cls in self.class_names],
            'F1-Score': [report[cls]['f1-score'] for cls in self.class_names]
        }
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        axes[1].bar(x - width, metrics_df['Precision'], width, label='Precision')
        axes[1].bar(x, metrics_df['Recall'], width, label='Recall')
        axes[1].bar(x + width, metrics_df['F1-Score'], width, label='F1-Score')
        
        axes[1].set_xlabel('Classes')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Classification Metrics by Class')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.class_names, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_results:
            plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath):
        """Save the complete model"""
        self.model.save(filepath)
        
        # Save class names and configuration
        config = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'input_shape': self.input_shape
        }
        
        with open(filepath.replace('.h5', '_config.json'), 'w') as f:
            json.dump(config, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved model"""
        # Load configuration
        config_path = filepath.replace('.h5', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create instance
        classifier = cls(
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            input_shape=tuple(config['input_shape'])
        )
        
        # Load model
        classifier.model = tf.keras.models.load_model(filepath)
        classifier.class_names = config['class_names']
        
        return classifier

# Utility functions
def compare_models(train_dir, val_dir, test_dir, models=['ResNet50', 'MobileNetV2', 'EfficientNetB0']):
    """
    Compare different pre-trained models
    """
    results = {}
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Initialize classifier
        classifier = PretrainedImageClassifier(model_name=model_name)
        classifier.build_model()
        
        # Create data generators
        generators = classifier.create_data_generators(train_dir, val_dir, test_dir)
        
        # Train model
        start_time = time.time()
        history = classifier.train_model(
            generators['train'], 
            generators['validation'], 
            epochs=10,
            fine_tune_epochs=5
        )
        training_time = time.time() - start_time
        
        # Evaluate
        evaluation = classifier.evaluate_model(generators['test'])
        
        results[model_name] = {
            'accuracy': evaluation['accuracy'],
            'training_time': training_time,
            'parameters': classifier.model.count_params(),
            'model_size_mb': classifier.model.count_params() * 4 / (1024 * 1024)  # Approximate
        }
    
    # Compare results
    comparison_df = pd.DataFrame(results).T
    print("\nModel Comparison:")
    print(comparison_df)
    
    return results

# Example usage
def main():
    """
    Example usage of the PretrainedImageClassifier
    """
    # Example with a simple dataset structure
    # dataset/
    #   train/
    #     class1/
    #     class2/
    #   validation/
    #     class1/
    #     class2/
    #   test/
    #     class1/
    #     class2/
    
    # Initialize classifier
    classifier = PretrainedImageClassifier(
        model_name='ResNet50',
        num_classes=10  # Will be automatically detected from data
    )
    
    # Build model
    model = classifier.build_model()
    
    # Uncomment these lines when you have actual data
    # generators = classifier.create_data_generators(
    #     train_dir='dataset/train',
    #     val_dir='dataset/validation',
    #     test_dir='dataset/test'
    # )
    
    # Train model
    # history = classifier.train_model(
    #     generators['train'],
    #     generators['validation'],
    #     epochs=20,
    #     fine_tune_epochs=10
    # )
    
    # Evaluate
    # evaluation = classifier.evaluate_model(generators['test'])
    
    # Single prediction
    # result = classifier.predict_single_image('path/to/image.jpg')
    # print(f"Predicted class: {result['class_name']} (confidence: {result['confidence']:.2f})")
    
    # Save model
    # classifier.save_model('my_classifier.h5')
    
    print("Classifier setup complete!")

if __name__ == "__main__":
    main()
```

### Key Features and Optimizations

#### **1. Multiple Pre-trained Models:**
- **ResNet50**: Good balance of accuracy and speed
- **MobileNetV2**: Lightweight for mobile deployment
- **EfficientNetB0**: State-of-the-art efficiency
- **VGG16**: Simple architecture, good for understanding
- **InceptionV3**: Excellent for complex image classification
- **DenseNet121**: Dense connections for feature reuse

#### **2. Transfer Learning Strategies:**
```python
# Feature extraction (freeze base model)
base_model.trainable = False

# Fine-tuning (unfreeze some layers)
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False
```

#### **3. Data Augmentation:**
```python
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

#### **4. Best Practices:**

1. **Progressive Training**: Start with frozen base, then fine-tune
2. **Learning Rate Scheduling**: Reduce LR for fine-tuning
3. **Data Augmentation**: Increase dataset diversity
4. **Early Stopping**: Prevent overfitting
5. **Model Checkpointing**: Save best performing models

#### **5. Common Pitfalls and Solutions:**

**Problem**: Overfitting on small datasets
**Solution**: Use strong data augmentation and dropout

**Problem**: Poor transfer learning performance
**Solution**: Match pre-training and target domains, adjust fine-tuning strategy

**Problem**: Slow training
**Solution**: Use mixed precision training and optimize batch size

#### **6. Usage Examples:**

```python
# Quick setup for binary classification
classifier = PretrainedImageClassifier(model_name='MobileNetV2', num_classes=2)
classifier.build_model()

# Compare multiple models
results = compare_models(train_dir, val_dir, test_dir, 
                        models=['ResNet50', 'MobileNetV2', 'EfficientNetB0'])

# Production deployment
classifier.save_model('production_model.h5')
loaded_classifier = PretrainedImageClassifier.load_model('production_model.h5')
```

This implementation provides a complete, production-ready solution for image classification using pre-trained CNNs with comprehensive evaluation, comparison tools, and deployment capabilities.nnyEdgeDetector:
    """
    A comprehensive Canny edge detection implementation with multiple approaches
    """
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.edges = None
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        try:
            # Read image
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB for matplotlib display
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            print(f"Image loaded successfully: {self.original_image.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def preprocess_image(self, image, gaussian_blur=True, kernel_size=5):
        """
        Preprocess image before edge detection
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        if gaussian_blur:
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        return gray
    
    def opencv_canny(self, low_threshold=50, high_threshold=150, aperture_size=3, l2_gradient=False):
        """
        Canny edge detection using OpenCV implementation
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Preprocess image
        gray = self.preprocess_image(self.original_image)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold, 
                         apertureSize=aperture_size, L2gradient=l2_gradient)
        
        return edges, gray
    
    def manual_canny(self, low_threshold=50, high_threshold=150, sigma=1.0):
        """
        Manual implementation of Canny edge detection for educational purposes
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Step 1: Convert to grayscale and apply Gaussian blur
        gray = self.preprocess_image(self.original_image)
        
        # Step 2: Calculate gradients using Sobel operators
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Step 3: Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Step 4: Non-maximum suppression
        suppressed = self._non_maximum_suppression(magnitude, direction)
        
        # Step 5: Double thresholding
        edges = self._double_threshold(suppressed, low_threshold, high_threshold)
        
        # Step 6: Edge tracking by hysteresis
        final_edges = self._hysteresis_tracking(edges)
        
        return final_edges, gray, magnitude, direction
    
    def _non_maximum_suppression(self, magnitude, direction):
        """
        Non-maximum suppression to thin edges
        """
        M, N = magnitude.shape
        suppressed = np.zeros((M, N), dtype=np.float32)
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q = 255
                    r = 255
                    
                    # Angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = magnitude[i, j+1]
                        r = magnitude[i, j-1]
                    # Angle 45
                    elif 22.5 <= angle[i,j] < 67.5:
                        q = magnitude[i+1, j-1]
                        r = magnitude[i-1, j+1]
                    # Angle 90
                    elif 67.5 <= angle[i,j] < 112.5:
                        q = magnitude[i+1, j]
                        r = magnitude[i-1, j]
                    # Angle 135
                    elif 112.5 <= angle[i,j] < 157.5:
                        q = magnitude[i-1, j-1]
                        r = magnitude[i+1, j+1]
                    
                    if magnitude[i,j] >= q and magnitude[i,j] >= r:
                        suppressed[i,j] = magnitude[i,j]
                    else:
                        suppressed[i,j] = 0
                        
                except IndexError:
                    pass
                    
        return suppressed
    
    def _double_threshold(self, img, low_threshold, high_threshold):
        """
        Apply double thresholding
        """
        high_threshold_ratio = 0.15
        low_threshold_ratio = 0.05
        
        if high_threshold is None:
            high_threshold = img.max() * high_threshold_ratio
        if low_threshold is None:
            low_threshold = high_threshold * low_threshold_ratio
        
        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)
        
        weak = np.int32(25)
        strong = np.int32(255)
        
        strong_i, strong_j = np.where(img >= high_threshold)
        zeros_i, zeros_j = np.where(img < low_threshold)
        weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        return res
    
    def _hysteresis_tracking(self, img):
        """
        Edge tracking by hysteresis
        """
        M, N = img.shape
        weak = 25
        strong = 255
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                if img[i,j] == weak:
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or 
                            (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or 
                            (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or 
                            (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError:
                        pass
        return img
    
    def adaptive_canny(self, auto_threshold=True, sigma=0.33):
        """
        Adaptive Canny with automatic threshold selection
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        # Preprocess image
        gray = self.preprocess_image(self.original_image)
        
        if auto_threshold:
            # Automatic threshold selection based on image statistics
            median = np.median(gray)
            lower = int(max(0, (1.0 - sigma) * median))
            upper = int(min(255, (1.0 + sigma) * median))
        else:
            lower, upper = 50, 150
        
        # Apply Canny
        edges = cv2.Canny(gray, lower, upper)
        
        return edges, gray, lower, upper
    
    def multi_scale_canny(self, scales=[1.0, 1.5, 2.0]):
        """
        Multi-scale Canny edge detection
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        gray = self.preprocess_image(self.original_image)
        all_edges = []
        
        for scale in scales:
            # Apply Gaussian blur with different scales
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=scale, sigmaY=scale)
            
            # Adaptive thresholding for each scale
            median = np.median(blurred)
            lower = int(max(0, 0.67 * median))
            upper = int(min(255, 1.33 * median))
            
            edges = cv2.Canny(blurred, lower, upper)
            all_edges.append(edges)
        
        # Combine edges from different scales
        combined_edges = np.zeros_like(gray)
        for edges in all_edges:
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        return combined_edges, all_edges
    
    def visualize_results(self, save_path=None):
        """
        Comprehensive visualization of edge detection results
        """
        if self.original_image is None:
            print("No image loaded!")
            return
        
        # Apply different methods
        opencv_edges, gray = self.opencv_canny()
        manual_edges, _, magnitude, direction = self.manual_canny()
        adaptive_edges, _, lower, upper = self.adaptive_canny()
        multiscale_edges, scale_edges = self.multi_scale_canny()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Original image
        axes[0,0].imshow(self.original_image)
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Grayscale
        axes[0,1].imshow(gray, cmap='gray')
        axes[0,1].set_title('Grayscale')
        axes[0,1].axis('off')
        
        # Gradient magnitude
        axes[0,2].imshow(magnitude, cmap='hot')
        axes[0,2].set_title('Gradient Magnitude')
        axes[0,2].axis('off')
        
        # OpenCV Canny
        axes[1,0].imshow(opencv_edges, cmap='gray')
        axes[1,0].set_title('OpenCV Canny (50, 150)')
        axes[1,0].axis('off')
        
        # Manual Canny
        axes[1,1].imshow(manual_edges, cmap='gray')
        axes[1,1].set_title('Manual Canny Implementation')
        axes[1,1].axis('off')
        
        # Adaptive Canny
        axes[1,2].imshow(adaptive_edges, cmap='gray')
        axes[1,2].set_title(f'Adaptive Canny ({lower}, {upper})')
        axes[1,2].axis('off')
        
        # Multi-scale Canny
        axes[2,0].imshow(multiscale_edges, cmap='gray')
        axes[2,0].set_title('Multi-scale Canny')
        axes[2,0].axis('off')
        
        # Gradient direction
        axes[2,1].imshow(direction, cmap='hsv')
        axes[2,1].set_title('Gradient Direction')
        axes[2,1].axis('off')
        
        # Edge overlay
        overlay = self.original_image.copy()
        overlay[opencv_edges > 0] = [255, 0, 0]  # Red edges
        axes[2,2].imshow(overlay)
        axes[2,2].set_title('Edges Overlay')
        axes[2,2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()
        
        return {
            'opencv_edges': opencv_edges,
            'manual_edges': manual_edges,
            'adaptive_edges': adaptive_edges,
            'multiscale_edges': multiscale_edges
        }

def compare_threshold_effects(image_path):
    """
    Compare different threshold combinations
    """
    detector = CannyEdgeDetector()
    detector.load_image(image_path)
    
    threshold_pairs = [
        (30, 100), (50, 150), (100, 200), (150, 250)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (low, high) in enumerate(threshold_pairs):
        edges, _ = detector.opencv_canny(low, high)
        axes[i].imshow(edges, cmap='gray')
        axes[i].set_title(f'Canny Edges (low={low}, high={high})')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def batch_process_images(input_folder, output_folder, method='opencv'):
    """
    Batch process multiple images
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    detector = CannyEdgeDetector()
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing {filename}...")
            
            image_path = os.path.join(input_folder, filename)
            detector.load_image(image_path)
            
            if method == 'opencv':
                edges, _ = detector.opencv_canny()
            elif method == 'adaptive':
                edges, _, _, _ = detector.adaptive_canny()
            elif method == 'multiscale':
                edges, _ = detector.multi_scale_canny()
            
            # Save edges
            output_path = os.path.join(output_folder, f"edges_{filename}")
            cv2.imwrite(output_path, edges)

# Example usage and testing
def main():
    # Initialize detector
    detector = CannyEdgeDetector()
    
    # For demonstration, create a simple test image
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(test_image, (100, 100), 30, (0, 0, 0), -1)
    
    detector.original_image = test_image
    
    # Apply different methods
    print("Applying OpenCV Canny...")
    opencv_edges, gray = detector.opencv_canny(50, 150)
    
    print("Applying Manual Canny...")
    manual_edges, _, _, _ = detector.manual_canny(50, 150)
    
    print("Applying Adaptive Canny...")
    adaptive_edges, _, lower, upper = detector.adaptive_canny()
    print(f"Adaptive thresholds: {lower}, {upper}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0,0].imshow(test_image)
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(opencv_edges, cmap='gray')
    axes[0,1].set_title('OpenCV Canny')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(manual_edges, cmap='gray')
    axes[1,0].set_title('Manual Canny')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(adaptive_edges, cmap='gray')
    axes[1,1].set_title('Adaptive Canny')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Command line interface
    parser = argparse.ArgumentParser(description='Canny Edge Detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--low', type=int, default=50, help='Low threshold')
    parser.add_argument('--high', type=int, default=150, help='High threshold')
    parser.add_argument('--method', type=str, default='opencv', 
                       choices=['opencv', 'manual', 'adaptive', 'multiscale'],
                       help='Edge detection method')
    parser.add_argument('--output', type=str, help='Output path for results')
    
    args = parser.parse_args()
    
    if args.image:
        detector = CannyEdgeDetector()
        if detector.load_image(args.image):
            if args.method == 'opencv':
                edges, _ = detector.opencv_canny(args.low, args.high)
            elif args.method == 'adaptive':
                edges, _, _, _ = detector.adaptive_canny()
            elif args.method == 'multiscale':
                edges, _ = detector.multi_scale_canny()
            elif args.method == 'manual':
                edges, _, _, _ = detector.manual_canny(args.low, args.high)
            
            if args.output:
                cv2.imwrite(args.output, edges)
                print(f"Edges saved to {args.output}")
            
            # Visualize
            detector.visualize_results()
    else:
        main()
```

### Key Features and Optimizations

#### **1. Multiple Implementation Approaches:**
- **OpenCV Implementation**: Fast, optimized C++ backend
- **Manual Implementation**: Educational step-by-step process
- **Adaptive Thresholding**: Automatic parameter selection
- **Multi-scale Detection**: Better edge detection at different scales

#### **2. Performance Considerations:**
- **Vectorized Operations**: Using NumPy for fast computation
- **Memory Efficiency**: Processing images in-place where possible
- **Batch Processing**: Handle multiple images efficiently
- **Parameter Tuning**: Automatic threshold selection based on image statistics

#### **3. Common Pitfalls and Solutions:**

**Problem**: Noisy edges due to image noise
**Solution**: Apply Gaussian blur preprocessing

**Problem**: Disconnected edges
**Solution**: Use lower threshold values or post-processing morphological operations

**Problem**: Too many false edges
**Solution**: Increase threshold values or use adaptive thresholding

#### **4. Usage Examples:**

```python
# Basic usage
detector = CannyEdgeDetector()
detector.load_image('your_image.jpg')
edges, gray = detector.opencv_canny(low_threshold=50, high_threshold=150)

# Adaptive thresholding
adaptive_edges, _, low, high = detector.adaptive_canny()

# Compare different methods
results = detector.visualize_results(save_path='edge_comparison.png')

# Batch processing
batch_process_images('input_folder/', 'output_folder/', method='adaptive')
```

#### **5. Best Practices:**

1. **Preprocessing**: Always apply Gaussian blur to reduce noise
2. **Parameter Selection**: Use adaptive thresholding for unknown images
3. **Post-processing**: Apply morphological operations to clean up edges
4. **Validation**: Visual inspection and quantitative metrics
5. **Scale Invariance**: Use multi-scale approach for robust detection

This implementation provides a comprehensive, production-ready solution for edge detection with multiple approaches, optimization strategies, and practical usage examples.

---

## Question 3
- [ ] Done

**Create a function to automaticallycrop imagescentering on the main object.**

**Answer:**

Automatic image cropping that centers on the main object is a crucial task in computer vision applications like photo editing, content-aware resizing, and automated image processing pipelines. This involves object detection, saliency detection, and intelligent cropping strategies.

### Complete Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import os
from typing import Tuple, List, Optional, Dict, Union
import warnings
warnings.filterwarnings('ignore')

class AutomaticImageCropper:
    """
    Comprehensive automatic image cropping system that centers on main objects
    """
    
    def __init__(self):
        self.saliency_detector = None
        self.object_detector = None
        self.edge_detector = None
        
    def crop_center_object(self, image: np.ndarray, method: str = 'saliency', 
                          target_ratio: Optional[float] = None, 
                          margin_ratio: float = 0.1) -> Dict:
        """
        Main function to crop image centering on the main object
        
        Args:
            image: Input image (numpy array)
            method: Cropping method ('saliency', 'contour', 'object_detection', 'combined')
            target_ratio: Target aspect ratio (width/height). If None, uses original ratio
            margin_ratio: Additional margin around detected object (0.0 to 0.5)
            
        Returns:
            Dictionary containing cropped image and metadata
        """
        
        if method == 'saliency':
            return self._crop_by_saliency(image, target_ratio, margin_ratio)
        elif method == 'contour':
            return self._crop_by_contour(image, target_ratio, margin_ratio)
        elif method == 'object_detection':
            return self._crop_by_object_detection(image, target_ratio, margin_ratio)
        elif method == 'combined':
            return self._crop_by_combined_method(image, target_ratio, margin_ratio)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _crop_by_saliency(self, image: np.ndarray, target_ratio: Optional[float], 
                         margin_ratio: float) -> Dict:
        """
        Crop image based on saliency detection
        """
        # Generate saliency map
        saliency_map = self._compute_saliency_map(image)
        
        # Find bounding box of salient region
        bbox = self._get_saliency_bbox(saliency_map, threshold=0.3)
        
        # Adjust bbox with margin and target ratio
        adjusted_bbox = self._adjust_bbox(bbox, image.shape, target_ratio, margin_ratio)
        
        # Crop image
        cropped_image = self._crop_image_bbox(image, adjusted_bbox)
        
        return {
            'cropped_image': cropped_image,
            'bbox': adjusted_bbox,
            'method': 'saliency',
            'saliency_map': saliency_map,
            'confidence': self._compute_crop_confidence(saliency_map, adjusted_bbox)
        }
    
    def _crop_by_contour(self, image: np.ndarray, target_ratio: Optional[float], 
                        margin_ratio: float) -> Dict:
        """
        Crop image based on contour detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection and contour finding
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find main object contour
        main_contour = self._find_main_contour(contours, image.shape)
        
        if main_contour is not None:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            bbox = (x, y, x + w, y + h)
        else:
            # Fallback to center crop
            h, w = image.shape[:2]
            bbox = (w//4, h//4, 3*w//4, 3*h//4)
        
        # Adjust bbox
        adjusted_bbox = self._adjust_bbox(bbox, image.shape, target_ratio, margin_ratio)
        
        # Crop image
        cropped_image = self._crop_image_bbox(image, adjusted_bbox)
        
        return {
            'cropped_image': cropped_image,
            'bbox': adjusted_bbox,
            'method': 'contour',
            'main_contour': main_contour,
            'edges': edges,
            'confidence': 0.7 if main_contour is not None else 0.3
        }
    
    def _crop_by_object_detection(self, image: np.ndarray, target_ratio: Optional[float], 
                                 margin_ratio: float) -> Dict:
        """
        Crop image based on object detection (simplified using color clustering)
        """
        # Color-based object detection (simplified approach)
        # In production, use YOLO, R-CNN, or similar models
        
        object_mask = self._detect_objects_by_color(image)
        
        # Find bounding box of detected objects
        bbox = self._get_mask_bbox(object_mask)
        
        # Adjust bbox
        adjusted_bbox = self._adjust_bbox(bbox, image.shape, target_ratio, margin_ratio)
        
        # Crop image
        cropped_image = self._crop_image_bbox(image, adjusted_bbox)
        
        return {
            'cropped_image': cropped_image,
            'bbox': adjusted_bbox,
            'method': 'object_detection',
            'object_mask': object_mask,
            'confidence': self._compute_mask_confidence(object_mask)
        }
    
    def _crop_by_combined_method(self, image: np.ndarray, target_ratio: Optional[float], 
                               margin_ratio: float) -> Dict:
        """
        Combine multiple methods for robust cropping
        """
        # Get results from different methods
        saliency_result = self._crop_by_saliency(image, target_ratio, margin_ratio)
        contour_result = self._crop_by_contour(image, target_ratio, margin_ratio)
        object_result = self._crop_by_object_detection(image, target_ratio, margin_ratio)
        
        # Weight results based on confidence
        results = [saliency_result, contour_result, object_result]
        weights = [r['confidence'] for r in results]
        
        # Combine bounding boxes
        combined_bbox = self._combine_bboxes(
            [r['bbox'] for r in results], 
            weights, 
            image.shape
        )
        
        # Adjust combined bbox
        final_bbox = self._adjust_bbox(combined_bbox, image.shape, target_ratio, margin_ratio)
        
        # Crop image
        cropped_image = self._crop_image_bbox(image, final_bbox)
        
        return {
            'cropped_image': cropped_image,
            'bbox': final_bbox,
            'method': 'combined',
            'individual_results': results,
            'confidence': np.mean(weights)
        }
    
    def _compute_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute saliency map using multiple methods
        """
        # Method 1: Static saliency using spectral residual
        saliency_spectral = self._spectral_residual_saliency(image)
        
        # Method 2: Fine-grained saliency
        saliency_fine = self._fine_grained_saliency(image)
        
        # Method 3: Color contrast saliency
        saliency_color = self._color_contrast_saliency(image)
        
        # Combine saliency maps
        combined_saliency = (saliency_spectral + saliency_fine + saliency_color) / 3.0
        
        # Normalize
        combined_saliency = cv2.normalize(combined_saliency, None, 0, 1, cv2.NORM_MINMAX)
        
        return combined_saliency
    
    def _spectral_residual_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Spectral residual saliency detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Resize for faster processing
        small_size = (64, 64)
        small = cv2.resize(gray, small_size)
        
        # FFT
        f = np.fft.fft2(small)
        log_amplitude = np.log(np.abs(f) + 1e-10)
        phase = np.angle(f)
        
        # Spectral residual
        residual = log_amplitude - cv2.boxFilter(log_amplitude, -1, (3, 3))
        
        # Inverse FFT
        saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
        
        # Resize back and smooth
        saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))
        saliency = cv2.GaussianBlur(saliency, (11, 11), 2.5)
        
        return saliency
    
    def _fine_grained_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Fine-grained saliency using OpenCV
        """
        # Convert to Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Compute mean values
        mean_l = np.mean(lab[:, :, 0])
        mean_a = np.mean(lab[:, :, 1])
        mean_b = np.mean(lab[:, :, 2])
        
        # Compute saliency
        saliency = (lab[:, :, 0] - mean_l) ** 2 + \
                  (lab[:, :, 1] - mean_a) ** 2 + \
                  (lab[:, :, 2] - mean_b) ** 2
        
        # Normalize
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
        
        return saliency
    
    def _color_contrast_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Color contrast based saliency
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Compute local contrast
        kernel = np.ones((5, 5), np.float32) / 25
        
        saliency_maps = []
        for i in range(3):
            channel = lab[:, :, i].astype(np.float32)
            local_mean = cv2.filter2D(channel, -1, kernel)
            contrast = np.abs(channel - local_mean)
            saliency_maps.append(contrast)
        
        # Combine channels
        saliency = np.mean(saliency_maps, axis=0)
        
        # Normalize
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
        
        return saliency
    
    def _get_saliency_bbox(self, saliency_map: np.ndarray, threshold: float = 0.3) -> Tuple[int, int, int, int]:
        """
        Get bounding box from saliency map
        """
        # Threshold saliency map
        binary = (saliency_map > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback to center
            h, w = saliency_map.shape
            return (w//4, h//4, 3*w//4, 3*h//4)
        
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, x + w, y + h)
    
    def _find_main_contour(self, contours: List, image_shape: Tuple) -> Optional[np.ndarray]:
        """
        Find the main object contour
        """
        if not contours:
            return None
        
        # Filter contours by area and position
        h, w = image_shape[:2]
        min_area = (w * h) * 0.01  # At least 1% of image
        max_area = (w * h) * 0.8   # At most 80% of image
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Check if contour is not too close to edges
                x, y, cw, ch = cv2.boundingRect(contour)
                if x > w * 0.05 and y > h * 0.05 and (x + cw) < w * 0.95 and (y + ch) < h * 0.95:
                    valid_contours.append(contour)
        
        if not valid_contours:
            return None
        
        # Return largest valid contour
        return max(valid_contours, key=cv2.contourArea)
    
    def _detect_objects_by_color(self, image: np.ndarray) -> np.ndarray:
        """
        Simplified object detection using color clustering
        """
        # Reshape image for clustering
        data = image.reshape((-1, 3))
        
        # K-means clustering
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Find dominant colors
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Create mask for non-background colors
        # Assume background is the most frequent color
        background_label = unique_labels[np.argmax(counts)]
        
        # Create object mask
        object_mask = (labels != background_label).reshape(image.shape[:2])
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        object_mask = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        
        return object_mask
    
    def _get_mask_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box from binary mask
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback to center
            h, w = mask.shape
            return (w//4, h//4, 3*w//4, 3*h//4)
        
        # Get bounding box of all contours
        all_contours = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_contours)
        
        return (x, y, x + w, y + h)
    
    def _adjust_bbox(self, bbox: Tuple[int, int, int, int], image_shape: Tuple, 
                    target_ratio: Optional[float], margin_ratio: float) -> Tuple[int, int, int, int]:
        """
        Adjust bounding box with margin and target aspect ratio
        """
        x1, y1, x2, y2 = bbox
        h, w = image_shape[:2]
        
        # Add margin
        margin_x = int((x2 - x1) * margin_ratio)
        margin_y = int((y2 - y1) * margin_ratio)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Adjust for target ratio
        if target_ratio is not None:
            current_width = x2 - x1
            current_height = y2 - y1
            current_ratio = current_width / current_height
            
            if current_ratio > target_ratio:
                # Too wide, increase height
                new_height = current_width / target_ratio
                height_diff = new_height - current_height
                y1 = max(0, y1 - height_diff // 2)
                y2 = min(h, y2 + height_diff // 2)
            else:
                # Too tall, increase width
                new_width = current_height * target_ratio
                width_diff = new_width - current_width
                x1 = max(0, x1 - width_diff // 2)
                x2 = min(w, x2 + width_diff // 2)
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _combine_bboxes(self, bboxes: List[Tuple], weights: List[float], 
                       image_shape: Tuple) -> Tuple[int, int, int, int]:
        """
        Combine multiple bounding boxes using weighted average
        """
        if not bboxes:
            h, w = image_shape[:2]
            return (w//4, h//4, 3*w//4, 3*h//4)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(bboxes)
            total_weight = len(bboxes)
        
        weights = [w / total_weight for w in weights]
        
        # Weighted average
        x1 = sum(bbox[0] * w for bbox, w in zip(bboxes, weights))
        y1 = sum(bbox[1] * w for bbox, w in zip(bboxes, weights))
        x2 = sum(bbox[2] * w for bbox, w in zip(bboxes, weights))
        y2 = sum(bbox[3] * w for bbox, w in zip(bboxes, weights))
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _crop_image_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop image using bounding box
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def _compute_crop_confidence(self, saliency_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Compute confidence score for the crop
        """
        x1, y1, x2, y2 = bbox
        
        # Saliency within crop
        crop_saliency = saliency_map[y1:y2, x1:x2]
        crop_mean = np.mean(crop_saliency)
        
        # Saliency outside crop
        full_mean = np.mean(saliency_map)
        
        # Confidence based on contrast
        confidence = min(1.0, crop_mean / (full_mean + 1e-10))
        
        return confidence
    
    def _compute_mask_confidence(self, mask: np.ndarray) -> float:
        """
        Compute confidence based on mask quality
        """
        # Ratio of object pixels
        object_ratio = np.sum(mask) / mask.size
        
        # Confidence based on object size (not too small, not too large)
        if 0.1 <= object_ratio <= 0.7:
            confidence = 0.8
        elif 0.05 <= object_ratio <= 0.9:
            confidence = 0.6
        else:
            confidence = 0.3
        
        return confidence
    
    def batch_crop_images(self, image_paths: List[str], output_dir: str, 
                         method: str = 'combined', target_ratio: Optional[float] = None) -> List[Dict]:
        """
        Batch process multiple images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Crop image
                result = self.crop_center_object(image, method, target_ratio)
                
                # Save cropped image
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_cropped{ext}")
                
                cropped_rgb = result['cropped_image']
                cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, cropped_bgr)
                
                result['input_path'] = image_path
                result['output_path'] = output_path
                results.append(result)
                
                print(f"Processed {i+1}/{len(image_paths)}: {filename}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_cropping_process(self, image: np.ndarray, method: str = 'combined', 
                                  target_ratio: Optional[float] = None) -> None:
        """
        Visualize the cropping process
        """
        result = self.crop_center_object(image, method, target_ratio)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Draw bounding box on original
        bbox = result['bbox']
        image_with_bbox = image.copy()
        cv2.rectangle(image_with_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
        axes[0, 1].imshow(image_with_bbox)
        axes[0, 1].set_title(f'Detected Region (Confidence: {result["confidence"]:.2f})')
        axes[0, 1].axis('off')
        
        # Cropped result
        axes[0, 2].imshow(result['cropped_image'])
        axes[0, 2].set_title('Cropped Result')
        axes[0, 2].axis('off')
        
        # Method-specific visualizations
        if 'saliency_map' in result:
            axes[1, 0].imshow(result['saliency_map'], cmap='hot')
            axes[1, 0].set_title('Saliency Map')
            axes[1, 0].axis('off')
        
        if 'edges' in result:
            axes[1, 1].imshow(result['edges'], cmap='gray')
            axes[1, 1].set_title('Edge Detection')
            axes[1, 1].axis('off')
        
        if 'object_mask' in result:
            axes[1, 2].imshow(result['object_mask'], cmap='gray')
            axes[1, 2].set_title('Object Mask')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

# Utility functions
def smart_crop_for_social_media(image_path: str, platform: str = 'instagram') -> np.ndarray:
    """
    Smart cropping for social media platforms
    """
    # Platform aspect ratios
    ratios = {
        'instagram': 1.0,      # Square
        'facebook': 1.91,      # Landscape
        'twitter': 16/9,       # Wide
        'pinterest': 2/3,      # Portrait
        'tiktok': 9/16         # Vertical
    }
    
    target_ratio = ratios.get(platform, 1.0)
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Crop
    cropper = AutomaticImageCropper()
    result = cropper.crop_center_object(image, method='combined', target_ratio=target_ratio)
    
    return result['cropped_image']

def compare_cropping_methods(image: np.ndarray) -> Dict:
    """
    Compare different cropping methods
    """
    cropper = AutomaticImageCropper()
    
    methods = ['saliency', 'contour', 'object_detection', 'combined']
    results = {}
    
    for method in methods:
        result = cropper.crop_center_object(image, method=method)
        results[method] = result
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Method results
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
    for i, method in enumerate(methods):
        row, col = positions[i]
        axes[row, col].imshow(results[method]['cropped_image'])
        axes[row, col].set_title(f'{method.title()} (Conf: {results[method]["confidence"]:.2f})')
        axes[row, col].axis('off')
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Example usage
def main():
    """
    Example usage of the AutomaticImageCropper
    """
    # Create sample image for testing
    sample_image = np.zeros((300, 400, 3), dtype=np.uint8)
    sample_image[50:250, 100:300] = [255, 255, 255]  # White rectangle
    sample_image[100:200, 150:250] = [255, 0, 0]     # Red square in center
    
    # Initialize cropper
    cropper = AutomaticImageCropper()
    
    # Test different methods
    print("Testing different cropping methods...")
    
    # Saliency-based cropping
    saliency_result = cropper.crop_center_object(sample_image, method='saliency')
    print(f"Saliency method confidence: {saliency_result['confidence']:.2f}")
    
    # Combined method
    combined_result = cropper.crop_center_object(sample_image, method='combined')
    print(f"Combined method confidence: {combined_result['confidence']:.2f}")
    
    # Visualize process
    cropper.visualize_cropping_process(sample_image, method='combined')
    
    # Compare methods
    compare_results = compare_cropping_methods(sample_image)
    
    print("Cropping demonstration complete!")

if __name__ == "__main__":
    main()
```

### Key Features and Optimizations

#### **1. Multiple Cropping Approaches:**
- **Saliency Detection**: Uses spectral residual and color contrast
- **Contour Analysis**: Edge-based object boundary detection
- **Object Detection**: Color clustering for object segmentation
- **Combined Method**: Weighted fusion of all approaches

#### **2. Smart Bounding Box Adjustment:**
```python
# Automatic margin addition
margin_x = int((x2 - x1) * margin_ratio)

# Target aspect ratio enforcement
if current_ratio > target_ratio:
    new_height = current_width / target_ratio
```

#### **3. Confidence Scoring:**
```python
def _compute_crop_confidence(self, saliency_map, bbox):
    crop_saliency = saliency_map[y1:y2, x1:x2]
    crop_mean = np.mean(crop_saliency)
    full_mean = np.mean(saliency_map)
    confidence = min(1.0, crop_mean / (full_mean + 1e-10))
    return confidence
```

#### **4. Social Media Integration:**
```python
# Platform-specific aspect ratios
ratios = {
    'instagram': 1.0,      # Square
    'facebook': 1.91,      # Landscape
    'twitter': 16/9,       # Wide
    'pinterest': 2/3,      # Portrait
    'tiktok': 9/16         # Vertical
}
```

#### **5. Best Practices:**

1. **Multi-method Approach**: Combine different detection techniques
2. **Confidence Scoring**: Evaluate crop quality automatically
3. **Aspect Ratio Control**: Maintain desired proportions
4. **Margin Addition**: Prevent cutting off important parts
5. **Batch Processing**: Handle multiple images efficiently

#### **6. Common Pitfalls and Solutions:**

**Problem**: False positive object detection
**Solution**: Use multiple methods and confidence scoring

**Problem**: Cutting off important parts
**Solution**: Add margins and validate crop boundaries

**Problem**: Poor performance on complex scenes
**Solution**: Combine saliency, contour, and color-based methods

#### **7. Usage Examples:**

```python
# Basic usage
cropper = AutomaticImageCropper()
result = cropper.crop_center_object(image, method='combined', target_ratio=1.0)
cropped_image = result['cropped_image']

# Social media cropping
instagram_crop = smart_crop_for_social_media('photo.jpg', 'instagram')

# Batch processing
results = cropper.batch_crop_images(image_paths, 'output_folder', method='saliency')

# Method comparison
comparison = compare_cropping_methods(image)
```

#### **8. Performance Considerations:**

- **Saliency Computation**: Use smaller image sizes for faster processing
- **Contour Detection**: Optimize edge detection parameters
- **Memory Usage**: Process images in batches for large datasets
- **Real-time Processing**: Cache preprocessing results

This implementation provides a comprehensive, production-ready solution for automatic image cropping with multiple detection methods, confidence scoring, and platform-specific optimizations.

---

## Question 4
- [ ] Done

**Write code to segment an image intosuperpixelsusingSLIC (Simple Linear Iterative Clustering).**

**Answer:**

SLIC (Simple Linear Iterative Clustering) is one of the most popular superpixel segmentation algorithms. It groups pixels into perceptually meaningful atomic regions that can replace the rigid pixel grid structure. This is crucial for object recognition, segmentation, and various computer vision applications.

### Complete Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift
from skimage.measure import regionprops, label
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage
from typing import Tuple, List, Optional, Dict, Union
import time
import warnings
warnings.filterwarnings('ignore')

class SuperpixelSLIC:
    """
    Comprehensive SLIC superpixel segmentation implementation
    """
    
    def __init__(self):
        self.segments = None
        self.original_image = None
        self.lab_image = None
        self.centers = None
        self.distances = None
        
    def segment_image(self, image: np.ndarray, n_segments: int = 300, 
                     compactness: float = 10.0, max_iter: int = 10,
                     sigma: float = 0, method: str = 'slic') -> np.ndarray:
        """
        Main segmentation function using SLIC algorithm
        
        Args:
            image: Input RGB image
            n_segments: Approximate number of superpixels
            compactness: Balance between color and spatial proximity
            max_iter: Maximum number of iterations
            sigma: Gaussian smoothing before segmentation
            method: 'slic', 'custom', 'felzenszwalb', or 'quickshift'
            
        Returns:
            Segmented image with superpixel labels
        """
        self.original_image = image.copy()
        
        if method == 'slic':
            return self._slic_scikit(image, n_segments, compactness, max_iter, sigma)
        elif method == 'custom':
            return self._slic_custom(image, n_segments, compactness, max_iter, sigma)
        elif method == 'felzenszwalb':
            return self._felzenszwalb_segmentation(image, sigma)
        elif method == 'quickshift':
            return self._quickshift_segmentation(image, sigma)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _slic_scikit(self, image: np.ndarray, n_segments: int, 
                    compactness: float, max_iter: int, sigma: float) -> np.ndarray:
        """
        SLIC using scikit-image implementation
        """
        # Convert to LAB color space for better perceptual distance
        lab_image = rgb2lab(image)
        
        # Apply SLIC
        segments = slic(lab_image, 
                       n_segments=n_segments,
                       compactness=compactness,
                       max_iter=max_iter,
                       sigma=sigma,
                       start_label=1,
                       slic_zero=True)
        
        self.segments = segments
        self.lab_image = lab_image
        
        return segments
    
    def _slic_custom(self, image: np.ndarray, n_segments: int, 
                    compactness: float, max_iter: int, sigma: float) -> np.ndarray:
        """
        Custom SLIC implementation from scratch
        """
        # Convert to LAB color space
        lab_image = rgb2lab(image)
        h, w, c = lab_image.shape
        
        # Apply Gaussian smoothing if specified
        if sigma > 0:
            for i in range(c):
                lab_image[:, :, i] = ndimage.gaussian_filter(lab_image[:, :, i], sigma)
        
        # Initialize cluster centers
        step = int(np.sqrt(h * w / n_segments))
        centers = self._initialize_cluster_centers(lab_image, step)
        
        # Initialize labels and distances
        labels = np.full((h, w), -1, dtype=np.int32)
        distances = np.full((h, w), np.inf, dtype=np.float64)
        
        # SLIC iterations
        for iteration in range(max_iter):
            old_centers = centers.copy()
            
            # Reset distances for current iteration
            distances.fill(np.inf)
            
            # For each cluster center
            for i, center in enumerate(centers):
                self._update_cluster_assignment(lab_image, center, i, labels, distances, 
                                              step, compactness)
            
            # Update cluster centers
            centers = self._update_cluster_centers(lab_image, labels, len(centers))
            
            # Check convergence
            if self._check_convergence(old_centers, centers, tolerance=1e-3):
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Post-process to enforce connectivity
        labels = self._enforce_connectivity(labels, h, w)
        
        self.segments = labels
        self.lab_image = lab_image
        self.centers = centers
        self.distances = distances
        
        return labels
    
    def _initialize_cluster_centers(self, lab_image: np.ndarray, step: int) -> List[List[float]]:
        """
        Initialize cluster centers on a regular grid
        """
        h, w, c = lab_image.shape
        centers = []
        
        for y in range(step // 2, h - step // 2, step):
            for x in range(step // 2, w - step // 2, step):
                # Move center to lowest gradient position in 3x3 neighborhood
                min_grad = np.inf
                best_pos = (y, x)
                
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            # Compute gradient magnitude
                            grad = self._compute_gradient(lab_image, ny, nx)
                            if grad < min_grad:
                                min_grad = grad
                                best_pos = (ny, nx)
                
                y_best, x_best = best_pos
                center = [lab_image[y_best, x_best, 0],  # L
                         lab_image[y_best, x_best, 1],   # a
                         lab_image[y_best, x_best, 2],   # b
                         x_best,                         # x
                         y_best]                         # y
                centers.append(center)
        
        return centers
    
    def _compute_gradient(self, lab_image: np.ndarray, y: int, x: int) -> float:
        """
        Compute gradient magnitude at given position
        """
        h, w = lab_image.shape[:2]
        
        # Compute gradients in L channel
        grad_x = 0
        grad_y = 0
        
        if x > 0 and x < w - 1:
            grad_x = lab_image[y, x + 1, 0] - lab_image[y, x - 1, 0]
        
        if y > 0 and y < h - 1:
            grad_y = lab_image[y + 1, x, 0] - lab_image[y - 1, x, 0]
        
        return grad_x ** 2 + grad_y ** 2
    
    def _update_cluster_assignment(self, lab_image: np.ndarray, center: List[float], 
                                  cluster_id: int, labels: np.ndarray, distances: np.ndarray,
                                  step: int, compactness: float) -> None:
        """
        Update pixel assignments to clusters
        """
        h, w = lab_image.shape[:2]
        l_c, a_c, b_c, x_c, y_c = center
        
        # Search in 2S x 2S region around cluster center
        search_size = step
        
        y_min = max(0, int(y_c - search_size))
        y_max = min(h, int(y_c + search_size))
        x_min = max(0, int(x_c - search_size))
        x_max = min(w, int(x_c + search_size))
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # Color distance in LAB space
                dl = lab_image[y, x, 0] - l_c
                da = lab_image[y, x, 1] - a_c
                db = lab_image[y, x, 2] - b_c
                dc = np.sqrt(dl**2 + da**2 + db**2)
                
                # Spatial distance
                dx = x - x_c
                dy = y - y_c
                ds = np.sqrt(dx**2 + dy**2)
                
                # Combined distance
                distance = dc + (compactness * ds / step)
                
                # Update assignment if distance is smaller
                if distance < distances[y, x]:
                    distances[y, x] = distance
                    labels[y, x] = cluster_id
    
    def _update_cluster_centers(self, lab_image: np.ndarray, labels: np.ndarray, 
                               num_centers: int) -> List[List[float]]:
        """
        Update cluster centers based on current assignments
        """
        h, w = lab_image.shape[:2]
        new_centers = []
        
        for i in range(num_centers):
            # Find all pixels assigned to this cluster
            mask = (labels == i)
            
            if np.any(mask):
                # Compute mean color and position
                l_mean = np.mean(lab_image[mask, 0])
                a_mean = np.mean(lab_image[mask, 1])
                b_mean = np.mean(lab_image[mask, 2])
                
                y_coords, x_coords = np.where(mask)
                x_mean = np.mean(x_coords)
                y_mean = np.mean(y_coords)
                
                new_centers.append([l_mean, a_mean, b_mean, x_mean, y_mean])
            else:
                # Keep old center if no pixels assigned
                new_centers.append([50.0, 0.0, 0.0, w//2, h//2])
        
        return new_centers
    
    def _check_convergence(self, old_centers: List, new_centers: List, 
                          tolerance: float = 1e-3) -> bool:
        """
        Check if algorithm has converged
        """
        if len(old_centers) != len(new_centers):
            return False
        
        total_change = 0
        for old, new in zip(old_centers, new_centers):
            # Check spatial change
            dx = old[3] - new[3]
            dy = old[4] - new[4]
            spatial_change = np.sqrt(dx**2 + dy**2)
            total_change += spatial_change
        
        return total_change / len(old_centers) < tolerance
    
    def _enforce_connectivity(self, labels: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Enforce connectivity of superpixels
        """
        # Create new labels with proper connectivity
        new_labels = np.zeros_like(labels)
        label_count = 0
        
        # For each unique label
        unique_labels = np.unique(labels)
        
        for label_id in unique_labels:
            if label_id == -1:
                continue
                
            # Find connected components for this label
            mask = (labels == label_id)
            labeled_mask, num_components = ndimage.label(mask)
            
            for component_id in range(1, num_components + 1):
                component_mask = (labeled_mask == component_id)
                new_labels[component_mask] = label_count
                label_count += 1
        
        return new_labels
    
    def _felzenszwalb_segmentation(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Felzenszwalb's efficient graph-based segmentation
        """
        segments = felzenszwalb(image, scale=100, sigma=sigma, min_size=50)
        self.segments = segments
        return segments
    
    def _quickshift_segmentation(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Quick shift segmentation
        """
        segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5, sigma=sigma)
        self.segments = segments
        return segments
    
    def visualize_superpixels(self, image: np.ndarray, segments: np.ndarray, 
                             method: str = 'boundaries') -> np.ndarray:
        """
        Visualize superpixels with different methods
        """
        if method == 'boundaries':
            # Show boundaries
            marked = mark_boundaries(image, segments, color=(1, 0, 0), mode='thick')
            return marked
        
        elif method == 'average_color':
            # Fill each superpixel with average color
            segmented = np.zeros_like(image)
            
            for segment_id in np.unique(segments):
                mask = (segments == segment_id)
                if np.any(mask):
                    avg_color = np.mean(image[mask], axis=0)
                    segmented[mask] = avg_color
            
            return segmented
        
        elif method == 'random_colors':
            # Assign random colors to superpixels
            num_segments = len(np.unique(segments))
            colors = np.random.rand(num_segments, 3)
            
            segmented = np.zeros_like(image)
            for i, segment_id in enumerate(np.unique(segments)):
                mask = (segments == segment_id)
                segmented[mask] = colors[i]
            
            return segmented
        
        else:
            raise ValueError(f"Unknown visualization method: {method}")
    
    def extract_superpixel_features(self, image: np.ndarray, segments: np.ndarray) -> Dict:
        """
        Extract features for each superpixel
        """
        features = {
            'mean_color': [],
            'std_color': [],
            'area': [],
            'centroid': [],
            'perimeter': [],
            'solidity': [],
            'eccentricity': []
        }
        
        # Convert segments to labeled image
        labeled_segments = label(segments)
        
        # Extract region properties
        props = regionprops(labeled_segments, intensity_image=image)
        
        for prop in props:
            # Color statistics
            mask = (segments == prop.label - 1)
            if np.any(mask):
                region_colors = image[mask]
                features['mean_color'].append(np.mean(region_colors, axis=0))
                features['std_color'].append(np.std(region_colors, axis=0))
            else:
                features['mean_color'].append([0, 0, 0])
                features['std_color'].append([0, 0, 0])
            
            # Geometric properties
            features['area'].append(prop.area)
            features['centroid'].append(prop.centroid)
            features['perimeter'].append(prop.perimeter)
            features['solidity'].append(prop.solidity)
            features['eccentricity'].append(prop.eccentricity)
        
        return features
    
    def merge_similar_superpixels(self, image: np.ndarray, segments: np.ndarray, 
                                 color_threshold: float = 20.0) -> np.ndarray:
        """
        Merge similar adjacent superpixels
        """
        # Extract features
        features = self.extract_superpixel_features(image, segments)
        mean_colors = np.array(features['mean_color'])
        
        # Find adjacency relationships
        adjacency = self._find_adjacency(segments)
        
        # Merge similar superpixels
        merged_segments = segments.copy()
        segment_mapping = {}
        
        for segment_id in np.unique(segments):
            if segment_id in segment_mapping:
                continue
                
            current_color = mean_colors[segment_id]
            
            # Find similar adjacent superpixels
            for adj_id in adjacency.get(segment_id, []):
                if adj_id in segment_mapping:
                    continue
                    
                adj_color = mean_colors[adj_id]
                color_distance = np.linalg.norm(current_color - adj_color)
                
                if color_distance < color_threshold:
                    # Merge superpixels
                    segment_mapping[adj_id] = segment_id
                    merged_segments[segments == adj_id] = segment_id
        
        return merged_segments
    
    def _find_adjacency(self, segments: np.ndarray) -> Dict[int, List[int]]:
        """
        Find adjacent superpixels
        """
        h, w = segments.shape
        adjacency = {}
        
        # Check horizontal and vertical neighbors
        for y in range(h):
            for x in range(w):
                current_id = segments[y, x]
                
                if current_id not in adjacency:
                    adjacency[current_id] = set()
                
                # Check neighbors
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor_id = segments[ny, nx]
                        if neighbor_id != current_id:
                            adjacency[current_id].add(neighbor_id)
        
        # Convert sets to lists
        return {k: list(v) for k, v in adjacency.items()}
    
    def adaptive_slic(self, image: np.ndarray, min_segments: int = 100, 
                     max_segments: int = 1000, quality_threshold: float = 0.8) -> np.ndarray:
        """
        Adaptive SLIC that automatically determines optimal number of superpixels
        """
        best_segments = None
        best_quality = 0
        best_n_segments = min_segments
        
        # Test different numbers of superpixels
        for n_segments in range(min_segments, max_segments + 1, 50):
            segments = self.segment_image(image, n_segments=n_segments, method='slic')
            quality = self._evaluate_segmentation_quality(image, segments)
            
            if quality > best_quality:
                best_quality = quality
                best_segments = segments
                best_n_segments = n_segments
            
            # Early stopping if quality is good enough
            if quality > quality_threshold:
                break
        
        print(f"Best segmentation: {best_n_segments} superpixels, quality: {best_quality:.3f}")
        return best_segments
    
    def _evaluate_segmentation_quality(self, image: np.ndarray, segments: np.ndarray) -> float:
        """
        Evaluate quality of superpixel segmentation
        """
        # Compute within-superpixel variance
        total_variance = 0
        total_pixels = 0
        
        for segment_id in np.unique(segments):
            mask = (segments == segment_id)
            if np.any(mask):
                region_pixels = image[mask]
                if len(region_pixels) > 1:
                    variance = np.var(region_pixels, axis=0).sum()
                    total_variance += variance * len(region_pixels)
                total_pixels += len(region_pixels)
        
        # Normalize variance
        avg_variance = total_variance / total_pixels if total_pixels > 0 else float('inf')
        
        # Quality is inverse of variance (lower variance = higher quality)
        quality = 1.0 / (1.0 + avg_variance / 100.0)
        
        return quality
    
    def batch_segment_images(self, image_paths: List[str], output_dir: str, 
                           **kwargs) -> List[Dict]:
        """
        Batch process multiple images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Segment image
                start_time = time.time()
                segments = self.segment_image(image, **kwargs)
                processing_time = time.time() - start_time
                
                # Visualize superpixels
                visualized = self.visualize_superpixels(image, segments, method='boundaries')
                
                # Save result
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_superpixels{ext}")
                
                # Convert to BGR for saving
                output_bgr = (visualized * 255).astype(np.uint8)
                output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, output_bgr)
                
                # Store results
                result = {
                    'input_path': image_path,
                    'output_path': output_path,
                    'num_superpixels': len(np.unique(segments)),
                    'processing_time': processing_time,
                    'segments': segments
                }
                results.append(result)
                
                print(f"Processed {i+1}/{len(image_paths)}: {filename} "
                      f"({result['num_superpixels']} superpixels, {processing_time:.2f}s)")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'error': str(e)
                })
        
        return results

# Utility functions
def compare_segmentation_methods(image: np.ndarray) -> Dict:
    """
    Compare different superpixel segmentation methods
    """
    segmenter = SuperpixelSLIC()
    
    methods = {
        'SLIC (scikit)': {'method': 'slic', 'n_segments': 300},
        'SLIC (custom)': {'method': 'custom', 'n_segments': 300},
        'Felzenszwalb': {'method': 'felzenszwalb', 'sigma': 0.5},
        'Quickshift': {'method': 'quickshift', 'sigma': 1.0}
    }
    
    results = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Compare methods
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
    for i, (method_name, params) in enumerate(methods.items()):
        start_time = time.time()
        segments = segmenter.segment_image(image, **params)
        processing_time = time.time() - start_time
        
        # Visualize
        visualized = segmenter.visualize_superpixels(image, segments, method='boundaries')
        
        # Plot
        row, col = positions[i]
        axes[row, col].imshow(visualized)
        axes[row, col].set_title(f'{method_name}\n'
                               f'{len(np.unique(segments))} superpixels\n'
                               f'{processing_time:.2f}s')
        axes[row, col].axis('off')
        
        results[method_name] = {
            'segments': segments,
            'num_superpixels': len(np.unique(segments)),
            'processing_time': processing_time,
            'visualization': visualized
        }
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

def visualize_superpixel_properties(image: np.ndarray, segments: np.ndarray) -> None:
    """
    Visualize different superpixel properties
    """
    segmenter = SuperpixelSLIC()
    
    # Extract features
    features = segmenter.extract_superpixel_features(image, segments)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original with boundaries
    boundaries = segmenter.visualize_superpixels(image, segments, method='boundaries')
    axes[0, 0].imshow(boundaries)
    axes[0, 0].set_title('Original with Boundaries')
    axes[0, 0].axis('off')
    
    # Average colors
    avg_colors = segmenter.visualize_superpixels(image, segments, method='average_color')
    axes[0, 1].imshow(avg_colors)
    axes[0, 1].set_title('Average Colors')
    axes[0, 1].axis('off')
    
    # Random colors
    random_colors = segmenter.visualize_superpixels(image, segments, method='random_colors')
    axes[0, 2].imshow(random_colors)
    axes[0, 2].set_title('Random Colors')
    axes[0, 2].axis('off')
    
    # Area visualization
    area_viz = np.zeros_like(image)
    areas = np.array(features['area'])
    normalized_areas = (areas - areas.min()) / (areas.max() - areas.min())
    
    for i, segment_id in enumerate(np.unique(segments)):
        mask = (segments == segment_id)
        area_val = normalized_areas[i]
        area_viz[mask] = [area_val, area_val, area_val]
    
    axes[1, 0].imshow(area_viz, cmap='viridis')
    axes[1, 0].set_title('Area Visualization')
    axes[1, 0].axis('off')
    
    # Solidity visualization
    solidity_viz = np.zeros_like(image)
    solidities = np.array(features['solidity'])
    
    for i, segment_id in enumerate(np.unique(segments)):
        mask = (segments == segment_id)
        solidity_val = solidities[i]
        solidity_viz[mask] = [solidity_val, solidity_val, solidity_val]
    
    axes[1, 1].imshow(solidity_viz, cmap='plasma')
    axes[1, 1].set_title('Solidity Visualization')
    axes[1, 1].axis('off')
    
    # Statistics
    stats_text = f"""Superpixel Statistics:
    Total Superpixels: {len(np.unique(segments))}
    Avg Area: {np.mean(areas):.1f} pixels
    Avg Solidity: {np.mean(solidities):.3f}
    Color Std: {np.mean([np.mean(std) for std in features['std_color']]):.2f}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, 
                   verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
def main():
    """
    Example usage of SuperpixelSLIC
    """
    # Create sample image
    sample_image = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Add some colored regions
    sample_image[50:150, 50:150] = [255, 0, 0]      # Red square
    sample_image[75:125, 175:225] = [0, 255, 0]     # Green square
    sample_image[25:75, 200:250] = [0, 0, 255]      # Blue square
    sample_image[125:175, 75:125] = [255, 255, 0]   # Yellow square
    
    # Add some noise
    noise = np.random.randint(0, 50, sample_image.shape)
    sample_image = np.clip(sample_image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Initialize segmenter
    segmenter = SuperpixelSLIC()
    
    print("Testing SLIC superpixel segmentation...")
    
    # Basic segmentation
    segments = segmenter.segment_image(sample_image, n_segments=50, method='slic')
    print(f"Generated {len(np.unique(segments))} superpixels")
    
    # Visualize results
    boundaries = segmenter.visualize_superpixels(sample_image, segments, method='boundaries')
    
    # Compare methods
    print("\nComparing different segmentation methods...")
    comparison = compare_segmentation_methods(sample_image)
    
    # Visualize properties
    print("\nVisualizing superpixel properties...")
    visualize_superpixel_properties(sample_image, segments)
    
    # Test adaptive segmentation
    print("\nTesting adaptive segmentation...")
    adaptive_segments = segmenter.adaptive_slic(sample_image, min_segments=20, max_segments=100)
    
    print("SLIC superpixel segmentation demonstration complete!")

if __name__ == "__main__":
    main()
```

### Key Features and Optimizations

#### **1. Multiple SLIC Implementations:**
- **Scikit-image SLIC**: Fast, optimized implementation
- **Custom SLIC**: From-scratch implementation for understanding
- **Alternative Methods**: Felzenszwalb and Quickshift for comparison

#### **2. Advanced SLIC Algorithm:**
```python
# Distance computation in SLIC
dc = np.sqrt(dl**2 + da**2 + db**2)  # Color distance in LAB
ds = np.sqrt(dx**2 + dy**2)          # Spatial distance
distance = dc + (compactness * ds / step)  # Combined distance
```

#### **3. Connectivity Enforcement:**
```python
def _enforce_connectivity(self, labels, h, w):
    # Ensure each superpixel is connected
    labeled_mask, num_components = ndimage.label(mask)
    # Relabel disconnected components
```

#### **4. Feature Extraction:**
```python
# Comprehensive superpixel features
features = {
    'mean_color': [],     # Average color
    'std_color': [],      # Color variation
    'area': [],           # Size in pixels
    'centroid': [],       # Center position
    'perimeter': [],      # Boundary length
    'solidity': [],       # Shape compactness
    'eccentricity': []    # Shape elongation
}
```

#### **5. Best Practices:**

1. **LAB Color Space**: Better perceptual distance than RGB
2. **Gradient-based Initialization**: Start centers at low-gradient positions
3. **Connectivity Enforcement**: Ensure connected superpixels
4. **Adaptive Segmentation**: Automatically find optimal parameters
5. **Post-processing**: Merge similar adjacent superpixels

#### **6. Common Pitfalls and Solutions:**

**Problem**: Over-segmentation in textured regions
**Solution**: Increase compactness parameter or apply pre-smoothing

**Problem**: Disconnected superpixels
**Solution**: Use connectivity enforcement step

**Problem**: Poor boundary adherence
**Solution**: Lower compactness value and use LAB color space

#### **7. Usage Examples:**

```python
# Basic usage
segmenter = SuperpixelSLIC()
segments = segmenter.segment_image(image, n_segments=300, compactness=10.0)

# Adaptive segmentation
adaptive_segments = segmenter.adaptive_slic(image, min_segments=100, max_segments=500)

# Feature extraction
features = segmenter.extract_superpixel_features(image, segments)

# Batch processing
results = segmenter.batch_segment_images(image_paths, 'output_folder')
```

#### **8. Performance Considerations:**

- **Image Size**: Larger images need more superpixels
- **Compactness**: Higher values create more regular shapes
- **Iterations**: Usually converges in 5-10 iterations
- **Memory Usage**: Store only necessary intermediate results

This implementation provides a comprehensive solution for superpixel segmentation with SLIC, including custom implementation, feature extraction, visualization tools, and performance optimizations.

---

## Question 5
- [ ] Done

**Implement basicfacial landmark detectionusingDliborOpenCV.**

**Answer:**

Facial landmark detection is a crucial task in computer vision that identifies key points on faces such as eyes, nose, mouth, and jawline. This technology is fundamental for face recognition, emotion detection, face swapping, and augmented reality applications.

### Complete Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import imutils
from scipy.spatial import distance as dist
from collections import OrderedDict
import os
from typing import Tuple, List, Optional, Dict, Union
import time
import warnings
warnings.filterwarnings('ignore')

class FacialLandmarkDetector:
    """
    Comprehensive facial landmark detection system using Dlib and OpenCV
    """
    
    def __init__(self, predictor_path: Optional[str] = None):
        """
        Initialize the facial landmark detector
        
        Args:
            predictor_path: Path to dlib's facial landmark predictor model
        """
        # Initialize face detector
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Initialize landmark predictor
        if predictor_path and os.path.exists(predictor_path):
            self.landmark_predictor = dlib.shape_predictor(predictor_path)
            self.predictor_loaded = True
        else:
            print("Warning: Dlib predictor not loaded. Using OpenCV cascade for basic detection.")
            self.landmark_predictor = None
            self.predictor_loaded = False
            # Load OpenCV face cascade as fallback
            self.cv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                       'haarcascade_frontalface_default.xml')
            self.cv_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                       'haarcascade_eye.xml')
            self.cv_mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                        'haarcascade_smile.xml')
        
        # Define facial landmark regions
        self.facial_landmarks_idxs = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 35)),
            ("jaw", (0, 17))
        ])
        
        # Colors for visualization
        self.colors = {
            "mouth": (255, 0, 0),
            "right_eyebrow": (0, 255, 0),
            "left_eyebrow": (0, 255, 0),
            "right_eye": (0, 0, 255),
            "left_eye": (0, 0, 255),
            "nose": (255, 255, 0),
            "jaw": (255, 0, 255)
        }
    
    def detect_landmarks_dlib(self, image: np.ndarray, draw_landmarks: bool = True) -> Dict:
        """
        Detect facial landmarks using Dlib
        
        Args:
            image: Input image
            draw_landmarks: Whether to draw landmarks on image
            
        Returns:
            Dictionary containing landmarks, faces, and annotated image
        """
        if not self.predictor_loaded:
            raise ValueError("Dlib predictor not loaded. Use detect_landmarks_opencv instead.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect faces
        faces = self.face_detector(gray)
        
        all_landmarks = []
        annotated_image = image.copy()
        
        for face in faces:
            # Detect landmarks
            landmarks = self.landmark_predictor(gray, face)
            landmarks_np = face_utils.shape_to_np(landmarks)
            
            all_landmarks.append({
                'face_bbox': (face.left(), face.top(), face.right(), face.bottom()),
                'landmarks': landmarks_np,
                'landmark_regions': self._extract_landmark_regions(landmarks_np)
            })
            
            if draw_landmarks:
                annotated_image = self._draw_landmarks_dlib(annotated_image, landmarks_np, face)
        
        return {
            'faces_detected': len(faces),
            'landmarks': all_landmarks,
            'annotated_image': annotated_image,
            'method': 'dlib'
        }
    
    def detect_landmarks_opencv(self, image: np.ndarray, draw_landmarks: bool = True) -> Dict:
        """
        Detect facial landmarks using OpenCV (simplified approach)
        
        Args:
            image: Input image
            draw_landmarks: Whether to draw landmarks on image
            
        Returns:
            Dictionary containing detected features and annotated image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect faces
        faces = self.cv_face_cascade.detectMultiScale(gray, 1.3, 5)
        
        all_landmarks = []
        annotated_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi_color = annotated_image[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = self.cv_eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            
            # Detect mouth in face region
            mouth = self.cv_mouth_cascade.detectMultiScale(face_roi, 1.1, 3)
            
            # Convert relative coordinates to absolute
            eye_landmarks = []
            for (ex, ey, ew, eh) in eyes:
                eye_center = (x + ex + ew//2, y + ey + eh//2)
                eye_landmarks.append(eye_center)
            
            mouth_landmarks = []
            for (mx, my, mw, mh) in mouth:
                mouth_center = (x + mx + mw//2, y + my + mh//2)
                mouth_landmarks.append(mouth_center)
            
            # Estimate nose position (simple heuristic)
            nose_x = x + w//2
            nose_y = y + int(h * 0.6)
            nose_landmarks = [(nose_x, nose_y)]
            
            # Create simplified landmark structure
            simplified_landmarks = {
                'face_bbox': (x, y, x+w, y+h),
                'eyes': eye_landmarks,
                'mouth': mouth_landmarks,
                'nose': nose_landmarks,
                'face_center': (x + w//2, y + h//2)
            }
            
            all_landmarks.append(simplified_landmarks)
            
            if draw_landmarks:
                annotated_image = self._draw_landmarks_opencv(annotated_image, simplified_landmarks)
        
        return {
            'faces_detected': len(faces),
            'landmarks': all_landmarks,
            'annotated_image': annotated_image,
            'method': 'opencv'
        }
    
    def _extract_landmark_regions(self, landmarks: np.ndarray) -> Dict:
        """
        Extract different facial regions from landmarks
        """
        regions = {}
        
        for (name, (i, j)) in self.facial_landmarks_idxs.items():
            if j <= len(landmarks):
                regions[name] = landmarks[i:j]
            
        return regions
    
    def _draw_landmarks_dlib(self, image: np.ndarray, landmarks: np.ndarray, 
                           face_rect: dlib.rectangle) -> np.ndarray:
        """
        Draw Dlib landmarks on image
        """
        # Draw face bounding box
        cv2.rectangle(image, (face_rect.left(), face_rect.top()), 
                     (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
        
        # Draw landmarks by region
        for (name, (i, j)) in self.facial_landmarks_idxs.items():
            if j <= len(landmarks):
                points = landmarks[i:j]
                color = self.colors.get(name, (255, 255, 255))
                
                # Draw points
                for (x, y) in points:
                    cv2.circle(image, (x, y), 2, color, -1)
                
                # Draw connecting lines for certain regions
                if name in ["mouth", "right_eyebrow", "left_eyebrow", "jaw"]:
                    for k in range(len(points) - 1):
                        cv2.line(image, tuple(points[k]), tuple(points[k + 1]), color, 1)
                    
                    # Close the loop for mouth
                    if name == "mouth":
                        cv2.line(image, tuple(points[-1]), tuple(points[0]), color, 1)
                
                elif name in ["right_eye", "left_eye"]:
                    # Draw eye contour
                    cv2.polylines(image, [points], True, color, 1)
        
        return image
    
    def _draw_landmarks_opencv(self, image: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Draw OpenCV landmarks on image
        """
        # Draw face bounding box
        x1, y1, x2, y2 = landmarks['face_bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw eyes
        for eye_point in landmarks['eyes']:
            cv2.circle(image, eye_point, 5, (0, 0, 255), -1)
            cv2.circle(image, eye_point, 10, (0, 0, 255), 2)
        
        # Draw mouth
        for mouth_point in landmarks['mouth']:
            cv2.circle(image, mouth_point, 5, (255, 0, 0), -1)
            cv2.circle(image, mouth_point, 8, (255, 0, 0), 2)
        
        # Draw nose
        for nose_point in landmarks['nose']:
            cv2.circle(image, nose_point, 3, (255, 255, 0), -1)
        
        # Draw face center
        center = landmarks['face_center']
        cv2.circle(image, center, 3, (255, 255, 255), -1)
        
        return image
    
    def analyze_facial_features(self, landmarks_data: Dict) -> Dict:
        """
        Analyze facial features from landmarks
        """
        if landmarks_data['method'] == 'dlib':
            return self._analyze_features_dlib(landmarks_data)
        else:
            return self._analyze_features_opencv(landmarks_data)
    
    def _analyze_features_dlib(self, landmarks_data: Dict) -> Dict:
        """
        Analyze facial features from Dlib landmarks
        """
        analysis = {}
        
        for face_data in landmarks_data['landmarks']:
            landmarks = face_data['landmarks']
            regions = face_data['landmark_regions']
            
            face_analysis = {}
            
            # Eye aspect ratio (for blink detection)
            if 'left_eye' in regions and 'right_eye' in regions:
                left_ear = self._eye_aspect_ratio(regions['left_eye'])
                right_ear = self._eye_aspect_ratio(regions['right_eye'])
                face_analysis['eye_aspect_ratio'] = {
                    'left': left_ear,
                    'right': right_ear,
                    'average': (left_ear + right_ear) / 2.0,
                    'blink_detected': (left_ear + right_ear) / 2.0 < 0.2
                }
            
            # Mouth aspect ratio (for yawn/smile detection)
            if 'mouth' in regions:
                mar = self._mouth_aspect_ratio(regions['mouth'])
                face_analysis['mouth_aspect_ratio'] = {
                    'value': mar,
                    'yawn_detected': mar > 0.6,
                    'smile_detected': self._detect_smile(regions['mouth'])
                }
            
            # Face orientation
            if 'nose' in regions and len(regions['nose']) > 0:
                nose_tip = regions['nose'][4]  # Nose tip is usually point 30 (index 4 in nose region)
                face_bbox = face_data['face_bbox']
                face_center_x = (face_bbox[0] + face_bbox[2]) / 2
                
                nose_deviation = nose_tip[0] - face_center_x
                face_analysis['head_pose'] = {
                    'nose_deviation': nose_deviation,
                    'facing_direction': 'left' if nose_deviation < -5 else 'right' if nose_deviation > 5 else 'center'
                }
            
            # Face dimensions
            face_analysis['face_dimensions'] = self._calculate_face_dimensions(landmarks)
            
            analysis[f'face_{len(analysis)}'] = face_analysis
        
        return analysis
    
    def _analyze_features_opencv(self, landmarks_data: Dict) -> Dict:
        """
        Analyze facial features from OpenCV landmarks (simplified)
        """
        analysis = {}
        
        for i, face_data in enumerate(landmarks_data['landmarks']):
            face_analysis = {}
            
            # Basic feature counts
            face_analysis['features_detected'] = {
                'eyes': len(face_data['eyes']),
                'mouth': len(face_data['mouth']),
                'nose': len(face_data['nose'])
            }
            
            # Face dimensions
            x1, y1, x2, y2 = face_data['face_bbox']
            face_analysis['face_dimensions'] = {
                'width': x2 - x1,
                'height': y2 - y1,
                'area': (x2 - x1) * (y2 - y1)
            }
            
            # Eye distance (if 2 eyes detected)
            if len(face_data['eyes']) == 2:
                eye1, eye2 = face_data['eyes']
                eye_distance = dist.euclidean(eye1, eye2)
                face_analysis['eye_distance'] = eye_distance
            
            analysis[f'face_{i}'] = face_analysis
        
        return analysis
    
    def _eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate eye aspect ratio for blink detection
        """
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def _mouth_aspect_ratio(self, mouth_landmarks: np.ndarray) -> float:
        """
        Calculate mouth aspect ratio for yawn detection
        """
        # Compute the euclidean distances between the vertical
        # mouth landmarks
        A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])  # 51, 59
        B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])   # 53, 57
        
        # Compute the euclidean distance between the horizontal
        # mouth landmarks
        C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])   # 49, 55
        
        # Compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)
        
        return mar
    
    def _detect_smile(self, mouth_landmarks: np.ndarray) -> bool:
        """
        Simple smile detection based on mouth shape
        """
        # Check if mouth corners are higher than center
        left_corner = mouth_landmarks[0]   # Point 48
        right_corner = mouth_landmarks[6]  # Point 54
        mouth_center = mouth_landmarks[3]  # Point 51
        
        # Calculate if corners are above center
        corner_height = (left_corner[1] + right_corner[1]) / 2
        smile_detected = corner_height < mouth_center[1] - 2
        
        return smile_detected
    
    def _calculate_face_dimensions(self, landmarks: np.ndarray) -> Dict:
        """
        Calculate various face dimensions
        """
        # Face width (jaw points)
        jaw_width = dist.euclidean(landmarks[0], landmarks[16])
        
        # Face height (top to bottom)
        face_height = dist.euclidean(landmarks[19], landmarks[8])  # Eyebrow to chin
        
        # Eye distance
        left_eye_center = np.mean(landmarks[42:48], axis=0)
        right_eye_center = np.mean(landmarks[36:42], axis=0)
        eye_distance = dist.euclidean(left_eye_center, right_eye_center)
        
        # Nose length
        nose_length = dist.euclidean(landmarks[27], landmarks[33])
        
        # Mouth width
        mouth_width = dist.euclidean(landmarks[48], landmarks[54])
        
        return {
            'jaw_width': jaw_width,
            'face_height': face_height,
            'eye_distance': eye_distance,
            'nose_length': nose_length,
            'mouth_width': mouth_width,
            'face_ratio': face_height / jaw_width if jaw_width > 0 else 0
        }
    
    def real_time_detection(self, camera_index: int = 0, duration: int = 30) -> None:
        """
        Real-time facial landmark detection from camera
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print(f"Starting real-time detection for {duration} seconds...")
        print("Press 'q' to quit early")
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks
            if self.predictor_loaded:
                result = self.detect_landmarks_dlib(frame_rgb, draw_landmarks=True)
            else:
                result = self.detect_landmarks_opencv(frame_rgb, draw_landmarks=True)
            
            # Convert back to BGR for display
            annotated_frame = cv2.cvtColor(result['annotated_image'], cv2.COLOR_RGB2BGR)
            
            # Add performance info
            current_time = time.time()
            fps = frame_count / (current_time - start_time) if current_time > start_time else 0
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Faces: {result['faces_detected']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Method: {result['method']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Facial Landmark Detection', annotated_frame)
            
            frame_count += 1
            
            # Check for exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if time.time() - start_time > duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nReal-time detection completed:")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")
    
    def batch_process_images(self, image_paths: List[str], output_dir: str, 
                           method: str = 'auto') -> List[Dict]:
        """
        Batch process multiple images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = []
        
        # Determine method
        if method == 'auto':
            detection_method = 'dlib' if self.predictor_loaded else 'opencv'
        else:
            detection_method = method
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect landmarks
                start_time = time.time()
                if detection_method == 'dlib' and self.predictor_loaded:
                    result = self.detect_landmarks_dlib(image, draw_landmarks=True)
                else:
                    result = self.detect_landmarks_opencv(image, draw_landmarks=True)
                
                processing_time = time.time() - start_time
                
                # Analyze features
                analysis = self.analyze_facial_features(result)
                
                # Save annotated image
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_landmarks{ext}")
                
                # Convert to BGR for saving
                output_bgr = cv2.cvtColor(result['annotated_image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, output_bgr)
                
                # Store results
                result_data = {
                    'input_path': image_path,
                    'output_path': output_path,
                    'faces_detected': result['faces_detected'],
                    'processing_time': processing_time,
                    'method': detection_method,
                    'analysis': analysis
                }
                results.append(result_data)
                
                print(f"Processed {i+1}/{len(image_paths)}: {filename} "
                      f"({result['faces_detected']} faces, {processing_time:.2f}s)")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_landmark_analysis(self, image: np.ndarray, method: str = 'auto') -> None:
        """
        Visualize comprehensive landmark analysis
        """
        # Determine method
        if method == 'auto':
            detection_method = 'dlib' if self.predictor_loaded else 'opencv'
        else:
            detection_method = method
        
        # Detect landmarks
        if detection_method == 'dlib' and self.predictor_loaded:
            result = self.detect_landmarks_dlib(image, draw_landmarks=True)
        else:
            result = self.detect_landmarks_opencv(image, draw_landmarks=True)
        
        # Analyze features
        analysis = self.analyze_facial_features(result)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Annotated image
        axes[1].imshow(result['annotated_image'])
        axes[1].set_title(f'Landmarks ({result["method"]}) - {result["faces_detected"]} faces')
        axes[1].axis('off')
        
        # Add analysis text
        if analysis:
            analysis_text = "Feature Analysis:\n"
            for face_id, face_data in analysis.items():
                analysis_text += f"\n{face_id.upper()}:\n"
                
                if 'eye_aspect_ratio' in face_data:
                    ear_data = face_data['eye_aspect_ratio']
                    analysis_text += f"  Blink: {'Yes' if ear_data['blink_detected'] else 'No'}\n"
                    analysis_text += f"  EAR: {ear_data['average']:.3f}\n"
                
                if 'mouth_aspect_ratio' in face_data:
                    mar_data = face_data['mouth_aspect_ratio']
                    analysis_text += f"  Yawn: {'Yes' if mar_data['yawn_detected'] else 'No'}\n"
                    analysis_text += f"  Smile: {'Yes' if mar_data['smile_detected'] else 'No'}\n"
                
                if 'head_pose' in face_data:
                    pose_data = face_data['head_pose']
                    analysis_text += f"  Head: {pose_data['facing_direction']}\n"
            
            plt.figtext(0.02, 0.02, analysis_text, fontsize=10, 
                       verticalalignment='bottom', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

# Utility functions
def download_dlib_predictor(output_path: str = "shape_predictor_68_face_landmarks.dat") -> bool:
    """
    Helper to download dlib's facial landmark predictor
    Note: In practice, you need to download this manually from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    """
    print("To use Dlib landmarks, download the predictor from:")
    print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("Extract and provide the path to the .dat file")
    return False

def compare_detection_methods(image: np.ndarray, predictor_path: Optional[str] = None) -> None:
    """
    Compare Dlib and OpenCV detection methods
    """
    # Initialize detectors
    detector = FacialLandmarkDetector(predictor_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # OpenCV detection
    opencv_result = detector.detect_landmarks_opencv(image, draw_landmarks=True)
    axes[1].imshow(opencv_result['annotated_image'])
    axes[1].set_title(f'OpenCV Detection\n{opencv_result["faces_detected"]} faces')
    axes[1].axis('off')
    
    # Dlib detection (if available)
    if detector.predictor_loaded:
        dlib_result = detector.detect_landmarks_dlib(image, draw_landmarks=True)
        axes[2].imshow(dlib_result['annotated_image'])
        axes[2].set_title(f'Dlib Detection\n{dlib_result["faces_detected"]} faces')
    else:
        axes[2].text(0.5, 0.5, 'Dlib Predictor\nNot Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Dlib Detection (N/A)')
    
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
def main():
    """
    Example usage of FacialLandmarkDetector
    """
    # Create sample face image (in practice, use real images)
    sample_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw a simple face
    cv2.circle(sample_image, (200, 200), 100, (200, 200, 200), -1)  # Face
    cv2.circle(sample_image, (170, 170), 10, (0, 0, 0), -1)         # Left eye
    cv2.circle(sample_image, (230, 170), 10, (0, 0, 0), -1)         # Right eye
    cv2.ellipse(sample_image, (200, 200), (15, 10), 0, 0, 180, (0, 0, 0), 2)  # Nose
    cv2.ellipse(sample_image, (200, 240), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    # Initialize detector
    detector = FacialLandmarkDetector()
    
    print("Testing facial landmark detection...")
    
    # OpenCV detection (always available)
    print("\nTesting OpenCV detection...")
    opencv_result = detector.detect_landmarks_opencv(sample_image, draw_landmarks=True)
    print(f"OpenCV: {opencv_result['faces_detected']} faces detected")
    
    # Analyze features
    analysis = detector.analyze_facial_features(opencv_result)
    print(f"Analysis: {len(analysis)} faces analyzed")
    
    # Visualize results
    detector.visualize_landmark_analysis(sample_image, method='opencv')
    
    # Compare methods
    print("\nComparing detection methods...")
    compare_detection_methods(sample_image)
    
    print("\nFacial landmark detection demonstration complete!")
    print("\nFor better results with Dlib:")
    print("1. Download shape_predictor_68_face_landmarks.dat")
    print("2. Initialize with: FacialLandmarkDetector('path/to/predictor.dat')")

if __name__ == "__main__":
    main()
```

### Key Features and Optimizations

#### **1. Dual Detection Methods:**
- **Dlib**: 68-point accurate facial landmarks
- **OpenCV**: Simplified cascade-based detection as fallback

#### **2. Comprehensive Landmark Analysis:**
```python
# Eye aspect ratio for blink detection
ear = (A + B) / (2.0 * C)

# Mouth aspect ratio for yawn detection
mar = (A + B) / (2.0 * C)

# Smile detection based on mouth corner positions
smile_detected = corner_height < mouth_center[1] - 2
```

#### **3. Real-time Processing:**
```python
def real_time_detection(self, camera_index=0, duration=30):
    # Live camera feed with landmark detection
    # FPS monitoring and performance display
```

#### **4. Feature Regions Mapping:**
```python
facial_landmarks_idxs = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])
```

#### **5. Best Practices:**

1. **Multi-method Support**: Both Dlib and OpenCV implementations
2. **Real-time Optimization**: Efficient processing for video streams
3. **Feature Analysis**: Extract meaningful facial characteristics
4. **Robust Detection**: Fallback methods when primary fails
5. **Visualization Tools**: Comprehensive result display

#### **6. Common Pitfalls and Solutions:**

**Problem**: Missing Dlib predictor model
**Solution**: Provide fallback OpenCV detection with download instructions

**Problem**: Poor lighting affecting detection
**Solution**: Histogram equalization and adaptive preprocessing

**Problem**: Multiple faces in single image
**Solution**: Process each face separately with individual analysis

#### **7. Usage Examples:**

```python
# Basic usage
detector = FacialLandmarkDetector('shape_predictor_68_face_landmarks.dat')
result = detector.detect_landmarks_dlib(image)

# Real-time detection
detector.real_time_detection(camera_index=0, duration=30)

# Batch processing
results = detector.batch_process_images(image_paths, 'output_folder')

# Feature analysis
analysis = detector.analyze_facial_features(result)
```

#### **8. Performance Considerations:**

- **Model Loading**: Cache predictor models for repeated use
- **Image Preprocessing**: Optimize grayscale conversion and resizing
- **Real-time Processing**: Balance accuracy vs. speed for video
- **Memory Management**: Process large batches efficiently

This implementation provides a complete facial landmark detection solution with both Dlib and OpenCV support, comprehensive feature analysis, real-time capabilities, and production-ready optimizations.

---

