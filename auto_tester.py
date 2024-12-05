import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from seismic_purifier.config import BATCH_SIZE
from seismic_purifier.representation_learning_models import (
    RepresentationLearningSingleAutoencoder,
    RepresentationLearningDenoisingSingleAutoencoder,
    RepresentationLearningMultipleAutoencoder
)
from seismic_purifier.classifier_models import (
    ClassifierAutocovariance, 
    ClassifierAugmentedAutoencoder, 
    ClassifierMultipleAutoencoder
)
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# ============================
# Configuration
# ============================

# Paths to your data
TEST_DATA_PATH = 'data/X_test_1280sample.npy'  # Replace with your actual path
TEST_LABEL_PATH = 'data/Y_test_1280sample.npy'  # Replace with your actual path

# Paths to your models
MODEL_PATHS = ['models/rep_learning_autoencoder.h5','models/rep_learning_autoencoder_ensemble.h5','models/rep_learning_denoising_autoencoder.h5']

def test_model(model, MODEL_PATH, X_test, Y_test):
    print(model.name)
    model.compile()
    model(X_test)
    model.load_weights(MODEL_PATH)

    # ============================
    # Classifier Model Instantiation
    # ============================

    # Choose the model for classification. This is just for convenience, these models are actually wrappers around
    #representation learning models.
    # For example, using RepresentationLearningSingleAutoencoder
    # model_classifier = ClassifierAutocovariance(model)

    # Alternatively, you can choose other wrappers. 
    # model_classifier = ClassifierAugmentedAutoencoder(model)
    if model.name=='rep_learning_autoencoder_ensemble':
        model_classifier = ClassifierMultipleAutoencoder(model)
    else: 
        model_classifier = ClassifierAugmentedAutoencoder(model)

    '''
    Note: One should be careful about the compatibility of the classifier wrappers with the models. 
    RepresentationLearningSingleAutoencoder and RepresentationLearningDenoising
    Autoencoder are compatible with ClassifierAutocovariance, ClassifierAugmentedAutoencoder. 
    However, RepresentationLearningMultipleAutoencoder is only compatible with 
    ClassifierMultipleAutoencoder. 
    '''

    # ============================
    # Obtain earthquake probabilities
    # ============================
    earthquake_scores = model_classifier(X_test)

    # ============================
    # Plot ROC curve.
    # ============================
    fpr, tpr, __ = roc_curve(Y_test, earthquake_scores)
    auc_score = roc_auc_score(y_true=Y_test, y_score=earthquake_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')

    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(model.name+'_ROC.png')
    print(auc_score)

# ============================
# Data loading
# ============================
X_test = np.load(TEST_DATA_PATH)
print(f"Test data shape: {X_test.shape}")

Y_test = np.load(TEST_LABEL_PATH)  # Expected shape: (num_samples)
print(f"Test label shape: {Y_test.shape}")

# ============================
# Representation Learning Model Instantiation
# ============================

# Choose the model you want to train
# For example, using RepresentationLearningSingleAutoencoder

single_model = RepresentationLearningSingleAutoencoder(
    name="rep_learning_autoencoder"
)

# Alternatively, you can choose other models:
denoising_single_model = RepresentationLearningDenoisingSingleAutoencoder(
     name="rep_learning_denoising_autoencoder",
     input_noise_std=1e-6,
     denoising_noise_std=2e-1
 )
multiple_model = RepresentationLearningMultipleAutoencoder(
     name="rep_learning_autoencoder_ensemble",
     input_noise_std=1e-6,
     eps=1e-27
)

models=[single_model,denoising_single_model,multiple_model]

# ============================
# Model Compilation
# ============================
for i in range(len(models)):
    test_model(model=models[i],MODEL_PATH=MODEL_PATHS[i],X_test=X_test,Y_test=Y_test)

