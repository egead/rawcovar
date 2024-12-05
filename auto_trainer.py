from tensorflow import keras
import numpy as np
import os
from seismic_purifier.config import BATCH_SIZE
from seismic_purifier.representation_learning_models import (
    RepresentationLearningSingleAutoencoder,
    RepresentationLearningDenoisingSingleAutoencoder,
    RepresentationLearningMultipleAutoencoder
)
from tensorflow.keras.callbacks import LambdaCallback

# ============================
# Configuration
# ============================

# Paths to your data
TRAIN_DATA_PATH = 'data/X_train_1280sample.npy'  # Replace with your actual path
TEST_DATA_PATH = 'data/X_test_1280sample.npy'  # Replace with your actual path
TRAIN_LABEL_PATH = 'data/Y_train_1280sample.npy'  # Replace with your actual path
TEST_LABEL_PATH = 'data/Y_test_1280sample.npy'  # Replace with your actual path

# Directory to save checkpoints
CHECKPOINT_DIR = 'checkpoints'

# Training parameters
EPOCHS = 50
LEARNING_RATE = 1e-3

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_save_model(model,EPOCHS,BATCH_SIZE):
    # ============================
    # Model Compilation
    # ============================
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE) 
    model.compile(optimizer=optimizer)

    # ============================
    # Callbacks Setup
    # ============================
    # Define callbacks for saving checkpoints, early stopping.

    #Define callback for saving activations in each layer
    stored_activations = {}
    activation_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: [stored_activations.setdefault(layer.name, []).append(
                tf.keras.Model(inputs=model.input,outputs=layer.output).predict(x_val)
            ) for layer in model.layers if layer.name in layer_names
        ]
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, model.name+'_epoch_{epoch:02d}.h5'),
            save_weights_only=True,
            save_freq='epoch',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        activation_callback
    ]

    fit_result = model.fit(X_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        callbacks=callbacks, 
                        shuffle=False)

    MODEL_SAVE_PATH = 'models/'+model.name+'.h5'
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    model.save_weights(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


# ============================
# Data loading
# ============================
X_train = np.load(TRAIN_DATA_PATH)  # Expected shape: (num_samples, 3000, 3)
print(f"Training data shape: {X_train.shape}")

# ============================
# Representation Learning Models Instantiation
# ============================

single_model = RepresentationLearningSingleAutoencoder(
    name="rep_learning_autoencoder"
)
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
# Training the Representation Learning Models and Save.
# ============================
for model in models:
    train_save_model(model,EPOCHS,BATCH_SIZE)
