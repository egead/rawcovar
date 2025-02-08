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
# Current configuration: Train and save model seperately for each station
# ============================

DATA_BASE_PATH = 'data/silivri'  # Base directory containing station folders
CHECKPOINT_DIR = 'checkpoints'# Directory to save checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



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
# Main training loop
# ============================
for station in os.listdir(DATA_BASE_PATH):
    station_path = os.path.join(DATA_BASE_PATH, station)
    
    if not os.path.isdir(station_path):
        continue  

    # Load all .npy files for current station
    npy_files = [f for f in os.listdir(station_path) if f.endswith('.npy')]
    if not npy_files:
        print(f"No NPY files found in {station}, skipping...")
        continue

    # Load and concatenate station data
    X_train = np.concatenate(
        [np.load(os.path.join(station_path, f)) for f in npy_files],
        axis=0
    )
    print(f"\nTraining on {station} with data shape: {X_train.shape}")

    # Initialize models with station-specific names
    models = [
        RepresentationLearningSingleAutoencoder(
            name=f"{station}_single_autoencoder"
        ),
        RepresentationLearningDenoisingSingleAutoencoder(
            name=f"{station}_denoising_autoencoder",
            input_noise_std=1e-6,
            denoising_noise_std=2e-1
        ),
        RepresentationLearningMultipleAutoencoder(
            name=f"{station}_multiple_autoencoder",
            input_noise_std=1e-6,
            eps=1e-27
        )
    ]

    # Train and save each model
    for model in models:
        print(f"\nTraining {model.name}")
        train_save_model(
            model=model,
            X_train=X_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
