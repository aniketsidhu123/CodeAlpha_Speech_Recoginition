import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional

# ==============================================================================
# Custom Attention Layer for CRNN
# ==============================================================================

@tf.keras.utils.register_keras_serializable(package="SER")
class TemporalAttention(layers.Layer):
    """
    Bahdanau-style temporal attention layer to pool sequence outputs
    by learning attention weights over time steps.
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            name="attention_weight",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, feature_dim)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)  # (batch_size, time_steps, 1)
        ait = tf.squeeze(ait, -1)                # (batch_size, time_steps)
        weights = tf.nn.softmax(ait, axis=-1)    # (batch_size, time_steps)
        weights_expanded = tf.expand_dims(weights, -1)
        output = tf.reduce_sum(inputs * weights_expanded, axis=1)  # (batch_size, feature_dim)
        return output

    def get_config(self):
        return super(TemporalAttention, self).get_config()

# ==============================================================================
# Model 1: 1D CNN Architecture (on 1D acoustic features)
# ==============================================================================

def build_1d_cnn_model(
    input_dim: int,
    num_classes: int = 8,
    learning_rate: float = 0.001
) -> models.Model:
    """
    High-capacity Residual Deep Neural Network for multi-feature acoustic vectors.
    Input shape: (input_dim, 1) or (input_dim,)
    """
    inputs = layers.Input(shape=(input_dim, 1), name="audio_features_1d")
    x = layers.Flatten()(inputs)
    
    # Dense Projection
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Residual Block 1
    res1 = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    res1 = layers.BatchNormalization()(res1)
    res1 = layers.Dropout(0.3)(res1)
    
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(res1)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.add([res1, x])
    
    # Residual Block 2
    res2 = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    res2 = layers.BatchNormalization()(res2)
    res2 = layers.Dropout(0.3)(res2)
    
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(res2)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.add([res2, x])
    
    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_1D_SER")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==============================================================================
# Model 2: Bidirectional LSTM / RNN Model (on sequence features)
# ==============================================================================

def build_bilstm_model(
    input_shape: Tuple[int, int] = (130, 93),
    num_classes: int = 8,
    learning_rate: float = 0.001
) -> models.Model:
    """
    Bidirectional LSTM network modeling temporal phonetic/emotion dynamics.
    Input shape: (timesteps, feature_dim)
    """
    inputs = layers.Input(shape=input_shape, name="sequence_features")
    
    # Layer 1: BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(inputs)
    x = layers.BatchNormalization()(x)
    
    # Layer 2: BiLSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.2))(x)
    x = layers.BatchNormalization()(x)
    
    # Dense Classifier
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="BiLSTM_SER")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==============================================================================
# Model 3: Hybrid CRNN with Attention (CNN + BiLSTM + Attention Pooling)
# ==============================================================================

def build_crnn_attention_model(
    input_shape: Tuple[int, int] = (130, 93),
    num_classes: int = 8,
    learning_rate: float = 0.001
) -> models.Model:
    """
    Hybrid Convolutional Recurrent Neural Network with Attention.
    1. Conv1D extracts localized spectro-temporal features.
    2. Bidirectional LSTM captures contextual temporal evolution.
    3. Self-Attention weights crucial emotional segments.
    """
    inputs = layers.Input(shape=input_shape, name="crnn_sequence_input")
    
    # Convolutional front-end
    x = layers.Conv1D(64, kernel_size=5, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Recurrent middle-end (BiLSTM)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.2))(x)
    x = layers.BatchNormalization()(x)
    
    # Temporal Attention pooling
    x = TemporalAttention()(x)
    
    # Dense Classifier
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CRNN_Attention_SER")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==============================================================================
# Model Factory
# ==============================================================================

def get_model(
    model_type: str = "cnn",
    input_dim_or_shape = None,
    num_classes: int = 8,
    learning_rate: float = 0.001
) -> models.Model:
    """
    Factory function to instantiate models: 'cnn', 'lstm', 'crnn'.
    """
    model_type = model_type.lower()
    if model_type in ["cnn", "1d_cnn", "conv1d"]:
        input_dim = input_dim_or_shape if isinstance(input_dim_or_shape, int) else 289
        return build_1d_cnn_model(input_dim=input_dim, num_classes=num_classes, learning_rate=learning_rate)
    elif model_type in ["lstm", "bilstm"]:
        shape = input_dim_or_shape if isinstance(input_dim_or_shape, tuple) else (130, 93)
        return build_bilstm_model(input_shape=shape, num_classes=num_classes, learning_rate=learning_rate)
    elif model_type in ["crnn", "crnn_attention", "hybrid"]:
        shape = input_dim_or_shape if isinstance(input_dim_or_shape, tuple) else (130, 93)
        return build_crnn_attention_model(input_shape=shape, num_classes=num_classes, learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from 'cnn', 'lstm', 'crnn'.")
