from pathlib import Path
import tensorflow as tf
from sklearn.metrics import confusion_matrix

class ModelTrainer_DL:
    def __init__(self):
        pass

    def build_cnn_model_param1(self, input_shape):
        """
        Builds a Convolutional Neural Network (CNN) model.
        Note: The input shape is hard-coded to (N, 128, 1).
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),    # (time_steps, channels)

            tf.keras.layers.Conv1D(2, 8, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_cnn_model_param2(self, input_shape):
        """
        Builds a Convolutional Neural Network (CNN) model.
        Note: The input shape is hard-coded to (N, 128, 1).
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(4, 8, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_cnn_model_param3(self, input_shape):
        """
        Builds a Convolutional Neural Network (CNN) model.
        Note: The input shape is hard-coded to (N, 128, 1).
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(8, 8, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Trains and evaluates the CNN model.
        Assumes X_train and X_test are already correctly shaped.
        """
        results = {}

        input_shape = (256, 1) # This shape is expected by the model
        
        cnn_model = self.build_cnn_model_param3(input_shape)
        cnn_model.summary()

        print("--- Starting Model Training ---")
        cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        print("--- Model Training Finished ---")

        # Save model
        cnn_model.save(Path.cwd() / (str(cnn_model.name) + ".h5"))
        print("--- Model Saved ---")

        # Predict
        print("--- Evaluating on Test Data ---")
        y_pred_proba = cnn_model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Evaluate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate metrics
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (fn + tp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        results['CNN'] = {
            'FPR': fpr,
            'FNR': fnr,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
        }
        
        # F-score
        beta = [1, 2, 0.5]
        for b in beta:
            key_name = f'F{b}'
            key_value = (1 + b**2) * (recall * precision) / (b**2 * precision + recall)
            results['CNN'][key_name] = key_value
            
        return results