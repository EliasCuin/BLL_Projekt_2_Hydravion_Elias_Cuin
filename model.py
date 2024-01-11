import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Inspirationsquelle des folgenden Programmiercodes:
# Neveu, T.:  Apprentissage par renforcement, Episodes 1-9,
# www.youtube.com/playlist?list=PLpEPgC7cUJ4YPZlfUu0vQTwPraVKPASUa


class model():
  # "modelFileName" sei der Name des Modells
  def __init__(self, modelFileName="model.h5"):
    self.model = None
    self.modelFileName = modelFileName

     # Lädt das Modell falls vorhanden sonst "else"
    if(os.path.isfile(self.modelFileName)):
      self.model = load_model(self.modelFileName)
    else:
    # Erstellung und Speicherung eines neuen KI-Modells
      tf.keras.layers.LSTM(64, return_sequences=True, stateful=True, unroll=True)
      input_layer = tf.keras.layers.Input(batch_shape=(None, 21))
      hidden_layer1 = tf.keras.layers.Dense(units=32, activation='relu', use_bias=True)(input_layer)
      hidden_layer2 = tf.keras.layers.Dense(units=44, activation='relu', use_bias=True)(hidden_layer1)
      output_layer = tf.keras.layers.Dense(units=3, activation='softmax', use_bias=True)(hidden_layer2)
      self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
      self.model.save(self.modelFileName)

    self.model_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    self.model.compile(optimizer=self.model_optimizer, loss='mean_squared_error')

  # Speichert das aktuelle Modell.
  def save(self):
     self.model.save(self.modelFileName)

  # Berechnen der Loss von einem Batch, mithilfe der zuvor ausgerechneten Bestandteile der Loss Formel (tf_states, tf_actions, Qtargets (bzw. Target-Value))
  def model_loss(self, tf_states, tf_actions, Qtargets):
    # Berechnet die Loss mithilfe der Loss Formel von Reinforcement Learning.
    predictions = self.model(tf_states)
    squared_diff = tf.square(predictions - Qtargets)

    #Die Loss wird hierbei für die relevanten "Actions" berechnet (deswegen wird mit tf_actions multipliziert).
    weighted_loss = tf_actions * squared_diff

    # Ein Durchschnittswert der Loss wird für den gegebenen Batch dieser Funktion berechnet.
    mean_loss = tf.reduce_mean(weighted_loss)

    return mean_loss

  def train_model(self, states, actions, rewards, next_states):
      # Konvertiert die Erfahrungs-Arrays in Tensoren für Verarbeitung mit TensorFlow.
      tf_states = tf.constant(states, dtype=tf.float32)
      tf_actions = tf.constant(actions, dtype=tf.float32)
      tf_rewards = tf.constant(rewards, dtype=tf.float32)
      tf_next_states = tf.constant(next_states, dtype=tf.float32)

      # Die Prediction-Werte (auch Q Werte) werden für den nächsten State ausgerechnet
      Qnext_states = self.model(tf_next_states)

      # Berechnet die Target-Values für die Aktualisierung des Modells.
      Qtargets =tf.math.reduce_max(Qnext_states, axis = 1, keepdims = True) * 0.99 + tf.expand_dims(tf_rewards,1)

      # Modell in batches trainieren
      # (in kleinen Teilchen wofür die loss ausgerechnet (Durchschnitt ) wird und so optimiert wird)
      batch_size = 32
      losses = []

      for batch_start in range(0, len(next_states), batch_size ):
        # Die Erfahrungen werden, wenn möglich, immer in Batches der Länge 32 aufgeteilt.
        batch_ende = batch_start + batch_size if batch_start + batch_size < len(next_states) else len(next_states)

        # Der ":"-Teil von Tensorflow bedeutet, dass alle Spalten ausgewählt werden sollen (Es handelt sich hierbei um einen mehrdimensionalen TensorFlow-Array)
        batch_states = tf_states[batch_start: batch_ende, :]
        batch_actions = tf_actions[batch_start: batch_ende, :]
        # (Qtargets ist das gleiche wie "Target_Values" )
        batch_Qtargets  = Qtargets[batch_start: batch_ende]

        # Berechnen der Loss mithilfe der definierten Loss-Funktion.
        loss = self.model_loss(batch_states,batch_actions,batch_Qtargets )

        # .numpy() wandelt ein TensorFlow-Array in einen normalen numpy Array um.
        losses.append(loss.numpy())

        # Berechnung der Loss und Aufrufen der trainierbaren Parametern mittels GradientTape()
        with tf.GradientTape() as tape:
            # Durchschnitt der Loss wird für die Erfahrungen dieses "Batch" berechnet
            loss = self.model_loss(tf_states, tf_actions, Qtargets)

        # Mithilfe dieser errechneten "Loss" werden die jeweiligen Gradienten der Loss mit Rücksicht auf alle Parameter des NN berechnet.
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))



      print("durchschnittlicher Verlust", np.mean(losses))
