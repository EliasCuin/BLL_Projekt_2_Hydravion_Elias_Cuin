import model
from tensorflow import keras
import tensorflow as tf
import numpy as np
import math
from statistics import mean

# Inspirationsquelle des folgenden Programmiercodes:
# Neveu, T.:  Apprentissage par renforcement, Episodes 1-9,
# www.youtube.com/playlist?list=PLpEPgC7cUJ4YPZlfUu0vQTwPraVKPASUa

class agent():
  def __init__(self,env, modelName="model.h5"):
     # Initialisiert das Tic-Tac-Toe-Environment und das KI-Modell.
    self.env = env()

    # Erstellen einer Modellinstanz, von der "Modell" Klasse.
    self.model = model.model(modelName)

  # Generiert "Beispieldaten", bzw. Trainingsdaten  und übergibt diese der KI.
  def train(self, epStart = 1):
    # Startet mit dem Default Epsilon-Wert 1. (bzw. 10, weil in main.py diese Funktion mit epStart = 10 aufgerufen wird )
    eps  = epStart

    # Erfahrungslisten definieren ( U(d) ).
    states = [];
    rewards = [];
    next_states = [];
    actions = [];

    old_rewards = []

   # Aktuellen State von dem Environment abfragen.
    currentState = self.env.getState()

    # Das Training beinhaltet 10.000 Episoden, in der jeweils jede Episode 500 Schritte beinhaltet. In jedem Schritt wird eine Erfahrungseinheit gesammelt.
    # Das Modell wird nach jeweils 5 Episoden mit den gesammelten Erfahrungen der Erfahrungslisten trainiert.
    for episodes in range(5_00):
      reward = 0
      step = 0

      while (step < 500):
        # Erfahrungwerte sammeln, indem die aktuelle KI (und auch Epsilon-Greedy-Strategie) sich auf dem Environment fortbewegen.
        act = self.pick_action(currentState, eps);
        reward = self.env.step(act);
        st2 = self.env.getState()
        old_rewards.append(reward)

        # Setzt das Spiel zurück, wenn es beendet ist.
        if(self.env.gameOver):
          self.env.reset()

        # Fügt die gesammelten Erfahrungen zufällig in die Erfahrungsliste hinzu.
        speicherIndex = math.floor(np.random.rand() * len(states));
        states.insert(speicherIndex, currentState)
        rewards.insert(speicherIndex,reward)
        next_states.insert(speicherIndex,st2 )
        actions.insert(speicherIndex, act )

        # Die gesammelten Erfahrungen bleiben immer bei 100.000, damit das Modell nicht jedes Mal zu viel auf einmal lernen muss.
        if(len(states) > 10_000_0000):
          states = states[1:]
          rewards = rewards[1:]
          next_states = next_states[1:]
          actions = actions[1:]
        # Den aktuellen State der KI aktualisieren.
        currentState = self.env.getState()
        step +=1

      # Ausführen der Epsilon Greedy Strategie.
      eps = max(0.1, eps * 0.9980);

      # Trainiert das Modell alle 5 Episoden mithilfe der gesammelten Daten.
      old_rewards = [mean(old_rewards)]

      if(episodes % 5 == 0):
        # Das Modell wird mithilfe der gesammelten Erfahrungen trainiert.
        self.model.train_model(states,actions,rewards,next_states)
        # Das Modell wird auf der Festplatte aktualisiert
        self.model.save()

  # Hier kann eine KI-Prediction durchgeführt werden - es wird hier ein State als 9-langes Array benötigt.
  # Epsilon beschreibt dabei, mit welcher Wahrscheinlichkeit die Vorhersage nicht von der KI stammen soll (sondern von einem zufälligen Action-Generator).
  # Zurückgegeben wird die Aktion der KI als 9-längigen Array, gefüllt mit Nullen, außer an einer Stelle, wo eine 1 ist.
  # Der Index dieser 1 im Array gibt an, an welcher Stelle des Spielfelds das Zeichen der KI platziert werden soll.

  def pick_action(self, state, epsilon):
    st_tensor = np.array(state).reshape((1, -1))
    act = [0,0,0]
    if np.random.rand() < epsilon:
        act[np.random.randint(3)] = 1
    else:
        result = self.model.model.predict(st_tensor)
        argmax = np.argmax(result[0])
        act[argmax] = 1
        del result
        del argmax

    return act

  # Vorhersage von KI (ohne Epsilon Greedy) mit allen genauen Q-Values.
  def vanilla_prediction(self, state):
    st_tensor = np.array(state).reshape((1, -1))
    return self.model.model.predict(st_tensor)[0]
