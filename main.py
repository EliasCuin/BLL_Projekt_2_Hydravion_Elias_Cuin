import agent
import environment
import pygame
import sys

#_____________________________________
# Modusauswahl:

MODELPATH = "model.h5" # Pfad zum Speichern oder Laden des KI-Modells (Modelle haben das Format .h5).
MODE = "PLAY"

# MODE = "PLAY": aktiviert das Spielen gegen eine KI, geladen aus MODELPATH.
# MODE = "TRAIN" oder sonstiges: steht für das Tranieren einer KI, dabei wird das KI-Modell in "MODELPATH" (z. 8) gespeichert.

#_____________________________________

# Funktion zum Visualisieren des Spiels der KI innerhalb des Environments.
def play():
  global agent
  global environment

  # Initialisieren des "Agent" und Starten des Environment.
  agent = agent.agent(environment.Environment, MODELPATH)
  game = environment.Environment()
  game.startVisualisation()

  # Einstellungen der FPS des Spiels:
  deltaFrame = 25

  # Schleife zum Visualisieren des Spiels mithilfe von PyGame
  while True:
    startVisualisationTime = pygame.time.get_ticks()
    if(game.gameOver):
      game.reset()
    state = game.getState()
    action = agent.pick_action(state, epsilon=0)
    game.step(action)
    game.visualiseCurrentFrame()
    passedTime = pygame.time.get_ticks() - startVisualisationTime
    if(deltaFrame - passedTime > 0):
      pygame.time.wait(deltaFrame - passedTime)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

# Startet Visualisierung bzw. Trainieren basierend auf eingestellten Modus.
if(MODE == "PLAY"):
   play()
else:
  agent = agent.agent(environment.Environment, MODELPATH)
  agent.train(epStart=10 )