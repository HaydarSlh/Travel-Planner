# Tool: classify_destination
# Input validated by schemas.tools.ClassifyDestinationInput before execution.
# Runs the loaded joblib Pipeline (injected via dependency, never reloaded per-call)
# to predict the travel style label for a given destination's feature vector.
# Returns the predicted label and per-class probabilities for the agent to use in ranking.
# On failure, returns a structured error dict — never raises into the agent loop.
