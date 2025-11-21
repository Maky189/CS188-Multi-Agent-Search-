The function I implemented for question 5 is shown below exactly as it appears in `multiAgents.py`. I include it here so the reader can see the implementation and then I explain, in continuous prose, what each part does and why it was included.

```python
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: implemented a compact heuristic that combines the game's
    built-in `score` with a strong penalty for remaining food, a distance
    penalty to the nearest food, and a ghost-related term that rewards
    reachable scared ghosts while accounting for the closest active ghost.
    """
    ghostStates = currentGameState.getGhostStates()
    pacmanPosition = currentGameState.getPacmanPosition()
    counter = currentGameState.getNumFood()
    score = currentGameState.getScore()
    huge = len(currentGameState.getCapsules())

    food = currentGameState.getFood()
    foodPosition = food.asList()
    foodPosition = sorted(foodPosition, key=lambda position : manhattanDistance(pacmanPosition, position))

    closestDistanceToFood = 0
    if len(foodPosition) > 0:
        closestDistanceToFood = manhattanDistance(foodPosition[0], pacmanPosition)

    closeGhost = float("inf")
    ghostEval = 0

    for ghost in ghostStates:
        ghostPosition = ghost.getPosition()
        manhattan_Distance = manhattanDistance(pacmanPosition, ghostPosition)

        if ghost.scaredTimer == 0:
            if manhattan_Distance < closeGhost:
                closeGhost = manhattan_Distance
        elif ghost.scaredTimer > manhattan_Distance:
            ghostEval += 200 - manhattan_Distance

    if closeGhost == float("inf"):
        closeGhost = 0
    ghostEval += closeGhost

    return score - 10 * counter + 1 * ghostEval * huge - 2 * closestDistanceToFood
```

The code begins by extracting information from the `GameState` that will be useful for determining how good a state is. It fetches the ghost states because whether ghosts are near or scared fundamentally changes what is safe and desirable. It fetches Pacman's position since distances to objects (food, ghosts) are computed relative to Pacman. It also reads the number of food pellets remaining into `counter` and the current environment `score` so the heuristic can combine progress toward the goal with the game’s own scoring.

Next the implementation counts power capsules and stores that count in the variable named `huge`. This value is used later as a multiplier for the ghost-related term; the intent is to make ghost-chasing behaviour more significant when capsules are present on the board, though as discussed below that multiplier is a tunable design choice.

To handle food, the function converts the food grid to a list of coordinates and sorts that list by Manhattan distance from Pacman. The first element (if any food exists) is therefore the nearest food pellet and the code computes `closestDistanceToFood` as that Manhattan distance. If there is no food left `closestDistanceToFood` remains zero. This single nearest-food distance is used to encourage movement toward the closest pellet by subtracting twice that distance from the final evaluation, so smaller distances increase the returned value.

The ghost handling is a two-part procedure. First, `closeGhost` is initialized to positive infinity and will be reduced to the minimum Manhattan distance to any non-scared ghost. This means the heuristic tracks how close the most dangerous (non-scared) ghost is. Second, a separate accumulator `ghostEval` collects bonuses for scared ghosts that are both near and reachable while they remain scared. Inside the loop over `ghostStates`, the code computes the Manhattan distance to each ghost and checks whether the ghost is scared. If a ghost is not scared and it is closer than any previously observed non-scared ghost, `closeGhost` is updated to that smaller distance. If a ghost is scared and its `scaredTimer` is greater than the Manhattan distance (loosely, the ghost will remain scared long enough to reach it), the implementation adds a reward equal to `200 - manhattan_Distance` to `ghostEval`. This reward grows when the ghost is closer and when it is certainly catchable before the scared timer expires.

After the loop, if `closeGhost` was never updated it is set to zero so it does not remain an infinite value. The current value of `closeGhost` is added into `ghostEval` so that ghost-related considerations include both the distance to the nearest active ghost and any large positive incentives to chase scared ghosts.

Finally, the function returns a linear combination of the features. It starts with the built-in `score`, subtracts `10 * counter` to penalize remaining food strongly, adds the product `ghostEval * huge` (so ghost-related incentives are amplified in proportion to how many capsules remain), and subtracts `2 * closestDistanceToFood` to encourage closing on the nearest pellet. In effect, the returned number balances safety, immediate opportunities to eat scared ghosts, and progress toward clearing the board.

The specific numeric choices are pragmatic and intended to produce a certain style of play rather than to be universally optimal. The `-10 * counter` term gives a heavy reward for reducing the number of remaining pellets; that makes the agent prioritize finishing the level, sometimes at the cost of taking riskier moves. The `200 - manhattan_Distance` bonus for reachable scared ghosts is deliberately large to make chasing edible ghosts attractive, but if that leads the agent to risk death, the constant can be reduced. The multiplication of `ghostEval` by `huge` aggressively increases ghost-chasing incentives when capsules are present, but it also means ghost considerations vanish when no capsules remain; changing this multiplier to a boolean indicator (there are capsules or not) or to a different function of remaining capsules may give more consistent behaviour across different maps.

In practice, small refinements can improve robustness: replacing the single nearest-food distance with a combination of the k nearest foods or with a sum of inverse distances reduces myopic actions; normalizing distances by map size keeps weights portable between layouts; and explicitly penalizing immediate ghost proximity with a very large negative value ensures the agent values survival above small gains. Running the autograder or playing a number of automated games while sweeping the key constants (`10`, `200`, `2`) will reveal which adjustments produce the best trade-off for your testing layouts.

If you want, I will now mark this task done and optionally run the autograder for question 5 or play a few trial games to measure average performance and suggest tuned weights.
