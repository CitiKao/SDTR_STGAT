# Dispatch RL Architecture Diagrams

This document summarizes the main dispatch/routing schemes currently implemented in the STDR workspace.

## 0. Overall Taxi Dispatch And Routing Pipeline

```mermaid
flowchart LR
    A[Historical NYC traffic data<br/>edge_speeds, demand, supply] --> B[STGAT speed predictor]
    A --> C[Demand / supply state]
    B --> D[Predicted edge speeds<br/>15 / 30 / 60 min]
    C --> E[Greedy dispatch module]
    E --> F[Dispatch OD pairs<br/>origin, destination, count]
    D --> G[Routing / reranking module]
    F --> G
    G --> H[Selected route]
    H --> I[Realized evaluation<br/>future edge_speeds]
    I --> J[Metrics<br/>travel time, win rate, regret]
```

Role split:

- Dispatch module: generates vehicle relocation OD pairs from demand/supply imbalance.
- Speed prediction module: predicts future edge speeds.
- Routing/RL module: chooses a route under prediction uncertainty.
- Evaluation: uses realized future speed, not predicted speed.

## 1. Original Edge-Level DDQN Routing

```mermaid
flowchart TB
    A[STGAT predicted speeds] --> B[Dynamic routing graph]
    C[Dispatch OD pair] --> D[RoutingEnv reset]
    B --> D
    D --> E[State<br/>current node, destination,<br/>neighbor speed, neighbor length, time slot]
    E --> F[DDQN Q-network]
    F --> G[Action<br/>choose next outgoing edge]
    G --> H[Move to next node]
    H --> I{Reached destination<br/>or max steps?}
    I -- no --> E
    I -- yes --> J[Route result]
    J --> K[Reward from realized speeds]
    K --> L[Replay buffer + DDQN update]
```

Key point:

- Action is local: choose the next edge.
- It directly competes with Dijkstra.
- In experiments, it learned reachability but did not beat Dijkstra-pred.

## 2. K-Candidate Route DDQN Reranker

```mermaid
flowchart TB
    A[STGAT predicted speeds] --> B[Predicted edge travel time]
    B --> C[K-shortest paths / Yen algorithm]
    D[Dispatch OD pair] --> C
    C --> E[Top-K feasible candidate routes<br/>K = 6 by default]
    E --> F[Route feature builder]
    F --> G[Route features<br/>pred time, distance, hops,<br/>min speed, mean speed, rank]
    G --> H[DDQN reranker]
    H --> I[Action<br/>choose one route from K]
    I --> J[Selected route]
    J --> K[Reward from realized future speeds]
    K --> L[Replay buffer + DDQN update]
```

Key point:

- Top-K here means Top-K feasible candidate routes.
- Top-K generation is algorithmic, not attention-based.
- DDQN does not build the path edge by edge; it reranks candidate routes.
- This is the main useful RL design found so far.

Main result on unseen opportunity samples:

- Val300: DDQN / Dijkstra-pred = 0.9805
- Test300: DDQN / Dijkstra-pred = 0.9799

## 3. Pattern Top-K Attention Enhanced Reranker

```mermaid
flowchart TB
    A[STGAT predicted edge speed profiles] --> B[Candidate route set<br/>from K-shortest paths]
    A --> C[All graph edge speed patterns]
    B --> D[Route pattern query<br/>mean speed curve of route edges]
    C --> E[Pattern attention search]
    D --> E
    E --> F[Top-K similar traffic-pattern edges<br/>not required to be connected]
    F --> G[Interpretable pattern features<br/>similarity, ref speed,<br/>low-speed ratio, attention entropy]
    B --> H[Base route features]
    G --> I[Feature concatenation]
    H --> I
    I --> J[DDQN reranker]
    J --> K[Choose one candidate route]
    K --> L[Realized-speed reward]
```

Interpretability:

- The pattern Top-K edges are reference traffic patterns, not route edges.
- They may be spatially disconnected.
- Each decision can report which similar-pattern edges were referenced.

Current effect:

- Base reranker test120: DDQN / Dijkstra-pred = 0.988512
- Pattern Top-K attention test120: DDQN / Dijkstra-pred = 0.988002
- It slightly improves win rate and interpretability, but the average travel-time gain is small.

## 4. Mixed Training + Direct Regret Reward

```mermaid
flowchart TB
    A[Training candidate pool] --> B{Sample type}
    B -->|70 percent| C[Opportunity cases<br/>Dijkstra-pred is not candidate oracle]
    B -->|30 percent| D[Normal cases<br/>general route distribution]
    C --> E[DDQN route reranker]
    D --> E
    E --> F[Chosen route]
    F --> G[Realized travel time]
    G --> H[Direct regret reward]
    H --> I["reward = (DijkstraPredRealTime - ChosenRealTime) / DijkstraPredRealTime"]
    I --> J[DDQN update]
```

Purpose:

- Opportunity cases teach the model how to correct Dijkstra-pred failures.
- Normal cases reduce over-specialization to only failure cases.
- Direct regret reward optimizes average realized travel-time improvement.

Current test120 effect:

- Baseline K6 shaped opportunity-only: DDQN / Dijkstra-pred = 0.988512
- Mixed + direct regret: DDQN / Dijkstra-pred = 0.984870
- Extra average improvement over baseline: 0.364 percentage points
- Tradeoff: win rate drops from 73.33 percent to 58.33 percent.

## 5. Current Method Comparison

```mermaid
flowchart LR
    A[Original edge-level DDQN] -->|lost to Dijkstra| B[Not main method]
    C[K-candidate route reranker] -->|main effective design| D[Recommended core method]
    E[Pattern Top-K attention] -->|interpretable, slight gain| F[Use as explainability module]
    G[Mixed + direct regret] -->|best average time, lower win rate| H[Use as average-time optimization variant]
```

Recommended paper framing:

- Main method: K-candidate route DDQN reranker.
- Explainability extension: Pattern Top-K traffic-pattern attention.
- Training strategy variant: mixed opportunity/normal training.
- Avoid claiming that RL globally beats Dijkstra on all OD cases.

