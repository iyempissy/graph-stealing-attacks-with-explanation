# Private Graph Extraction via Feature Explanations
### by anonymous authors
This repository contains additional details and reproducibility of the stated results. 
We will make the repository publicly available upon acceptance. 

<!-- ## Motivation
<p align=center><img style="vertical-align:middle" width="500" height="290" src="https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/motivation.png" /></p>

### Problem Definition: 
Given the explanation and/or some auxiliary information, can we reconstruct the private graph?

### Attack example: Graph Stealing Attacks with Explanation and Features (GSEF)
![alt text](https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/GSEF.png?raw=true)


## Attack taxonomy based on attacker’s knowledge
<p align=center><img style="vertical-align:middle" width="300" height="200" src="https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/attacktaxonomy.png" /></p> -->

## Results

| Exp       | Method      | Cora |    | CoraML |    | Bitcoin |    |
|-----------|-------------|:----:|:--:|:------:|:--:|:-------:|:--:|
|           |             | AUC  | AP | AUC    | AP | AUC     | AP |
| **Baselines** | FeatureSim  | 0.796 $\pm$ 0.02	| 0.822 $\pm$ 0.02 | 0.736 $\pm$ 0.	| 0.776 $\pm$ 0. | 0.536 $\pm$ 0.05	| 0.476 $\pm$ 0.01|
|           | LSA         | 0.794 $\pm$ 0.03	| 0.829 $\pm$ 0.02 | 0.728 $\pm$ 0.04	| 0.759 $\pm$ 0.01 | 0.530 $\pm$ 0.05	| 0.500 $\pm$ 0.06|
|           | GraphMI     | 0.859 $\pm$ 0.01	| 0.834 $\pm$ 0.01 | 0.815 $\pm$ 0.02	| 0.810 $\pm$ 0.03 | 0.583 $\pm$ 0.06	| 0.515 $\pm$ 0.05|
|           | SLAPS       | 0.716 $\pm$ 0.05	| 0.757 $\pm$ 0.04 | 0.682 $\pm$ 0.02	| 0.738 $\pm$ 0.04 | 0.590 $\pm$ 0.04	| 0.557 $\pm$ 0.07|
|           |             |      |    |        |    |         |    |
|    **Grad**   | GSEF-Concat | 0.694 $\pm$ 0.03 | 0.733 $\pm$ 0.03  | 0.685 $\pm$ 0.02 | 0.749 $\pm$ 0.02 | 0.447 $\pm$ 0.04 | 0.476 $\pm$ 0.04|
|           | GSEF-Mult   |0.692 $\pm$ 0.02 | 0.749 $\pm$ 0.01 | 0.683 $\pm$ 0.04 | 0.762 $\pm$ 0.04 | 0.266 $\pm$ 0.06 | 0.381 $\pm$ 0.06|
|           | GSEF        |$\underline{0.947}$ $\pm$ 0.04 | $\underline{0.955}$ $\pm$ 0.03 | $\bf{0.902}$ $\pm$ 0.02 | $\underline{0.832}$ $\pm$ 0.03 | $\bf{0.700}$ $\pm$ 0.05 | $\bf{0.715}$ $\pm$ 0.05|
|           | GSE         |0.870 $\pm$ 0.03 | 0.893 $\pm$ 0.02 | 0.689 $\pm$ 0.05 | 0.761 $\pm$ 0.04 | 0.254 $\pm$ 0.07 | 0.376 $\pm$ 0.06|
|           | ExplainSim  | $\bf{0.983}$ $\pm$ 0.01 | $\bf{0.980}$ $\pm$ 0.01 | 0.900 $\pm$ 0.01 | $\bf{0.904}$ $\pm$ 0.01 | $\underline{0.694}$ $\pm$ 0.05 | $\underline{0.656}$ $\pm$ 0.03|
|           |             |      |    |        |    |         |    |
|    **Grad-I**   | GSEF-Concat | 0.700 $\pm$ 0.02  | 0.755 $\pm$ 0.02  | 0.703 $\pm$ 0.05 	| 0.753 $\pm$ 0.05  | 0.522 $\pm$ 0.08 	| 0.526 $\pm$ 0.06 |
|           | GSEF-Mult   | 0.665 $\pm$ 0.04  | 0.702 $\pm$ 0.04  | 0.710 $\pm$ 0.03  | 0.743 $\pm$ 0.05  | 0.228 $\pm$ 0.07 	| 0.363 $\pm$ 0.07 |
|           | GSEF        | $\underline{0.914}$ $\pm$ 0.03 	| $\underline{0.917}$ $\pm$ 0.02  | $\underline{0.802}$ $\pm$ 0.02 	| $\bf{0.842}$ $\pm$ 0.05  | $\bf{0.710}$ $\pm$ 0.04 	| $\bf{0.725}$ $\pm$ 0.05 |
|           | GSE         | 0.872 $\pm$ 0.02 	| 0.900 $\pm$ 0.01  |0.725 $\pm$ 0. 	| 0.790 $\pm$ 0.  | 0.256 $\pm$ 0. 	| 0.377 $\pm$ 0. |
|           | ExplainSim  | $\bf{0.983}$  $\pm$ 0.01 	| $\bf{0.978}$  $\pm$ 0.01  | $\bf{0.908}$  $\pm$ 0.02 	| $\bf{0.911}$  $\pm$ 0.02  | 0.690 $\pm$ 0.05 	| 0.651 $\pm$ 0.07 |
|           |             |      |    |        |    |         |    |
|    **Zorro**   | GSEF-Concat | 0.823 $\pm$ 0.04	| 0.860 $\pm$ 0.05          | 0.735 $\pm$ 0.02	| 0.786 $\pm$ 0.01 | $\underline{0.575}$ $\pm$ 0.03 | 0.529 $\pm$ 0.05|
|           | GSEF-Mult   | 0.723 $\pm$ 0.	| 0.756 $\pm$ 0. | 0.681 $\pm$ 0.	| 0.697 $\pm$ 0. | 0.399 $\pm$ 0. | 0.449 $\pm$ 0.|
|           | GSEF        | $\bf{0.884}$ $\pm$ 0.03	| $\bf{0.880}$ $\pm$ 0.04 | $\underline{0.776}$ $\pm$ 0.03	| $\underline{0.820}$ $\pm$ 0.02 | 0.537 $\pm$ 0.05 | $\underline{0.527}$ $\pm$ 0.04|
|           | GSE         | 0.779 $\pm$ 0.04	| 0.810 $\pm$ 0.01 | 0.722 $\pm$ 0.02	| 0.777 $\pm$ 0.02 | $\bf{0.596}$ $\pm$ 0.03 | $\bf{0.561}$ $\pm$ 0.03|
|           | ExplainSim  | $\underline{0.871}$ $\pm$ 0.02	| $\underline{0.873}$ $\pm$ 0.02 | $\bf{0.806}$ $\pm$ 0.02 | $\bf{0.829}$ $\pm$ 0.03 | 0.427 $\pm$ 0.06 | 0.485 $\pm$ 0.05|
|           |             |      |    |        |    |         |    |
|    **Zorro-S**   | GSEF-Concat | 0.881 $\pm$ 0.03 | 0.913 $\pm$ 0.04 | 0.751 $\pm$ 0.03	| 0.804 $\pm$ 0.03 | 0.602 $\pm$ 0.05 | 0.586 $\pm$ 0.04 |
|           | GSEF-Mult   | 0.752 $\pm$ 0.05 | 0.784 $\pm$ 0.05 | 0.710 $\pm$ 0.03	| 0.727 $\pm$ 0.02 | 0.536 $\pm$ 0.04 | 0.524 $\pm$ 0.04 |
|           | GSEF        | $\bf{0.921}$ $\pm$ 0.02	| $\bf{0.918}$ $\pm$ 0.01 | $\bf{0.797}$ $\pm$ 0.02	| $\bf{0.801}$ $\pm$ 0.01 | $\underline{0.595}$ $\pm$ 0.05 | $\underline{0.572}$ $\pm$ 0.08|
|           | GSE         | 0.891 $\pm$ 0.03	| 0.916 $\pm$ 0.02 | 0.774 $\pm$ 0.02	| 0.818 $\pm$ 0.02 | 0.560 $\pm$ 0.08 | 0.561 $\pm$ 0.07 |
|           | ExplainSim  | $\underline{0.912}$ $\pm$ 0.03	| $\underline{0.932}$ $\pm$ 0.02 | 0.732 $\pm$ 0.02	| 0.804 $\pm$ 0.01 | 0.480 $\pm$ 0.05 | 0.489 $\pm$ 0.06 |
|           |             |      |    |        |    |         |    |
|    **GLime**   | GSEF-Concat | $\underline{0.634}$    $\pm$ 0.03       | $\underline{0.685}$ $\pm$ 0.05          | $\underline{0.627}$ $\pm$ 0.05	| $\underline{0.664}$ $\pm$ 0.03 | $\underline{0.536}$ $\pm$ 0.08 | $\underline{0.538}$ $\pm$ 0.04
|           | GSEF-Mult   |0.517 $\pm$ 0.02 | 0.529 $\pm$ 0.02 | 0.563 $\pm$ 0.04	| 0.570 $\pm$ 0.05 | 0.238 $\pm$ 0.08 | 0.362 $\pm$ 0.08 |
|           | GSEF        | $\bf{0.769}$ $\pm$ 0.02	| $\bf{0.800}$ $\pm$ 0.03 | $\bf{0.681}$ $\pm$ 0.03	| $\bf{0.740}$ $\pm$ 0.04 | $\bf{0.548}$ $\pm$ 0.05 | $\bf{0.542}$ $\pm$ 0.07 |
|           | GSE         | 0.559 $\pm$ 0.05	| 0.588 $\pm$ 0.04 | 0.503 $\pm$ 0.03	| 0.565 $\pm$ 0.02 | 0.262 $\pm$ 0.13 | 0.371 $\pm$ 0.16|
|           | ExplainSim  | 0.513 $\pm$ 0.05	| 0.535 $\pm$ 0.03 | 0.522 $\pm$ 0.03	| 0.515 $\pm$ 0.03 | 0.502 $\pm$ 0.08 | 0.498 $\pm$ 0.05|
|           |             |      |    |        |    |         |    |
|    **GNNExp**   | GSEF-Concat | 0.600 $\pm$ 0.06 | 0.639  $\pm$ 0.05 | 0.649 $\pm$ 0.05	| 0.677 $\pm$ 0.04 | 0.418 $\pm$ 0.07 | 0.459 $\pm$ 0.09|
|           | GSEF-Mult    | $\underline{0.703}$ $\pm$ 0.05 | $\underline{0.750}$ $\pm$ 0.03 | $\underline{0.661}$ $\pm$ 0.03	| $\underline{0.720}$ $\pm$ 0.02 | 0.391 $\pm$ 0.08 | 0.451 $\pm$ 0.05|
|           | GSEF        | $\bf{0.790}$ $\pm$ 0.	| $\bf{0.808}$ $\pm$ 0. | $\bf{0.700}$ $\pm$ 0.	| $\bf{0.732}$ $\pm$ 0. | $\bf{0.605}$ $\pm$ 0. | $\bf{0.573}$ $\pm$ 0. |
|           | GSE         |0.514 $\pm$ 0.	| 0.540 $\pm$ 0. | 0.461 $\pm$ 0.	| 0.494 $\pm$ 0. | 0.322 $\pm$ 0. | 0.406 $\pm$ 0. |
|           | ExplainSim  |0.517 $\pm$ 0.	| 0.513 $\pm$ 0. | 0.498 $\pm$ 0.	| 0.499 $\pm$ 0. | $\underline{0.539}$ $\pm$ 0. | $\underline{0.523}$ $\pm$ 0. |


## Parameters
AutoEncoder

## Running Explanation Attacks

### Running GSEF-Concat
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explanation} -ntrials 10 -attack_type gsef_concat 
```

### Running GSEF-Mult
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explanation} -ntrials 10 -attack_type gsef_mult
```

### Running GSEF
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explanation} -use_exp_as_reconstruction_loss 1 -ntrials 10 -attack_type gsef
```

### Running GSE
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explanation} -ntrials 10 -attack_type gse
```

### Running ExplainSim
```bash
python3 main.py -model pairwise_sim -dataset {dataset-name} -explanation_method {explanation} -ntrials 10 -attack_type explainsim
```

## Running Baselines:

### Running SLAPS
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explanation} -ntrials 10 -attack_type slaps
```

### Running FeatureSim
```bash
python3 main.py -model pairwise_sim -dataset {dataset-name} -explanation_method {explanation} -ntrials 10 -attack_type featuresim
```

### Running GraphMI
Download code from [their repository](https://github.com/zaixizhang/GraphMI) and use our data pipeline

### Running LSA
Download code from [their repository](https://github.com/xinleihe/link_stealing_attack) (attack-2) and use our data pipeline


## Running Defense
```bash
python3 main.py -model pairwise_sim -dataset {dataset-name} -explanation_method zorro-hard -ntrials 10 -attack_type explainsim -use_defense 5 -epsilon {eps}
```

## Parameters
Parameters for running the code are enclosed in {}. The take the following values:
- dataset-name ==> ['cora', 'cora_ml', 'bitcoin']
- explanation ==> ['grad', 'gradinput', 'zorro-soft', 'zorro-hard', 'graphlime', 'gnn-explainer']
- eps ==> [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
