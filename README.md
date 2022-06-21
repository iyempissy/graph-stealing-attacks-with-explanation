# Anonymized code for the paper: Private Graph Extraction via Feature Explanations

<!-- ## Motivation
<p align=center><img style="vertical-align:middle" width="500" height="290" src="https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/motivation.png" /></p>

### Problem Definition: 
Given the explanation and/or some auxiliary information, can we reconstruct the private graph?

### Attack example: Graph Stealing Attacks with Explanation and Features (GSEF)
![alt text](https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/GSEF.png?raw=true)


## Attack taxonomy based on attacker’s knowledge
<p align=center><img style="vertical-align:middle" width="300" height="200" src="https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/attacktaxonomy.png" /></p> -->


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
