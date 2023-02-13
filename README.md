# Private Graph Extraction via Feature Explanations
### by Iyiola E. Olatunji, Mandeep Rathee, Thorben Funke, and Megha Khosla
This repository contains additional details and reproducibility of the stated results.
*Accepted in PETS 2023*

## Motivation
<p align=center><img style="vertical-align:middle" width="500" height="290" src="https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/motivation.png" /></p>

### Problem Definition: 
Given the explanation and/or some auxiliary information, can we reconstruct the private graph?

### Attack example: Graph Stealing Attacks with Explanation and Features (GSEF)
![alt text](https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/GSEF.png?raw=true)


## Attack taxonomy based on attacker’s knowledge
<p align=center><img style="vertical-align:middle" width="300" height="200" src="https://github.com/iyempissy/graph-stealing-attacks-with-explanation/blob/main/images/attacktaxonomy.png" /></p>

## Modules in GSEF attack and it's variants explained

### Generator:
- The generator takes the explanations and/or node features and outputs an adjacency matrix
- First, it assigns weights to all possible edges of each node and is fully parameterized.
- Then optimizes these weights (parameters) over the training epochs via the final loss function that combines both the classification task loss and the self-supervision task loss
- The final adjacency matrix is the one that minimizes the final loss function

### Self-supervision:
- Some nodes might not be connected to a labelled nodes. In such case, they will retain their initially assigned weights. Hence, the self-supervision module helps in learning reasonable edge weights for all possible edges
- The goal is to reconstruct clean features from noisy features 
- It consists of a *GCN* denoising autoencoder that takes as input the generated adjacency and noisy features

### Classifcation module:
- Utilize the knowledge of the labels to further optimize the adjacency matrix
- The classification module is a *GCN* model that takes as input the the node features and the newly optimized adjacency matrix via self-supervision


## Results
All experiments were run 10 times. We present the mean and the standard deviation below.




| Exp       | Method      | Cora |    | CoraML |    | Bitcoin |    |
|-----------|-------------|:----:|:--:|:------:|:--:|:-------:|:--:|
|           |             | AUC  | AP | AUC    | AP | AUC     | AP |
| **Baselines** | FeatureSim  |$0.799 \pm 0.04$	| $0.827 \pm 0.04$| $0.706 \pm 0.08$	| $0.753 \pm 0.07$| $0.535 \pm 0.03$	| $0.478 \pm 0.02$|
|           | LSA         |$0.795 \pm 0.03$	| $0.810 \pm 0.02$| $0.725 \pm 0.04$	| $0.760 \pm 0.01$| $0.532 \pm 0.05$	| $0.500 \pm 0.06$|
|           | GraphMI     |$0.856 \pm 0.01$	| $0.830 \pm 0.01$| $0.808 \pm 0.02$	| $0.814 \pm 0.03$| $0.585 \pm 0.05$	| $0.518 \pm 0.06$|
|           | SLAPS       |$0.736 \pm 0.05$	| $0.776 \pm 0.05$| $0.649 \pm 0.06$	| $0.702 \pm 0.07$| $0.597 \pm 0.09$	| $0.577 \pm 0.07$|
|           |             |      |    |        |    |         |    |
|    **Grad**   | GSEF-Concat |$0.734 \pm 0.05$	| $0.773 \pm 0.04$     | $0.640 \pm 0.05$	| $0.705 \pm 0.04$| $0.527 \pm 0.04$	| $0.515 \pm 0.03$|
|           | GSEF-Mult   |$0.678 \pm 0.04$	| $0.737 \pm 0.03$| $0.666 \pm 0.06$	| $0.730 \pm 0.05$| $0.264 \pm 0.07$	| $0.383 \pm 0.04$|
|           | GSEF        |$\underline{0.948 \pm 0.02}$	| $\underline{0.953 \pm 0.01}$ | $\mathbf{0.902 \pm 0.08}$	| $\underline{0.833 \pm 0.07}$ | $\mathbf{0.700 \pm 0.05}$	| $\mathbf{0.715 \pm 0.04}$|
|           | GSE         |$0.924 \pm 0.03$	| $0.939 \pm 0.02$| $0.699 \pm 0.07$	| $0.768 \pm 0.05$| $0.229 \pm 0.03$	| $0.365 \pm 0.02$|
|           | ExplainSim  |$\mathbf{0.984 \pm 0.01}$	| $\mathbf{0.978 \pm 0.01}$| $\underline{0.890 \pm 0.04}$ | $\mathbf{0.891 \pm 0.04}$| $\underline{0.681 \pm 0.03}$	| $\underline{0.644 \pm 0.03}$|
|           |             |      |    |        |    |         |    |
|    **Grad-I**   | GSEF-Concat |$0.734 \pm 0.06$         | $0.775 \pm 0.04$         | $0.674 \pm 0.05$	| $0.724 \pm 0.04$| $0.525 \pm 0.09$	| $0.527 \pm 0.05$|
|           | GSEF-Mult   |$0.691 \pm 0.02$| $0.742 \pm 0.02$| $0.717 \pm 0.05$| $0.756 \pm 0.06$| $0.252 \pm 0.03$	| $0.380 \pm 0.02$|
|           | GSEF        |$\underline{0.949 \pm 0.02}$	| $\underline{0.950 \pm 0.02}$ | $\underline{0.787 \pm 0.08}$	| $\underline{0.832 \pm 0.07}$ | $\mathbf{0.709 \pm 0.04}$	| $\mathbf{0.723 \pm 0.03}$|
|           | GSE         |$0.903 \pm 0.04$	| $0.923 \pm 0.04$|$0.717 \pm 0.08$	| $0.781 \pm 0.06$| $0.256 \pm 0.03$	| $0.380 \pm 0.02$|
|           | ExplainSim  |$\mathbf{0.984 \pm 0.01}$	| $\mathbf{0.979 \pm 0.01}$| $\mathbf{0.903 \pm 0.04}$	| $\mathbf{0.899 \pm 0.04}$| $\underline{0.681 \pm 0.03}$	| $\underline{0.644 \pm 0.03}$|
|           |             |      |    |        |    |         |    |
|    **Zorro**   | GSEF-Concat | $0.823 \pm 0.04$	| $0.860 \pm 0.05$          | $0.735 \pm 0.02$	| $0.786 \pm 0.01$ | $\underline{0.575 \pm 0.03}$ | $0.529 \pm 0.05$|
|           | GSEF-Mult   |$0.723 \pm 0.03$	| $0.756 \pm 0.03$ | $0.681 \pm 0.02$	| $0.697 \pm 0.04$ | $0.399 \pm 0.07$ | $0.449 \pm 0.05$|
|           | GSEF        | $\mathbf{0.884 \pm 0.03}$	| $\mathbf{0.880 \pm 0.04}$ | $\underline{0.776 \pm 0.03}$	| $\underline{0.820 \pm 0.02}$ | $0.537 \pm 0.05$ | $\underline{0.527 \pm 0.04}$|
|           | GSE         | $0.779 \pm 0.04$	| $0.810 \pm 0.01$ | $0.722 \pm 0.02$	| $0.777 \pm 0.02$ | $\mathbf{0.596 \pm 0.03}$ | $\mathbf{0.561 \pm 0.03}$|
|           | ExplainSim  |$\underline{0.871 \pm 0.02}$	| $\underline{0.873 \pm 0.02}$ | $\mathbf{0.806 \pm 0.02}$ | $\mathbf{0.829 \pm 0.03}$ | $0.427 \pm 0.06$ | $0.485 \pm 0.05$|
|           |             |      |    |        |    |         |    |
|    **Zorro-S**   | GSEF-Concat | $0.907 \pm 0.03$          | $0.922 \pm 0.02$          | $\underline{0.747 \pm 0.06}$	| $\underline{0.791 \pm 0.05}$ | $\mathbf{0.601 \pm 0.06}$ | $\mathbf{0.590 \pm 0.05}$|
|           | GSEF-Mult   | $0.794 \pm 0.06$ | $0.815 \pm 0.06$ | $0.712 \pm 0.06$	| $0.740 \pm 0.06$ | $0.490 \pm 0.08$ | $0.491 \pm 0.05$|
|           | GSEF        | $\mathbf{0.918 \pm 0.02}$	| $\underline{0.923 \pm 0.02}$ | $\mathbf{0.776 \pm 0.06}$	| $0.819 \pm 0.05$ | $\underline{0.598 \pm 0.03}$ | $\underline{0.565 \pm 0.03}$|
|           | GSE         | $0.893 \pm 0.04$	| $0.915 \pm 0.02$ | $0.742 \pm 0.06$	| $\mathbf{0.784 \pm 0.05}$ | $0.571 \pm 0.03$ | $0.564 \pm 0.04$|
|           | ExplainSim  |$\underline{0.908 \pm 0.03}$	| $\mathbf{0.934 \pm 0.02}$ | $0.732 \pm 0.05$	| $0.787 \pm 0.03$ | $0.484 \pm 0.04$ | $0.496 \pm 0.03$|
|           |             |      |    |        |    |         |    |
|    **GLime**   | GSEF-Concat | $\underline{0.643 \pm 0.05}$          | $\underline{0.710 \pm 0.04}$          | $\underline{0.610 \pm 0.05}$	| $\underline{0.652 \pm 0.04}$ | $\underline{0.473 \pm 0.07}$ | $\underline{0.492 \pm 0.05}$|
|           | GSEF-Mult   |$0.516 \pm 0.06$ | $0.522 \pm 0.04$ | $0.517 \pm 0.05$	| $0.528 \pm 0.04$ | $0.264 \pm 0.03$ | $0.371 \pm 0.01$|
|           | GSEF        |$\mathbf{0.730 \pm 0.05}$	| $\mathbf{0.774 \pm 0.03}$ | $\mathbf{0.681 \pm 0.05}$	| $\mathbf{0.740 \pm 0.05}$ | $\mathbf{0.542 \pm 0.05}$ | $\mathbf{0.525 \pm 0.03}$|
|           | GSE         |$0.558 \pm 0.06$	| $0.571 \pm 0.05$ | $0.540 \pm 0.06$	| $0.555 \pm 0.05$ | $0.236 \pm 0.04$ | $0.361 \pm 0.02$|
|           | ExplainSim  |$0.505 \pm 0.04$	| $0.524 \pm 0.04$ | $0.520 \pm 0.04$	| $0.523 \pm 0.04$ | $0.504 \pm 0.05$ | $0.512 \pm 0.03$|
|           |             |      |    |        |    |         |    |
|    **GNNExp**   | GSEF-Concat |$0.614 \pm 0.04$          | $0.650 \pm 0.04$          | $0.653 \pm 0.05$	| $0.705 \pm 0.04$ | $0.467 \pm 0.11$ | $0.489 \pm 0.06$|
|           | GSEF-Mult     |$\underline{0.724 \pm 0.05}$ | $\underline{0.760 \pm 0.05}$ | $\underline{0.637 \pm 0.05}$	| $\underline{0.692 \pm 0.05}$ | $0.390 \pm 0.10$ | $0.454 \pm 0.05$|
|           | GSEF        |$\mathbf{0.762 \pm 0.04}$	| $\mathbf{0.796 \pm 0.03}$ | $\mathbf{0.700 \pm 0.07}$	| $\mathbf{0.796 \pm 0.06}$ | $\mathbf{0.590 \pm 0.04}$ | $\mathbf{0.563 \pm 0.03}$|
|           | GSE         |$0.517 \pm 0.04$	| $0.552 \pm 0.03$ | $0.490 \pm 0.05$	| $0.508 \pm 0.04$ | $0.386 \pm 0.11$ | $0.451 \pm 0.07$|
|           | ExplainSim  |$0.537 \pm 0.05$	| $0.541 \pm 0.04$ | $0.484 \pm 0.05$	| $0.508 \pm 0.04$ | $\underline{0.551 \pm 0.04}$ | $\underline{0.545 \pm 0.03}$ |


## Hyperparameters
### Self-supervision
- num_layers - 2
- learning rate - 0.01
- epochs - 2000
- hidden_size - 512
- dropout - 0.5
- noisy_mask_ratio - 20

### Classification module
- num_layers - 2
- learning rate - 0.001
- epochs - 200
- hidden_size - 32
- dropout - 0.5

### Target model
- num_layers - 2
- learning rate - 0.01
- epochs - 200
- hidden_size - 32
- dropout - 0.5
- weight_decay - 5e-4
 

## Getting started
To simulate a realistic setting (and as described in the paper), we seperate the explanation generation environment and the attack environment where the explanations and attacks are executed by different entities (although installing graphlime ```pip install graphlime``` on the attack environment should be sufficient or alternatively, running all experiments on the explanation environment).  

### Server config
- OS and version: *Debian 10.3*
- Python version: *Python 3.8*
- Anaconda version: *2021.05*
- Cuda: *Cuda 11.1* (explanation environment), *Cuda 10.1* (attack environment)


### Explanation Generation Module
**Note:** All explanations used for the experiments are in the [Explanations](https://github.com/iyempissy/graph-stealing-attacks-with-explanation/tree/main/Explanations) folders to *avoid regenerating explanations*. They only need to be unzipped if necessary. Hence, step 1 and 2 can be skipped.

- **Step 1:** Setup explanation environement
```bash 
./explanation_environment.sh
```


- **Step 2:** Generate expalanations and save the model  
See [Generating Explanations](#generating-explanations) section

### Attack Module
- **Step 3:** Setup attack environment  
Install libraries in *attack_requirements.txt*  
Refer to [https://pytorch.org/get-started/previous-versions/#linux-and-windows-20](https://pytorch.org/get-started/previous-versions/#linux-and-windows-20) on installing the PyTorch versions.

- **Step 4:** Run attack  
See [Running Explanation Attacks](#running-explanation-attacks) section

### Sample output:
```bash
reconstructed auroc mean 0.9239545119093648
reconstructed avg_prec mean 0.9385418527248227

reconstructed auroc std 0.03037829443212373
reconstructed avg_prec std 0.023833043366627218

average attacker advantage 0.8359649121761322
std attacker advantage 0.023077240761854497

total time 501.8652458667755127
```

## Code args
Parameters for running the code are enclosed in {}. The take the following values:
- dataset-name ==> ['cora', 'cora_ml', 'bitcoin', 'citeseer', 'credit', 'pubmed']
- explainer ==> ['grad', 'gradinput', 'zorro-soft', 'zorro-hard', 'graphlime', 'gnn-explainer']
- eps ==> [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]


## Generating Explanations
```bash
python3 explanations.py --model gcn --dataset {dataset-name} --explainer {explainer} --save_exp 
```


## Running Explanation Attacks

### Running GSEF-Concat
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explainer} -ntrials 10 -attack_type gsef_concat 
```

### Running GSEF-Mult
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explainer} -ntrials 10 -attack_type gsef_mult
```

### Running GSEF
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explainer} -use_exp_as_reconstruction_loss 1 -ntrials 10 -attack_type gsef
```

### Running GSE
```bash
python3 main.py -model end2end -dataset {dataset-name} -explanation_method {explainer} -ntrials 10 -attack_type gse
```

### Running ExplainSim
```bash
python3 main.py -model pairwise_sim -dataset {dataset-name} -explanation_method {explainer} -ntrials 10 -attack_type explainsim
```

## Running Baselines:

### Running SLAPS
```bash
python3 main.py -model end2end -dataset {dataset-name} -ntrials 10 -attack_type slaps
```

### Running FeatureSim
```bash
python3 main.py -model pairwise_sim -dataset {dataset-name} -ntrials 10 -attack_type featuresim
```

### Running GraphMI
Download code from [their repository](https://github.com/zaixizhang/GraphMI) and use our data pipeline

### Running LSA
Download code from [their repository](https://github.com/xinleihe/link_stealing_attack) (attack-2) and use our data pipeline


## Running Defense
```bash
python3 main.py -model pairwise_sim -dataset {dataset-name} -explanation_method zorro-hard -ntrials 10 -attack_type explainsim -use_defense 5 -epsilon {eps}
```

### Running Fidelity
```bash
python3 main.py -model fidelity -get_fidelity 1 -dataset {dataset-name} -explanation_method {explainer} -ntrials 10 -use_defense 5 -epsilon {eps}
```

### Running Sparsity and Intersection
```bash
python3 main.py -model exp_intersection -get_intersection 1 -dataset {dataset-name} -explanation_method {explainer} -ntrials 10 -use_defense 5 -epsilon {eps}
```

### License
Copyright © 2022, Olatunji Iyiola Emmanuel.
Released under the [MIT license](LICENSE).
