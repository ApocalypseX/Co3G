# Co3G

This is the repository for the paper:  **Conservatism Controllable Compositional Guidance for Offline Safe Reinforcement Learning**.

## Algorithm projects

### Installation Instructions

1. Create Environment
    Please first create the conda environments by using:
    
    ```bash
    conda create -n co3g python=3.9.17
    ```
    
    Then, install Pytorch and packages in requirements by using:
    
    ```bash
    conda activate co3g
    pip install -r requirements.txt
    ```
    
    For jaxlib, you may need to manually download it from the official website and then install it. For example:

    ```bash
    pip install jax==0.4.9 \
    jaxlib==0.4.9+cuda12.cudnn88 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
    
2. Download Datasets
    
    First, download the datasets of OSRL tasks from [http://data.offline-saferl.org/download](http://data.offline-saferl.org/download).
    
    Then, modify the 23rd line in `DSRL/dsrl/offline_env.py`, substitute the 'XXX’ with the real downloaded data root.
    

### Co3G Training and Evaluation

First, activate the conda environment:

```bash
conda activate co3g
```

Then, train Co3G by

```bash
bash run_co3g.sh
```

Subsequently, put the path of learned models into the list `diffusor_list` in all train of evaluation python files in `launcher/examples/`.

Next, for Co3G RL, train the latent RL guider by:

```bash
bash run_co3g_guider.sh
```

Finally, for Co3G Manual and RS, evaluate the policy by:

```bash
bash run_co3g_evaluate_manual/rs.sh
```



## Note

The implementation is based on [[FISOR](https://github.com/ZhengYinan-AIR/FISOR)] which is open-sourced.