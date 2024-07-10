# Benchmarking Overview

We believe that using Bio Foundation models requires a thorough evaluation of the capabilities of each model on different downstream tasks. In contrast to Language modeling, biology has a lot of diverse applications for which these models can be used and tested. Therefore, we have implemented a benchmarking system which helps us evaluate different models on a series of tasks.
The evaluation of large language models has played a pivotal role in the development of new applications and it will be identical for Bio Foundation Models.

To evaluate the models, we have three choices:
1. Fine-Tune the Model
2. Probe the Model & Train a Model Head
3. Zero-Shot Evaluation

```{image} ./assets/Fine-Tune_Probing.jpg
:alt: Fine_Tuning_vs_Probing
:width: 200px
:align: center
```

The code for the benchmarking system is available in our [GitHub repository](https://github.com/helicalAI/helical/tree/release/helical/benchmark).

To enable easy benchmarking we have built a framework with the following parts:

```{image} ./assets/Benchmarking.jpg
:alt: Benchmarking_Setup
:width: 200px
:align: center
```
