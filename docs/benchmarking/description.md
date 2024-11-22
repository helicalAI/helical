# Benchmarking Overview

We believe that using Bio Foundation models requires a thorough evaluation of the capabilities of each model on different downstream tasks. In contrast to Language modeling, biology has a lot of diverse applications for which these models can be used and tested. Therefore, we have implemented a benchmarking system which helps us evaluate different models on a series of tasks.
The evaluation of large language models has played a pivotal role in the development of new applications and it will be identical for Bio Foundation Models.

To evaluate the models, we have three choices:

- Fine-tune the model
- Probe the model & train a model Head
- Zero-shot evaluation

![Fine-tuning vs Probing](./assets/Fine-Tune_Probing.jpg)
*Fine-tuning vs probing comparison for bio foundation models.*

The code for the benchmarking system is available in our [GitHub repository](https://github.com/helicalAI/helical/tree/release/helical/benchmark).

To enable easy benchmarking we have built a framework with the following parts:

![Benchmarking Setup](./assets/Benchmarking.jpg)
*Framework setup for easy benchmarking of bio foundation models.*
