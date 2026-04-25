# LoRA Fine-Tuned LLM for extraction of aspect-based opinions from customer reviews

**Author:** Yassine BLAIECH

## Description of the Work

This project explores **aspect-based sentiment extraction** using a **LoRA fine-tuned large language model**. The task is to predict sentiment labels for three restaurant-review aspects:

- `Price`
- `Food`
- `Service`

The selected approach was **Approach 2: LoRA fine-tuned LLM**, using **Qwen/Qwen3-0.6B** as the base model. This model was chosen because it provides a strong balance between natural language reasoning ability and computational efficiency, making it suitable for rapid experimentation while remaining lightweight enough for standard GPU memory limits.

## Approach Selected

Rather than fully fine-tuning the entire model, the project relies on **Low-Rank Adaptation (LoRA)**. This makes it possible to adapt the model to the extraction task while training only a very small fraction of the full parameter count.

The main idea was to preserve the general language knowledge already present in the base model and specialize it for structured aspect-level sentiment prediction.

## Implementation and Formatting Strategy

To make the extraction task explicit, each review was transformed into a prompt-completion style training example:

```text
Review: [Text]
Extraction:
Price: [Val], Food: [Val], Service: [Val]
```

This formatting strategy encourages the model to learn a precise and repeatable output structure. During inference, the generated response is parsed to recover the predicted labels. If the output contains an invalid class, the system safely defaults that aspect to **`No Opinion`**.

This design improves robustness while keeping the generation format easy to evaluate.

## Architecture and Trainable Parameters

The project uses LoRA adapters inserted into the attention layers of the model, specifically targeting:

- `q_proj`
- `v_proj`

With this setup, the original model weights remain frozen while only the adapter parameters are updated.

This leads to a major reduction in computational cost:

| Experiment | Configuration | Trainable Parameters | Share of Total Parameters |
| --- | --- | ---: | ---: |
| Experiment 1 | `r = 8` | ~0.69M | ~0.11% |
| Experiment 2 | `r = 16` | ~1.38M | ~0.22% |

The second experiment doubles the LoRA rank, increasing expressivity while still remaining extremely parameter-efficient.

## Experimental Results

### Experiment 1: Baseline LoRA

**Hyperparameters**

- `r = 8`
- `lora_alpha = 16`
- `learning_rate = 2e-4`
- `epochs = 3`
- effective batch size = `4`

**Observations**

The validation loss decreased steadily:

`2.50 -> 2.44 -> 2.43`

This shows that the model learned the task consistently and established a strong baseline. Performance was already solid across all aspects, although the **Food** category remained slightly harder than **Price** and **Service**.

**Aspect-wise Accuracy**  
Averaged over 5 runs:

| Metric | Accuracy (%) |
| --- | ---: |
| Price | 86.17 |
| Food | 85.63 |
| Service | 88.07 |
| Overall Macro Average | 86.62 |

### Experiment 2: Optimized LoRA (Final Model)

**Hyperparameters**

- `r = 16`
- `lora_alpha = 32`
- `learning_rate = 3e-4`
- `epochs = 3`
- effective batch size = `16`

**Observations**

By increasing both the LoRA rank and scaling factor, the model gained more representational flexibility for aspect-specific nuances. A slightly higher learning rate also helped improve convergence within the same number of epochs.

This configuration improved performance across every category while still avoiding overfitting.

**Aspect-wise Accuracy**  
Averaged over 2 runs:

| Metric | Accuracy (%) |
| --- | ---: |
| Price | 86.92 |
| Food | 86.42 |
| Service | 88.50 |
| Overall Macro Average | 87.28 |

## Dev Data Accuracy

The final macro-average accuracy obtained on the development split is:

## **87.28%**

This confirms that the optimized LoRA setup produced a measurable improvement over the baseline while preserving the efficiency benefits of parameter-efficient fine-tuning.

## Possible Extensions and Future Work

If more time or computational resources were available, the project could be extended in several promising directions:

### 1. Targeting Additional Modules

The current setup only adapts `q_proj` and `v_proj`. Expanding LoRA to additional linear layers such as `k_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` could push performance closer to full fine-tuning.

### 2. Dynamic Learning Rate

Using a cosine scheduler with warmup instead of a fixed learning rate could produce smoother optimization and potentially lead to better final convergence.

### 3. Advanced Prompt Engineering

Stronger prompting strategies, such as few-shot formatting or more explicit instruction framing, may help the model better distinguish difficult sentiment classes such as **Mixed** and **Negative**.

### 4. Scaling the Base Model

The same optimized LoRA configuration could be applied to larger Qwen variants such as **Qwen/Qwen3-1.7B** or **Qwen/Qwen3-4B**, which may further improve extraction quality while keeping the fine-tuning process manageable.

## Repository Structure

```text
|-- src/
|   |-- config.py
|   |-- ftlora_extractor.py
|   `-- runproject.py
`-- readme.md
```
