To achieve a dynamic adjustment of the learning rate (`lr`) and learning rate warmup proportion (`lr_warmup`) during training based on the observed `f1` score, you can implement a monitoring mechanism in your training loop. Below is a Markdown representation of the steps, as well as an explanation of how to integrate it into your code.

---

### 1) Dynamic Adjustment of Learning Rate and Warmup Proportion During Training

1. **Initialization**:

   - Define default values for `lr` and `lr_warmup` in the argument parser or configuration file.
   - Include a mechanism to monitor the `f1` score during training.

2. **Monitoring `f1` Score**:

   - During the training process, calculate the `f1` score after a validation step or a predefined number of iterations/epochs.
   - Compare the current `f1` score to a threshold to decide if adjustments are needed.

3. **Adjusting Hyperparameters**:

   - If the `f1` score is not improving or starts to degrade:
     - Reduce the `lr` value (e.g., from `5e-5` to `5e-6`).
     - Increase the `lr_warmup` proportion (e.g., from `0.1` to `0.2`).

4. **Implementation in Code**:
   - Add conditional logic to modify these values dynamically in the training loop.
   - Ensure the optimizer is updated with the new `lr` value after adjustment.

### Notes

- Adjustments should be gradual to avoid instability during training.
- Logging the changes in `lr` and `lr_warmup` can help track the impact of these modifications on model performance.
- Use a scheduler if available in your training framework (e.g., PyTorch's `ReduceLROnPlateau`) to simplify this process.

This ensures a systematic response to performance fluctuations and enhances the overall training robustness.

### Handling Memory Errors During Training: Adjusting Batch Size

When encountering a **memory error** during training (e.g., `CUDA out of memory`), the primary step is to reduce the batch size systematically. Below are structured instructions for addressing memory issues.

---

### 2) Steps to Handle Memory Errors by Reducing Batch Size

1. **Identify the Issue**:

   - A memory error usually occurs due to the model's inability to process the current batch size with the available GPU memory.

2. **Initial Configuration**:

   - Start with a batch size suitable for your hardware (e.g., `16`).
   - Include an argument in your configuration to set the batch size dynamically.

3. **Dynamic Adjustment**:

   - Upon encountering a memory error:
     1. Halve the batch size (e.g., from `16` to `8`).
     2. Retry the training process.
     3. If the error persists, halve the batch size again until the issue is resolved or the batch size reaches `1`.

4. **Log Changes**:

   - Record the changes in the batch size for later analysis and debugging.

5. **Iterative Check**:
   - Monitor the model's performance after batch size adjustments. Smaller batch sizes may require compensatory adjustments to learning rate or gradient accumulation steps.
