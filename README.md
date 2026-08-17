---
library_name: transformers
license: apache-2.0
base_model: openai/whisper-medium
tags:
- generated_from_trainer
datasets:
- wjbmattingly/zulu_merged_audio
metrics:
- wer
model-index:
- name: whisper-zulu-medium
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: wjbmattingly/zulu_merged_audio
      type: wjbmattingly/zulu_merged_audio
    metrics:
    - name: Wer
      type: wer
      value: 0.1993037098042152
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# whisper-zulu-medium

This model is a fine-tuned version of [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) on the wjbmattingly/zulu_merged_audio dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2949
- Wer: 0.1993

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- total_train_batch_size: 64
- total_eval_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 400
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 0.8378        | 1.25  | 100  | 0.7290          | 0.5605 |
| 0.3624        | 2.5   | 200  | 0.4048          | 0.2791 |
| 0.2279        | 3.75  | 300  | 0.3236          | 0.2187 |
| 0.1524        | 5.0   | 400  | 0.2949          | 0.1993 |


### Framework versions

- Transformers 4.45.0.dev0
- Pytorch 2.0.1+cu117
- Datasets 2.21.0
- Tokenizers 0.19.1
