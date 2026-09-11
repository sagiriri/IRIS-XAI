# IRIS-XAI — Final Results Summary

Auto-generated from the master benchmark CSV. No experiments were rerun.

## Model Performance

| dataset    | model        |   accuracy |   precision_macro |   recall_macro |   f1_macro |   auc_macro_ovr |   avg_batch_inference_time_sec |   model_size_mb |
|:-----------|:-------------|-----------:|------------------:|---------------:|-----------:|----------------:|-------------------------------:|----------------:|
| CIFAR10    | efficientnet |     0.9653 |          0.965404 |         0.9653 | 0.965294   |        0.998979 |                     0.007615   |        15.3364  |
| CIFAR10    | resnet18     |     0.9518 |          0.952308 |         0.9518 | 0.951737   |        0.998259 |                     0.002921   |        42.6546  |
| CIFAR10    | resnet50     |     0.1225 |          0.104144 |         0.1225 | 0.0678099  |        0.522329 |                     0.00685873 |        89.7542  |
| CIFAR10    | simplecnn    |     0.6392 |          0.700592 |         0.6392 | 0.635263   |        0.952654 |                     0.001025   |         1.49516 |
| FUNNYBIRDS | efficientnet |     0.972  |          0.973859 |         0.972  | 0.971904   |        0.999878 |                     0.010587   |        15.5319  |
| FUNNYBIRDS | resnet18     |     0.966  |          0.969136 |         0.966  | 0.966015   |        0.999788 |                     0.00249    |        42.7329  |
| FUNNYBIRDS | resnet50     |     0.012  |          0.012441 |         0.012  | 0.00655178 |        0.492837 |                     0.00532693 |        90.0668  |
| FUNNYBIRDS | simplecnn    |     0.964  |          0.968952 |         0.964  | 0.963855   |        0.999914 |                     0.235027   |         1.53437 |


## Explainability Score Matrix (0-100)

| dataset    | model        |   eg_gradcam |   gradcam |   intgrad |   lime |   shap |
|:-----------|:-------------|-------------:|----------:|----------:|-------:|-------:|
| CIFAR10    | efficientnet |         61.6 |      57.9 |      44.6 |   48.7 |   46.4 |
| CIFAR10    | resnet18     |         73.9 |      67.7 |      55.1 |   50.3 |   54.6 |
| CIFAR10    | resnet50     |         51.8 |      50.4 |      46.5 |   49.2 |   53.6 |
| CIFAR10    | simplecnn    |         72   |      67.2 |      51.2 |   48.5 |   56.8 |
| FUNNYBIRDS | efficientnet |         67.6 |      62.9 |      69.8 |   50.7 |   53.1 |
| FUNNYBIRDS | resnet18     |         74.7 |      71.3 |      69.5 |   45.4 |   66.9 |
| FUNNYBIRDS | resnet50     |         57.5 |      55.2 |      49   |   52.9 |   52.9 |
| FUNNYBIRDS | simplecnn    |         89.2 |      82.8 |      55.6 |   56.1 |   77.4 |


## FunnyBirds Ground-Truth Comparison

| model        | xai_method   |   part_overlap_ratio |   clutter_leakage_ratio |   explanation_runtime_sec |
|:-------------|:-------------|---------------------:|------------------------:|--------------------------:|
| efficientnet | eg_gradcam   |            0.0871125 |               0.0488797 |                 0.0312614 |
| efficientnet | gradcam      |            0.087087  |               0.048894  |                 0.054426  |
| efficientnet | intgrad      |            0.100734  |               0.235466  |                 0.61418   |
| efficientnet | lime         |            0.056517  |               0.077366  |                 0.907069  |
| efficientnet | shap         |            0.059693  |               0.203989  |                14.1223    |
| resnet18     | eg_gradcam   |            0.0787404 |               0.0612167 |                 0.0266803 |
| resnet18     | gradcam      |            0.078762  |               0.061251  |                 0.027239  |
| resnet18     | intgrad      |            0.103358  |               0.236666  |                 0.683185  |
| resnet18     | lime         |            0.014466  |               0.07599   |                 0.913121  |
| resnet18     | shap         |            0.086263  |               0.259791  |                 9.47061   |
| resnet50     | eg_gradcam   |            0.0266346 |               0.0598281 |                 1.63079   |
| resnet50     | gradcam      |            0.027395  |               0.0609979 |                 1.75622   |
| resnet50     | intgrad      |            0.0963429 |               0.251012  |                 9.24339   |
| resnet50     | lime         |            0.013873  |               0.0767796 |                 5.61054   |
| resnet50     | shap         |            0.0922389 |               0.23999   |               232.865     |
| simplecnn    | eg_gradcam   |            0.0891572 |               0.0722388 |                 0.0125472 |
| simplecnn    | gradcam      |            0.097656  |               0.091957  |                 0.155629  |
| simplecnn    | intgrad      |            0.106734  |               0.141498  |                 0.156486  |
| simplecnn    | lime         |            0.031214  |               0.069289  |                 0.886403  |
| simplecnn    | shap         |            0.107717  |               0.178773  |                 6.93345   |


## Overall Method Ranking (Grad-CAM vs SHAP vs LIME)

| xai_method   |   avg_explainability_score |   avg_runtime_sec |   avg_part_overlap_ratio |   avg_clutter_leakage_ratio |   rank_by_score |
|:-------------|---------------------------:|------------------:|-------------------------:|----------------------------:|----------------:|
| eg_gradcam   |                    68.5301 |          0.239711 |                0.0704112 |                   0.0605408 |               1 |
| gradcam      |                    64.4528 |          0.275443 |                0.072725  |                   0.065775  |               2 |
| shap         |                    57.7071 |         35.7203   |                0.086478  |                   0.220636  |               3 |
| intgrad      |                    55.1822 |          1.67363  |                0.101792  |                   0.216161  |               4 |
| lime         |                    50.2063 |          1.5181   |                0.0290175 |                   0.0748561 |               5 |

