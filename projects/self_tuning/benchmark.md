# Benchmark

We provide benchmarks of semi supervised learning algorithms included in this project on fine-grained classification
datasets [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
, [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
and [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).

### CUB-200-2011 on ResNet-50 (Supervised Pre-trained)

| Methods      | 15%  | 30%  | 50%  | Avg  |
| ------------ | ---- | ---- | ---- | ---- |
| baseline     | 46.8 | 59.3 | 69.8 | 58.6 |
| pseudo_label | 53.2 | 64.9 | 72.5 | 63.5 |
| pi_model     | 47.3 | 59.2 | 70.0 | 58.8 |
| mean_teacher | 63.8 | 72.2 | 78.4 | 71.5 |
| uda          | 46.8 | 62.3 | 72.2 | 60.4 |
| fix_match    | 49.8 | 67.3 | 75.9 | 64.3 |
| self_tuning  | 64.4 | 73.6 | 78.8 | 72.3 |

### Stanford Cars on ResNet-50 (Supervised Pre-trained)

| Methods      | 15%  | 30%  | 50%  | Avg  |
| ------------ | ---- | ---- | ---- | ---- |
| baseline     | 38.3 | 60.7 | 72.6 | 57.2 |
| pseudo_label | 45.6 | 68.4 | 78.6 | 64.2 |
| pi_model     | 38.0 | 59.7 | 71.6 | 56.4 |
| mean_teacher | 70.5 | 83.6 | 89.1 | 81.1 |
| uda          | 49.3 | 69.6 | 79.8 | 66.2 |
| fix_match    | 51.4 | 76.1 | 81.1 | 69.5 |
| self_tuning  | 74.7 | 85.2 | 89.3 | 83.1 |

### FGVC Aircraft on ResNet-50 (Supervised Pre-trained)

| Methods      | 15%  | 30%  | 50%  | Avg  |
| ------------ | ---- | ---- | ---- | ---- |
| baseline     | 42.4 | 59.0 | 68.2 | 56.5 |
| pseudo_label | 49.1 | 68.4 | 78.6 | 65.4 |
| pi_model     | 43.0 | 58.5 | 69.0 | 56.8 |
| mean_teacher | 61.4 | 76.0 | 81.2 | 72.9 |
| uda          | 50.5 | 66.1 | 73.0 | 63.2 |
| fix_match    | 50.0 | 66.0 | 71.9 | 62.6 |
| self_tuning  | 66.0 | 79.0 | 83.6 | 76.2 |
