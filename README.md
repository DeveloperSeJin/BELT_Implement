BELT: Bootstrapped EEG-to-Language Training
This repository contains an unofficial implementation of the paper "BELT: Bootstrapped EEG-to-Language Training by Natural Language Supervision" (Zhou et al., IEEE TNSRE 2024).
Overview
BELT is a novel approach for decoding natural language from EEG signals using a Discrete Conformer (D-Conformer) encoder and bootstrapped training with language supervision. This implementation aims to reproduce the key components and methodology described in the paper.
Note: This is an unofficial implementation as the original code has not been released by the authors.
Paper Information

Title: BELT: Bootstrapped EEG-to-Language Training by Natural Language Supervision
Authors: Jinzhao Zhou, Yiqun Duan, Yu-Cheng Chang, Yu-Kai Wang, Chin-Teng Lin
Published: IEEE Transactions on Neural Systems and Rehabilitation Engineering, Vol. 32, 2024
DOI: 10.1109/TNSRE.2024.3450795


Implementation Status
This is an independent implementation based on the methodology described in the paper. Key implementation details include:

✅ D-Conformer architecture with Conformer blocks
✅ Vector quantization module
✅ Three bootstrapping strategies
✅ EEG preprocessing pipeline
✅ Training objectives for multiple tasks

Results (from paper)
TaskMetricBELTBaselineSentence DecodingBLEU-142.31%36.86%Sentiment ClassificationAccuracy69.3%55.3%Word ClassificationTop-10 Acc31.04%25.26%

Disclaimer
This is an unofficial implementation created for research and educational purposes. The original authors have not released their code at the time of this implementation. Results may differ from those reported in the paper due to implementation details and hyperparameter choices.

Acknowledgments
This implementation is based on the methodology described in the BELT paper by Zhou et al. We thank the authors for their detailed description of the approach.
