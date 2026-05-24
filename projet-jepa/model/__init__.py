"""
I-JEPA Model Components
=======================
This package contains the three core neural network components of I-JEPA:

1. **Context Encoder** (encoder.py): A Vision Transformer (ViT) that encodes
   visible (unmasked) image patches into latent representations.

2. **Predictor** (predictor.py): A narrow transformer that predicts the latent
   representations of masked target patches using context embeddings and
   positional information of the targets.

3. **Target Encoder** (target_encoder.py): An EMA (Exponential Moving Average)
   copy of the context encoder that produces the prediction targets. This
   encoder receives NO gradients and is updated only via momentum.

KEY INSIGHT (What makes I-JEPA different from MAE):
- MAE reconstructs PIXELS of masked patches (operates in pixel space)
- I-JEPA predicts REPRESENTATIONS of masked patches (operates in latent space)
- This forces the model to learn semantic features rather than low-level textures
"""
