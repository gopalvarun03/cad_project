# DeepCAD Architecture Explanation

DeepCAD treats 3D CAD models as a **sequence of construction commands** (e.g., Line, Arc, Circle, Extrude) rather than meshes or point clouds. It uses a two-stage approach:
1.  **Autoencoder (AE):** Compresses CAD sequences into a latent vector $z$.
2.  **Latent GAN (LGAN):** Generates new latent vectors $z$ from random noise.

## 1. Data Representation
*   **Sequence:** A CAD model is a sequence of $N$ commands.
*   **Tokens:**
    *   **Command:** 6 types (`Line`, `Arc`, `Circle`, `EOS`, `SOL`, `Ext`).
    *   **Arguments:** Parameters (coords, angles, etc.) quantized into 256 bins.
    *   **Groups:** Loops and extrusions are grouped.

## 2. Autoencoder (AE)
Located in `model/autoencoder.py`.

### Encoder
*   **Input:** Sequence of (Command, Args).
*   **Embedding (`CADEmbedding`):**
    *   Embeds commands and quantized arguments.
    *   Adds **Positional Encoding**.
    *   Adds optional **Group Embedding** to distinguish loops/extrusions.
*   **Transformer:** 4-layer `TransformerEncoder`.
*   **Bottleneck:** Aggregates the sequence (sum/mean) and projects it to a 256-dim latent code $z$ using `Tanh` activation.

### Decoder
*   **Input:** Latent code $z$.
*   **Embedding (`ConstEmbedding`):** Uses learned constant queries + positional encoding (does not use shifted inputs like standard seq2seq).
*   **Transformer:** 4-layer `TransformerDecoder` that attends to $z$ globally.
*   **Heads (`FCN`):**
    *   **Command Head:** Predicts command type.
    *   **Argument Head:** Predicts argument bins (256 classes).

### Loss (`trainer/loss.py`)
*   **Reconstruction Loss:** Cross-Entropy for both commands and arguments.

## 3. Latent GAN (LGAN)
Located in `model/latentGAN.py`.

*   **Objective:** Learn the distribution of $z$ produced by the AE.
*   **Generator:** 5-layer MLP (`Linear` -> `LeakyReLU` ... -> `Tanh`). Maps noise to $z$.
*   **Discriminator:** 5-layer MLP. Distinguishes real $z$ from fake $z$.
*   **Training:** Uses WGAN-GP (Wasserstein GAN with Gradient Penalty).
