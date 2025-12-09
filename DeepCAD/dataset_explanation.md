# DeepCAD Dataset Explanation

The DeepCAD dataset pipeline transforms raw Onshape JSON data into a quantized vector sequence suitable for the Transformer model.

## 1. Raw Data Source (`cad_json`)
*   **Format:** JSON files derived from Onshape Public Public datasets.
*   **Content:** Hierarchical construction history (Feature Script).
    *   **Sequence:** Ordered list of features (e.g., `ExtrudeFeature`).
    *   **Entities:** Definitions for sketches, planes, and parameters.

## 2. Data Processing Pipeline
The conversion logic is primarily in `dataset/json2vec.py` and `cadlib/extrude.py`.

### A. Object Representation (`CADSequence`)
The raw JSON is parsed into a structured Python object `CADSequence`, which consists of a list of `Extrude` operations.
Each `Extrude` operation contains:
1.  **Sketch Profile (`Profile`):** A 2D loop of curves (`Line`, `Arc`, `Circle`).
2.  **Sketch Plane (`CoordSystem`):** The 3D plane defined by origin and orientation (`theta`, `phi`, `gamma`).
3.  **Extrusion Parameters:**
    *   **Operation:** Boolean type (New Body, Join, Cut, Intersect).
    *   **Extents:** Depth of extrusion (`extent_one`, `extent_two`).
    *   **Type:** One-sided, Symmetric, or Two-sided.

### B. Normalization
*   The entire 3D model is scaled to fit within a **unit cube** $[-1, 1]^3$.
*   This ensures consistent scale across all models for the neural network.

### C. Quantization (Numericalization)
*   Continuous floating-point values (coordinates, angles, lengths) are **quantized** into **256 discrete bins** (integers 0-255).
*   This converts the regression problem into a classification problem, which is often more stable for Transformers.

### D. Vectorization (`to_vector`)
The object hierarchy is flattened into a 1D sequence of command vectors.
*   **Structure:** `[Command_Token, Arg_1, Arg_2, ..., Arg_N]`
*   **Sequence Pattern:**
    ```
    [SOL, Curve_1, Curve_2, ..., Extrude_Cmd]  <-- Feature 1
    [SOL, Curve_1, Curve_2, ..., Extrude_Cmd]  <-- Feature 2
    ...
    [EOS]                                      <-- End of Sequence
    ```
*   **Command Tokens:** `Line`, `Arc`, `Circle`, `EOS`, `SOL` (Start of Loop), `Ext` (Extrude).

## 3. Final Dataset Format (`cad_vec`)
*   **File Format:** HDF5 (`.h5`).
*   **Data:** A single numpy array `vec` of shape `(N, 1 + N_ARGS)`.
    *   **Column 0:** Command Index.
    *   **Columns 1-End:** Quantized Arguments (0-255). Unused arguments for a specific command are padded with `-1`.

## 4. Training Augmentation
Implemented in `dataset/cad_dataset.py`.
*   **Feature Swapping:** Since the order of independent features (e.g., two separate holes) doesn't affect the final shape, the data loader randomly swaps the order of extrusion blocks during training to encourage order invariance where appropriate.
