# CAD Project - Drawing to 3D CAD Model Conversion

This project focuses on converting 2D engineering drawings to 3D CAD models using deep learning approaches. It integrates two main systems: **DeepCAD** and **Drawing2CAD** to process, evaluate, and visualize CAD data.

## 📋 Overview

The project converts 2D SVG drawings into 3D STEP files and provides comprehensive visualization and evaluation tools to compare generated models against ground truth.

## 🗂️ Project Structure

### Core Scripts

- **`step_to_iso.py`** - Batch converter for STEP files to isometric PNG images
  - Renders STEP files in isometric view
  - Generates HTML galleries for visualization
  - Uses OpenCascade (pythonOCC) for 3D rendering

- **`step_to_iso_compare.py`** - Comparison tool with metrics
  - Side-by-side comparison of predicted vs. ground truth models
  - Computes accuracy metrics:
    - Command type accuracy (ACCcmd)
    - Parameter accuracy (ACCparam) 
    - Intersection over Union (IoU)
  - Generates comparative HTML galleries with color-coded highlighting

- **`step_to_iso_flask.py`** - Interactive Flask web application
  - Live comparison viewer
  - Real-time metric display
  - Interactive navigation through test results
  - Serves both predicted and ground truth visualizations

- **`compare_gallery.py`** - Enhanced gallery generation
- **`rename_svg.ipynb`** - SVG file preprocessing

### Notebooks

- **`cad_project.ipynb`** - Main project workflow and data processing
- **`flask.ipynb`** - Flask app development and testing
- **`tt.ipynb`** - Testing and experimentation

### Dependencies

- **DeepCAD/** - Deep learning model for CAD generation
- **Drawing2CAD/** - Drawing to CAD conversion framework

## 🔧 Key Features

### 1. STEP File Rendering
- Converts 3D STEP files to 2D isometric PNG images
- Offscreen rendering for batch processing
- Multiple view options (isometric, front, top, side)

### 2. Metric Computation
The comparison tools compute several accuracy metrics:

- **ACCcmd**: Command type accuracy (percentage of correct CAD commands)
- **ACCparam**: Parameter accuracy (percentage of correct command parameters within tolerance)
- **IoU**: Intersection over Union between predicted and ground truth

**Metric Parameters:**
- `TOL = 3` - Parameter tolerance threshold
- `CAD_EOS_IDX = 2` - End-of-sequence marker
- `PAD_PARAM = -1` - Padding marker

### 3. Visualization Tools

#### Static Gallery (`step_to_iso_compare.py`)
- Generates HTML galleries with side-by-side comparisons
- Color-coded highlighting:
  - 🟢 Green: Correct predictions
  - 🔴 Red: Incorrect predictions
- Displays metrics for each sample

#### Interactive Flask App (`step_to_iso_flask.py`)
- Web-based interface for browsing results
- Real-time metric computation
- Navigate through test samples
- View predicted models, ground truth, and SVG inputs

## 📁 Data Paths

### Input Directories
```
Drawing2CAD/proj_log/epoch_100_shivank/
├── evaluation_results/test/  # Predicted H5 files
└── test_results/             # Generated STEP files

DeepCAD/data2/
├── svg_raw/                  # Original SVG drawings
└── cad_vec/                  # Ground truth CAD vectors
```

### Output Directories
```
Drawing2CAD/proj_log/epoch_100_shivank/
├── output_pngs/             # Rendered PNG images
└── new_pngs/                # Comparison outputs
```

## 🚀 Usage

### Batch Rendering
```python
python step_to_iso.py
```
Converts all STEP files in the input directory to isometric PNG images and generates an HTML gallery.

### Generate Comparison Gallery
```python
python step_to_iso_compare.py
```
Creates a comprehensive comparison gallery with metrics for all test samples.

### Launch Interactive Viewer
```python
python step_to_iso_flask.py
```
Then navigate to `http://localhost:5000` in your browser.

## 🛠️ Technical Stack

- **Python 3.x**
- **pythonOCC** - OpenCascade bindings for 3D CAD manipulation
- **Flask** - Web framework for interactive visualization
- **h5py** - HDF5 file handling for CAD vectors
- **NumPy** - Numerical computations
- **OpenCascade** - 3D CAD kernel

## 📊 Data Format

### H5 Files
- **`GENERATED_DATASET_NAME = "out_vec"`** - Predicted CAD vectors
- **`TRUTH_DATASET_NAME = "vec"`** - Ground truth CAD vectors

### CAD Vector Structure
Each vector contains:
- Command types (line, arc, circle, etc.)
- Parameters (coordinates, radii, angles)
- Padding markers

## 🎯 Workflow

1. **Input**: SVG drawing files
2. **Processing**: DeepCAD/Drawing2CAD models generate 3D STEP files
3. **Rendering**: Convert STEP to isometric PNG views
4. **Evaluation**: Compare predictions against ground truth
5. **Visualization**: View results in HTML galleries or Flask app

## 📝 Notes

- The project uses offscreen rendering for batch processing efficiency
- Anti-aliasing is disabled for faster rendering
- Color coding helps quickly identify prediction accuracy
- Supports multiple CAD primitives and complex geometries

## 🔍 Evaluation Metrics Explained

- **Command Accuracy**: Measures if the right CAD operation was predicted (e.g., line vs. arc)
- **Parameter Accuracy**: Checks if command parameters are within acceptable tolerance
- **IoU**: Measures geometric overlap between predicted and ground truth shapes

## 🐛 Known Issues

- Some STEP files may produce empty or invalid shapes (logged during processing)
- Large batch processing may require significant memory

## 📚 Related Directories

- **`Readmes/cad_project/`** - This documentation
- **`DeepCAD/`** - Core deep learning model
- **`Drawing2CAD/`** - Conversion framework
