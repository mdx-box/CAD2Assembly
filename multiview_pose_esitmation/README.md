# MultiView

MultiView is a multi-view RGB-D object pose tracking and evaluation pipeline. It supports `left`, `right`, and `both` camera views, optional two-view fusion, pose visualization, and BOP-style metrics including ADD, ADD-S, VSD, MSSD, MSPD, and AR.

## Installation

The recommended environment is Linux with an NVIDIA GPU, CUDA, and Python 3.9.

```bash
conda create -n multiview python=3.9 -y
conda activate multiview

# Install a PyTorch build compatible with your CUDA version first.
pip install torch torchvision

pip install numpy pandas scipy opencv-python trimesh tqdm openpyxl pytorch-lightning

# Obtain and install SAM2 in this module directory.
git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2
```

Before running, also make sure that:

- `nvdiffrast` is installed and can be imported as `nvdiffrast.torch`.
- `bop_toolkit_lib` is available in the Python environment for VSD, MSSD, and MSPD evaluation.
- The FoundationPose source package is available as `foundationpose`, because `estimater.py` reuses its geometry and prediction modules.
- `estimater.py` is in the project root and exports `MultiView`, `ScorePredictor`, and `PoseRefinePredictor`.
- The SAM2 checkpoint is downloaded separately to `sam2/checkpoints/sam2.1_hiera_large.pt`, or another checkpoint is provided with `--sam2_checkpoint`. Model weights are excluded from Git because the default checkpoint is larger than GitHub's normal file-size limit.

## Dataset layout

The dataset root should contain the two evaluation sequences and the object model:

```text
<dataset_root>/
├── evaluation/
│   ├── <object_name>_left/
│   └── <object_name>_right/
└── models/
    └── <object_name>/
        └── <object_mesh>.obj
```

Each sequence should contain RGB images, depth data, masks, camera intrinsics, ground-truth poses, and per-frame metadata required by the evaluator.

## Run

Run the evaluation from the project root:

```bash
python multiview_bop_eval.py \
  --dataset_root "<dataset_root>" \
  --object_name "<object_name>" \
  --obj_mesh "<object_mesh_path>" \
  --view both \
  --calibrate_axis_from_first_frame \
  --axis_calibration_view right \
  --axis_calibration_output "<axis_calibration_output_path>" \
  --save_video \
  --save_images \
  --no_display
```

The command processes both views, calibrates the frozen prediction branch from the first frame, saves videos and frame images, and runs without an interactive display. By default, results are written under `results/real_scene_multiview_tracking/`.

## Demo videos

### Two-view pose-fusion example 1

![Two-view pose-fusion example for the external plane](Video/ext_plane_two_view_fusion.webp)

### Two-view pose-fusion example 2

![Two-view pose-fusion example for the payload](Video/play_load_two_view_fusion.webp)

The animated previews play directly on the GitHub page. The original MP4 files are stored in the same directory.

Use `python multiview_bop_eval.py --help` to see all available options.
