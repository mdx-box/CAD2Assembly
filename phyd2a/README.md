# PhyD2A

PhyD2A uses LLM-derived assembly knowledge to guide Genesis-based physical disassembly search. Valid disassembly sequences and trajectories are reversed to generate assembly sequences and paths.

## Installation

```bash
conda create -n phyd2a python=3.10 -y
conda activate phyd2a

# Install a PyTorch build suitable for the local CPU/CUDA platform first.
pip install torch
pip install -r requirements.txt
```

## Asset layout

```text
assets/
└── <assembly_name>/
    └── <assembly_id>/
        ├── <part_1>.obj
        ├── <part_2>.obj
        └── ...
```

## Mesh preprocessing

```bash
python assets/process_mesh.py \
  --source-dir "assets/<assembly_name>/<raw_assembly_id>" \
  --target-dir "assets/<assembly_name>/<processed_assembly_id>" \
  --subdivide \
  --max-edge 0.3
```

```bash
python assets/subdivide_batch.py \
  --source-dir "assets/<assembly_name>/<processed_assembly_id>" \
  --target-dir "assets/<assembly_name>/<subdivided_assembly_id>" \
  --max-edge 0.3 \
  --num-proc 8
```

The current source snapshot expects the project helper modules `assets/subdivide.py` and `utils/parallel.py`. Add these helpers before running the two preprocessing commands.

## Planning

```bash
python examples/phyd2a_benchmark.py \
  --assets-root "assets" \
  --dir "<assembly_name>" \
  --id "<subdivided_assembly_id>" \
  --base-part-id "<base_part_id>" \
  --approach full_phyd2a \
  --seed 1 \
  --no-vis \
  --no-save-video \
  --no-save-diagnostic-video \
  --path-max-time 300
```

Use `--llm-prior-file "<prior_json>"` to provide a custom semantic prior. Results are written under the selected assembly directory unless `--output-dir` is supplied.
