# When the local machine is not enough: Google Colab through VS Code

Everything measured in this repository fits the local testbed (i7-14650HX, 24 GB RAM, RTX 5060 Laptop 8 GB): BitNet b1.58 2B4T
needs 1.1 GB of weights, runs on CPU and GPU, and the experiments are inference-time. These items would exceed it and should be
moved to Colab (or another GPU host) instead of being attempted locally:

| Work | Why it does not fit | Suggested Colab runtime |
|---|---|---|
| Ternary quantisation-aware training or distillation of models above ~300M parameters (BitDistill, Tequila, Sherry recipes) | optimiser state and activations exceed 8 GB VRAM and take many GPU-hours | A100 40/80 GB (Colab Pro+), 8-24 h in chunks with checkpoints on Drive |
| Post-training ternarisation of 14B+ models (CAT-Q reports 8-60 h on 8xA100) | RAM and VRAM | not practical on Colab; use a multi-GPU cloud box |
| Perplexity / lm-eval sweeps of 27B ternary models | 27B at ~1.75 bit is 5.9 GB of weights but the FP16 reference needs ~54 GB | A100 80 GB for the reference, local machine for the ternary side |
| Full-length long-context (>= 64K) evaluations | KV cache and time | A100 |

## Setup (VS Code Colab extension)

1. Install the **Google Colab** extension in VS Code, open a notebook, and choose **Select Kernel -> Colab -> New Colab Server**
   (sign in with the Google account; pick the GPU runtime in the dialog).
2. In the first cell install tooling with hash-checked or pinned packages (see `security.md` for the CA-incident rules):
   ```python
   !pip install --quiet "pip>=26.2.2" "certifi>=2026.10.19"
   !git clone https://github.com/Quiet-Intelligence/TernixEngine.git && cd TernixEngine && git checkout develop
   ```
3. Mount Drive for checkpoints (`from google.colab import drive; drive.mount('/content/drive')`) and write every experiment to a
   JSON under `/content/drive/MyDrive/ternix/` with the commit hash, as `docs/research/EXPERIMENTS.md` requires.
4. Colab sessions disconnect: make training loops resumable (save every N steps) and keep the experiment entry in
   `EXPERIMENTS.md` open until the raw output is committed.
5. Copy results back into the repository, commit them on a `research/*` branch, and merge into `develop` as usual.
