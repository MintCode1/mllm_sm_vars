
#!/bin/bash
set -e

echo "🚀 Starting clean vLLM setup..."

# -------------------------------
# 1. Clean workspace
# -------------------------------
cd /workspace
rm -rf mllm_sm_vars

# -------------------------------
# 2. Clone repo
# -------------------------------
git clone https://github.com/MintCode1/mllm_sm_vars.git
cd mllm_sm_vars

# -------------------------------
# 3. System dependencies
# -------------------------------
echo "🔧 Installing system dependencies..."
apt update
apt install -y build-essential cmake ninja-build git

# -------------------------------
# 4. Python build dependencies
# -------------------------------
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install \
  setuptools \
  setuptools_scm \
  wheel \
  packaging \
  ninja \
  cmake \
  typing_extensions \
  filelock \
  sympy \
  numpy

# -------------------------------
# 5. Install PyTorch (CUDA)
# -------------------------------
echo "🔥 Installing PyTorch..."
pip uninstall -y torch torchvision torchaudio || true
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify torch
python - <<PY
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available()
PY

# -------------------------------
# 6. Patch XPU bug everywhere
# -------------------------------
echo "🩹 Patching torch.version.xpu..."
grep -rl "torch.version.xpu" vllm | xargs sed -i 's/torch.version.xpu/getattr(torch.version, "xpu", None)/g'

# -------------------------------
# 7. Fix pyproject.toml
# -------------------------------
echo "📄 Fixing pyproject.toml..."
sed -i 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/g' vllm/pyproject.toml
sed -i '/license-files/d' vllm/pyproject.toml

# -------------------------------
# 8. Install vLLM (PRECOMPILED — critical)
# -------------------------------
echo "📦 Installing vLLM..."
export VLLM_USE_PRECOMPILED=1
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

pip install -e vllm --no-build-isolation

# -------------------------------
# 9. Verify install
# -------------------------------
echo "✅ Verifying installation..."
python - <<PY
import vllm
print("vLLM import successful")
PY

echo "🎉 Setup complete. You're ready to run experiments."

