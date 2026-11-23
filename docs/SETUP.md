# rag-agent-gpu: Windows setup guide

Native Windows 10/11, PowerShell, Anaconda at `C:\Users\<USER>\anaconda3`. No WSL2.

## 1. Audit
```powershell
nvidia-smi                      # GPU, driver, max CUDA version
where.exe conda                 # empty if conda is not on PATH (fine; setup.ps1 uses the full path)
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" info
Get-Command cl.exe              # not required, all packages are prebuilt wheels
```

## 2. Install
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1            # add -Recreate to rebuild
```
Creates the env (conda-forge, Python 3.11), installs CUDA 12.8 PyTorch wheels, then `requirements.txt`,
runs `pip check`, registers the Jupyter kernel, and creates `.env`.

Notes
- `faiss-cpu` is used: `faiss-gpu` is not available on native Windows. Embeddings still run on GPU through
  sentence-transformers / PyTorch; only the FAISS index search is on CPU.
- `numpy<2.0` is pinned to avoid binary ABI crashes with compiled extensions.
- Ollama is not a Python dependency: install the Ollama Windows app to serve models (`ollama pull qwen3-coder:30b`).

## 3. Verify
```powershell
& "$env:USERPROFILE\anaconda3\envs\rag-agent-gpu\python.exe" verify_env.py
```

## 4. PyCharm
1. **Interpreter:** File > Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter >
   Conda Environment > *Use existing environment*. Conda executable:
   `C:\Users\<USER>\anaconda3\condabin\conda.bat`. Select `rag-agent-gpu`
   (`...\anaconda3\envs\rag-agent-gpu\python.exe`).
2. **Terminal:** File > Settings > Tools > Terminal > Shell path `powershell.exe`, tick *Activate virtualenv*.
   If conda isn't hooked, run once: `& "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"; conda activate rag-agent-gpu`.
3. **Run/Debug configs:** install the *EnvFile* plugin, then Run > Edit Configurations > EnvFile > enable and add `.env`.
   Variables set in `.env`: `HF_HOME=C:\hf_cache` (short path, avoids MAX_PATH), `CUDA_VISIBLE_DEVICES=0`,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `python-dotenv` also loads it in code.
