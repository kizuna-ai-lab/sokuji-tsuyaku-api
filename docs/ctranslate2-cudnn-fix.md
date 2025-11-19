# CTranslate2 cuDNN Loading Issue - Root Cause and Fix

## Problem Summary

When running the application under Uvicorn, CTranslate2 (used by faster-whisper) crashes during CUDA inference with:
```
Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
Invalid handle. Cannot load symbol cudnnCreateConvolutionDescriptor
IOT instruction (core dumped)
```

## Root Cause Analysis

### 1. Incomplete CTranslate2 Package
The CTranslate2 wheel package bundles cuDNN 9.1.0 libraries but is **missing** `libcudnn_cnn.so.9.1.0`. It only includes the main cuDNN library:
```bash
$ ls ctranslate2.libs/
libcudnn-74a4c495.so.9.1.0  # Only this file is bundled
# libcudnn_cnn.so.9.1.0 is MISSING!
```

### 2. RPATH Takes Priority Over LD_LIBRARY_PATH
CTranslate2's compiled binary has a hardcoded RPATH:
```bash
$ readelf -d _ext.cpython-312-x86_64-linux-gnu.so | grep RPATH
RPATH: [$ORIGIN/../ctranslate2.libs]
```

Linux dynamic linker search order:
1. **RPATH** (hardcoded in binary) ← Highest priority
2. LD_LIBRARY_PATH (environment variable)
3. System default paths

Since RPATH points to `ctranslate2.libs/` but the file is missing there, the loader fails even though we:
- Copied the file from system cuDNN 9.15.0 to `ctranslate2.libs/`
- Set `LD_LIBRARY_PATH` to point to `ctranslate2.libs/`

### 3. Uvicorn-Specific Failure
The issue only manifests in Uvicorn's environment:
- ✅ Standalone Python scripts work (even without ctypes preload)
- ❌ Uvicorn with LD_LIBRARY_PATH fails
- ✅ Uvicorn with ctypes preload works

The exact reason why Uvicorn's async/ASGI environment causes LD_LIBRARY_PATH fallback to fail is unclear, but likely related to how libraries are loaded across different event loop contexts and the internal httpx.AsyncClient ASGI calls.

## Solution

Preload `libcudnn_cnn.so.9.1.0` into the global symbol table using `ctypes.CDLL` with `RTLD_GLOBAL` flag **before** any library tries to load it:

```python
import ctypes
import pathlib

cudnn_file = pathlib.Path(".../.venv/lib/python3.12/site-packages/ctranslate2.libs/libcudnn_cnn.so.9.1.0")
if cudnn_file.exists():
    ctypes.CDLL(str(cudnn_file), mode=ctypes.RTLD_GLOBAL)
```

### Why This Works

1. **Preloading**: Loads the library immediately during app initialization
2. **RTLD_GLOBAL**: Places all symbols in the global symbol table, making them visible to all subsequently loaded libraries
3. **Priority**: When CTranslate2 later tries to load cuDNN symbols, they're already available globally

## Setup Requirements

### 1. Copy Missing cuDNN Library
```bash
# The system has cuDNN 9.15.0, which is compatible with 9.1.0 API
cp /lib/x86_64-linux-gnu/libcudnn_cnn.so.9.15.0 \
   .venv/lib/python3.12/site-packages/ctranslate2.libs/libcudnn_cnn.so.9.1.0
```

### 2. Upgrade CTranslate2
```bash
# CTranslate2 4.6.1 has better cuDNN compatibility
uv pip install --upgrade "ctranslate2>=4.6.0"
```

### 3. Add PyTorch Dependency
CTranslate2 also needs PyTorch for NLLB models:
```toml
# pyproject.toml
dependencies = [
    "ctranslate2>=4.6.0",
    "torch>=2.0.0",
    ...
]
```

## Verification

Check that the fix is working:
```bash
# Start server (ctypes preload happens automatically in create_app)
uvicorn speaches.main:create_app --factory

# Check logs for confirmation
# Expected: "Preloaded cuDNN CNN library from ..."
```

## Alternative Solutions (Not Used)

### ❌ LD_LIBRARY_PATH Only
Doesn't work because RPATH has higher priority.

### ❌ Symbolic Links
Failed in Uvicorn environment for unknown reasons (possibly dlopen behavior differences).

### ❌ Downgrade cuDNN to 9.1.0
Would break other applications requiring cuDNN 9.15.0 (like PyTorch).

### ❌ Use CPU Mode
```bash
WHISPER__INFERENCE_DEVICE=cpu
```
Works but sacrifices CUDA performance.

## References

- CTranslate2 GitHub: https://github.com/OpenNMT/CTranslate2
- Related issue: https://github.com/OpenNMT/CTranslate2/issues/1234 (example)
- Linux dynamic linking: `man ld.so`
