# RunPod LoRA Trainer - Windows

One-click LoRA training on RunPod cloud GPUs via SSH from your local Windows machine.

## Prerequisites

- Python 3.10+
- [Git for Windows](https://gitforwindows.org/) (provides `ssh.exe` and `scp.exe`)

## Setup

### 1. SSH Key

Open PowerShell or Git Bash and generate an ed25519 key pair:

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519"
```

Then add the **public** key to RunPod:

1. Copy the key: `type $env:USERPROFILE\.ssh\id_ed25519.pub`
2. Go to [RunPod SSH Settings](https://www.runpod.io/console/user/settings) -> SSH Public Keys
3. Paste the key and save

### 2. RunPod API Key

1. Go to [RunPod API Keys](https://www.runpod.io/console/user/settings) -> API Keys
2. Create a new key and copy it
3. Set it as an environment variable:

**PowerShell (current session):**
```powershell
$env:RUNPOD_API_KEY = "your_key_here"
```

**Permanent (System Environment Variables):**
1. Search "Environment Variables" in Windows
2. Add a new User variable: `RUNPOD_API_KEY` = `your_key_here`

### 3. Configure Paths

Edit `config.yaml` and fill in the `LOCAL_*` paths to match your system:

```yaml
LOCAL_CONFIG1_PATH: D:\path\to\your\train_lora.yaml
LOCAL_DATASET_DIR: D:\path\to\your\dataset
LOCAL_DOWNLOAD_DIR: D:\path\to\your\output
LOCAL_RESUME_SOURCE_DIR:   # leave empty unless resuming from a previous run
```

Also adjust GPU type, cloud type, and training parameters as needed.

### 4. SSH/SCP Paths

If Git for Windows is not installed at the default location, edit the `SSH_EXE_PATH` and `SCP_EXE_PATH` variables in `TrainV4.py`:

```python
SSH_EXE_PATH = r"C:\Program Files\Git\usr\bin\ssh.exe"
SCP_EXE_PATH = r"C:\Program Files\Git\usr\bin\scp.exe"
```

### 5. HF Token

Edit `atw.txt` and set your Hugging Face token:

```
HF_TOKEN=hf_YourTokenHere
```

Get one at https://huggingface.co/settings/tokens if you don't have one.

## Usage

Double-click `train_runpod.bat` or run it from a terminal:

```cmd
train_runpod.bat
```

This will:
1. Create a Python venv and install dependencies (first run only)
2. Spin up a RunPod GPU pod
3. Install ai-toolkit on the pod
4. Upload your dataset, config, and HF token
5. Run training while continuously syncing checkpoints back to your machine
6. Download remaining output files
7. Terminate the pod on success (prompts on error)

## Monitoring

The script attempts to open a PowerShell window that tails the remote training log. If that doesn't work, you can manually SSH into the pod and run `tail -f /workspace/training_run.log`.
