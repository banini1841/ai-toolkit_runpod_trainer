# RunPod LoRA Trainer - Linux

One-click LoRA training on RunPod cloud GPUs via SSH from your local Linux machine.

## Setup

### 1. SSH Key

Generate an ed25519 key pair (press Enter for defaults, no passphrase needed):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
```

Then add the **public** key to RunPod:

1. Copy the key: `cat ~/.ssh/id_ed25519.pub`
2. Go to [RunPod SSH Settings](https://www.runpod.io/console/user/settings) -> SSH Public Keys
3. Paste the key and save

### 2. RunPod API Key

1. Go to [RunPod API Keys](https://www.runpod.io/console/user/settings) -> API Keys
2. Create a new key and copy it
3. Add it to your shell config:

**Fish** (`~/.config/fish/config.fish`):
```fish
set -gx RUNPOD_API_KEY "your_key_here"
```

**Bash** (`~/.bashrc`):
```bash
export RUNPOD_API_KEY="your_key_here"
```

Then reload your shell or run the export command directly.

### 3. Configure Paths

Edit `config.yaml` and fill in the `LOCAL_*` paths to match your system:

```yaml
LOCAL_CONFIG1_PATH: /path/to/your/train_lora_config.yaml
LOCAL_DATASET_DIR: /path/to/your/dataset-folder
LOCAL_DOWNLOAD_DIR: /path/to/your/output
LOCAL_RESUME_SOURCE_DIR:   # leave empty unless resuming from a previous run
```

Also adjust GPU type, cloud type, and training parameters as needed.

### 4. HF Token

Edit `atw.txt` and set your Hugging Face token:

```
HF_TOKEN=hf_YourTokenHere
```

Get one at https://huggingface.co/settings/tokens if you don't have one.

## Usage

```bash
./train_runpod.sh
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

If you run inside tmux, a monitoring pane opens automatically. Otherwise the script prints an SSH command you can run in a second terminal to tail the training log.
