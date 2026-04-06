import tempfile
import subprocess
import os
import shlex
import time
import json
import runpod
from runpod.error import QueryError
import shutil
import glob
import threading
import hashlib
import yaml
import base64


print_lock = threading.Lock()

# --- Resolve script directory so config/atw.txt work from any cwd ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Read Config ---
with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# --- Configuration --------------------------------------------------------------------------------------------------------------------------------------

# ++ Pod Creation Parameters (FILL THESE IN ACCURATELY!) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
POD_NAME = None # Optional name for the pod
GPU_TYPE_ID = cfg["GPU_TYPE_ID"] # Crucial: Find exact ID via RunPod API/UI e.g. "NVIDIA RTX A6000"
GPU_COUNT = 1
# Choose ONE: Template ID or Image Name
TEMPLATE_ID = None # e.g "YOUR_TEMPLATE_ID_HERE" - Set to None if using imageName
IMAGE_NAME = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" # Set to None if using templateId
CONTAINER_DISK_GB = int(cfg["CONTAINER_DISK_GB"]) # Temporary disk space
VOLUME_DISK_GB = 0 # Permanent disk space for /workspace (Set 0 if using existing VOLUME_ID)
# Optional: Network Volume (If using existing - VOLUME_DISK_GB might be ignored/not needed)
VOLUME_ID = "" # Set to YOUR_VOLUME_ID if using existing network volume, leave "" otherwise
VOLUME_MOUNT_PATH = "/workspace" # Usually /workspace for network volume or default pod storage
# Cloud Type
CLOUD_TYPE = cfg["CLOUD_TYPE"] # Use "COMMUNITY" or "SECURE"
# Data Center ID (Optional, find via API/UI if needed)
DATA_CENTER_ID = None # e.g. "US-EAST-1". Set to None or omit if not needed
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# --- Local Paths (FILL THESE IN!) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
LOCAL_CONFIG1_PATH = cfg["LOCAL_CONFIG1_PATH"] # Ensure these filenames are correct
LOCAL_DATASET_DIR = cfg["LOCAL_DATASET_DIR"] # Path to your dataset folder
LOCAL_DOWNLOAD_DIR = cfg["LOCAL_DOWNLOAD_DIR"] # Where to save the final output locally (subfolder recommended), it will be saved in LOCAL_DOWNLOAD_DIR\output
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++ Resume Training +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Optional: Specify a local directory containing previous output/checkpoints to upload for resuming
LOCAL_RESUME_SOURCE_DIR = cfg["LOCAL_RESUME_SOURCE_DIR"] # SET THIS PATH to resume, or leave ""/None to skip
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

max_pod_wait_time = 1000

# --- More Local Paths & Settings (Verify these) -------------------------------------------------------------------
LOCAL_ENV_FILE_PATH = os.path.join(SCRIPT_DIR, "atw.txt") # Path to your env file to upload

SSH_KEY_PATH = os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519") # Full path to your PRIVATE SSH key
SSH_EXE_PATH = r"C:\Program Files\Git\usr\bin\ssh.exe" # Full path to ssh.exe (from Git for Windows)
SCP_EXE_PATH = r"C:\Program Files\Git\usr\bin\scp.exe" # Full path to scp.exe (from Git for Windows)

# Dynamically get config filenames from local paths
LOCAL_CONFIG2_PATH = "" # Ensure these filenames are correct
# -----------------------------------------------------------------------------------------------------------------


# --- Script Internal Variables (Should be automatically set or configured below) -------------------------------------------
# SSH Details (Filled by script)
POD_IP_OR_HOSTNAME = "" # Filled by script
POD_USER = "root"       # Default for RunPod standard templates
POD_PORT = ""           # Filled by script
POD_ID = ""             # Filled by script

# Remote Paths (Modify if necessary)
REMOTE_WORKSPACE_DIR = VOLUME_MOUNT_PATH # Use volume mount path from config
REMOTE_TOOLKIT_DIR = os.path.join(REMOTE_WORKSPACE_DIR, "ai-toolkit").replace("\\", "/")
REMOTE_ENV_FILE_PATH = os.path.join(REMOTE_TOOLKIT_DIR, ".env").replace("\\", "/")
REMOTE_DATASET_DIR = os.path.join(REMOTE_WORKSPACE_DIR, "dataset").replace("\\", "/")
REMOTE_CONFIG_DIR = os.path.join(REMOTE_TOOLKIT_DIR, "config").replace("\\", "/")
REMOTE_OUTPUT_DIR = os.path.join(REMOTE_TOOLKIT_DIR, "output").replace("\\", "/")

LOG_FILE = os.path.join(REMOTE_WORKSPACE_DIR, "training_run.log").replace("\\", "/") # Log in workspace
REMOTE_PID_FILE = os.path.join(REMOTE_WORKSPACE_DIR, "train.pid").replace("\\", "/")
REMOTE_EXIT_FILE = os.path.join(REMOTE_WORKSPACE_DIR, "train_exit").replace("\\", "/")
REMOTE_TRAIN_SCRIPT = os.path.join(REMOTE_WORKSPACE_DIR, "run_train.sh").replace("\\", "/")

# Check existence of local config files
config_1_exists = os.path.isfile(LOCAL_CONFIG1_PATH)
config_2_exists = os.path.isfile(LOCAL_CONFIG2_PATH)
if not config_1_exists:
    print(f"ERROR: Local config file not found: {LOCAL_CONFIG1_PATH}")
    exit(1)
CONFIG_FILE_1 = os.path.join("config", os.path.basename(LOCAL_CONFIG1_PATH)).replace("\\", "/")
if config_2_exists:
    CONFIG_FILE_2 = os.path.join("config", os.path.basename(LOCAL_CONFIG2_PATH)).replace("\\", "/")
else:
    CONFIG_FILE_2 = None # Set to None if file doesn't exist locally

# --- End Configuration --------------------------------------------------------------------------------------------------------------



# --- API Key Check -----------------------------------------------------------------------------------------------------------
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("ERROR: RUNPOD_API_KEY environment variable not set.")
    print("Please set it before running the script (e.g., $env:RUNPOD_API_KEY='your_key' or set RUNPOD_API_KEY=your_key)")
    exit(1)
try:
    runpod.api_key = RUNPOD_API_KEY
    print("Testing RunPod API key...")
    runpod.get_gpus() # Simple read call to test authentication
    print("RunPod API key appears valid.")
except QueryError as err:
     print(f"ERROR: RunPod API QueryError during initial check: {err}")
     print("Please ensure your RUNPOD_API_KEY environment variable is set correctly and has permissions.")
     exit(1)
except Exception as e:
    print(f"ERROR: Unexpected error initializing RunPod SDK: {e}")
    exit(1)
# --- End API Key Check --------------------------------------------------------------------------------------------------------



# --- Helper Function Definitions -----------------------------------------------------------------------------------------------------------------------------------
# Helper function to build user@host string (will be updated)
# USER_HOST = ""

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)


def run_command(cmd_string, check=True, capture=True, use_shell=True):
    """Runs a command string using subprocess, default shell=True."""
    print(f"\n---> Running Command ({'shell' if use_shell else 'direct'}): {cmd_string}")
    try:
        # Use shell=True as confirmed working for user's SSH key setup
        result = subprocess.run(cmd_string, shell=use_shell, capture_output=capture, text=True, check=check, encoding='utf-8', errors='replace')
        if capture:
            stdout = result.stdout.strip() if result.stdout else ""
            stderr = result.stderr.strip() if result.stderr else ""
            if stdout: print("STDOUT:", stdout)
            if stderr: print("STDERR:", stderr)
        # check=True will raise CalledProcessError on non-zero exit
        return result.returncode
    except FileNotFoundError:
        # Extract command name carefully for error message
        cmd_name = shlex.split(cmd_string)[0] if isinstance(cmd_string, str) else "Unknown"
        print(f"ERROR: Command not found (via shell): Check if '{cmd_name}' is in PATH or if the shell works.")
        raise # Re-raise after printing helpful message
    except subprocess.CalledProcessError as e:
         print(f"ERROR Subprocess: Command failed with code {e.returncode}")
         raise # Re-raise the error to be caught by the main try/except block
    except Exception as e:
        print(f"ERROR: An unexpected error occurred running command: {e}")
        raise # Re-raise the error


def run_remote_command(remote_cmd, check=True):
    """Runs a command remotely via SSH using shell=True."""
    global USER_HOST
    if not POD_IP_OR_HOSTNAME or not POD_PORT:
         print("ERROR: Pod IP/Port not set for remote command.")
         raise RuntimeError("Pod IP/Port not set before remote command execution.")
    USER_HOST = f"{POD_USER}@{POD_IP_OR_HOSTNAME}"
    known_hosts_file = os.path.join(tempfile.gettempdir(), "ssh_known_hosts")
    # Added common SSH options for automation
    cmd_str = f'"{SSH_EXE_PATH}" -q -p {POD_PORT} -o ConnectTimeout=15 -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile={known_hosts_file} -i "{SSH_KEY_PATH}" "{USER_HOST}" "{remote_cmd}"'
    print(f"\n---> Running Remote Command: {remote_cmd} on {USER_HOST}:{POD_PORT}")
    # Rely on run_command to raise exception if check=True and command fails
    return run_command(cmd_str, check=check, capture=True, use_shell=True)


def transfer_file_or_dir(local_path, remote_target_path, is_directory=False, upload=True, check=False): # Added 'check=True' parameter
    """Uploads or downloads using SCP using shell=True, allowing check control."""
    global USER_HOST
    if not POD_IP_OR_HOSTNAME or not POD_PORT:
        print("ERROR: Pod IP/Port not set for file transfer.")
        raise RuntimeError("Pod IP/Port not set before file transfer.")
    USER_HOST = f"{POD_USER}@{POD_IP_OR_HOSTNAME}"
    known_hosts_file = os.path.join(tempfile.gettempdir(), "ssh_known_hosts")
    # Use -q for quiet, remove if debugging needed
    flags = f"-P {POD_PORT} -q -o StrictHostKeyChecking=no -o UserKnownHostsFile={known_hosts_file} -i \"{SSH_KEY_PATH}\""
    if is_directory:
        flags += " -r"

    quoted_local_path = shlex.quote(local_path)
    quoted_remote_target = shlex.quote(remote_target_path) # Quote remote path too
    remote_spec = f"{USER_HOST}:{quoted_remote_target}" # Use quoted remote path

    if upload:
        source = quoted_local_path
        destination = remote_spec
    else: # Download
        source = remote_spec
        destination = quoted_local_path

    # Construct command string using full path to scp.exe
    cmd_str = f'"{SCP_EXE_PATH}" {flags} {source} {destination}'

    action = "Uploading" if upload else "Downloading"
    # Using log_message (if defined) or print for status
    status_message = f"---> {action} {'directory' if is_directory else 'file'}: {os.path.basename(local_path if upload else remote_target_path)} using port {POD_PORT}"
    # Check if log_message exists before calling it, otherwise use print
    if 'log_message' in globals() and callable(log_message):
        log_message(status_message)
    else:
        print(status_message)

    # Pass the 'check' variable received by this function down to run_command
    return run_command(cmd_str, check=check, capture=True, use_shell=True)


def terminate_pod(pod_id_to_terminate):
     """Terminates (removes) the pod using the RunPod Python SDK."""
     if not pod_id_to_terminate:
         print("ERROR: No Pod ID provided for termination.")
         return -1 # Indicate failure, but don't raise exception from here
     print(f"\n---> Terminating (Removing) pod via SDK: {pod_id_to_terminate}")
     try:
         # Attempt termination directly
         termination_result = runpod.terminate_pod(pod_id_to_terminate)
         print("Termination API call initiated. Result:", termination_result)

         # Wait and verify termination status
         max_check_attempts = 12 ; check_delay = 10
         print(f"Waiting up to {max_check_attempts * check_delay}s for termination confirmation...")
         for attempt in range(max_check_attempts):
             time.sleep(check_delay)
             print(f"Verifying termination (Attempt {attempt + 1}/{max_check_attempts})...")
             try:
                 pod_info_after = runpod.get_pod(pod_id_to_terminate)
                 if not pod_info_after:
                      print(f"Pod {pod_id_to_terminate} successfully terminated (no longer exists).")
                      return 0
                 else:
                      current_status = pod_info_after.get('desiredStatus', 'UNKNOWN')
                      print(f"Pod {pod_id_to_terminate} still exists. Status: {current_status}")
                      if current_status in ['TERMINATED']:
                           print("Pod status confirmed as TERMINATED.")
                           return 0
             except QueryError as err:
                  if "graphql: pod not found" in str(err).lower():
                       print(f"Pod {pod_id_to_terminate} successfully terminated (no longer found via API).")
                       return 0
                  else:
                      print(f"Warning: API Error checking pod status after termination: {err}")

         print(f"Warning: Pod {pod_id_to_terminate} termination not confirmed after {max_check_attempts * check_delay}s.")
         return -1

     except QueryError as err:
         if "graphql: pod not found" in str(err).lower():
              print(f"Info: Pod {pod_id_to_terminate} not found via API (already terminated?).")
              return 0
         print(f"ERROR: RunPod API QueryError during termination attempt: {err}")
         if hasattr(err, 'query'): print("Query details:", err.query)
         return -1
     except Exception as e:
         print(f"ERROR: An unexpected error occurred during termination: {e}")
         return -1


def wait_for_ssh(max_wait_time=360, delay=20):
    """Waits for SSH connection to the pod to become available."""
    print(f"\n--- Waiting for SSH connection to {POD_IP_OR_HOSTNAME}:{POD_PORT} (max {max_wait_time}s) ---")
    global USER_HOST
    if not POD_IP_OR_HOSTNAME or not POD_PORT:
         print("ERROR: Cannot wait for SSH without Pod IP and Port.")
         return False
    USER_HOST = f"{POD_USER}@{POD_IP_OR_HOSTNAME}"
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        ssh_test_cmd = "echo SSH is Ready"
        # Use check=False to evaluate return code manually
        rc = run_remote_command(ssh_test_cmd, check=False)
        if rc == 0:
            print("SSH connection successful!")
            return True
        print(f"Connection attempt failed (Code: {rc}), waiting {delay}s...")
        time.sleep(delay)
    print(f"ERROR: SSH connection readiness check timed out after {max_wait_time}s.")
    return False


def launch_monitoring_window():
    """(Optional) Attempts to launch a new terminal window for live log monitoring."""
    # This function remains experimental due to quoting/platform issues
    if not all([POD_IP_OR_HOSTNAME, POD_PORT, SSH_KEY_PATH, LOG_FILE, USER_HOST]):
        print("WARNING: Missing details required to launch monitoring window.")
        return
    print("\n--- Attempting to launch monitoring window ---")
    try:
        remote_command_for_ssh = f"tail -f {LOG_FILE}"
        # Using PowerShell - command quoting is tricky. This attempts to pass it correctly.
        # Encapsulate the ssh command and its arguments in single quotes for PowerShell -Command
        inner_ssh_cmd = f"& '{SSH_EXE_PATH}' -p {POD_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile='{os.path.join(tempfile.gettempdir(), 'ssh_known_hosts')}' -i '{SSH_KEY_PATH}' '{USER_HOST}' '{remote_command_for_ssh}'"
        # Escape inner single quotes for the outer double quotes if necessary, or use different quoting.
        # Using """ for the outer command string to allow internal quotes more easily:
        full_start_cmd = f"""start powershell -NoExit -Command "{inner_ssh_cmd.replace('"', '""')}" """ # Replace potential internal double quotes for PS

        print(f"Executing: {full_start_cmd}")
        subprocess.Popen(full_start_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        print("Monitoring window launch command sent (Check new PowerShell window).")
    except Exception as e:
        print(f"ERROR: Failed to launch monitoring window: {e}")


# --- Sync state (defined before continuous_output_sync which uses it) ---
sync_thread = None
sync_stop_event = threading.Event()


# --- Continuous Output Sync ---
def continuous_output_sync(poll_interval=60, optimizer_check_interval=2, optimizer_max_wait=20):
    """
    Continuously downloads new safetensors files from pod output.
    Downloads optimizer safely: waits until optimizer file stops changing content (hash) before downloading.
    Keeps structure: output/<lora_name>/files locally.
    """

    print("\n--- Continuous output sync started ---")

    downloaded_files = set()
    optimizer_hash_state = {}  # folder -> last downloaded hash

    known_hosts_file = os.path.join(tempfile.gettempdir(), "ssh_known_hosts")
    user_host = f"{POD_USER}@{POD_IP_OR_HOSTNAME}"

    while not sync_stop_event.is_set():
        try:
            # Find all checkpoints
            find_cmd = f"find {REMOTE_OUTPUT_DIR} -type f -name '*.safetensors'"
            ssh_cmd = (
                f'"{SSH_EXE_PATH}" -q -p {POD_PORT} '
                f'-o StrictHostKeyChecking=no '
                f'-o UserKnownHostsFile={known_hosts_file} '
                f'-i "{SSH_KEY_PATH}" '
                f'"{user_host}" "{find_cmd}"'
            )

            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True
            )

            remote_files = result.stdout.strip().splitlines()

            for remote_file in remote_files:
                if not remote_file or remote_file in downloaded_files:
                    continue

                # --- download checkpoint ---
                rel_path = os.path.relpath(remote_file, REMOTE_OUTPUT_DIR).replace("\\", "/")
                local_path = os.path.join(
                    LOCAL_DOWNLOAD_DIR,
                    os.path.basename(REMOTE_OUTPUT_DIR),
                    rel_path
                )
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                safe_print(f"New checkpoint detected → {rel_path}")
                transfer_file_or_dir(local_path, remote_file, upload=False, check=True)
                downloaded_files.add(remote_file)

                # --- check optimizer in same folder ---
                remote_dir = os.path.dirname(remote_file)
                optimizer_remote = f"{remote_dir}/optimizer.pt"
                optimizer_local = os.path.join(os.path.dirname(local_path), "optimizer.pt")

                last_hash = None
                waited = 0

                while waited < optimizer_max_wait:
                    current_hash = sha256_of_remote_file(optimizer_remote)

                    if not current_hash:
                        safe_print("Optimizer file not found yet, retrying...")
                        time.sleep(optimizer_check_interval)
                        waited += optimizer_check_interval
                        continue

                    if last_hash == current_hash:
                        # Hash stable -> download if changed since last download
                        if optimizer_hash_state.get(remote_dir) != current_hash:
                            safe_print("Optimizer content stable -> downloading")
                            transfer_file_or_dir(
                                optimizer_local,
                                optimizer_remote,
                                upload=False,
                                check=True
                            )
                            optimizer_hash_state[remote_dir] = current_hash
                        break
                    else:
                        # Hash changed -> wait and re-check
                        last_hash = current_hash
                        time.sleep(optimizer_check_interval)
                        waited += optimizer_check_interval
                else:
                    safe_print("Warning: optimizer file did not stabilize in time, skipping this round.")

        except Exception as e:
            safe_print(f"[Sync Warning] {e}")

        sync_stop_event.wait(poll_interval)

    safe_print("--- Continuous output sync stopped ---")


def sha256_of_remote_file(remote_file):
    """Get SHA256 of remote file via SSH statelessly."""
    stat_cmd = f'sha256sum {remote_file} 2>/dev/null | awk \'{{print $1}}\''
    ssh_stat_cmd = (
        f'"{SSH_EXE_PATH}" -q -p {POD_PORT} '
        f'-o StrictHostKeyChecking=no '
        f'-o UserKnownHostsFile={os.path.join(tempfile.gettempdir(), "ssh_known_hosts")} '
        f'-i "{SSH_KEY_PATH}" '
        f'"{POD_USER}@{POD_IP_OR_HOSTNAME}" "{stat_cmd}"'
    )
    result_stat = subprocess.run(ssh_stat_cmd, shell=True, capture_output=True, text=True)
    return result_stat.stdout.strip()


def sha256_of_local_file(local_file):
    """Compute SHA256 hash of local file."""
    h = hashlib.sha256()
    with open(local_file, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_remote_command_output(remote_cmd):
    """Runs a command remotely via SSH and returns (returncode, stdout).
    Does not raise on failure - returns (-1, '') instead."""
    if not POD_IP_OR_HOSTNAME or not POD_PORT:
        return -1, ""
    USER_HOST = f"{POD_USER}@{POD_IP_OR_HOSTNAME}"
    known_hosts_file = os.path.join(tempfile.gettempdir(), "ssh_known_hosts")
    cmd_str = (
        f'"{SSH_EXE_PATH}" -q -p {POD_PORT} '
        f'-o ConnectTimeout=15 -o LogLevel=ERROR '
        f'-o StrictHostKeyChecking=no '
        f'-o UserKnownHostsFile={known_hosts_file} '
        f'-i "{SSH_KEY_PATH}" '
        f'"{USER_HOST}" "{remote_cmd}"'
    )
    try:
        result = subprocess.run(
            cmd_str, shell=True, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=30
        )
        return result.returncode, (result.stdout.strip() if result.stdout else "")
    except Exception:
        return -1, ""


def try_ssh_connection():
    """Test if SSH connection to the pod works."""
    rc, output = run_remote_command_output("echo ok")
    return rc == 0 and "ok" in output


def handle_disconnection():
    """Handle SSH disconnection: reconnect in background while prompting user.
    Returns True if reconnected, False if user chose to terminate."""
    reconnected = threading.Event()
    user_terminate = threading.Event()

    def reconnect_worker():
        attempt = 0
        while not user_terminate.is_set():
            attempt += 1
            safe_print(f"[Reconnect] Attempt {attempt}...")
            if try_ssh_connection():
                reconnected.set()
                return
            user_terminate.wait(20)

    def input_worker():
        while not reconnected.is_set() and not user_terminate.is_set():
            try:
                answer = input(f"\nConnection lost. Terminate pod {POD_ID}? (y/n): ").lower().strip()
                if answer == 'y':
                    user_terminate.set()
                    return
                elif answer == 'n':
                    safe_print("Continuing reconnection attempts...")
            except EOFError:
                user_terminate.set()
                return

    reconnect_t = threading.Thread(target=reconnect_worker, daemon=True)
    input_t = threading.Thread(target=input_worker, daemon=True)
    reconnect_t.start()
    input_t.start()

    while not reconnected.is_set() and not user_terminate.is_set():
        time.sleep(0.5)

    if reconnected.is_set():
        safe_print("\n--- Connection re-established! Resuming... ---")
        return True
    return False


def launch_training_nohup(config_file, stage_label):
    """Write a training wrapper script to the pod and launch it with nohup.
    Training survives SSH disconnects. Returns the remote PID."""
    safe_print(f"\n--- Launching {stage_label} with nohup ---")

    # Clean previous state
    run_remote_command(f"rm -f {REMOTE_EXIT_FILE} {REMOTE_PID_FILE} {REMOTE_TRAIN_SCRIPT}", check=False)

    # Create wrapper script via base64 to avoid shell quoting issues
    script_content = (
        f"#!/bin/bash\n"
        f"cd {REMOTE_TOOLKIT_DIR}\n"
        f"./venv/bin/python run.py {config_file} >> {LOG_FILE} 2>&1\n"
        f"echo $? > {REMOTE_EXIT_FILE}\n"
    )
    encoded = base64.b64encode(script_content.encode()).decode()
    run_remote_command(
        f"echo {encoded} | base64 -d > {REMOTE_TRAIN_SCRIPT} && chmod +x {REMOTE_TRAIN_SCRIPT}",
        check=True
    )

    # Launch with nohup and save PID
    run_remote_command(
        f"nohup {REMOTE_TRAIN_SCRIPT} > /dev/null 2>&1 & echo $! > {REMOTE_PID_FILE}",
        check=True
    )

    # Read back the PID
    rc, pid = run_remote_command_output(f"cat {REMOTE_PID_FILE}")
    if rc != 0 or not pid.strip().isdigit():
        raise RuntimeError(f"Failed to get training PID. rc={rc}, output='{pid}'")

    pid = pid.strip()
    safe_print(f"Training process launched with PID {pid}")
    return pid


def monitor_training(stage_label, poll_interval=15):
    """Monitor a nohup training process, handling SSH disconnects gracefully.
    Returns the exit code of the training process (0 = success)."""
    global sync_thread, sync_stop_event

    while True:
        try:
            check_cmd = (
                f"if [ -f {REMOTE_EXIT_FILE} ]; then "
                f"echo DONE $(cat {REMOTE_EXIT_FILE}); "
                f"elif [ -f {REMOTE_PID_FILE} ] && kill -0 $(cat {REMOTE_PID_FILE}) 2>/dev/null; then "
                f"echo RUNNING; "
                f"else echo ERROR; fi"
            )
            rc, output = run_remote_command_output(check_cmd)

            if rc != 0:
                raise ConnectionError(f"SSH command failed (rc={rc})")

            if output.startswith("DONE"):
                parts = output.split()
                exit_code = int(parts[1]) if len(parts) > 1 else -1
                safe_print(f"{stage_label} finished with exit code {exit_code}")
                return exit_code
            elif output.startswith("RUNNING"):
                pass  # still running, continue polling
            elif output.startswith("ERROR"):
                safe_print(f"WARNING: Training process not found and no exit file. It may have crashed.")
                return -1
            else:
                safe_print(f"WARNING: Unexpected status output: {output}")

            time.sleep(poll_interval)

        except Exception as e:
            safe_print(f"\n!!! SSH connection lost during {stage_label}: {e} !!!")

            # Stop sync while disconnected
            sync_stop_event.set()
            if sync_thread and sync_thread.is_alive():
                sync_thread.join(timeout=5)

            if handle_disconnection():
                # Reconnected - restart sync and monitoring
                safe_print("Restarting continuous checkpoint sync...")
                sync_stop_event = threading.Event()
                sync_thread = threading.Thread(
                    target=continuous_output_sync, args=(5,), daemon=True
                )
                sync_thread.start()
                launch_monitoring_window()
                safe_print(f"Resuming {stage_label} monitoring...")
                continue
            else:
                raise RuntimeError(f"User chose to terminate during {stage_label} disconnection")


def set_yaml_value(data, path, value):
    keys = path.split(".")
    d = data
    for key in keys[:-1]:
        # list index?
        if key.isdigit():
            key = int(key)
            while len(d) <= key:
                d.append({})
            d = d[key]
        else:
            if key not in d or not isinstance(d[key], (dict, list)):
                d[key] = {}
            d = d[key]
    last = keys[-1]
    if last.isdigit():
        last = int(last)
        while len(d) <= last:
            d.append(None)
        d[last] = value
    else:
        d[last] = value

# --- End Helper Function Definitions -----------------------------------------------------------------------------------------------------------------------------------



# --- MODIFIED Main Workflow with Detailed Error Handling ---

with open(LOCAL_CONFIG1_PATH) as f:
    acfg = yaml.safe_load(f)
set_yaml_value(acfg, "config.process.0.training_folder", "output")
set_yaml_value(acfg, "config.name", cfg["name"])
set_yaml_value(acfg, "config.process.0.trigger_word", cfg["trigger_word"])
set_yaml_value(acfg, "config.process.0.train.steps", cfg["steps"])
set_yaml_value(acfg, "config.process.0.save.save_every", cfg["save_every"])
set_yaml_value(acfg, "config.process.0.save.max_step_saves_to_keep", cfg["max_step_saves_to_keep"])
set_yaml_value(acfg, "config.process.0.datasets.0.folder_path", "/workspace/dataset")
with open(LOCAL_CONFIG1_PATH, "w") as f:
    yaml.safe_dump(acfg, f, sort_keys=False)


print(">>> Starting Automated LoRA Training Workflow (with SDK Pod Creation) <<<")

# Flags/Variables to track state
pod_was_created = False
training_completed = False
error_occurred = False
terminate_decision = False # Default to not terminating unless successful

# Define POD_ID outside try block so finally block can access it
POD_ID = ""


try:
    # --- Step 0: Create Pod via SDK ---
    print("\n--- Step 0: Creating RunPod Pod via SDK ---")
    pod_params = {
        "name": POD_NAME, "image_name": IMAGE_NAME, "gpu_type_id": GPU_TYPE_ID,
        "cloud_type": CLOUD_TYPE, "gpu_count": GPU_COUNT, "volume_in_gb": VOLUME_DISK_GB,
        "container_disk_in_gb": CONTAINER_DISK_GB, "start_ssh": True, "support_public_ip": True, "ports": "22/tcp"
    }
    if TEMPLATE_ID: pod_params["template_id"] = TEMPLATE_ID; pod_params.pop("image_name", None)
    if VOLUME_ID:
        pod_params["volume_mount_path"] = VOLUME_MOUNT_PATH
        pod_params["volume_id"] = VOLUME_ID
        if "volume_in_gb" in pod_params: pod_params.pop("volume_in_gb", None)
    if DATA_CENTER_ID: pod_params["data_center_id"] = DATA_CENTER_ID

    print(f"Calling runpod.create_pod with params: {json.dumps(pod_params, indent=2)}")
    start_time = time.time()
    last_exception = None
    while time.time() - start_time < max_pod_wait_time:
        try:
            new_pod_info = runpod.create_pod(**pod_params)
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Error occurred: {e}")
            last_exception = e
            time.sleep(2)  # Wait 2 seconds before retrying
    else:
        raise last_exception
    print("--- Pod Creation Initiated ---")

    # print(json.dumps(new_pod_info, indent=2)) # Verbose
    POD_ID = new_pod_info.get('id')
    if not POD_ID: raise ValueError("Failed to get Pod ID from creation response.")
    pod_was_created = True # Mark that the API call succeeded and we have an ID
    print(f"Pod successfully created with ID: {POD_ID}")


    # --- Step 0.1: Fetching Pod Connection Details ---
    print("\n--- Step 0.1: Fetching Pod Connection Details ---")
    max_fetch_attempts = 18; fetch_delay = 10; pod_details = None; ssh_details_found = False
    for attempt in range(max_fetch_attempts):
        print(f"Fetching pod details (Attempt {attempt + 1}/{max_fetch_attempts})...")
        time.sleep(fetch_delay)
        pod_details = runpod.get_pod(POD_ID)
        # print(f"DEBUG Pod details: {json.dumps(pod_details, indent=2)}")
        runtime_info = pod_details.get('runtime')
        pod_status = pod_details.get('desiredStatus', 'UNKNOWN')
        print(f"Pod status: {pod_status}.")
        if pod_status in ['TERMINATED', 'FAILED']: raise RuntimeError(f"Pod entered {pod_status} state during startup.")

        if runtime_info and isinstance(runtime_info.get('ports'), list):
            for port_map in runtime_info['ports']:
                if port_map.get('privatePort') == 22 and port_map.get('publicPort') and port_map.get('ip') and port_map.get('type') == 'tcp':
                    # Use the IP associated with the pod details if available and matches port mapping IP
                    pod_ip = pod_details.get('ip')
                    if pod_ip and pod_ip == port_map.get('ip'):
                        POD_IP_OR_HOSTNAME = port_map['ip']
                        POD_PORT = str(port_map['publicPort'])
                        print(f"Found SSH connection details: IP={POD_IP_OR_HOSTNAME}, Port={POD_PORT}")
                        ssh_details_found = True; break
                    else:
                         # Fallback or warning if IPs don't match? Or just use port_map['ip']?
                         # Let's use the port map IP but warn if it differs from pod IP
                         POD_IP_OR_HOSTNAME = port_map['ip']
                         POD_PORT = str(port_map['publicPort'])
                         if pod_ip and pod_ip != POD_IP_OR_HOSTNAME:
                              print(f"Warning: Pod IP ({pod_ip}) differs from Port Map IP ({POD_IP_OR_HOSTNAME}). Using Port Map IP.")
                         print(f"Found SSH connection details: IP={POD_IP_OR_HOSTNAME}, Port={POD_PORT}")
                         ssh_details_found = True; break
            if ssh_details_found: break
        print(f"Runtime/port info not yet available, waiting {fetch_delay}s...")
    if not ssh_details_found: raise RuntimeError(f"Failed to fetch SSH connection details after {max_fetch_attempts} attempts.")


    # --- Step 0.5: Wait for SSH Readiness ---
    if not wait_for_ssh():
        raise RuntimeError("SSH readiness check failed.")


    # --- Step 0.6: Pre-create Log File ---
    print(f"\n--- Pre-creating remote log file: {LOG_FILE} ---")
    if run_remote_command(f"touch {LOG_FILE}", check=False) != 0: # check=True raises error on failure
        print("Remote log file could not be created, continues without.")
    else:
        print("Remote log file created/touched successfully.")
        launch_monitoring_window()


    # --- Step 1: Installing ai-toolkit ---
    print("\n--- Step 1: Installing ai-toolkit ---")
    run_remote_command(f"echo '--- Step 1: Installing ai-toolkit ---' >> {LOG_FILE} 2>&1", check=False)
    setup_commands = [
        f"{{ if [ ! -d {REMOTE_TOOLKIT_DIR} ]; then git clone https://github.com/ostris/ai-toolkit.git {REMOTE_TOOLKIT_DIR}; else echo 'Toolkit dir {REMOTE_TOOLKIT_DIR} exists, skipping clone.'; fi; }} > {LOG_FILE} 2>&1",
        f"{{ cd {REMOTE_TOOLKIT_DIR} && git submodule update --init --recursive; }} >> {LOG_FILE} 2>&1",
        f"{{ cd {REMOTE_TOOLKIT_DIR} && python -m venv venv; }} >> {LOG_FILE} 2>&1",
        f"{{ cd {REMOTE_TOOLKIT_DIR} && ./venv/bin/python -m pip install --upgrade pip; }} >> {LOG_FILE} 2>&1",
        f"{{ cd {REMOTE_TOOLKIT_DIR} && ./venv/bin/python -m pip install torch torchvision torchaudio; }} >> {LOG_FILE} 2>&1",
        f"{{ cd {REMOTE_TOOLKIT_DIR} && ./venv/bin/python -m pip install -r requirements.txt; }} >> {LOG_FILE} 2>&1"
    ]
    for cmd in setup_commands:
        print(f"Executing setup command remotely (logging to {LOG_FILE})...")
        run_remote_command(cmd, check=True) # Rely on check=True to raise error
    print("--- Finished Step 1: Installing ai-toolkit ---")
    run_remote_command(f"echo '--- Finished Step 1: Installing ai-toolkit ---' >> {LOG_FILE} 2>&1", check=False)


    # --- Step 2: Set Environment Key ---
    print("\n--- Step 2: Set Environment Key ---")
    run_remote_command(f"echo '--- Step 2: Uploading Environment File ---' >> {LOG_FILE} 2>&1", check=False)
    transfer_file_or_dir(LOCAL_ENV_FILE_PATH, REMOTE_TOOLKIT_DIR, upload=True, check=True) # check=True raises error
    # Rename command
    src_path = os.path.join(REMOTE_TOOLKIT_DIR, os.path.basename(LOCAL_ENV_FILE_PATH)).replace("\\", "/")
    rename_cmd = f"{{ mv {src_path} {REMOTE_ENV_FILE_PATH}; }} >> {LOG_FILE} 2>&1"
    run_remote_command(rename_cmd, check=True)
    print("\n--- Finished Step 2: Upload Env File ---")
    run_remote_command(f"echo '--- Finished Step 2: Set Environment Key ---' >> {LOG_FILE} 2>&1", check=False)


    # --- Step 2.5: Upload Previous Output/Checkpoints for Resuming (Optional) ---
    # Use LOCAL_RESUME_SOURCE_DIR as defined in config
    if LOCAL_RESUME_SOURCE_DIR and os.path.isdir(LOCAL_RESUME_SOURCE_DIR):
        resume_folder_name = os.path.basename(LOCAL_RESUME_SOURCE_DIR)
        print(f"\n--- Step 2.5: Uploading Resume Data from '{resume_folder_name}' ---")
        run_remote_command(f"echo '--- Starting Step 2.5: Upload Resume Data ({resume_folder_name}) ---' >> {LOG_FILE} 2>&1", check=False)
        transfer_file_or_dir(LOCAL_RESUME_SOURCE_DIR, REMOTE_OUTPUT_DIR, is_directory=True, upload=True) # check=True raises error
        print(f"\n--- Finished Step 2.5: Upload Resume Data ---")
        run_remote_command(f"echo '--- Finished Step 2.5: Upload Resume Data ---' >> {LOG_FILE} 2>&1", check=False)
    elif LOCAL_RESUME_SOURCE_DIR:
        print(f"WARNING: Specified local resume directory not found at '{LOCAL_RESUME_SOURCE_DIR}'. Skipping resume data upload.")
        run_remote_command(f"echo '--- Warning: Local resume directory not found ({LOCAL_RESUME_SOURCE_DIR}), Skipping resume data upload ---' >> {LOG_FILE} 2>&1", check=False)
    else:
        print("\n--- Step 2.5: No resume data directory specified, skipping upload. ---")
        run_remote_command(f"echo '--- Step 2.5: No resume data directory specified, skipping upload. ---' >> {LOG_FILE} 2>&1", check=False)


    # --- Step 3: Uploading Dataset ---
    print("\n--- Step 3: Uploading Dataset ---")
    run_remote_command(f"echo '--- Step 3: Uploading Dataset ---' >> {LOG_FILE} 2>&1", check=False)
    cleanup_cmd = f"{{ rm -rf {REMOTE_DATASET_DIR}; }} >> {LOG_FILE} 2>&1"
    print(f"Running cleanup...")
    run_remote_command(cleanup_cmd, check=False)
    transfer_file_or_dir(LOCAL_DATASET_DIR, REMOTE_DATASET_DIR, is_directory=True, upload=True) # check=True raises error
    print("\n--- Finished Step 3: Upload Dataset ---")
    run_remote_command(f"echo '--- Finished Step 3: Upload Dataset ---' >> {LOG_FILE} 2>&1", check=False)


    # --- Step 3.5: Upload Config Files ---
    print("\n--- Step 3.5: Upload Config Files ---")
    run_remote_command(f"echo '--- Step 3.5: Upload Config Files ---' >> {LOG_FILE} 2>&1", check=False)
    transfer_file_or_dir(LOCAL_CONFIG1_PATH, REMOTE_CONFIG_DIR, upload=True, check=True) # check=True raises error
    if config_2_exists:
        transfer_file_or_dir(LOCAL_CONFIG2_PATH, REMOTE_CONFIG_DIR, upload=True) # check=True raises error
    print("\n--- Finished Step 3.5: Upload Config Files ---")
    run_remote_command(f"echo '--- Finished Step 3.5: Upload Config Files ---' >> {LOG_FILE} 2>&1", check=False)



    # --- Step 4: Running Training ---
    print("\n--- Step 4: Running Training ---")
    # Start continuous checkpoint syncing
    sync_thread = threading.Thread(target=continuous_output_sync, args=(5,), daemon=True)
    sync_thread.start()
    run_remote_command(f"echo '--- Step 4: Running Training ---' >> {LOG_FILE} 2>&1", check=False)

    # Stage 1 - launched with nohup so training survives SSH disconnects
    run_remote_command(f"echo -e '\\n--- Starting Training Stage 1 ---\\n' >> {LOG_FILE} 2>&1", check=False)
    print(f"\n--- Running Stage 1 (Output appended to {LOG_FILE} on pod) ---")
    launch_training_nohup(CONFIG_FILE_1, "Training Stage 1")
    exit_code = monitor_training("Training Stage 1")
    if exit_code != 0:
        raise RuntimeError(f"Training Stage 1 failed with exit code {exit_code}")

    # Stage 2 (if applicable)
    if config_2_exists:
        run_remote_command(f"echo -e '\\n--- Starting Training Stage 2 ---\\n' >> {LOG_FILE} 2>&1", check=False)
        print(f"\n--- Running Stage 2 (Output appended to {LOG_FILE} on pod) ---")
        launch_training_nohup(CONFIG_FILE_2, "Training Stage 2")
        exit_code = monitor_training("Training Stage 2")
        if exit_code != 0:
            raise RuntimeError(f"Training Stage 2 failed with exit code {exit_code}")

    training_completed = True # Mark training success
    # Stop checkpoint syncing
    sync_stop_event.set()
    if sync_thread:
        sync_thread.join()
    print("--- Successfully Finished Step 4: Training ---")
    run_remote_command(f"echo '--- Successfully Finished Step 4: Training ---' >> {LOG_FILE} 2>&1", check=False)


    # --- Step 5: Downloading Output ---
    print("\n--- Step 5: Download remaining output ---")
    run_remote_command(f"echo '--- Step 5: Download remaining output ---' >> {LOG_FILE} 2>&1", check=False)

    # Folder on pod: output/<lora_name>
    lora_output_remote = os.path.join(REMOTE_OUTPUT_DIR).replace("\\", "/")
    lora_output_local = os.path.join(LOCAL_DOWNLOAD_DIR, os.path.basename(REMOTE_OUTPUT_DIR))

    os.makedirs(lora_output_local, exist_ok=True)

    # List all files on remote pod
    list_files_cmd = f'find {lora_output_remote} -type f'
    ssh_list_cmd = (
        f'"{SSH_EXE_PATH}" -q -p {POD_PORT} '
        f'-o StrictHostKeyChecking=no '
        f'-o UserKnownHostsFile={os.path.join(tempfile.gettempdir(), "ssh_known_hosts")} '
        f'-i "{SSH_KEY_PATH}" '
        f'"{POD_USER}@{POD_IP_OR_HOSTNAME}" "{list_files_cmd}"'
    )

    result = subprocess.run(ssh_list_cmd, shell=True, capture_output=True, text=True)
    remote_files = result.stdout.strip().splitlines()

    for remote_file in remote_files:
        if not remote_file:
            continue

        # Skip checkpoints (handled by continuous sync)
        if remote_file.endswith(".safetensors"):
            continue

        # Compute local path
        rel_path = os.path.relpath(remote_file, lora_output_remote).replace("\\", "/")
        local_path = os.path.join(lora_output_local, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # --- Special handling for optimizer ---
        if remote_file.endswith("optimizer.pt"):
            download_needed = False

            if not os.path.exists(local_path):
                download_needed = True
            else:
                try:
                    remote_hash = sha256_of_remote_file(remote_file)
                    local_hash = sha256_of_local_file(local_path)
                    if remote_hash != local_hash:
                        download_needed = True
                except Exception:
                    download_needed = True  # fallback if hash fails

            if download_needed:
                print(f"Downloading updated optimizer.pt → {rel_path}")
                transfer_file_or_dir(local_path, remote_file, upload=False, check=True)
            else:
                print(f"Optimizer already up to date: {rel_path}")

            continue  # optimizer handled, skip normal logic

        # --- Normal files ---
        if os.path.exists(local_path):
            print(f"Skipping already downloaded file: {rel_path}")
            continue

        print(f"Downloading new output file: {rel_path}")
        transfer_file_or_dir(local_path, remote_file, upload=False, check=True)

    print("\n--- Finished Step 5: Remaining output downloaded ---")
    run_remote_command(f"echo '--- Finished Step 5: Remaining output downloaded ---' >> {LOG_FILE} 2>&1", check=False)


    # If we reach here without exception
    print("\n>>> All workflow steps completed successfully. <<<")
    run_remote_command(f"echo '>>> All workflow steps completed successfully. <<<' >> {LOG_FILE} 2>&1", check=False)
    terminate_decision = True # Mark for termination on success


except Exception as e:
    error_occurred = True
    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! WORKFLOW ERROR: {e}")
    # Print traceback for more detailed debugging
    import traceback
    traceback.print_exc()
    print(f"!!! Script stopped prematurely.           !!!")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    # Decision logic happens in finally block



finally:
    # --- Step 6: Terminating Pod ---
    print("\n--- Step 6: Checking Pod Termination ---")
    if pod_was_created and POD_ID: # Check if we know a pod was created by this script run AND we got its ID
        if error_occurred:
            if not training_completed: # Error happened BEFORE or DURING training finished
                print("An error occurred before or during training.")
                # --- Code for interactive prompt (commented out for default non-termination) ---
                while True:
                    try:
                        answer = input(f"Terminate pod {POD_ID}? (y/n): ").lower().strip()
                        if answer == 'y':
                            terminate_decision = True
                            break
                        elif answer == 'n':
                            terminate_decision = False
                            print("Pod will NOT be terminated.")
                            break
                        else:
                            print("Invalid input. Please enter 'y' or 'n'.")
                    except EOFError: # Handle non-interactive session
                         print("Non-interactive session detected. Defaulting to terminating pod on pre-training error.")
                         terminate_decision = True
                         break
                # --- End interactive prompt code ---
            else: # Error happened AFTER training finished (e.g., download, move)
                print("An error occurred AFTER training finished (download/move).")
                print("Pod will NOT be terminated automatically to allow inspection/recovery.")
                terminate_decision = False
        elif not error_occurred and training_completed: # Ensure training completed successfully
            print("Workflow completed successfully.")
            terminate_decision = True # Mark for termination on success
        else: # No error, but training didn't complete? Should have been caught by exceptions
             print("Workflow ended without error, but training completion flag not set? Not terminating.")
             terminate_decision = False

        # Actual termination based on decision
        if terminate_decision:
            print(f"Attempting termination of pod {POD_ID}...")
            if terminate_pod(POD_ID) == 0:
                print(">>> Pod Termination Successful <<<")
            else:
                print("!!! Pod termination failed or could not be confirmed! Please check RunPod dashboard manually. !!!")
        else:
             print(f"--- Pod {POD_ID} termination skipped based on error stage or user input/default. ---")

    elif POD_ID: # We have a POD_ID but pod_was_created might be false if creation failed early
         print(f"\n--- Info: Pod with ID {POD_ID} might exist but wasn't confirmed fully created/operational by this script. ---")
         print("--- Manual check/termination recommended on RunPod dashboard. ---")
    else: # No POD_ID was ever obtained
         print("\n--- Info: Pod creation failed or Pod ID not obtained. No termination attempted. ---")
