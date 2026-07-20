# Colab CLI - Key Commands for CharaConsist

## Session Management

### Create Session
```bash
colab new -s <session_name> [--gpu T4|L4|A100|H100] [--tpu v5e1|v6e1]
```
- Allocates a fresh Colab VM
- Registers the session locally with keep-alive daemon
- Options:
  - `-s, --session`: Give the session a name
  - `--gpu`: Request GPU (T4, L4, A100, H100)
  - `--tpu`: Request TPU (v5e1, v6e1)

### Check Session Status
```bash
colab status -s <session_name>
```
- Shows resource usage, uptime, hardware details

### List All Sessions
```bash
colab sessions
```
- Lists all active sessions on your account

### Stop Session
```bash
colab stop -s <session_name>
```
- Terminates the VM and kills keep-alive daemon
- Cleans up from billing

---

## Code Execution

### Execute Python Code
```bash
colab exec -s <session_name> "python code here"
```
- One-shot code execution
- Use stdin piping for multi-line scripts:
  ```bash
  cat script.py | colab exec -s <session_name>
  ```

### Interactive REPL
```bash
colab repl -s <session_name>
```
- Jupyter kernel-based interactive Python shell
- Type `exit()` or Ctrl-D to quit

### Raw Terminal
```bash
colab console -s <session_name>
```
- Direct bash shell on the VM (bypasses Jupyter)
- Full terminal control

### One-Shot Run (No Session Management)
```bash
colab run [--gpu L4] [--keep] script.py [args...]
```
- Auto-allocates VM, runs script, tears down
- `--keep`: Don't stop VM after execution (for debugging)
- Perfect for quick tests

---

## File Management

### Upload Files
```bash
colab upload -s <session_name> ../local/path /vm/remote/path
```
- Transfers files from local machine to Colab VM

### Download Files
```bash
colab download -s <session_name> /vm/remote/path ./local/output
```
- Retrieves files from Colab VM to local machine

### List Remote Files
```bash
colab ls -s <session_name> [/remote/path]
```
- Lists files on the Colab VM

### Delete Remote Files
```bash
colab rm -s <session_name> /vm/file/path
```
- Removes files from the Colab VM

---

## Authentication

### Setup Once
```bash
colab auth --auth=adc
```
- Google Application Default Credentials (recommended)
- Or: `colab auth --auth=oauth`

---

## Logging

### View Session Logs
```bash
colab log -s <session_name>
```
- Structured logs with keep-alive events, errors, etc.

---

## CharaConsist Batch Workflow

The original `run.sh` script has a Colab CLI wrapper. The Bash wrapper below
is the maintained batch workflow; `run_colab.py` remains a legacy
cross-platform prototype.

### Legacy Python prototype
```bash
python run_colab.py prompts/stress_test --gpu L4
python run_colab.py prompts/stress_test --session my-session --gpu L4 --keep
```

### Bash batch runner (Linux/macOS/WSL)
```bash
bash run_colab.sh prompts/stress_test my-session \
  --model-path /content/drive/MyDrive/models/flux-dev --gpu L4
bash run_colab.sh prompts/stress_test my-session \
  --model-path /content/drive/MyDrive/models/flux-dev --gpu L4 --keep
```

`--model-path` is required and is a path **on the Colab VM**. The wrapper
uploads the local source tree and the entire prompt-folder tree, but does not
upload FLUX weights. Put the model on the VM first (for example, by mounting
Google Drive) because model uploads are large and a new Colab VM is empty.

### Workflow Steps (Automated)
1. **Create session** with specified GPU
2. **Upload** the source files and prompt-folder contents to
   `/content/CharaConsist`
3. **Run inference** for every `.txt` file in that uploaded folder (including
   nested folders), with `--save_mask` enabled
4. **Download results** to `./results_colab`
5. **Cleanup** - stop session (unless `--keep` flag)

### Key Differences from Original
| Aspect | Original `run.sh` | `run_colab.sh` |
|--------|------|---|
| Execution | Local GPU (requires GPU hardware) | Remote Colab GPU (cloud-based) |
| Setup | Single machine | Any machine + internet |
| Cost | Free if you own GPU, else cloud cost | Free (Colab free tier) or paid QuotaIncrease |
| Model path | `../model/flux-dev` (relative) | Explicit remote path passed with `--model-path` |
| Scaling | Limited by single GPU VRAM | Can request multiple VMs via multiple sessions |

---

## Tips & Tricks

### Monitor Long-Running Task
```bash
# Terminal 1: Start inference
colab run --keep --gpu L4 inference.py

# Terminal 2: Monitor (in another shell)
colab status -s run-xxxxxx
colab exec -s run-xxxxxx "nvidia-smi"
```

### Debug a Failed Run
```bash
# Keep the session alive
colab run --keep script.py

# Then inspect interactively
colab console -s <session-name>
cd /root/CharaConsist
ls -la results/
```

### Batch Multiple Prompts with Progress
```bash
# Python version includes progress indicators
python run_colab.py prompts/stress_test --gpu L4
```

### Download Only Part of Results
```bash
# Download specific output directory
colab download -s my-session "/root/CharaConsist/results/bg_fg/test_001" ./local_results
```

### Reuse Session for Multiple Runs
```bash
# Create once
colab new -s my-session --gpu L4

# Run multiple times
colab exec -s my-session "python inference.py --prompts_file ..."
colab exec -s my-session "python another_task.py"

# Clean up when done
colab stop -s my-session
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Session not found" | Check `colab sessions` to list active sessions |
| Upload timeout | Try uploading smaller files or use `--timeout 60s` in exec |
| Keep-alive expired | Session auto-stops after ~90 mins of inactivity; use `--keep` to override |
| GPU quota exceeded | Request quota increase or switch to T4 (free tier) |
| Authentication fails | Run `colab auth --auth=adc` or `gcloud auth application-default login` |

