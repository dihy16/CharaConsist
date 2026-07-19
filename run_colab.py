#!/usr/bin/env python3
"""
CharaConsist Batch Inference Runner using Colab CLI
Allocates a Colab VM, uploads prompts, runs inference jobs, downloads results.

Usage:
  python run_colab.py prompts/stress_test --gpu L4
  python run_colab.py prompts/stress_test --session my-session --gpu L4
  python run_colab.py prompts/stress_test --gpu L4 --keep

"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


class ColabBatchRunner:
    def __init__(self, prompts_folder, session_name=None, gpu=None, keep_session=False):
        self.prompts_folder = Path(prompts_folder)
        self.session_name = session_name or f"characonsist-{datetime.now().strftime('%s')}"
        self.gpu = gpu
        self.keep_session = keep_session
        
        # Validate prompts folder
        if not self.prompts_folder.exists():
            raise FileNotFoundError(f"Prompts folder not found: {self.prompts_folder}")
        
        self.prompt_files = sorted(self.prompts_folder.glob("*.txt"))
        if not self.prompt_files:
            raise ValueError(f"No .txt files found in {self.prompts_folder}")
        
        print("=" * 50)
        print("CharaConsist Batch Inference on Colab CLI")
        print("=" * 50)
        print(f"Session: {self.session_name}")
        print(f"Prompts folder: {self.prompts_folder}")
        print(f"Found {len(self.prompt_files)} prompt file(s)")
        if self.gpu:
            print(f"GPU: {self.gpu}")
        print(f"Keep session: {self.keep_session}")
        print()
    
    def run_command(self, cmd, description=""):
        """Execute a shell command and return output."""
        print(f"Running: {' '.join(cmd)}")
        if description:
            print(f"  {description}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Command failed with exit code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        
        return result.stdout
    
    def step1_create_session(self):
        """Step 1: Create Colab session."""
        print("[1/4] Creating Colab session...")
        cmd = ["colab", "new", "-s", self.session_name]
        if self.gpu:
            cmd.extend(["--gpu", self.gpu])
        
        self.run_command(cmd, f"Session: {self.session_name}")
        print("✓ Session created\n")
    
    def step2_upload_files(self):
        """Step 2: Upload prompts to session."""
        print("[2/4] Uploading files to session...")
        
        # Create remote directory
        exec_cmd = """import os; os.makedirs('/root/CharaConsist/prompts_batch', exist_ok=True)"""
        self.run_command(["colab", "exec", "-s", self.session_name, exec_cmd])
        
        # Upload each prompt file
        for i, prompt_file in enumerate(self.prompt_files, 1):
            basename = prompt_file.name
            remote_path = f"/root/CharaConsist/prompts_batch/{basename}"
            print(f"  [{i}/{len(self.prompt_files)}] Uploading: {basename}")
            
            self.run_command(
                ["colab", "upload", "-s", self.session_name, str(prompt_file), remote_path],
                f"Remote: {remote_path}"
            )
        
        print("✓ Files uploaded\n")
    
    def step3_run_inference(self):
        """Step 3: Run inference jobs."""
        print("[3/4] Running inference jobs...")
        
        for i, prompt_file in enumerate(self.prompt_files, 1):
            basename_no_ext = prompt_file.stem
            basename = prompt_file.name
            print(f"  [{i}/{len(self.prompt_files)}] Processing: {basename_no_ext}")
            
            # Build the Python code to execute on Colab
            exec_code = f"""
import subprocess
import sys
import os

prompt_file = "/root/CharaConsist/prompts_batch/{basename}"
output_dir = "/root/CharaConsist/results/bg_fg/{basename_no_ext}"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Run inference
subprocess.run([
    sys.executable,
    "inference.py",
    "--init_mode", "0",
    "--prompts_file", prompt_file,
    "--model_path", "../model/flux-dev",
    "--out_dir", output_dir
], cwd="/root/CharaConsist", check=True)

print(f"✓ Completed: {basename_no_ext}")
"""
            
            self.run_command(
                ["colab", "exec", "-s", self.session_name, exec_code],
                f"Inference job: {basename_no_ext}"
            )
        
        print("✓ All inference jobs completed\n")
    
    def step4_download_results(self):
        """Step 4: Download results."""
        print("[4/4] Downloading results...")
        
        output_dir = Path("results_colab")
        output_dir.mkdir(exist_ok=True)
        
        self.run_command(
            ["colab", "download", "-s", self.session_name, "/root/CharaConsist/results", str(output_dir)],
            f"Results directory: {output_dir}"
        )
        
        print(f"✓ Results downloaded to {output_dir}\n")
    
    def cleanup(self):
        """Step 5: Cleanup."""
        if self.keep_session:
            print(f"✓ Session '{self.session_name}' is still running")
            print(f"  Use 'colab stop -s {self.session_name}' to stop it later")
        else:
            print(f"Cleaning up: stopping session '{self.session_name}'...")
            self.run_command(["colab", "stop", "-s", self.session_name])
            print("✓ Session stopped")
        
        print()
    
    def run(self):
        """Execute the full pipeline."""
        try:
            self.step1_create_session()
            self.step2_upload_files()
            self.step3_run_inference()
            self.step4_download_results()
            self.cleanup()
            
            print("=" * 50)
            print("✓ Batch inference completed successfully!")
            print("=" * 50)
            return 0
        
        except Exception as e:
            print(f"\n✗ Error during execution:", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            
            # Attempt cleanup on error
            try:
                if not self.keep_session:
                    print(f"\nAttempting to stop session '{self.session_name}'...", file=sys.stderr)
                    subprocess.run(["colab", "stop", "-s", self.session_name], capture_output=True)
                    print("✓ Session stopped", file=sys.stderr)
            except:
                pass
            
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run CharaConsist batch inference on Colab using CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_colab.py prompts/stress_test --gpu L4
  python run_colab.py prompts/stress_test --session my-session --gpu T4
  python run_colab.py prompts/stress_test --gpu L4 --keep
        """
    )
    
    parser.add_argument(
        "prompts_folder",
        help="Path to folder containing prompt .txt files"
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--gpu",
        choices=["T4", "L4", "A100", "H100"],
        default=None,
        help="GPU type (T4, L4, A100, or H100)"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the session running after completion (useful for debugging)"
    )
    
    args = parser.parse_args()
    
    try:
        runner = ColabBatchRunner(
            prompts_folder=args.prompts_folder,
            session_name=args.session,
            gpu=args.gpu,
            keep_session=args.keep
        )
        return runner.run()
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
