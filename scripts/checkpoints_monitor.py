#!/usr/bin/env python3

import argparse
import json
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from subprocess import Popen, PIPE
from threading import Lock
from typing import Dict, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default paths - can be overridden via command line arguments
DEFAULT_CONVERTER_SCRIPT = "/workspace/projects/ai-tookit/scripts/convert_diffusers_to_comfy.py"
DEFAULT_TRAINING_FOLDER = "/data/models/current_train/v001"
DEFAULT_REF_MODEL = "/data/models/unet/flux1-dev-fp8.safetensors"


@dataclass
class CheckpointInfo:
    """Stores information about a checkpoint folder and its conversion status"""
    folder_path: Path
    safetensors_path: Path
    total_size: Optional[int] = None
    is_converting: bool = False
    is_done: bool = False
    last_check: datetime = datetime.now()


class CheckpointMonitor:
    def __init__(
            self,
            training_folder: Path,
            converter_script: Path,
            ref_model: Path,
            check_interval: int = 30
    ):
        self.training_folder = Path(training_folder)
        self.converter_script = Path(converter_script)
        self.ref_model = Path(ref_model)
        self.check_interval = check_interval
        self.dest_folder = self.training_folder.parent

        # Thread-safe storage for checkpoint information
        self.checkpoints: Dict[str, CheckpointInfo] = {}
        self.checkpoints_lock = Lock()

        # Set for tracking folders to skip
        self.skip_folders: Set[str] = {"samples"}

        # Executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Signal handling for graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown signal received. Cleaning up...")
        self.running = False
        self.executor.shutdown(wait=True)
        sys.exit(0)

    def get_safetensors_path(self, checkpoint_folder: Path) -> Path:
        """Generate the expected safetensors output path for a checkpoint folder"""
        return self.dest_folder / f"{checkpoint_folder.name}.safetensors"

    def verify_checkpoint_integrity(self, checkpoint_folder: Path) -> Optional[int]:
        """
        Verify the integrity of checkpoint files.
        Returns the expected total size if verification passes, None otherwise.
        """
        transformer_path = checkpoint_folder / "transformer"
        if not transformer_path.exists():
            logger.warning(f"Transformer folder missing in {checkpoint_folder}")
            return None

        # Read expected total size from index file
        index_file = transformer_path / "diffusion_pytorch_model.safetensors.index.json"
        try:
            with open(index_file) as f:
                index_data = json.load(f)
                total_size = index_data["metadata"]["total_size"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read index file in {checkpoint_folder}: {e}")
            return None

        # Verify all safetensors files are present and sum their sizes
        actual_size = 0
        part = 1
        while True:
            part_file = transformer_path / f"diffusion_pytorch_model-{part:05d}-of-00002.safetensors"
            if not part_file.exists():
                break
            actual_size += part_file.stat().st_size
            part += 1

        # Allow for small size discrepancies (0.1% tolerance)
        size_difference = abs(actual_size - total_size)
        tolerance = total_size * 0.001  # 0.1% tolerance

        if part != 3:  # We expect exactly 2 parts
            logger.warning(
                f"Incorrect number of parts in {checkpoint_folder}. "
                f"Expected: 2, Got: {part-1}"
            )
            return None

        if size_difference > tolerance:
            logger.warning(
                f"Size mismatch in {checkpoint_folder}. "
                f"Expected: {total_size}, Got: {actual_size} "
                f"(difference: {size_difference} bytes, {(size_difference/total_size)*100:.4f}%)"
            )
            return None

        if actual_size != total_size:
            logger.info(
                f"Minor size variation in {checkpoint_folder} "
                f"(within tolerance). Expected: {total_size}, "
                f"Got: {actual_size} "
                f"(difference: {size_difference} bytes, {(size_difference/total_size)*100:.4f}%)"
            )

        return total_size

    def convert_checkpoint(self, checkpoint: CheckpointInfo) -> bool:
        """
        Convert a checkpoint using the converter script.
        Returns True if conversion was successful.
        """
        try:
            with self.checkpoints_lock:
                if checkpoint.is_converting or checkpoint.is_done:
                    return False
                checkpoint.is_converting = True

            logger.info(f"Starting conversion of {checkpoint.folder_path}")

            process = Popen([
                "python",
                str(self.converter_script),
                str(checkpoint.folder_path),
                str(self.ref_model),
                str(checkpoint.safetensors_path)
            ], stdout=PIPE, stderr=PIPE)

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(
                    f"Conversion failed for {checkpoint.folder_path}\n"
                    f"stdout: {stdout.decode()}\n"
                    f"stderr: {stderr.decode()}"
                )
                with self.checkpoints_lock:
                    checkpoint.is_converting = False
                return False

            logger.info(f"Successfully converted {checkpoint.folder_path}")
            with self.checkpoints_lock:
                checkpoint.is_converting = False
                checkpoint.is_done = True
            return True

        except Exception as e:
            logger.error(f"Error during conversion of {checkpoint.folder_path}: {e}")
            with self.checkpoints_lock:
                checkpoint.is_converting = False
            return False

    def process_checkpoint(self, folder_path: Path) -> None:
        """Process a single checkpoint folder"""
        if folder_path.name in self.skip_folders:
            return

        with self.checkpoints_lock:
            checkpoint = self.checkpoints.get(str(folder_path))
            if checkpoint is None:
                safetensors_path = self.get_safetensors_path(folder_path)
                checkpoint = CheckpointInfo(folder_path, safetensors_path)
                self.checkpoints[str(folder_path)] = checkpoint

            # Update last check time
            checkpoint.last_check = datetime.now()

        # Skip if already done or converting
        if checkpoint.is_done or checkpoint.is_converting:
            return

        # Check if safetensors file already exists
        if checkpoint.safetensors_path.exists():
            with self.checkpoints_lock:
                checkpoint.is_done = True
            logger.info(f"Safetensors file already exists for {folder_path}")
            return

        # Verify checkpoint integrity
        total_size = self.verify_checkpoint_integrity(folder_path)
        if total_size is None:
            logger.warning(f"Integrity check failed for {folder_path}")
            return

        # Store the verified total size
        with self.checkpoints_lock:
            checkpoint.total_size = total_size

        # Submit conversion task to thread pool
        self.executor.submit(self.convert_checkpoint, checkpoint)

    def monitor(self):
        """Main monitoring loop"""
        logger.info(
            f"Starting checkpoint monitor\n"
            f"Training folder: {self.training_folder}\n"
            f"Converter script: {self.converter_script}\n"
            f"Reference model: {self.ref_model}"
        )

        while self.running:
            try:
                # Process all checkpoint folders
                for folder_path in self.training_folder.iterdir():
                    if folder_path.is_dir():
                        self.process_checkpoint(folder_path)

                # Sleep for the check interval
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor and convert model checkpoints")
    parser.add_argument(
        "--training-folder",
        type=Path,
        default=DEFAULT_TRAINING_FOLDER,
        help="Path to the training folder to monitor"
    )
    parser.add_argument(
        "--converter-script",
        type=Path,
        default=DEFAULT_CONVERTER_SCRIPT,
        help="Path to the converter script"
    )
    parser.add_argument(
        "--ref-model",
        type=Path,
        default=DEFAULT_REF_MODEL,
        help="Path to the reference model"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Interval between checks in seconds"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.converter_script.exists():
        logger.error(f"Converter script not found: {args.converter_script}")
        sys.exit(1)
    if not args.training_folder.exists():
        logger.error(f"Training folder not found: {args.training_folder}")
        sys.exit(1)
    if not args.ref_model.exists():
        logger.error(f"Reference model not found: {args.ref_model}")
        sys.exit(1)

    # Start monitoring
    monitor = CheckpointMonitor(
        args.training_folder,
        args.converter_script,
        args.ref_model,
        args.check_interval
    )
    monitor.monitor()


if __name__ == "__main__":
    main()
