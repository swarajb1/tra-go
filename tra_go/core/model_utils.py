"""Utility helpers for model file operations.

This module centralizes file-moving logic used when a model file is found
to be corrupted or otherwise needs to be moved to the discarded folder.
"""

from __future__ import annotations

import os
import shutil

from core.logger import logger

from database.enums import ModelLocationType


def move_model_to_discarded(src_dir: str, file_name: str) -> bool:
    """Move a model file from src_dir to the discarded models folder.

    Returns True if the file was moved, False otherwise.
    """
    try:
        src_path = os.path.join(src_dir, file_name)
        dst_dir = ModelLocationType.DISCARDED.value
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, file_name)

        if not os.path.exists(src_path):
            logger.debug("Source model does not exist, cannot move: %s", src_path)
            return False

        # If destination exists, try to remove it first to ensure move succeeds
        if os.path.exists(dst_path):
            try:
                os.remove(dst_path)
            except OSError:
                # ignore removal error and attempt move anyway
                logger.debug("Could not remove existing destination file: %s", dst_path)

        shutil.move(src_path, dst_path)
        logger.info("Moved problematic model to discarded: %s", dst_path)
        return True
    except OSError as mv_err:
        logger.warning("Failed to move file %s to discarded: %s", file_name, mv_err)
        return False


def move_model_path(src_path: str) -> str | None:
    """Move a model file specified by an absolute path to discarded folder.

    Returns destination path if moved, otherwise None.
    """
    try:
        if not os.path.exists(src_path):
            logger.debug("Source model path does not exist: %s", src_path)
            return None

        file_name = os.path.basename(src_path)
        dst_dir = ModelLocationType.DISCARDED.value
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, file_name)

        if os.path.exists(dst_path):
            try:
                os.remove(dst_path)
            except OSError:
                logger.debug("Could not remove existing destination file: %s", dst_path)

        shutil.move(src_path, dst_path)
        logger.info("Moved problematic model to discarded: %s", dst_path)
        return dst_path
    except OSError as mv_err:
        logger.warning("Failed to move file %s to discarded: %s", src_path, mv_err)
        return None
