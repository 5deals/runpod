import os
import sys
import logging
from typing import Dict, Any

import runpod

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pow.compute.compute import Compute
from pow.models.utils import Params
from pow.random import get_target

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global compute instance (loaded once, reused across requests)
COMPUTE = None
CURRENT_BLOCK_HASH = None


def initialize_compute(
    block_hash: str,
    block_height: int,
    params_dict: Dict[str, Any] = None,
    devices: list = None,
) -> Compute:
    """Initialize the compute model for a specific block hash"""
    global COMPUTE, CURRENT_BLOCK_HASH

    # If compute already initialized for this block_hash, reuse it
    if COMPUTE is not None and CURRENT_BLOCK_HASH == block_hash:
        logger.info(f"Reusing existing compute for block_hash={block_hash}")
        return COMPUTE

    logger.info(f"Initializing compute for block_hash={block_hash}")

    # Default params if not provided
    if params_dict is None:
        params_dict = {}

    params = Params(**params_dict)

    # Default devices if not provided
    if devices is None:
        devices = ["cuda:0"]

    # Create compute instance
    COMPUTE = Compute(
        params=params,
        block_hash=block_hash,
        block_height=block_height,
        public_key="",  # Will be overridden per request
        r_target=0.0,   # Will be overridden per request
        devices=devices,
        node_id=0,      # Serverless doesn't need node_id
    )

    CURRENT_BLOCK_HASH = block_hash
    logger.info("Compute initialized successfully")

    return COMPUTE


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runpod handler function

    Expected input format:
    {
        "block_hash": "string",
        "block_height": int,
        "public_key": "string",
        "nonces": [int, int, ...],
        "r_target": float,
        "params": {  # optional
            "dim": int,
            "n_layers": int,
            ...
        },
        "devices": ["cuda:0", ...]  # optional
    }

    Returns:
    {
        "public_key": "string",
        "block_hash": "string",
        "block_height": int,
        "nonces": [int, ...],  # filtered nonces
        "dist": [float, ...],   # distances
        "node_id": int
    }
    """
    try:
        # Extract input data
        input_data = event.get("input", {})

        block_hash = input_data["block_hash"]
        block_height = input_data["block_height"]
        public_key = input_data["public_key"]
        nonces = input_data["nonces"]
        r_target = input_data["r_target"]

        params_dict = input_data.get("params", {})
        devices = input_data.get("devices", ["cuda:0"])

        logger.info(f"Received request: block_hash={block_hash}, public_key={public_key[:10]}..., nonces={len(nonces)}, r_target={r_target}")

        # Initialize compute (or reuse existing)
        compute = initialize_compute(
            block_hash=block_hash,
            block_height=block_height,
            params_dict=params_dict,
            devices=devices,
        )

        # Get target for this block
        target = get_target(block_hash, compute.params.vocab_size)

        # Compute distances
        logger.info(f"Computing distances for {len(nonces)} nonces...")
        proof_batch = compute(
            nonces=nonces,
            public_key=public_key,
            target=target,
        )

        # Filter by r_target
        filtered_batch = proof_batch.sub_batch(r_target)

        logger.info(f"Computed {len(proof_batch)} nonces, {len(filtered_batch)} passed r_target filter")

        # Return result
        return {
            "public_key": filtered_batch.public_key,
            "block_hash": filtered_batch.block_hash,
            "block_height": filtered_batch.block_height,
            "nonces": filtered_batch.nonces,
            "dist": filtered_batch.dist,
            "node_id": filtered_batch.node_id,
            "total_computed": len(proof_batch),
            "total_valid": len(filtered_batch),
        }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


# Start the serverless handler
runpod.serverless.start({"handler": handler})
