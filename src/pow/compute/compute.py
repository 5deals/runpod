from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
    Tuple,
)

import numpy as np
import torch

from pow.compute.utils import Stats
from pow.data import ProofBatch
from pow.compute.model_init import ModelWrapper
from pow.models.utils import Params
from pow.random import (
    get_inputs,
    get_permutations,
    get_target,
)


class BaseCompute(ABC):
    def __init__(
        self,
        public_key: str,
        devices,
    ):
        self.public_key = public_key
        self.devices = devices

    @abstractmethod
    def __call__(
        self,
        nonces: List[int],
        public_key: str,
        target: np.ndarray,
    ) -> ProofBatch:
        pass

    @abstractmethod
    def validate(
        self,
        proof_batch: ProofBatch,
    ) -> ProofBatch:
        pass


class Compute(BaseCompute):
    def __init__(
        self,
        params: Params,
        block_hash: str,
        block_height: int,
        public_key: str,
        r_target: float,
        devices: List[str],
        node_id: int,
    ):
        self.public_key = public_key
        self.block_hash = block_hash
        self.block_height = block_height
        self.r_target = r_target
        self.params = params
        self.stats = Stats()
        self.devices = devices
        
        self.model = ModelWrapper.build(
            hash_=self.block_hash,
            params=params,
            stats=self.stats.time_stats,
            devices=self.devices,
            max_seq_len=self.params.seq_len,
        )
        self.target = get_target(
            self.block_hash,
            self.params.vocab_size
        )
        self.node_id = node_id
        
    def _prepare_batch(
        self,
        nonces: List[int],
        public_key: str,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        with self.stats.time_stats.time_gen_inputs():
            inputs = get_inputs(
                self.block_hash,
                public_key,
                nonces,
                dim=self.params.dim,
                seq_len=self.params.seq_len,
            )

        with self.stats.time_stats.time_gen_perms():
            permutations = get_permutations(
                self.block_hash,
                public_key,
                nonces,
                dim=self.params.vocab_size
            )

        return inputs, permutations

    def _process_batch(
        self,
        inputs: torch.Tensor,
        permutations: np.ndarray,
        target: np.ndarray,
        nonces: List[int],
        public_key: str = None,
    ) -> ProofBatch:
        if public_key is None:
            public_key = self.public_key

        self.stats.time_stats.next_iter()
        outputs = self.model(inputs, start_pos=0)
        with self.stats.time_stats.time_infer():
            with self.stats.time_stats.time_sync():
                for device in self.devices:
                    torch.cuda.synchronize(device=device)
            with self.stats.time_stats.time_numpy():
                outputs = outputs[:, -1, :].cpu().numpy()
        del inputs

        with self.stats.time_stats.time_perm():
            batch_indices = np.arange(outputs.shape[0])[:, None]
            outputs = outputs[batch_indices, permutations]

        with self.stats.time_stats.time_process():
            outputs = outputs / np.linalg.norm(outputs, axis=1, keepdims=True)
            distances = np.linalg.norm(
                outputs - target,
                axis=1
            )
            batch = ProofBatch(
                public_key=public_key,
                block_hash=self.block_hash,
                block_height=self.block_height,
                nonces=nonces,
                dist=distances.tolist(),
                node_id=self.node_id,
            )

        return batch

    def __call__(
        self,
        nonces: List[int],
        public_key: str,
        target: np.ndarray,
    ) -> ProofBatch:
        with self.stats.time_stats.time_total_gen():
            inputs, permutations = self._prepare_batch(nonces, public_key)

        return self._process_batch(inputs, permutations, target, nonces, public_key)

    def validate(
        self,
        proof_batch: ProofBatch,
    ) -> ProofBatch:
        nonces = proof_batch.nonces

        assert (
            proof_batch.block_hash == self.block_hash
        ), "Block hash must be the same as the one used to create the model"

        target_to_validate = get_target(proof_batch.block_hash, self.params.vocab_size)
        proof_batch = self(
            nonces=nonces,
            public_key=proof_batch.public_key,
            target=target_to_validate,
        )
        return proof_batch
