import threading
import queue
import numpy as np
from torch.utils.data import DataLoader
from config import JaxotronConfig
from data.nanoset import TokenizedBytesDataset

def jax_collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    tokens = np.stack([b["tokens"] for b in batch])
    return {
        "input_ids": tokens[:, :-1],
        "labels": tokens[:, 1:]
    }

class JaxPrefetcher:
    def __init__(self, dataloader: DataLoader, prefetch_size: int = 2):
        self.dataloader = dataloader
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            while not self.stop_event.is_set():
                for batch in self.dataloader:
                    if self.stop_event.is_set(): break
                    self.queue.put(batch)
        except Exception as e:
            pass
        finally:
            self.queue.put(None)

    def __iter__(self): return self

    def __next__(self) -> dict[str, np.ndarray]:
        batch = self.queue.get()
        if batch is None: raise StopIteration
        return batch

    def shutdown(self):
        self.stop_event.set()
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except queue.Empty: break
        if self.thread.is_alive(): self.thread.join()

def build_dataloader(config: JaxotronConfig, mesh, data_path: str) -> JaxPrefetcher:
    dataset = TokenizedBytesDataset(file_path=data_path, seq_len=config.training.sequence_length)
    
    loader = DataLoader(
        dataset,
        batch_size=config.training.global_batch_size,
        shuffle=True,          
        num_workers=0,  # FIX: Set to 0 to prevent JAX multithreading deadlocks
        collate_fn=jax_collate_fn,
        drop_last=True
    )
    return JaxPrefetcher(loader, prefetch_size=2)