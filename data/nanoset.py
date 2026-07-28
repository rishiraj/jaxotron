import os
import numpy as np
from torch.utils.data import Dataset

class TokenizedBytesDataset(Dataset):
    """Reads tokenized docs directly from memory-mapped binary files."""
    def __init__(self, file_path: str, seq_len: int, dtype=np.uint16):
        super().__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        
        file_size_bytes = os.path.getsize(file_path)
        self.total_tokens = file_size_bytes // self.dtype.itemsize
        self.tokens_per_sample = self.seq_len + 1
        self._len = self.total_tokens // self.tokens_per_sample
        
        self.mmap = np.memmap(self.file_path, dtype=self.dtype, mode='r')

    def __len__(self) -> int:
        return max(self._len, 1)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        idx = idx % max(self._len, 1) # Infinitely wrap the dataset
        
        start_idx = idx * self.tokens_per_sample
        end_idx = start_idx + self.tokens_per_sample
        
        tokens = self.mmap[start_idx:end_idx].astype(np.int32)
        
        # Guard for tiny dummy datasets
        if len(tokens) < self.tokens_per_sample:
            tokens = np.pad(tokens, (0, self.tokens_per_sample - len(tokens)), constant_values=0)
            
        return {"tokens": tokens}