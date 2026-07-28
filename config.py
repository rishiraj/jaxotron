from dataclasses import dataclass, field

@dataclass
class ParallelismConfig:
    dp: int = 1         # Data Parallelism
    fsdp: int = 1       # Fully Sharded Data Parallelism (ZeRO-3)
    tp: int = 1         # Tensor Parallelism
    
    @property
    def world_size(self) -> int:
        return self.dp * self.fsdp * self.tp

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    use_mup: bool = False

@dataclass
class TrainingConfig:
    global_batch_size: int = 512
    micro_batch_size: int = 4
    sequence_length: int = 4096
    train_steps: int = 10000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    seed: int = 42

@dataclass
class JaxotronConfig:
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)