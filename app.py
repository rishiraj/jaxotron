import os
import jax
from config import JaxotronConfig, ParallelismConfig, ModelConfig, TrainingConfig
from trainer.loop import train

if __name__ == "__main__":
    # Fake dataset setup for testing (Generates 2MB of dummy data)
    if jax.process_index() == 0 and not os.path.exists("dataset.bin"):
        with open("dataset.bin", "wb") as f:
            f.write(b"\x00\x01" * 1000000)

    # Automatically defaults fsdp to the number of available GPUs/TPUs
    config = JaxotronConfig(
        parallelism=ParallelismConfig(
            dp=1, 
            fsdp=jax.device_count(), 
            tp=1
        ),
        model=ModelConfig(hidden_size=1024, num_hidden_layers=8, use_mup=True),
        training=TrainingConfig(global_batch_size=32, micro_batch_size=4, train_steps=500)
    )
    
    train(config=config, data_path="dataset.bin", checkpoint_dir="./checkpoints")