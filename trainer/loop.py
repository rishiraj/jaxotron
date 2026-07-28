import jax
import time
import wandb
import equinox as eqx

from config import JaxotronConfig
from parallel.mesh import create_device_mesh
from parallel.sharding_utils import shard_model
from models.llama import LlamaForCausalLM
from optim.build import build_optimizer
from optim.mup import apply_mup_initialization, get_mup_lr_multiplier_tree
from data.collator import build_dataloader
from trainer.step import make_train_step
from serialize.orbax_utils import create_checkpointer, save_checkpoint, load_checkpoint_if_exists

def train(config: JaxotronConfig, data_path: str, checkpoint_dir: str):
    """The main Jaxotron training loop."""
    
    is_main_process = jax.process_index() == 0
    if is_main_process:
        wandb.init(project="jaxotron", config=config.__dict__)

    mesh = create_device_mesh(config.parallelism)
    if is_main_process: print(f"Created Mesh: {mesh}")

    # Initialize model on CPU first to save accelerator memory during setup
    init_key = jax.random.PRNGKey(config.training.seed)
    model_cpu = LlamaForCausalLM(config.model, key=init_key)

    if config.model.use_mup:
        model_cpu = apply_mup_initialization(model_cpu, config.model, init_key)
        lr_multipliers = get_mup_lr_multiplier_tree(model_cpu)
    else:
        lr_multipliers = None

    # jax.sharding context applies global mesh variables properly
    with mesh:
        model = shard_model(model_cpu, mesh)
        del model_cpu 

        optimizer = build_optimizer(model, config, lr_multipliers)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        checkpointer = create_checkpointer(checkpoint_dir)
        start_step, model, opt_state = load_checkpoint_if_exists(
            checkpointer, model_template=model, opt_state_template=opt_state
        )

        dataloader = build_dataloader(config, mesh, data_path)
        
        # Determine the number of microbatches expected for gradient accumulation
        replicas = config.parallelism.dp * config.parallelism.fsdp
        assert config.training.global_batch_size % (config.training.micro_batch_size * replicas) == 0, \
            "Global batch size must be a multiple of (micro_batch_size * DP * FSDP)"
        num_microbatches = config.training.global_batch_size // (config.training.micro_batch_size * replicas)
        
        # Encapsulate the JIT'd train_step closure over our static optimizer configs
        train_step = make_train_step(optimizer)

        if is_main_process: print("Starting training...")
        start_time = time.perf_counter()
        
        for step in range(start_step, config.training.train_steps):
            batch = next(dataloader)
            
            # XLA auto-compiles this lazily on its first call
            model, opt_state, metrics = train_step(
                model, opt_state, batch, num_microbatches
            )
            
            if step % 10 == 0:
                # Force async execution completion to log metrics
                loss_val = metrics["loss"].item() 
                elapsed = time.perf_counter() - start_time
                
                tok_per_sec = (config.training.global_batch_size * config.training.sequence_length * 10) / elapsed
                
                if is_main_process:
                    print(f"Step {step} | Loss: {loss_val:.4f} | Tok/s: {tok_per_sec:.2f}")
                    wandb.log({
                        "train/loss": loss_val,
                        "train/perplexity": metrics["perplexity"].item(),
                        "perf/tokens_per_sec": tok_per_sec,
                    }, step=step)
                
                start_time = time.perf_counter()

            if step > 0 and step % 1000 == 0:
                if is_main_process: print(f"Saving checkpoint at step {step}...")
                save_checkpoint(checkpointer, step, model, opt_state)
        
        dataloader.shutdown()
        checkpointer.wait_until_finished() 
        if is_main_process:
            wandb.finish()