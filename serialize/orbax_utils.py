import os
import orbax.checkpoint as ocp
import jax

def create_checkpointer(checkpoint_dir: str, keep_last: int = 3) -> ocp.CheckpointManager:
    abs_path = os.path.abspath(checkpoint_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=keep_last, create=True)
    return ocp.CheckpointManager(abs_path, options=options)

def save_checkpoint(mngr: ocp.CheckpointManager, step: int, model: jax.tree_util.PyTreeDef, opt_state: jax.tree_util.PyTreeDef):
    state = {"model": model, "opt_state": opt_state}
    # FIX: Use PyTreeSave for modern Orbax (>= 0.4.x)
    mngr.save(step, args=ocp.args.PyTreeSave(state))

def load_checkpoint_if_exists(mngr: ocp.CheckpointManager, model_template, opt_state_template):
    latest_step = mngr.latest_step()
    if latest_step is None:
        print("No checkpoint found. Starting from scratch.")
        return 0, model_template, opt_state_template

    print(f"Restoring from checkpoint step {latest_step}...")
    target = {"model": model_template, "opt_state": opt_state_template}
    
    # FIX: Use PyTreeRestore for modern Orbax compatibility 
    restored = mngr.restore(latest_step, args=ocp.args.PyTreeRestore(target))
    
    return latest_step, restored["model"], restored["opt_state"]