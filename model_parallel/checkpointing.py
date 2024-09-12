import os

import orbax.checkpoint as ocp


def save_checkpoint(state, log_dir):
    options = ocp.CheckpointManagerOptions(
        step_prefix="checkpoint",
        cleanup_tmp_directories=True,
        create=True,
    )
    metadata = {
        "model": "test",
    }
    item_handlers = {
        "params": ocp.StandardCheckpointHandler(),
        "metadata": ocp.JsonCheckpointHandler(),
    }
    # item_handlers["optimizer"] = ocp.StandardCheckpointHandler()
    manager = ocp.CheckpointManager(
        directory=os.path.abspath(os.path.join(log_dir, "checkpoints/")),
        item_names=tuple(item_handlers.keys()),
        item_handlers=item_handlers,
        options=options,
    )
    save_items = {
        "params": ocp.args.StandardSave(state.params),
        "metadata": ocp.args.JsonSave(metadata),
    }
    # save_items["optimizer"] = ocp.args.StandardSave(state.optimizer)
    eval_metrics = {"loss": 0.0}
    save_items = ocp.args.Composite(**save_items)
    manager.save(0, args=save_items, metrics=eval_metrics)
    manager.wait_until_finished()
    manager.close()
