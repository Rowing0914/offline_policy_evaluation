import tensorflow as tf


def eager_setup():
    """
    it enables an eager execution in tensorflow with config that allows us to flexibly access to a GPU
    from multiple python scripts
    """

    # === before TF 2.0 ===
    # config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
    #                                   intra_op_parallelism_threads=1,
    #                                   inter_op_parallelism_threads=1)
    # config.gpu_options.allow_growth = True
    # tf.compat.v1.enable_eager_execution(config=config)
    # tf.compat.v1.enable_resource_variables()

    # === For TF 2.0 ===
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession(config=config)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # TODO: if you don't need it, remove!!


def create_checkpoint(model, optimizer, model_dir, verbose=False):
    """ Create a checkpoint for managing a model

    :param model: TF Neural Network
    :param optimizer: TF optimisers
    :param model_dir: a directory to save the optimised weights and other checkpoints
    :return manager: a manager to control the save timing
    """
    checkpoint_dir = model_dir
    check_point = tf.train.Checkpoint(optimizer=optimizer,
                                      model=model,
                                      optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    manager = tf.train.CheckpointManager(check_point, checkpoint_dir, max_to_keep=3)

    if verbose:
        # try re-loading the previous training progress!
        try:
            print("Try loading the previous training progress")
            check_point.restore(manager.latest_checkpoint)
            print("===================================================\n")
            print("Restored the model from {}".format(checkpoint_dir))
            print("Currently we are on Epoch: {}".format(tf.compat.v1.train.get_global_step().numpy()))
            print("\n===================================================")
        except:
            print("===================================================\n")
            print("Previous Training files are not found in Directory: {}".format(checkpoint_dir))
            print("\n===================================================")
    return manager