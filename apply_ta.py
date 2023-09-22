from task_vectors.src.task_vectors import TaskVector


def get_model(pretrained_checkpoint, list_of_task_checkpoints, scaling_coef):
    task_vector_list = [
        TaskVector(pretrained_checkpoint, task_checkpoint)
        for task_checkpoint in list_of_task_checkpoints
    ]
    vector_sum = sum(task_vector_list)
    model = vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    return model
