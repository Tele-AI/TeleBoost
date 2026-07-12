from teleboost.training.core.loop import epoch_for_training_step, should_continue_training


def test_epoch_index_for_one_based_training_steps():
    assert [epoch_for_training_step(step, 1) for step in range(1, 6)] == [0, 1, 2, 3, 4]
    assert [epoch_for_training_step(step, 2) for step in range(1, 6)] == [0, 0, 1, 1, 2]


def test_total_training_steps_drives_loop_beyond_configured_epochs():
    seen_steps = []
    global_step = 1
    while should_continue_training(global_step, 5):
        seen_steps.append(global_step)
        global_step += 1

    assert seen_steps == [1, 2, 3, 4, 5]
