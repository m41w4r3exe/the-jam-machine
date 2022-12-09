import os


def define_generation_dir(model_repo_path):
    #### to remove later ####
    if model_repo_path == "models/model_2048_fake_wholedataset":
        model_repo_path = "misnaej/the-jam-machine"
    #### to remove later ####
    generated_sequence_files_path = f"midi/generated/{model_repo_path}"
    if not os.path.exists(generated_sequence_files_path):
        os.makedirs(generated_sequence_files_path)
    return generated_sequence_files_path


def bar_count_check(sequence, n_bars):
    """check if the sequence contains the right number of bars"""
    sequence = sequence.split(" ")
    # find occurences of "BAR_START" in a str
    bar_count = 0
    for seq in sequence:
        if seq == "BAR_END":
            bar_count += 1
    bar_count_matches = bar_count == n_bars
    if not bar_count_matches:
        print(f"Bar count is {bar_count} - but should be {n_bars}")
        print(f"-------------------")
    return bar_count_matches, bar_count
