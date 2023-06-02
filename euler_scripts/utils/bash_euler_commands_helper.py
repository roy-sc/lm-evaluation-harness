import json
import randomname
from datetime import datetime
from typing import List
from argparse import ArgumentParser


def get_config_dict(path):
    with open(path, 'r') as json_config_file:
        json_config = json.load(json_config_file)
        return json_config


def get_euler_command(
        log_file: str = "euler_run",
        n_cpus: int = 16,
        time: str = "48:00",
        memory: int = 2,
        n_gpus: int = 1,
        gpu_model: str = "NVIDIATITANRTX",  # "NVIDIAGeForceGTX1080Ti",
        group: str = "es_mtc",
        main_command: str = False
):
    """ Builds a command to run on euler. """

    command = "bsub "
    command += f" -o {log_file}"
    if time is not None:
        command += f" -W {time} "
    if n_cpus is not None:
        command += f" -n {n_cpus} "
    if group is not None:
        command += f" -G {group} "
    if memory is not None:
        command += f" -R \"rusage[mem={memory}G]\" "
    if n_gpus is not None:
        command += f" -R \"rusage[ngpus_excl_p={n_gpus}]\" "
    if gpu_model is not None:
        command += f" -R \"select[gpu_model0=={gpu_model}]\" "
    if main_command is not None:
        command += main_command
    command = command.replace("  ", " ")
    return command


def get_slurm_euler_command(
        log_file: str = "euler_run",
        n_cpus: int = 16,
        time: str = "48:00",
        memory_per_core: int = 4,
        n_gpus: int = 1,
        gpu_model: str = "NVIDIATITANRTX",
        memory_per_gpu: int = 20,
        group: str = "es_mtc",
        main_command: str = False
):
    """ Builds a slurm command to run on euler. """

    command = "sbatch "
    command += f" -o {log_file}"
    if time is not None:
        command += f" --time={time}:00 "
    if n_cpus is not None:
        command += f" -n {n_cpus} "
    if group is not None:
        command += f" -A {group} "
    if memory_per_core is not None:
        command += f" --mem-per-cpu={memory_per_core}G "
    if gpu_model is not None:
        command += f" --gpus={gpu_model}:{n_gpus} "
    else:
        if n_gpus is not None:
            command += f" --gpus={n_gpus} "
        if memory_per_gpu is not None:
            command += f" --gres=gpumem:{memory_per_gpu}G "
    if main_command is not None:
        command += f" --wrap=\"{main_command}"
    command = command.replace("  ", " ")
    return command


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to euler config file")
    parser.add_argument("--test_locally", action="store_true",
                        help="Whether to test the command locally on euler")

    args = parser.parse_args()
    json_config = get_config_dict(args.config_path)
    log_file_path = json_config["log_file_path"]
    num_cpus = json_config["num_cpus"]
    run_duration_hours = json_config["run_duration_hours"]
    shareholder_group = json_config["shareholder_group"]
    memory_per_cpu_gb = json_config["memory_per_cpu_gb"]
    gpu = json_config["gpu"]
    num_gpus = json_config["num_gpus"]
    main_commmand = json_config["main_command"]

    log_file = log_file_path + randomname.get_name() + "-" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.test_locally:
        command = main_commmand
        print("Local command executed")
    else:
        # command = get_euler_command(log_file=log_file, n_cpus=num_cpus, time=run_duration_hours, memory=memory_gb,
        #                                   gpu_model=gpu,
        #                                   group=shareholder_group, main_command=main_commmand)
        command = get_slurm_euler_command(log_file=log_file, n_cpus=num_cpus, time=run_duration_hours,
                                          memory_per_core=memory_per_cpu_gb,
                                          gpu_model=gpu, n_gpus=num_gpus,
                                          group=shareholder_group, main_command=main_commmand)
    print(command)


if __name__ == "__main__":
    main()
