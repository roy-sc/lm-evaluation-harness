import argparse
import json
import logging
import fnmatch
import os

import jsonargparse
import pandas as pd
import randomname as randomname
import wandb as wandb

from lm_eval import tasks, evaluator, utils

logging.getLogger("openai").setLevel(logging.WARNING)


def _is_json_task(task_name):
    return task_name == "json" or task_name.startswith("json=")


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0 and not _is_json_task(
                    value
            ):
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--prompt_version_per_task", type=str, default=None)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--wandb_on", type=bool, default=False)
    parser.add_argument('--config', action=jsonargparse.ActionConfigFile)

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        if _is_json_task(pattern):
            task_names.add(pattern)

        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    if args.wandb_on:
        wandb_mode = "online"
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        if not args.write_out:
            print("Wandb can only save task output, when write out is true. Therefore write out is set to true")
            args.write_out = True
    else:
        wandb_mode = "disabled"

    tasks_string = "TASK_" + "-".join(task_names)
    model_args_dict = utils.simple_parse_args_string(args.model_args)
    model_id = model_args_dict["pretrained"].replace("/", "-")
    model_string = f"MODEL_{model_id}"
    few_shot_string = f"{args.num_fewshot}-SHOT"

    args.output_base_path = os.path.join(args.output_base_path, model_id)

    wandb_run_name = randomname.get_name() + '_' + '_'.join(
        [tasks_string, model_string, few_shot_string])

    wandb_run_group_name = f"llm_leaderboard_{tasks_string}_group"
    # entity="background-tool",
    wandb.init(project="llm_leaderboard",  config=vars(args), name=wandb_run_name,
               mode=wandb_mode, group=wandb_run_group_name)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_id=model_args_dict["pretrained"],
        model_args=args.model_args,
        tasks=task_names,
        prompt_version_per_task=args.prompt_version_per_task,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )

    if args.wandb_on:
        wandb.log(results["results"])
        write_out_info = results["write_out_info"]
        wandb.Table.MAX_ARTIFACTS_ROWS = 600000
        for task_name, task_output in write_out_info.items():
            df = pd.DataFrame.from_dict(task_output)
            task_table = wandb.Table(dataframe=df)
            table_name = task_name + "_output_table"
            wandb.log({table_name: task_table})
        print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
