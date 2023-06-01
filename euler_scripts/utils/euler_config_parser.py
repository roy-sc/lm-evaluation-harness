import json
from argparse import ArgumentParser


def retreive(path, retreive_key):
    with open(path, "r") as f:
        json_f = json.loads(f.read())
        return json_f[retreive_key][:-1] if json_f[retreive_key][-1] == "/" \
            else json_f[retreive_key]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--retreive_key", type=str, help="Entry to retreive",
                        choices=["project_path", "code_path", "load_modules", "config_file"])
    parser.add_argument("--euler_config_path", type=str,
                        help="Path to euler config")
    args = parser.parse_known_args()[0]
    print(retreive(args.euler_config_path, args.retreive_key))
    exit(0)
