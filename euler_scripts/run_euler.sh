# Provide as first argument the path to the euler config file

# Provide as optional second argument the path to the inference prompt file

# original example: bash run_euler.sh configs/flan_t5_inference_config.json data/inference.json

# example: start locally (not on euler) inside the `euler_scripts` folder, and execute: (no need for an inference.json)
# bash run_euler.sh lm_eval_euler_config.json

if [ -z "$1" ]
  then
    echo "No euler_config.json path provided"
    exit 1
fi

CURR_BRANCH=$(git rev-parse --abbrev-ref HEAD)

LOAD_MODULES=$(python3 utils/euler_config_parser.py \
                       --euler_config_path "$1" \
                       --retreive_key load_modules)

PROJECT_PATH=$(python3 utils/euler_config_parser.py \
                       --euler_config_path "$1" \
                       --retreive_key project_path)

CODE_PATH=$(python3 utils/euler_config_parser.py \
                --euler_config_path "$1" \
                --retreive_key code_path)

CONFIG_FILE=$(python3 utils/euler_config_parser.py \
                --euler_config_path "$1" \
                --retreive_key config_file)

scp "$1" euler:$CODE_PATH/euler_scripts
#scp utils/bash_euler_commands_helper.py euler:$CODE_PATH/euler_scripts/utils/bash_euler_commands_helper.py
#scp ../lm_eval/tasks/xsum_faith_hallucination_classification.py euler:$CODE_PATH/lm_eval/tasks/xsum_faith_hallucination_classification.py
scp ../lm_eval/tasks/llm_summarization_mt.py euler:$CODE_PATH/lm_eval/tasks/llm_summarization_mt.py
#scp ../lm_eval/models/huggingface.py euler:$CODE_PATH/lm_eval/models/huggingface.py
#scp ../main.py euler:$CODE_PATH/main.py
scp ../"$CONFIG_FILE" euler:$CODE_PATH/$CONFIG_FILE

if [ -z "$2" ]
  then
    echo "No inference.json path provided"
  else
    scp ../"$2" euler:"$CODE_PATH"/"$2"
fi

#STR_COMMAND=$(python3 utils/bash_euler_commands_helper.py --config_path $1)
#echo $STR_COMMAND

ssh euler ARG1=\"$1\" \
          ARG4=\"$LOAD_MODULES\" \
          ARG5=\"$CODE_PATH\" \
          ARG6=\"$CURR_BRANCH\" \
          ARG7=\"$PROJECT_PATH\" \
          ARG8=\"$CONFIG_FILE\" \
          'bash -s' <<'ENDSSH'

    # Change to work dir
    echo "### Changing to project dir..."
    cd "$ARG7" || exit

    # Export to avoid relativ folder import errors in python
    echo "### Add to python path: $ARG7/"
    export PYTHONPATH="${PYTHONPATH}:$ARG7/"

    # Load all updates
    echo "### Pulling commits..."
    echo ""
    git pull
    git checkout "$ARG6"
    echo ""

    # LOAD MODULES AFTER ACTIVATING ENVIRONMENT TO AVOID LIBRARY ERRORS!
    echo "### Loading modules..."
    eval module load "$ARG4"
    # UNLOADING ETH_PROXY MODULES FOR WANDB
    # echo "Unloading eth_proxy module for wandb"
    # eval module unload eth_proxy
    echo ""

    # Activate environment
    echo "### Activating environment..."
    source env/bin/activate
    echo ""

    # Change to code dir
    echo "### Changing to code dir..."
    cd "$ARG5" || exit
    echo ""

    # Get commands
    echo "### Retrieving commands to execute..."
    echo "$ARG1"
    STR_COMMAND=$(python3 euler_scripts/utils/bash_euler_commands_helper.py --config_path euler_scripts/"$ARG1")
    echo ""

    # Run all the commands
    echo "### Running command..."
    CMD="$STR_COMMAND $ARG8 \""
    echo "$CMD"
    eval "$CMD"
    echo ""
    sleep 1

ENDSSH