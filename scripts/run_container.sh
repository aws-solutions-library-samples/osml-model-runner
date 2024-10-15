#!/bin/bash
#
# Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#

# This is a utility to run a local model runner container with an ENV imported from ECS.
# $1 = PATTERN = the pattern contained in the ECS task definition you want to import from
# $2 = CONTAINER_NAME = the name of the docker image you want to run
# $3 = AWS_REGION = region the ecs task definition is contained in

# Define colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Define log level colors
INFO_COLOR='\033[0;36m'    # Cyan for INFO
WARN_COLOR='\033[0;33m'    # Yellow for WARN
ERROR_COLOR='\033[0;31m'   # Red for ERROR
DEBUG_COLOR='\033[0;35m'   # Purple for DEBUG

echo -e "${CYAN}"
echo "    ____                    _                __                     __"
echo "   / __ \__  ______  ____  (_)___  ____ _   / /   ____  _________ _/ /"
echo "  / /_/ / / / / __ \/ __ \/ / __ \/ __  /  / /   / __ \/ ___/ __  / /"
echo " / _, _/ /_/ / / / / / / / / / / / /_/ /  / /___/ /_/ / /__/ /_/ / / "
echo "/_/ |_|\__,_/_/ /_/_/ /_/_/_/ /_/\__, /  /_____/\____/\___/\__,_/_/"
echo "    __  _______     ______      /____/__        _"
echo "   /  |/  / __ \   / ____/___  ____  / /_____ _(_)___  ___  _____"
echo "  / /|_/ / /_/ /  / /   / __ \/ __ \/ __/ __  / / __ \/ _ \/ ___/"
echo " / /  / / _, _/  / /___/ /_/ / / / / /_/ /_/ / / / / /  __/ /"
echo "/_/  /_/_/ |_|   \____/\____/_/ /_/\__/\__,_/_/_/ /_/\___/_/ "
echo -e "${NC}"

# Inputs
PATTERN="${1:-"MRDataplane"}"
IMAGE_NAME="${2:-"osml-model-runner:local"}"
AWS_REGION="${3:-"us-west-2"}"
LOG_FILE="model_runner.log"

# Get the latest task definition ARN based on a string pattern
LATEST_TASK_DEFINITION_ARN=$(aws ecs list-task-definitions --region "$AWS_REGION" --sort DESC | jq -r ".taskDefinitionArns[] | select(. | contains(\"$PATTERN\"))" | head -n 1)

if [ -z "$LATEST_TASK_DEFINITION_ARN" ]; then
    echo -e "${RED}No task definition found with pattern: $PATTERN${NC}"
    exit 1
fi

echo -e "${GREEN}Latest task definition ARN with pattern $PATTERN is:${NC} $LATEST_TASK_DEFINITION_ARN"

# Extract environment variables from ECS task definition using AWS CLI and jq
ENV_VARS=$(aws ecs describe-task-definition --region "$AWS_REGION" --task-definition "$LATEST_TASK_DEFINITION_ARN" | jq -r '.taskDefinition.containerDefinitions[0].environment[] | "-e " + .name + "=\"" + .value + "\"" ' | tr '\n' ' ')

# If aws cli or jq command fails
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to fetch environment variables from ECS task definition.${NC}"
    exit 1
fi

echo -e "${YELLOW}Running with environment variables:${NC} $ENV_VARS"

# Give the container permissions on aws credentials
echo -e "${YELLOW}Warning: Giving 777 permissions on ~/.aws/credentials - revert to desired permissions if needed${NC}"
chmod 777 ~/.aws/credentials

# Note: The `-d` flag runs the container in detached mode.
# Add it if you want to run the container in the background.
# Run Docker container with environment variables and mount AWS credentials
DOCKER_CMD="docker run \
  -p 8080:8080 \
  -v ~/.aws/credentials:/home/modelrunner/.aws/credentials:rw \
  $ENV_VARS \
  $IMAGE_NAME > $LOG_FILE 2>&1 &"

# Display the command to be run for debugging
echo -e "${CYAN}Executing Docker Command:${NC}"
echo "$DOCKER_CMD"

# Run Docker container with environment variables
eval "$DOCKER_CMD"

# Parse the log file for better readability and add colors
tail -f "$LOG_FILE" | awk -v info_color="$INFO_COLOR" -v warn_color="$WARN_COLOR" -v error_color="$ERROR_COLOR" -v debug_color="$DEBUG_COLOR" -v nc="$NC" '
{
    if ($0 ~ /INFO/) {print info_color "[INFO] " $0 nc}
    else if ($0 ~ /WARN/) {print warn_color "[WARN] " $0 nc}
    else if ($0 ~ /ERROR/) {print error_color "[ERROR] " $0 nc}
    else if ($0 ~ /DEBUG/) {print debug_color "[DEBUG] " $0 nc}
    else {print $0}
}'
