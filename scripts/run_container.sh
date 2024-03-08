#!/bin/bash
#
# Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#

# This is a utility to run a local model runner container with an ENV imported from ECS.
# $1 = PATTERN = the pattern contained in the ECS task definition you want to import from
# $2 = CONTAINER_NAME = the name of the docker image you want to run
# $3 = AWS_REGION = region the ecs task definition is contained in

echo "    ____                    _                __                     __";
echo "   / __ \__  ______  ____  (_)___  ____ _   / /   ____  _________ _/ /";
echo "  / /_/ / / / / __ \/ __ \/ / __ \/ __  /  / /   / __ \/ ___/ __  / /";
echo " / _, _/ /_/ / / / / / / / / / / / /_/ /  / /___/ /_/ / /__/ /_/ / / ";
echo "/_/ |_|\__,_/_/ /_/_/ /_/_/_/ /_/\__, /  /_____/\____/\___/\__,_/_/";
echo "    __  _______     ______      /____/__        _";
echo "   /  |/  / __ \   / ____/___  ____  / /_____ _(_)___  ___  _____";
echo "  / /|_/ / /_/ /  / /   / __ \/ __ \/ __/ __  / / __ \/ _ \/ ___/";
echo " / /  / / _, _/  / /___/ /_/ / / / / /_/ /_/ / / / / /  __/ /";
echo "/_/  /_/_/ |_|   \____/\____/_/ /_/\__/\__,_/_/_/ /_/\___/_/ ";


# Inputs
PATTERN="${1:-"MRDataplane"}"
IMAGE_NAME=PATTERN="${2:-"osml-model-runner:local"}"
AWS_REGION="${3:-"us-west-2"}"

# Get the latest task definition ARN based on a string pattern
LATEST_TASK_DEFINITION_ARN=$(aws ecs list-task-definitions --region $AWS_REGION --sort DESC | jq -r ".taskDefinitionArns[] | select(. | contains(\"$PATTERN\"))" | head -n 1)

if [ -z "$LATEST_TASK_DEFINITION_ARN" ]; then
    echo "No task definition found with pattern: $PATTERN"
    exit 1
fi

echo "Latest task definition ARN with pattern $PATTERN is: $LATEST_TASK_DEFINITION_ARN"

# Extract environment variables from ECS task definition using AWS CLI and jq
ENV_VARS=$(aws ecs describe-task-definition --region $AWS_REGION --task-definition "$LATEST_TASK_DEFINITION_ARN" | jq -r '.taskDefinition.containerDefinitions[0].environment[] | "-e " + .name + "=\"" + .value + "\"" ' | tr '\n' ' ')

# If aws cli or jq command fails
if [ $? -ne 0 ]; then
    echo "Failed to fetch environment variables from ECS task definition."
    exit 1
fi

echo "Running with env variables: $ENV_VARS"

# Give the container permissions on aws credentials
echo "Warning: Giving 777 permissions on ~/.aws/credentials - revert to desired permissions if needed"
chmod 777 ~/.aws/credentials

# Note: The `-d` flag runs the container in detached mode.
# Add it if you want to run the container in the background.
# Run Docker container with environment variables and mount AWS credentials
DOCKER_CMD="docker run \
  -p 8080:8080 \
  -v ~/.aws/credentials:/home/modelrunner/.aws/credentials:rw \
  $ENV_VARS \
  $IMAGE_NAME"

# Run Docker container with environment variables
eval "$DOCKER_CMD"
