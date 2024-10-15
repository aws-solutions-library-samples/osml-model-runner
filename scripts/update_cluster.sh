#!/bin/bash
#
# Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#

#
# Script to set the desired task count to 0 and disable autoscaling on all services in an ECS cluster
# that match a specified pattern.
# $1 = DESIRED_COUNT = the number of tasks to set for each service (default is 0)
# $2 = CLUSTER_NAME = the name of the ECS cluster (default is "MRCluster")
# $3 = SERVICE_PATTERN = the pattern to match services in the ECS cluster (default is "ModelRunner")
# $4 = AWS_REGION = the AWS region where the ECS cluster is located (default is "us-west-2")
#
# Example Usage:
#   ./update_cluster.sh 0 MRCluster ModelRunner us-west-2
#   This will set the desired task count to 0 and disable autoscaling on all services in the 'MRCluster' ECS cluster
#   that match the pattern "ModelRunner" in the 'us-west-2' region.
#

# Inputs
DESIRED_COUNT="${1:-0}"
CLUSTER_NAME="${2:-"MRCluster"}"
SERVICE_PATTERN="${3:-"ModelRunner"}"
AWS_REGION="${4:-"us-west-2"}"

# Get all services in the ECS cluster
SERVICES=$(aws ecs list-services --cluster "$CLUSTER_NAME" --region "$AWS_REGION" --query "serviceArns" --output text)
MODEL_RUNNER_SERVICE=$(echo "$SERVICES" | tr ' ' '\n' | grep "$SERVICE_PATTERN")

# Validate that at least one service was found
if [ -z "$MODEL_RUNNER_SERVICE" ]; then
    echo -e "${RED}No services found matching 'ModelRunner'. Exiting.${NC}"
    exit 1
else
    echo -e "${GREEN}Found the following services matching 'ModelRunner':${NC}"
    echo "$MODEL_RUNNER_SERVICE"
fi

# Set desired task count to 0 for each service
echo "Setting desired count to $DESIRED_COUNT for service: $SERVICE_NAME"
aws ecs update-service --cluster "$CLUSTER_NAME" --service "$MODEL_RUNNER_SERVICE" --desired-count $DESIRED_COUNT --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    echo "Failed to update service: $MODEL_RUNNER_SERVICE"
    exit 1
fi

echo "Desired count set to $DESIRED_COUNT for all services."

# Disable all scaling policies associated with the ECS services
echo "Disabling autoscaling policies..."

AUTO_SCALING_GROUPS=$(aws application-autoscaling describe-scalable-targets \
--service-namespace ecs --region "$AWS_REGION" \
--query "ScalableTargets[?ResourceId.contains(@, 'service/$CLUSTER_NAME')].ResourceId" \
--output text
)

for ASG in $AUTO_SCALING_GROUPS; do
    echo "Deregistering scalable target: $ASG"
    aws application-autoscaling deregister-scalable-target \
    --service-namespace ecs \
    --resource-id "$ASG" \
    --scalable-dimension ecs:service:DesiredCount \
    --region "$AWS_REGION"

    if [ $? -ne 0 ]; then
        echo "Failed to disable autoscaling for: $ASG"
        exit 1
    fi
done

echo "Autoscaling policies disabled."

echo "Cluster $CLUSTER_NAME is now set to have $DESIRED_COUNT running tasks, and autoscaling is turned off."
