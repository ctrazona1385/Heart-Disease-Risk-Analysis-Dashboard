#!/usr/bin/env bash
# Azure Container Apps deployment for Heart Disease Risk Dashboard
# Prerequisites: Azure CLI installed and logged in (az login)
# Usage: bash deploy_azure.sh
set -euo pipefail

# ---- Configuration — update these before running ----
RESOURCE_GROUP="heart-disease-rg"
LOCATION="eastus"
ACR_NAME="heartdashboardcyron"        # Must be globally unique, lowercase, 5-50 chars
APP_ENV="heart-disease-env"
APP_NAME="heart-disease-dashboard"
IMAGE_TAG="latest"

echo "=== Deploying Heart Disease Dashboard to Azure Container Apps ==="
echo "Resource group : $RESOURCE_GROUP"
echo "Location       : $LOCATION"
echo "ACR            : $ACR_NAME"
echo "App            : $APP_NAME"
echo ""

# 1. Create resource group
echo "[1/6] Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# 2. Create Azure Container Registry
echo "[2/6] Creating Azure Container Registry..."
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    --output none

# 3. Build and push image using ACR Tasks (no local Docker daemon required)
echo "[3/6] Building and pushing image via ACR Tasks..."
az acr build \
    --registry "$ACR_NAME" \
    --image "heart-dashboard:$IMAGE_TAG" \
    .

# 4. Retrieve ACR credentials
echo "[4/6] Retrieving registry credentials..."
ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer --output tsv)
ACR_USER=$(az acr credential show --name "$ACR_NAME" --query username --output tsv)
ACR_PASS=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" --output tsv)

# 5. Create Container Apps managed environment
echo "[5/6] Creating Container Apps environment..."
az containerapp env create \
    --name "$APP_ENV" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# 6. Deploy Container App
echo "[6/6] Deploying Container App..."
az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$APP_ENV" \
    --image "$ACR_SERVER/heart-dashboard:$IMAGE_TAG" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --target-port 8501 \
    --ingress external \
    --cpu 1.0 \
    --memory 2.0Gi \
    --min-replicas 0 \
    --max-replicas 3 \
    --output none

echo ""
echo "=== Deployment complete ==="
echo "App URL (HTTPS):"
az containerapp show \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.configuration.ingress.fqdn" \
    --output tsv | awk '{print "https://" $0}'
