#!/bin/bash

# Utility script to download datasets hosted on github respositories.

# Usage: bash scripts/download_data.sh <github_repo_url> <folder_path_in_repo> <local_target_directory>

REPO_URL="$1"
FOLDER_PATH="$2"
LOCAL_DIR="$3"

if [ -z "$REPO_URL" ] || [ -z "$FOLDER_PATH" ] || [ -z "$LOCAL_DIR" ]; then
  echo "Usage: $0 <github_repo_url> <folder_path_in_repo> <local_target_directory>"
  echo "Example: $0 https://github.com/user/repo path/to/folder ./my_downloads"
  exit 1
fi

REPO_API_PATH=$(echo "$REPO_URL" | sed -E 's|https://github.com/([^/]+)/([^/]+).*|\1/\2|')
OWNER=$(echo "$REPO_API_PATH" | cut -d/ -f1)
REPO=$(echo "$REPO_API_PATH" | cut -d/ -f2)

DEFAULT_BRANCH=$(curl -s "https://api.github.com/repos/$OWNER/$REPO" | jq -r .default_branch)

API_URL="https://api.github.com/repos/$OWNER/$REPO/contents/$FOLDER_PATH?ref=$DEFAULT_BRANCH"
RESPONSE=$(curl -s "$API_URL")

if echo "$RESPONSE" | grep -q '"message": "Not Found"'; then
  echo "❌ Folder not found. Check your repo URL and folder path."
  exit 1
fi

mkdir -p "$LOCAL_DIR"

echo "$RESPONSE" | jq -c '.[]' | while read -r item; do
  TYPE=$(echo "$item" | jq -r '.type')
  NAME=$(echo "$item" | jq -r '.name')
  DOWNLOAD_URL=$(echo "$item" | jq -r '.download_url')

  if [ "$TYPE" = "file" ]; then
    echo "⬇️  Downloading $NAME..."
    wget -q --show-progress "$DOWNLOAD_URL" -O "$LOCAL_DIR/$NAME"
  fi
done

echo "✅ Files downloaded to $LOCAL_DIR"
