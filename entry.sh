#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_DIR/duet-of-models}"
DATA_ARCHIVE_URL="${DATA_ARCHIVE_URL:-}"
DATA_ARCHIVE_ID="${DATA_ARCHIVE_ID:-1HnWofOd1lec7uO26PdurTEvli6LKgG7X}"
DATA_ARCHIVE_PATH="${DATA_ARCHIVE_PATH:-$WORKSPACE_DIR/data.tar.gz}"
DATA_EXTRACT_DIR="${DATA_EXTRACT_DIR:-$REPO_DIR}"
RUN_AFTER_SETUP="${RUN_AFTER_SETUP:-}"
KEEP_ALIVE="${KEEP_ALIVE:-1}"

mkdir -p "$WORKSPACE_DIR"

# Preparing repository
if [[ -n "$REPO_URL" ]]; then
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "Updating repository in $REPO_DIR"
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH"
  else
    echo "Cloning $REPO_URL into $REPO_DIR"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
  fi
elif [[ -d "$REPO_DIR" ]]; then
  echo "Using existing repository at $REPO_DIR"
else
  echo "REPO_URL is unset and $REPO_DIR does not exist. Skipping repository setup."
fi

# Downloading data
if [[ -n "$DATA_ARCHIVE_URL" || -n "$DATA_ARCHIVE_ID" ]]; then
  mkdir -p "$(dirname "$DATA_ARCHIVE_PATH")" "$DATA_EXTRACT_DIR"
  if [[ -n "$DATA_ARCHIVE_ID" ]]; then
    echo "Downloading Google Drive archive id $DATA_ARCHIVE_ID"
    gdown "$DATA_ARCHIVE_ID" --output "$DATA_ARCHIVE_PATH"
  else
    echo "Downloading Google Drive archive from $DATA_ARCHIVE_URL"
    gdown "$DATA_ARCHIVE_URL" --output "$DATA_ARCHIVE_PATH"
  fi
  # extracting data
  case "$DATA_ARCHIVE_PATH" in
  *.tar.gz | *.tgz)
    tar -xzf "$DATA_ARCHIVE_PATH" -C "$DATA_EXTRACT_DIR"
    ;;
  *.tar)
    tar -xf "$DATA_ARCHIVE_PATH" -C "$DATA_EXTRACT_DIR"
    ;;
  *.zip)
    unzip -o "$DATA_ARCHIVE_PATH" -d "$DATA_EXTRACT_DIR"
    ;;
  *)
    echo "Downloaded $DATA_ARCHIVE_PATH but do not know how to extract this extension."
    ;;
  esac
fi

# preparing xlstm env
if [[ -d "$REPO_DIR/envs/xlstm" ]]; then
  echo "Syncing xLSTM uv environment"
  uv sync --project "$REPO_DIR/envs/xlstm"
fi

if [[ -n "$RUN_AFTER_SETUP" ]]; then
  cd "$REPO_DIR"
  bash -lc "$RUN_AFTER_SETUP"
elif [[ "$KEEP_ALIVE" == "1" ]]; then
  echo "Setup complete. Keeping container alive for Vast.ai SSH."
  tail -f /dev/null
fi
