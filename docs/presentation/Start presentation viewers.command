#!/bin/zsh
set -e
task_project_dir="${0:A:h:h:h}"
cd -- "$task_project_dir"
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Python 3 is required to start the presentation viewers.'
  exit 1
fi
python3 src/neuristor/interfaces/presentation_server.py --present
