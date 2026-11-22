# Shared by build.sh and run.sh: pick the project's virtualenv Python, force UTF-8 I/O.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY=${PYTHON:-python3}; fi
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
[ -f .env ] || { echo "missing .env (copy .env.example and fill it in)"; exit 1; }
