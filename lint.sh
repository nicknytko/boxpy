#!/bin/bash

echo "... flake8 ..."
python3 -m flake8 --docstring-convention numpy --statistics boxpy && echo "flake8 passed."
echo
echo "... pylint ..."
pylint --rcfile setup.cfg boxpy
