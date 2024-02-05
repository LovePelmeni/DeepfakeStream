#!/bin/bash

# formatting code using autopep8
find -type f -name '*.py' ! -exec autopep8 --in-place '{}' \;

# running linter over source directories
pylint ./src 
pylint ./tests
pylint ./monitoring
