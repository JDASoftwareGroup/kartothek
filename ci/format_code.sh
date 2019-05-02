#!/bin/bash -xe
black .
isort -y -rc .
flake8
