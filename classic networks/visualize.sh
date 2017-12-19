#!/bin/bash

tensorboard --logdir="./graph" --port 6006
mv ./graph/events* ./bak

