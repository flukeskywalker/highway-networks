#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/highways/cifar10-19layers/cifar10_solver.prototxt