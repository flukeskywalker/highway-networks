#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/highways/cifar100-19layers/cifar100_solver.prototxt
