#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/highways/mnist-10layers/mnist_solver.prototxt
