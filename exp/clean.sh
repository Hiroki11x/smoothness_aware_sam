#!/bin/bash
directory="."
find "$directory" -type f -name "*.out" -exec rm -f {} \;
find "$directory" -type f -name "*.err" -exec rm -f {} \;
