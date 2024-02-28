#!/bin/bash

pyreverse -o dot -p concrete . --colorized

cd abstract_classes && pyreverse -o dot -p abstract .  --colorized && cd .. 

dot_files="$(find . -name "*.dot")"

for dot_file in $dot_files
do
	unflatten -f -l 4 -c 3 $dot_file | dot | gvpack -g | neato -s -n2 -Tpng -o ${dot_file%%.dot}.png
done
