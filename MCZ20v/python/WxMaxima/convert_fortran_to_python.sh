#!/bin/bash

# Arrays
perl -pe 's/KK20\((\d+),(\d+)\)/"KK[" . ($1-1) . "," . ($2-1) . "]"/eg' Cleaned_K_code.txt > K_python.txt
perl -pe 's/DD\((\d+),(\d+)\)/"DD[" . ($1-1) . "," . ($2-1) . "]"/eg' Cleaned_DD_code.txt > DD_python.txt
perl -pe 's/BBm\((\d+),(\d+)\)/"BB[" . ($1-1) . "," . ($2-1) . "]"/eg' Cleaned_BB_code.txt > BB_python.txt
perl -pe 's/DD\((\d+),(\d+)\)/"DD[" . ($1-1) . "," . ($2-1) . "]"/eg' Cleaned_D_code.txt > D_python.txt

# Vectors
perl -pe 's/\bFF20\s*\(\s*(\d+)\s*,\s*1\s*\)/"F[" . ($1 - 1) . "]"/eg' Cleaned_F_code.txt > F_python.txt


