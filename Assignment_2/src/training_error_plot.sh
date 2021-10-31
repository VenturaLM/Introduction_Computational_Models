#!/bin/bash

cat << _end_ | gnuplot
set terminal postscript eps color
set output "training_error_plot.eps"
set key right bottom box
set xlabel "Iterations"
set ylabel "Training error"
plot 'seed1.txt' using 1:2 t "Seed 1" w l lw 5, 'seed2.txt' using 1:2 t "Seed 2" w l lw 5, 'seed3.txt' using 1:2 t "Seed 3" w l lw 5
_end_