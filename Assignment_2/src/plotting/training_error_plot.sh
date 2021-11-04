#!/bin/bash

cat << _end_ | gnuplot
set terminal postscript eps color
set output "../../results/xor_i1000_l1_h8_e07_m1_f1_s/xor_i1000_l1_h8_e07_m1_f1_s.eps"
set key right bottom box
set xlabel "Iterations"
set ylabel "Training error"
plot '../../results/xor_i1000_l1_h8_e07_m1_f1_s/seed_1.txt' using 1:2 t "Seed 1" w l lw 5, \\
'../../results/xor_i1000_l1_h8_e07_m1_f1_s/seed_2.txt' using 1:2 t "Seed 2" w l lw 5, \\
'../../results/xor_i1000_l1_h8_e07_m1_f1_s/seed_3.txt' using 1:2 t "Seed 3" w l lw 5, \\
'../../results/xor_i1000_l1_h8_e07_m1_f1_s/seed_4.txt' using 1:2 t "Seed 4" w l lw 5, \\
'../../results/xor_i1000_l1_h8_e07_m1_f1_s/seed_5.txt' using 1:2 t "Seed 5" w l lw 5
_end_