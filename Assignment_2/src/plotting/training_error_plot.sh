#!/bin/bash

cat << _end_ | gnuplot
set terminal png
set output "../../results/divorce/divorce_i1000_l1_h4_e07_m1_f1_s/divorce_i1000_l1_h4_e07_m1_f1_s.png"
set key right bottom box
set xlabel "Iterations"
set ylabel "Training error"
plot '../../results/divorce/divorce_i1000_l1_h4_e07_m1_f1_s/seed_1.txt' using 1:2 t "Seed 1" w l lw 5, \\
'../../results/divorce/divorce_i1000_l1_h4_e07_m1_f1_s/seed_2.txt' using 1:2 t "Seed 2" w l lw 5, \\
'../../results/divorce/divorce_i1000_l1_h4_e07_m1_f1_s/seed_3.txt' using 1:2 t "Seed 3" w l lw 5, \\
'../../results/divorce/divorce_i1000_l1_h4_e07_m1_f1_s/seed_4.txt' using 1:2 t "Seed 4" w l lw 5, \\
'../../results/divorce/divorce_i1000_l1_h4_e07_m1_f1_s/seed_5.txt' using 1:2 t "Seed 5" w l lw 5
_end_