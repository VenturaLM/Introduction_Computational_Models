#!/bin/bash

cat << _end_ | gnuplot
set terminal png
set output "../../results/nomnist/extra3_nomnist_i500_l2_h16_e07_m1_f1_s/extra3_nomnist_i500_l2_h16_e07_m1_f1_s.png"
set key right top box
set xlabel "Iterations"
set ylabel "Training error"
plot '../../results/nomnist/extra3_nomnist_i500_l2_h16_e07_m1_f1_s/seed_1.txt' using 1:2 t "Seed 1" w l lw 2, \\
'../../results/nomnist/extra3_nomnist_i500_l2_h16_e07_m1_f1_s/seed_2.txt' using 1:2 t "Seed 2" w l lw 2, \\
'../../results/nomnist/extra3_nomnist_i500_l2_h16_e07_m1_f1_s/seed_3.txt' using 1:2 t "Seed 3" w l lw 2, \\
'../../results/nomnist/extra3_nomnist_i500_l2_h16_e07_m1_f1_s/seed_4.txt' using 1:2 t "Seed 4" w l lw 2, \\
'../../results/nomnist/extra3_nomnist_i500_l2_h16_e07_m1_f1_s/seed_5.txt' using 1:2 t "Seed 5" w l lw 2
_end_