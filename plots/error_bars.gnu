unset key
set xlabel "Number of training iterations" font "Helvetica,14"
set ylabel "Accuracy" font "Helvetica,14"
set tics font "Helvetica,14"
set yrange [0:1]
set xrange[0.6:19001]
set logscale x
plot "results/acc_per_its.log" using 1:2:3 with yerrorbars lw 2, "results/acc_per_its.log" with lines lw 5
pause -1
