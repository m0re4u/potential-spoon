unset key
set xlabel "Number of excitatory neurons" font "Helvetica,14"
set ylabel "Accuracy" font "Helvetica,14"
set tics font "Helvetica,14"
set xtics ("400" 400, "800" 800, "1600" 1600, "3200" 3200, "6400" 6400)
set yrange [0.5:1]
set xrange[350:8000]
set logscale x
plot "results/acc_per_N.log" using 1:2:3 with yerrorbars lw 2, "results/acc_per_N.log" with lines lw 5
pause -1
