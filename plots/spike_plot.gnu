clear
reset
unset key
set xlabel "Time" font "Helvetica,14"
set ylabel "Neuron index" font "Helvetica,14"
set yrange [0:1584]
set term qt size 560, 560

plot "results/spike_pattern.log" ls 7 lc rgb 'blue' ps 0.4, 400 ls 7 lc rgb 'black' dt (10, 7), 800 ls 7 lc rgb 'black' dt (10,7)

pause -1
