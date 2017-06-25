clear
reset
unset key
set xlabel "Time"
set ylabel "Neuron index"
set xrange [0:1584]
set term png

plot "results/spike_pattern.log" ls 7 lc rgb 'blue' ps 0.4, 400 ls 7 lc rgb 'black' dt (10, 7), 800 ls 7 lc rgb 'black' dt (10,7)

pause -1
