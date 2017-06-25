clear
reset
unset key
set xlabel "Time"
set ylabel "Trace value"

plot "results/trace_values.log" using ($1-0.3346):2 with linespoints

pause -1
