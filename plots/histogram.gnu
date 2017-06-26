clear
reset
set xrange [0:2000]
set yrange [0:103]
set xlabel "Image classification duration" font "Helvetica,14"
set ylabel "Count" font "Helvetica,14"
set key samplen 10 spacing 5 font ",14"
set tics font "Helvetica,14"


binwidth = 7
load 'hist.fct'
set style fill transparent solid 0.5
plot "../results/ATOM_ROBO_runtimes.log" i 0 @hist ls 2 title "Atom", "../results/JETSON_OPT_runtimes.log" i 0 @hist ls 7 title "Jetson", "../results/LAPTOP_OPT_runtimes.log" i 0 @hist ls 3 lw 0.1 title "i5


pause -1
