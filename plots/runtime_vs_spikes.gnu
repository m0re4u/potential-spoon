clear
reset
set xrange [0:65000]
set yrange [0:1500]
set xlabel "Number of spikes during image presentation" font "Helvetica,14"
set ylabel "Image classification duration" font "Helvetica,14"
set key samplen 10 spacing 5 font ",14"
set tics font "Helvetica,14"

f(x) = a*x + b
g(x) = c*x + d
# h(x) = e*x + f

fit f(x) "results/LAPTOP_OPT1_59999_30000_1000_4CORE.log" via a,b
fit g(x) "results/JETSON_OPT1_59999_30000_1000_4CORE.log" via c,d
# fit h(x) "results/ATOM_ROBO_59999_30000_1000_1CORE.log" via e,f

plot "results/LAPTOP_OPT1_59999_30000_1000_4CORE.log" title "i5" ps 1,\
     "results/JETSON_OPT1_59999_30000_1000_4CORE.log" title "Jetson" ps 1,\
     f(x) lw 3 title "i5 model",\
     g(x) lw 3 title "Jetson model"
     # "results/ATOM_ROBO_59999_30000_1000_1CORE.log" title "Atom" ps 1,\
     # h(x) lw 3

pause -1
