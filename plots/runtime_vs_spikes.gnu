clear
reset
set xrange [0:65000]
set yrange [0:2500]
set xlabel "Number of spikes during image presentation" font "Helvetica,14"
set ylabel "Image classification duration" font "Helvetica,14"
set key samplen 10 spacing 5 font ",14"
set tics font "Helvetica,14"

f(x) = a*x + b
g(x) = c*x + d
h(x) = e*x + f
fit f(x) "results/LAPTOP_OPT1_59999_30000_1000_4CORE.log" via a,b
fit g(x) "results/JETSON_OPT1_59999_30000_1000_4CORE.log" via c,d
fit h(x) "results/ATOM_ROBO_59999_30000_1000_1CORE.log" via e,f

# plot "results/ATOM_ROBO_59999_30000_1000_1CORE.log" title "Atom" pt 1 ps 1 lc rgb "slategrey",\
#      "results/LAPTOP_LIF_59999_30000_1000_4CORE.log" title "i5" pt 2 ps 1 lc rgb "blue",\
#      "results/JETSON_LIF_59999_30000_1000_4CORE.log" title "Jetson" pt 3 ps 1 lc rgb "sienna1",\
#      f(x) lc rgb "plum" lw 3 title "i5 model",\
#      g(x) lc rgb "yellow" lw 3 title "Jetson model",\
#      h(x) lc rgb "cyan" lw 3 title "Atom model"

 plot  "results/LAPTOP_OPT1_59999_30000_1000_4CORE.log" title "i5" pt 2 ps 1 lc rgb "blue",\
       "results/JETSON_OPT1_59999_30000_1000_4CORE.log" title "Jetson" pt 3 ps 1 lc rgb "sienna1" lw 0.6,\
       f(x) lc rgb "plum" lw 3 title "i5 model",\
       g(x) lc rgb "yellow" lw 3 title "Jetson model"


pause -1
