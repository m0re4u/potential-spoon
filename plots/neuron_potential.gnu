unset xtics
unset ytics
set xrange [0:70]
set yrange [0:1.2]
set terminal qt enhanced
set xlabel "Time" font "Helvectica,14"
set ylabel "Neuron potential" font "Helvectica,14"
set key samplen 10 spacing 5 font ",11"

f(x) = x < 10 ? 0 :  x < 20 ? 0.4 * exp(-(x-10)/10) : x < 25 ? 0.5 * exp(-(x-20)/10) : x < 33 ? 0.8 * exp(-(x-25)/10) : x < 33.5 ? 1.1 * exp(-(x-34)/10) : x < 50 ? 0 : 0.3 * exp(-(x-49)/10)

set style arrow 8 heads size screen 0.008,90 ls 2 lc rgb 'red' lw 3
set arrow from 34.5, 0.1 to 49, 0.1 as 8

plot  f(x) ls 1 lw 3 title "Neuron potential",\
      1 ls 4 lw 2 dt "-" title "V_{thres}"

pause -1
