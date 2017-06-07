clear
reset
set key off

#set xrange [0:10]
set xlabel 't'
set yrange [0:1584]
set ylabel 'neuron'
#set zrange [0:0.020]
set zlabel 'trace'
#splot "build/network.log" u 1:2:3 pt 7 ps 1
splot "build/network.log" u 1:2:3

pause -1
