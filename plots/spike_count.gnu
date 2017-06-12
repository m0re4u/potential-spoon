clear
reset

set xrange [800:1584]
set yrange [0:10]
set term png

plot 'build/network.log'
