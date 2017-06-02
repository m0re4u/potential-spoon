clear
reset
unset key

set title "STDP learning function"
set xlabel "dt"
set ylabel "dw"

f(dt) = exp(-dt/0.02)
plot [dt=0:0.04] f(dt)
pause -1
