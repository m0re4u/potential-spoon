clear
reset
unset key

#binwidth=5
#bin(x,width)=width*floor(x/width)
#plot '../build/network.log' using (bin($1,binwidth)):(1.0) smooth freq with boxes
plot '../build/network.sorted'
pause -1
