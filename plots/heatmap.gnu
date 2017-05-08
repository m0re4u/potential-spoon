# Create a 2D heat map from data in a file

set title "Heat Map generated from spike counts over 500 cycles"
unset key
set tic scale 0

# Color runs from white to green
set palette rgbformula -7,2,-7
set cbrange [0:5]
set cblabel "More green = more spikes"
unset cbtics

set xrange [0:27]
set yrange [27:0]

set view map
splot '../build/img.log' using 1:2:3 matrix with image
pause -1
