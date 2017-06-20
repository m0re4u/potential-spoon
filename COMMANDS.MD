#### For generating the files for graphing the V values
`./LIF 0 > neuron1.log && ./LIF 1 > neuron2.log && ./LIF 2 > neuron3.log && ./LIF 3 > neuron4.log && ./LIF 4 > neuron5.log && ./LIF 5 > neuron6.log`
#### Plotting V values
`plot 'neuron1.log' with lines lt rgb "red", 'neuron2.log' with lines lt rgb "blue", 'neuron3.log' with lines lt rgb "yellow", 'neuron4.log' with lines, 'neuron5.log' with lines, 'neuron6.log' with lines`

#### Create rectangle in gnuplot
`set object 1 rect from 0.0725,0 to 0.108,0.012 fc rgb "cyan" fs solid 0.3 noborder back`
