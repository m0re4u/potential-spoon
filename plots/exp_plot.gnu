# I(t) = Ir * exp((-r/l)*(t-t0))

# l = 0.0229      #Inductance (H)
# r = 3.34        #Resistance (Ohm)
# v = 5           #Voltage (V) DC
# i = v/r         #Peak current (A)
# tau = l/r       #Tau time constant
# a = tau * 4.4   #critical time value at which current is switched (switching occurs every a seconds)

set xrange [0:0.01]
plot exp(-1000 * x)
pause -1
