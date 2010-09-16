set terminal pdf color enhanced
set output "gpu_bandwidth.pdf"
set logscale x
set xrange[65536:12582912]

plot 'perfs-gpu0-dtoh.dat' using 1:2 with lines lw 2 title "Device to Host - gpu 0", 'perfs-gpu0-htod.dat' using 1:2 with lines lw 2 title "Host to Device - gpu 0", 'perfs-gpu1-dtoh.dat' using 1:2 with lines lw 2 title "Device to Host - gpu 1", 'perfs-gpu1-htod.dat' using 1:2 with lines lw 2 title "Host to Device - gpu 1"

