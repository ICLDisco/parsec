set terminal pdf color enhanced
set output "sgemm_bandwidth.pdf"
set logscale x

plot 'sgemm.dat' using 1:2 with lines lw 2 title "CUBLAS - gpu 0", 'sgemm.dat' using 1:3 with lines lw 2 title "Volkov - gpu 0"

