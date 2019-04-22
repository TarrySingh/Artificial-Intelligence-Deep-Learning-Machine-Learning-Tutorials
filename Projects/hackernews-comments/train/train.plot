# from http://www.gnuplotting.org/tag/png/
set style line 1 lc rgb '#8b1a0e' pt 1 ps 1 lt 1 lw 2 # --- red
set style line 2 lc rgb '#5e9c36' pt 6 ps 1 lt 2 lw 2 # --- green
set style line 3 lc rgb '#1f80ff' pt 3 ps 1 lt 3 lw 2 # --- blue
set style line 4 lc rgb '#ffa500' pt 3 ps 1 lt 4 lw 2 # --- orange
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12

set terminal pngcairo size 640,480 font 'Verdana,10'

set xlabel 'step'
set ylabel 'loss'

set key right top

a = 0
cumsum(x)=(a=a+x,a)
