# from http://www.gnuplotting.org/tag/png/
set style line 1 lc rgb '#8b1a0e' pt 1 ps 1 lt 1 lw 2 # --- red
set style line 2 lc rgb '#5e9c36' pt 6 ps 1 lt 1 lw 2 # --- green
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12

set terminal svg size 480,320 font 'Verdana,10'

set xrange [1:500]
set xlabel 'length'
set ylabel 'p(length)'

set key right center

a = 0
cumsum(x)=(a=a+x,a)
