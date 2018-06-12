set term pdfcairo enhanced dashed size 8.5in,11in font ',16'
set output 'd-modshift.pdf'
set multiplot layout 4,1 title "d, P = 11.024325 Days, E = 2982.147698 Days" font ',20'
a = 0.045
b = 0.0575
c = a+7.25*b
d = 0.08
e = 0.955
set arrow from screen (a-0.5*b),(e+0.01)       to screen (c+6.5*d),(e+0.01)       nohead lt 1 lw 5 lc 7
set arrow from screen (a-0.5*b),(e-0.025-0.01) to screen (c+6.5*d),(e-0.025-0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (a-0.5*b),(e-0.0125) to screen (c+6.5*d),(e-0.0125) nohead lt 1 lw 5 lc 7
set arrow from screen (a-0.5*b),(e+0.01) to screen (a-0.5*b),(e-0.025-0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (c+6.5*d),(e+0.01) to screen (c+6.5*d),(e-0.025-0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (a+0.5*b),(e-0.025-0.01) to screen (a+0.5*b),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (a+1.5*b),(e-0.025-0.01) to screen (a+1.5*b),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (a+2.5*b),(e-0.025-0.01) to screen (a+2.5*b),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (a+3.5*b),(e-0.025-0.01) to screen (a+3.5*b),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (a+4.5*b),(e-0.025-0.01) to screen (a+4.5*b),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (a+5.5*b),(e-0.025-0.01) to screen (a+5.5*b),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (a+6.5*b),(e-0.025-0.01) to screen (a+6.5*b),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (c+0.5*d),(e-0.025-0.01) to screen (c+0.5*d),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (c+1.5*d),(e-0.025-0.01) to screen (c+1.5*d),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (c+2.5*d),(e-0.025-0.01) to screen (c+2.5*d),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (c+3.5*d),(e-0.025-0.01) to screen (c+3.5*d),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (c+4.5*d),(e-0.025-0.01) to screen (c+4.5*d),(e+0.01) nohead lt 1 lw 5 lc 7
set arrow from screen (c+5.5*d),(e-0.025-0.01) to screen (c+5.5*d),(e+0.01) nohead lt 1 lw 5 lc 7
set label "{Pri}"         at screen (a+0*b),e center font ', 16'
set label "{Sec}"         at screen (a+1*b),e center font ', 16'
set label "{Ter}"         at screen (a+2*b),e center font ', 16'
set label "{Pos}"         at screen (a+3*b),e center font ', 16'
set label "{FA_{1}}"          at screen (a+4*b),e center font ', 16'
set label "{FA_{2}}"         at screen (a+5*b),e center font ', 16'
set label "F_{Red}"                   at screen (a+6*b),e center font ', 16'
set label "{Pri-Ter}"     at screen (c+0*d),e center font ', 16'
set label "{Pri-Pos}"     at screen (c+1*d),e center font ', 16'
set label "{Sec-Ter}"     at screen (c+2*d),e center font ', 16'
set label "{Sec-Pos}"     at screen (c+3*d),e center font ', 16'
set label "{Odd-Evn}"    at screen (c+4*d),e center font ', 16'
set label "{DMM}"    at screen (c+5*d),e center font ', 16'
set label "{Shape}"    at screen (c+6*d),e center font ', 16'
set label "57.6" at screen (a+0*b),(e-0.025) center font ',16' textcolor lt 7
set label "5.48" at screen (a+1*b),(e-0.025) center font ',16' textcolor lt 7
set label "4.34" at screen (a+2*b),(e-0.025) center font ',16' textcolor lt 7
set label "3.59" at screen (a+3*b),(e-0.025) center font ',16' textcolor lt 7
set label "5.02" at screen (a+4*b),(e-0.025) center font ',16' textcolor lt 7
set label "2.56" at screen (a+5*b),(e-0.025) center font ',16' textcolor lt 7
set label "1.42" at screen (a+6*b),(e-0.025) center font ',16' textcolor lt 7
set label "53.3" at screen (c+0*d),(e-0.025) center font ',16' textcolor lt 7
set label "54.1" at screen (c+1*d),(e-0.025) center font ',16' textcolor lt 7
set label "1.14" at screen (c+2*d),(e-0.025) center font ',16' textcolor lt 7
set label "1.89" at screen (c+3*d),(e-0.025) center font ',16' textcolor lt 7
set label "0.99" at screen (c+4*d),(e-0.025) center font ',16' textcolor lt 7
set label "1.00" at screen (c+5*d),(e-0.025) center font ',16' textcolor lt 7
set label "0.08" at screen (c+6*d),(e-0.025) center font ',16' textcolor lt 7
set origin 0.0,0.67
set xlabel 'Phase' offset 0,0.4
set xrange [-0.25 to 1.25]
set x2range [-0.25 to 1.25]
set xtics 0.25
set format x '%4.2f'
set format y '%6.0f'
set ylabel 'Flux (ppm)'
set object rect from 0.75,-1E7 to 1.25,1E7 fc rgb '#D3D3D3' lw 0
set label '' at 0.0004635423, graph 0.015 front point pt 9  ps 0.8
set label '' at 0.0610913118, graph 0.015 front point pt 9  ps 0.8
set label '' at -0.1324501656, graph 0.985 front point pt 11 ps 0.8
set label '' at 0.7383396392, graph 0.015 front point pt 8  ps 0.8
stats 'd-binned2.dat' u 2 nooutput
set yrange [1.0E6*STATS_min-0.5*(STATS_max-STATS_min) to 1.0E6*STATS_max+0.5*(STATS_max-STATS_min)]
set autoscale y
plot 'd-outfile1.dat' u 1:($2*1.0E6) pt 7 ps 0.1 lc 1 notitle, '' u ($1+1.0):($2*1.0E6) pt 7 ps 0.1 lc 1 notitle, '' u ($1+2.0):($2*1.0E6) pt 7 ps 0.1 lc 1 notitle, 'd-binned2.dat' u 1:($2*1.0E6) pt 7 ps 0.1 lc 3 notitle, '' u ($1+1.0):($2*1.0E6) pt 7 ps 0.1 lc 3 notitle, 'd-outfile1.dat' u 1:($3*1.0E6) with lines lt 1 lc 7 lw 5 notitle, '' u ($1+1.0):($3*1.0E6) with lines lt 1 lc 7 lw 5 notitle
set origin 0.0,0.435
set autoscale y
set xlabel 'Phase'
set xtics format '%3.1f'
unset arrow
unset xlabel
unset label
set xtics auto
set x2tics 0.25 mirror
set format x2 '%4.2f'
set ylabel 'Flux (ppm)'
set label '' at 0.0004635423, graph 0.015 front point pt 9  ps 0.8
set label '' at 0.0610913118, graph 0.015 front point pt 9  ps 0.8
set label '' at -0.1324501656, graph 0.985 front point pt 11 ps 0.8
set label '' at 0.7383396392, graph 0.015 front point pt 8  ps 0.8
set object rect from 0.75,-1E7 to 1.25,1E7 fc rgb '#D3D3D3' lw 0
plot 'd-outfile3.dat' u 1:(1E6*$4) with lines lt 1 lc 7 notitle, '' u ($1+1.0):(1E6*$4) with lines lt 1 lc 7 notitle, 0 with lines lt 2 lc 1 lw 5 notitle, 165.2167686434 with lines lt 2 lc 3 lw 5 notitle, -165.2167686434 with lines lt 2 lc 3 lw 5 notitle
unset arrow
unset object
unset label
set size square 0.375,0.275
set origin 0.0,0.2
set label 'Primary' at graph 0.5,0.925 center front
set xrange [-0.0156256243 to 0.0161286986]
set xtics 0.0105847743
set xtics format '%5.3f' mirror
set xlabel ' '
unset x2tics
unset x2label
unset y2tics
unset y2label
set autoscale y
set ytics format '%6.0f' mirror
set ytics auto
set ylabel 'Flux (ppm)'
plot 'd-binned2.dat' u 1:($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1+1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1-1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, 'd-outfile1.dat' u 1:($3*1E6) with lines lt 1 lc 7 notitle, '' u ($1+1.0):($3*1E6) with lines lt 1 lc 7 notitle, '' u ($1-1.0):($3*1E6) with lines lt 1 lc 7 notitle
unset label
set size square 0.375,0.275
set origin 0.315,0.2
set label 'Odd' at graph 0.5,0.925 center front
set xrange [-0.0156256243 to 0.0161286986]
set xtics 0.0105847743
set xtics format '%5.3f' mirror
set xlabel ' '
unset x2tics
unset x2label
unset y2tics
unset y2label
unset autoscale y
set ytics format '%6.0f' mirror
set ytics auto
set ylabel ' '
plot 'd-binned2-odd.dat' u 1:($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1+1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1-1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, 'd-outfile1.dat' u 1:($3*1E6*0.9680725361) with lines lt 1 lc 7 notitle, '' u ($1+1.0):($3*1E6*0.9680725361) with lines lt 1 lc 7 notitle, '' u ($1-1.0):($3*1E6*0.9680725361) with lines lt 1 lc 7 notitle
unset label
set size square 0.375,0.275
set origin 0.63,0.2
set label 'Even' at graph 0.5,0.925 center front
set xrange [-0.0156256243 to 0.0161286986]
set xtics 0.0105847743
set xtics format '%5.3f' mirror
set xlabel ' '
unset x2tics
unset x2label
unset y2tics
unset y2label
set ytics format '%6.0f' mirror
set ytics auto
set ylabel ' '
plot 'd-binned2-evn.dat' u 1:($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1+1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1-1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, 'd-outfile1.dat' u 1:($3*1E6*1.0024732980) with lines lt 1 lc 7 notitle, '' u ($1+1.0):($3*1E6*1.0024732980) with lines lt 1 lc 7 notitle, '' u ($1-1.0):($3*1E6*1.0024732980) with lines lt 1 lc 7 notitle
unset label
set size square 0.375,0.275
set origin 0.0, -0.015
set label 'Secondary' at graph 0.5,0.925 center front
set xrange [0.0450021453 to 0.0767564681]
set xtics 0.0105847743
set xtics format '%5.3f' mirror
set xlabel 'Phase'
unset x2tics
unset x2label
unset y2tics
unset y2label
set autoscale y
set ytics format '%6.0f' mirror
set ytics auto
unset ylabel
set ylabel 'Flux (ppm)'
plot 'd-binned2.dat' u 1:($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1+1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1-1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, 'd-outfile1.dat' u ($1+0.0610913118):((1.0000000000 + 0.0950677015*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle, '' u ($1+1.0+0.0610913118):((1.0000000000 + 0.0950677015*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle, '' u ($1-1.0+0.0610913118):((1.0000000000 + 0.0950677015*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle
unset label
set size square 0.375,0.275
set origin 0.315, -0.015
set label 'Tertiary' at graph 0.5,0.925 center front
set xtics 0.0105847743
set xtics format '%5.3f' mirror
set xrange [0.7222504727 to 0.7540047956]
set xlabel 'Phase'
unset x2tics
unset x2label
unset y2tics
unset y2label
set autoscale y
set ytics format '%6.0f' mirror
set ytics auto
set ylabel ' '
plot 'd-binned2.dat' u 1:($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1+1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1-1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, 'd-outfile1.dat' u ($1+0.7383396392):((1.0000000000 + 0.0752807109*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle, '' u ($1+1.0+0.7383396392):((1.0000000000 + 0.0752807109*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle, '' u ($1-1.0+0.7383396392):((1.0000000000 + 0.0752807109*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle
unset label
set size square 0.375,0.375
set origin 0.63, -0.065
set label 'Positive' at graph 0.5,0.075 center front
set xtics 0.0105847743
set xtics format '%5.3f' mirror
set xrange [-0.1485393321 to -0.1167850093]
set xlabel 'Phase'
unset x2tics
unset x2label
unset y2tics
unset y2label
set autoscale y
set ytics format '%6.0f' mirror
set ytics auto
set ylabel ' '
plot 'd-binned2.dat' u 1:($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1+1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, '' u ($1-1.0):($2*1E6):($3*1E6) with yerrorbars lt 1 pt 7 ps 0.5 lc 3 notitle, 'd-outfile1.dat' u ($1+-0.1324501656):((1.0000000000 + -0.0622620274*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle, '' u ($1+1.0+-0.1324501656):((1.0000000000 + -0.0622620274*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle, '' u ($1-1.0+-0.1324501656):((1.0000000000 + -0.0622620274*($3-1.0000000000))*1E6) with lines lt 1 lc 7 notitle
unset label
unset multiplot
