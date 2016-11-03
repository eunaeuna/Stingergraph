#$c = threshold, $d = epsilon
for (( c = 1; c < 21; c++))
do
for (( d = 5; d < 15; d++))
do
    ./bin/stinger_pagerank_sources -n 1 -b 10000 g.18.8.bin a.18.8.1000.bin $c $d
done
done
