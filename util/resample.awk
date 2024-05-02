# usage: cat abc.xyz | gawk -v seed="${RANDOM}" -f resample.awk
BEGIN { srand(seed); }
{ if (rand() < 0.1) { print $0; } }
