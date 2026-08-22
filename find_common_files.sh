# find ./results -maxdepth 1 -type f \( -iname "*.csv" -o -iname "*.json" \) > results.txt    
# find ./results/20260821-150621/ -maxdepth 1 -type f \( -iname "*.csv" -o -iname "*.json" \) > root.txt
sort results.txt > results_sorted.txt
sort root.txt > root_sorted.txt
# comm -12 results_sorted.txt root_sorted.txt > common_files.txt
comm -3 results_sorted.txt root_sorted.txt > outer_join_files.txt