file="tiles_GPU.txt"
while IFS= read line
do
    # display $line or do something with $line
    echo "$line"
    nohup bash run_arg.sh $line &
done <"$file"
