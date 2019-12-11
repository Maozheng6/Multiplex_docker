file="lines.txt"
while IFS= read line
do
    # display $line or do something with $line
    echo "$line"
    bash arg.sh $line
done <"$file"
