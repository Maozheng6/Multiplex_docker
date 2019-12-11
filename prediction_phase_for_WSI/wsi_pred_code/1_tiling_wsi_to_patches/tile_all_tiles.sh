file="tiles_to_tile.txt"
while IFS= read line
do
    # display $line or do something with $line
    echo "$line"
    wsi=`basename $line`
    nohup  python -u tiling_wsi.py $line > ./log_files/log_tiling_"$wsi".txt &
done <"$file"
