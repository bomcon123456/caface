#! /bin/bash

basedir="/home/termanteus/workspace/face/data/matcher/vinai/longrange/cctv/vinhomes/TNP/week2_aug_2022/cutted_aligend/tnp_aligned/pos"
enrolldir="/home/termanteus/workspace/face/data/matcher/vinai/enrolls/15112022_enroll/"
delimiter="_"
for d in $(ls -d $basedir/*); do
    dir_name=`basename $d`
    if [ -d $d ] && [ $dir_name != "unknown" ]; then
        # Split string by delimiter
        IFS=$delimiter read -ra split_string <<< "$dir_name"
        # Get last element of split string
        id="${split_string[-1]}"
        enroll_path=$enrolldir$id
        echo Handling $d with enroll path $enroll_path
        ./run.sh $d $enroll_path
    fi
done
# probe=$1
# gallery=$2
# python main.py $1 $2