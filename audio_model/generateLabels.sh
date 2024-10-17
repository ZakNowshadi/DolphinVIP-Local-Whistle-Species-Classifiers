SELECTION_DIR="../Audio/"
echo $SELECTION_DIR

cd $SELECTION_DIR

for species in `ls -d */`; do
    echo hi $species 
    for selection in `ls $species`; do
        echo $selection,$species >> labels.csv
    done
done

Audio/bottlenose