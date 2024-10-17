SELECTION_DIR = ../Audio
echo $SELECTION_DIR

for species in `ls $SELECTION_DIR`; do
    for selection in `ls $species`; do
        echo $selection,$species >> $SELECTION_DIR/annotations.csv
    done
done