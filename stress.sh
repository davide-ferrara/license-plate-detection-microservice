#!/bin/bash
#Usage : ./stress.sh dataset localhost 8080

host=$2
port=$3

dataset=$1

for file in "$dataset"/*; do
    # Controlla il file 
    if [ -f "$file" ]; then
        echo "Invio del file: $file"

        # Esegue lo script upload-file.sh per inviare il file alla pipeline
        ./upload-file.sh $file $host $port
        sleep 0.5
    fi
done