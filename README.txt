# README

Note: All relevant code for the music identification pipeline is provided in the `main.py` file. Code for plotting figures is provided in the `plot.py` file.

## Demo

Notice that one audio file is provided in the `./audio_files` directory. To run the code over the single file and default parameterization, simply run the following:

```python3 main.py 20 30 1```

This will save a file of the fingerprinted database (`20-30-1.pickle`), and generate and evaluate each of the tests. The final evaluation results over each test set (as reported accuracies) will be outputted as ('20-30-1.txt').



## Full Replication of Results

1. First, download the audio data from the MagnaTagATune dataset (https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) and extract all individual audio files into the `./audio_files` directory. Note that these must be converted into `.wav` format.

2. The code can be run as follows, where it requires the bin size, intensity threshold, and k parameters as input:

```python3 main.py [BIN_SIZE] [AMP_THRESH] [K]```


