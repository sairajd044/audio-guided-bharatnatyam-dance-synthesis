# Audio Guided Bharatnatyam Dance Synthesis
Pytorch implementation of my Master Thesis Project, Audio Guided Bharatnatyam Dance Synthesis in IIT KGP

## Requirement
First you have to set up an environment with the proper requirements. For that I recommend using conda. Install the requirements with:

`conda env create -f environment.yml -n your_env_name python=3.10.13`

## Datasets
For this project, I used two datasets:
* Bharatnatyam dataset: This dataset was provided by IIT KGP. [Processed dataset](https://drive.google.com/file/d/1hXEMQaGg8eAkwJdlMZFKahqC6D6lkAue/view?usp=drive_link)
* [Choreonet](https://github.com/abcyzj/ChoreoNet) dataset: A publicly available dataset to train the CAU Prediction Model. [Processed dataset](https://drive.google.com/file/d/125kql8YM6eF5FuTkD5O4jHEHQyDKV1aI/view?usp=drive_link)

Download the processed dataset, extract them and put in their respective directories.

## CAU Prediction Model

### Train
To start training from scratch:

`python CauPrediction/train.py -e NUM_EPOCHS -f dataset_directory -o output_directory -d cuda -w 10`

To resume training:

`python CauPrediction/train.py -e NUM_EPOCHS -f dataset_directory -o output_directory -d cuda -w 10 -r`

All flags
*   -r, --resume          Resumes the training
*   -e, --epoch EPOCH Number of epochs to train
*   -d, --device {cpu,cuda} (cpu / cuda)
*  -b, --bidirectional, use bidirectional GRU
*  -w, --window WINDOW window length for music feature extraction
*  -o, --ouptut_dir OUPTUT_DIR training result directory
*  -f, --dataset_dir DATASET_DIR dataset directory

### Inference
If you do not intend to train from scratch, the model weights can be downloaded from [here](https://drive.google.com/file/d/1DJ8L-u86IdE88VrlV-alQsttNU2XZWOm/view?usp=drive_link).

It takes a music file and outputs CAU sequence tokens

`python CauPrediction/inference.py -d cuda -w 10 -s 0 -m 1.mp3 -f dataset_directory`

All flags
*  -b, --bidirectional, use this if trained model used bidirectional GRU
*  -d, --device {cpu,cuda} (cpu / cuda)
*  -c, --saved_weights SAVED_WEIGHTS
*  -w, --window WINDOW window length for music feature extraction
*  -s, --start START starting time (in ms)
*  -e, --end END end time (in ms)
*  -m, --music_file MUSIC_FILE training result directory
*  -f, --dataset_dir DATASET_DIR dataset directory

## Motion Generation Model

### Train
Joint Rotation Inpainting Model

`python MotionGenerationModel/train.py -f dataset_directory -e NUM_EPOCHS -d cuda -s 192 -w 64 -b 8 -m J`

Root Point Inpainting Model

`python MotionGenerationModel/train.py -f dataset_directory -e NUM_EPOCHS -d cuda -s 192 -w 64 -b 16 -m R`

All flags
* -r, --resume          Resumes the training
 * -e, --epoch EPOCH
                        Number of epochs to train
 * -d, --device {cpu,cuda}
                        (cpu / cuda)
 * -s, --size SIZE  number of frames in a motion clip
 * -w, --window WINDOW
                        number of masked frames
 * -b, --batch_size BATCH_SIZE
                        size of training/validation batches
 * -m, --model {J,R}
                        (J)oint model/ (R)oot model
 * -o, --output_dir OUTPUT_DIR
                        training result directory
 * -f, --dataset_dir DATASET_DIR
                        dataset root directory

### Inference
If you do not intend to train from scratch, the model weights can be downloaded from [here](https://drive.google.com/file/d/1OyBVw1yxpvVHyEuyhpQ9cLrSfCd3ssTF/view?usp=drive_link).

To test on a random clip of motion capture data

`python MotionGenerationModel/test.py -o output_directory -w saved_weights_directory`

To join two dance together

`python MotionGenerationModel/inference.py -f dance1.pkl -s dance2.pkl -o output_directory -w saved_weights_directory`

## Visualize

To play a dance animation:

`python MotionGenerationModel/visualize.py data/modified_kinect1/1_tatta/1/Dancer1/tatta_1_Dancer1.pkl`

To save a dance animation as mp4

`python MotionGenerationModel/visualize.py data/modified_kinect1/1_tatta/1/Dancer1/tatta_1_Dancer1.pkl result.mp4`