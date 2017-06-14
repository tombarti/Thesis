# Meng Final Year Project
Meng Joint Mathematics and Computer Science Final Year Individual Project

### Project Structure

* __src/__: 
  * __data/__: contains scripts to download and convert the EmotioNet database to TFRecords
  * __slim/__: contains scripts, and libraries to train/fine-tune/evaluate models on datasets
* __tmp/__: directory to which model checkpoints and tf summaries are saved
* __report/__: latex files for project report
* __data/__: contains the dataset in different formats (e.g. plain images, TFRecords)
  * __images/__: contains all the images of the dataset
  * __partitioned/__: contains the images partitioned into train and validation sets
    * train/
    * validation/
  * __records/__: contains the images in `../train/` and `../validation` as TFRecords

### Fine-tunning/evaluating VGG_16 or Inception_V2
You need to execute one of the two scripts that are in __src/slim/scripts/__ whith the argument `train` to fine tune the model
or `evaluate` to perform evaluation. The script must be executed from the  __src/slim/__ directory:

Example: fine-tune vgg_16
```bash
cd src/slim/
./scripts/finetune_vgg_16_on_emotionet.sh train
```
