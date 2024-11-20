## Link to report 
### https://www.notion.so/ANLP-Project-Outline-1050aad568bd803da632c45d8a3c470f

## Link to the dataset
### https://drive.google.com/drive/folders/1xkCtRnbeKPg_-0qR7j8jtzV8lfEcZGJm?usp=sharing
### Run the following command to preprocess the dataset
```python
python3 para_preprocess.py # for ParaNMT dataset
python3 quora_preprocess.py # for QQPPos dataset
```

## Additional Files / Folders to Download from the net
```
glove.6B.300d.txt (file)
meteor-1.5.jar (file)
multi-bleu.perl (file)
stanford-corenlp-full-2018-10-05/ (folder)
paraphrase-en.gz (file, refer from https://github.com/malllabiisc/SGCP - used for evaluation and metric calculations)
```

## Run Instructions
### For training the model
```python
python train.py --dataset quora --model_save_path directory_to_save_model
```
### For evaluating the model
```python
python evaluate.py --dataset para/quora --model_save_path  your_saved_model  --idx epoch_num
```
### For metrics calculation
```python
python3 metrics.py -i model_save_path/trg_gen<idx>.txt -r quora/test_trg.txt -t model_save_path/exm<idx>.txt # these files are generated during evaluation
```