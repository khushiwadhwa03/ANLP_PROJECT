# ANLP Project
### Team Members - Khushi Wadhwa, Mahika Jain and Akshat Sanghvi (Team No. 41)

### This project addresses the challenge of improving Exemplar-Guided Paraphrase Generation (EGPG). This task aims to generate paraphrases that not only retain the original meaning but also mimic the style of a given "exemplar" sentence. 

## Link to report 
### https://www.notion.so/ANLP-Project-Outline-1050aad568bd803da632c45d8a3c470f

## Link to the presentation
### https://www.canva.com/design/DAGXChIlsnk/F1QSv6xX7UhwzT4-sSVINQ/edit?utm_content=DAGXChIlsnk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

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
python train.py --dataset <quora/para> --model_save_path directory_to_save_model
```
### For evaluating the model
```python
python evaluate.py --dataset <quora/para> --model_save_path  your_saved_model  --idx epoch_num
```
### For metrics calculation
```python
python3 metrics.py -i model_save_path/trg_gen<idx>.txt -r <quora/para>/test_trg.txt -t model_save_path/exm<idx>.txt # these files are generated during evaluation
```

### To infer the model, run the following command
```python
python3 inference.py --dataset <quora/para> --model_save_path  model_save_path  --idx <idx>
```
