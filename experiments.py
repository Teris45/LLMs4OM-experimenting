from ontomap.ontology.mse import * 
from ontomap.ontology.food import * 
from ontomap.ontology.phenotype import * 
from ontomap.ontology.biodiv import *
from ontomap.ontology.commonkg import *
from ontomap.ontology.anatomy import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import time
from ontomap.base import BaseConfig
from ontomap.encoder import IRILabelInRAGEncoder
from ontomap.ontology_matchers import Qwen2_5BGE_M3RAG, Qwen2_5_USER_BGE_M3RAG, Phi4_mini_USER_bge_m3RAG
from ontomap.postprocess import process
from ontomap.evaluation.evaluator import evaluator
import pandas as pd
import traceback
from contextlib import contextmanager


import os, sys

models_dict = {
    0: "Qwen_2.5 + bge_m3",
    1: "Qwen_2.5 + USER_bge_m3",
    2: "Phi4_mini + USER_bge_m3",

}


models = [
    # Qwen2_5BGE_M3RAG,
    # Qwen2_5_USER_BGE_M3RAG,
    Phi4_mini_USER_bge_m3RAG
]



# Context manager to suppress output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
            
config = BaseConfig(approach='rag').get_args(device='cuda', batch_size=16)

config_list = [
    config.Qwen2_5BGE_M3RAG,
    config.Qwen2_5_USER_BGE_M3RAG,
    config.Phi4_mini_USER_bge_m3RAG
]

index = 2

config.root_dir = "datasets"
results_list = list()

for mod in models:
    config_mod = config_list[index]
    model = mod(**config_mod)

    track_coutner = 0

    ontology_matching = {
        "anatomy": [
            MouseHumanOMDataset,
        ],
        "mse": [
            MaterialInformationEMMOOMDataset,
            MaterialInformationMatOntoMDataset
        ],
        "food": [
            CiqualSirenOMDataset,
        ],
        "phenotype": [
            DoidOrdoOMDataset,
            HpMpOMDataset
        ],
        "biodiv" : [
            EnvoSweetOMDataset,
            TaxrefldBacteriaNcbitaxonBacteriaOMDataset,
            TaxrefldChromistaNcbitaxonChromistaOMDataset,
            TaxrefldFungiNcbitaxonFungiOMDataset,
            TaxrefldPlantaeNcbitaxonPlantaeOMDataset,
            TaxrefldProtozoaNcbitaxonProtozoaOMDataset,
        ],
        "commonkg" : [
            NellDbpediaOMDataset,
            YagoWikidataOMDataset,
        ],
    }


    track_len = len(ontology_matching)


    try :
        for track in ontology_matching:
            track_coutner += 1
            
            print(
    f"""
                Current track: {track}
            {track_coutner} / {track_len}
                        
    """
                )
            
            onto_counter = 0
            onto_len = len(ontology_matching[track])
            for Onto in ontology_matching[track]:
                with suppress_stdout():
                    ontology = Onto().collect(root_dir=config.root_dir)
                
                ontology_name = ontology['dataset-info']['ontology-name']
                onto_counter += 1
                print(
    f"""
                    Current ontology: {ontology_name}
                {onto_counter} / {onto_len}

    """
                )
                start_time = time.perf_counter()
                encoded_inputs = IRILabelInRAGEncoder()(**ontology)
                predicts = model.generate(encoded_inputs)
                predicts2, _ = process.postprocess_hybrid(predicts=predicts,
                                    llm_confidence_th=0.7,
                                    ir_score_threshold=0.9)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                results = {'track': track, 'ontology': ontology_name, 'model': models_dict[index]}
                eval = evaluator(track=track,
                        predicts=predicts2,
                        references=ontology["reference"])
                results.update(eval)
                results['runtime'] = total_time
                print(f""" Printing Results 
                    
                    {results}
                    
                    
                    """)
                results_list.append(results)

        df = pd.DataFrame(results_list)
        df.to_csv('llms4om_results_multiple.csv',index=False, mode='a+', header=False)
        index += 1

    except Exception as e:
        traceback.print_exc()
        df = pd.DataFrame(results_list)
        df.to_csv('llms4om_results_multiple.csv',index=False, mode='a+', header=False)

    
        
        
        
        
        
        
        