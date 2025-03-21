from ontomap.ontology.mse import * 
from ontomap.ontology.food import * 
from ontomap.ontology.phenotype import * 
from ontomap.ontology.biodiv import *
from ontomap.ontology.commonkg import *
from ontomap.ontology.anatomy import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from ontomap.base import BaseConfig
from ontomap.encoder import IRILabelInRAGEncoder
from ontomap.ontology_matchers import Qwen2_5BGE_M3RAG
from ontomap.postprocess import process
from ontomap.evaluation.evaluator import evaluator
import pandas as pd
import traceback
from contextlib import contextmanager


import os, sys


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
config.root_dir = "datasets"
model = Qwen2_5BGE_M3RAG(**config.Qwen2_5BGE_M3RAG)

track_coutner = 0
results_list = list()

ontology_matching = {
    # "anatomy": [
        # MouseHumanOMDataset,
    # ],
    "mse": [
        # MaterialInformationEMMOOMDataset,
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
            encoded_inputs = IRILabelInRAGEncoder()(**ontology)
            predicts = model.generate(encoded_inputs)
            predicts2, _ = process.postprocess_hybrid(predicts=predicts,
                                llm_confidence_th=0.7,
                                ir_score_threshold=0.9)
            results = {'track': track, 'ontology': ontology_name}
            eval = evaluator(track=track,
                    predicts=predicts2,
                    references=ontology["reference"])
            results.update(eval)
            print(f""" Printing Results 
                
                {results}
                
                
                """)
            results_list.append(results)

    df = pd.DataFrame(results_list)
    df.to_csv('llms4om_results.csv',index=False, header=False, mode='a+')

except Exception as e:
    traceback.print_exc()
    df = pd.DataFrame(results_list)
    df.to_csv('llms4om_results.csv',index=False, header=False, mode='a+')
    
        
        
        
        
        
        
        