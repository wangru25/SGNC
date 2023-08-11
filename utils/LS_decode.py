# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-04 11:55:54
LastModifiedBy: Rui Wang
LastEditTime: 2022-12-05 10:57:41
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/LS_decode.py
Description: Module to extract contineous data-driven descriptors for a file of SMILES.
'''
import os
import sys
import argparse
import pandas as pd
import tensorflow as tf
from inference import InferenceModel
from preprocessing import preprocess_smiles
#from cddd.hyperparameters import DEFAULT_DATA_DIR
_default_model_dir = os.path.join('utils/data', 'default_model')
FLAGS = None

def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        None
    """
    parser.add_argument('-i',
                        '--input',
                        help='descriptor_file',
                        type=str)
    parser.add_argument('-o',
                        '--output',
                        help='output .csv file with a descriptor for each SMILES per row.',
                        type=str)
    parser.add_argument('--smiles_header',
                        help='if .csv, specify the name of the SMILES column header here.',
                        default="smiles",
                        type=str)
    parser.set_defaults(preprocess=True)
    parser.add_argument('--model_dir', default=_default_model_dir, type=str)
    parser.add_argument('--use_gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=True)
    parser.add_argument('--device', default="2", type=str)
    parser.add_argument('--cpu_threads', default=5, type=int)
    parser.add_argument('--batch_size', default=512, type=int)

def read_input(file):
    """Function that read teh provided file into a pandas dataframe.
    Args:
        file: File to read.
    Returns:
        pandas dataframe
    Raises:
        ValueError: If file is not a .smi or .csv file.
    """
    # des_df = pd.read_csv(file, index_col=0)
    des_df = pd.read_csv(file, header=None)
    des=des_df.values
    return des

def main(unused_argv):
    """Main function that extracts the contineous data-driven descriptors for a file of SMILES."""
#    if FLAGS.gpu:
#        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device)
    model_dir = FLAGS.model_dir
    file = FLAGS.input
    descriptors = read_input(file)
        
    print("start decoding...")
    infer_model = InferenceModel(model_dir=model_dir,
                                 use_gpu=FLAGS.gpu,
                                 batch_size=FLAGS.batch_size,
                                 cpu_threads=FLAGS.cpu_threads)
    decoded_smiles_list = infer_model.emb_to_seq(descriptors)
    print(decoded_smiles_list)
    print("finished calculating descriptors! %d out of %d input descriptors could be interpreted"
          %(len(decoded_smiles_list), len(descriptors)))
    df = pd.DataFrame(decoded_smiles_list, columns=["decoded_smiles"])
    print("writing descriptors to file...")
    df.to_csv(FLAGS.output)

def main_wrapper():
    global FLAGS
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
    
if __name__ == "__main__":
    main_wrapper()
