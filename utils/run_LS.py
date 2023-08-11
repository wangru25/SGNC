# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2022-11-04 11:52:23
LastModifiedBy: Rui Wang
LastEditTime: 2022-12-09 10:56:26
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/run_LS.py
Description: Module to extract contineous data-driven descriptors for a file of SMILES.
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from inference import InferenceModel
from preprocessing import preprocess_smiles
from preprocessing import canonical_removess_smiles
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
                        help='input file. Either .smi or .csv file.',
                        type=str)
    parser.add_argument('-o',
                        '--output',
                        help='output .csv file with a descriptor for each SMILES per row.',
                        type=str)
    parser.add_argument('--smiles_header',
                        help='if .csv, specify the name of the SMILES column header here.',
                        default="smiles",
                        type=str)
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--no-preprocess', dest='preprocess', action='store_false')
    parser.set_defaults(preprocess=False)
    parser.add_argument('--model_dir', default=_default_model_dir, type=str)
    parser.add_argument('--use_gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
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
    if file.endswith('.csv'):
        sml_df = pd.read_csv(file,index_col=0).rename({0:FLAGS.smiles_header, 1:"EXTREG"},
                                                         axis=1)
    elif file.endswith('.smi'):
        sml_df = pd.read_table(file, header=None).rename({0:FLAGS.smiles_header, 1:"EXTREG"},
                                                         axis=1)
    else:
        raise ValueError("use .csv or .smi format...")
    return sml_df

def main(unused_argv):
    """Main function that extracts the contineous data-driven descriptors for a file of SMILES."""
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device)
    model_dir = FLAGS.model_dir
    file = FLAGS.input
    df = read_input(file)
    print(df)
    if FLAGS.preprocess:
        print("start preprocessing SMILES...")
        df["new_smiles"] = df[FLAGS.smiles_header].map(preprocess_smiles)
        sml_list = df[~df.new_smiles.isna()].new_smiles.tolist()
        print("finished preprocessing SMILES!")
    else:
        df["new_smiles"] = df[FLAGS.smiles_header].map(canonical_removess_smiles)
        sml_list = df["new_smiles"].tolist()
    print("start calculating descriptors...")
    infer_model = InferenceModel(model_dir=model_dir,
                                 use_gpu=FLAGS.gpu,
                                 batch_size=FLAGS.batch_size,
                                 cpu_threads=FLAGS.cpu_threads)
    descriptors = infer_model.seq_to_emb(sml_list)
    decoded_smiles_list = infer_model.emb_to_seq(descriptors)
    print("finished calculating descriptors! %d out of %d input SMILES could be interpreted"
          %(len(sml_list), len(df)))
    if FLAGS.preprocess:
        df = df.join(pd.DataFrame(decoded_smiles_list,
                                  index=df[~df.new_smiles.isna()].index,
                                  columns=["decoded_smiles"]))
        df = df.join(pd.DataFrame(descriptors,
                                  index=df[~df.new_smiles.isna()].index,
                                  columns=["cddd_" + str(i+1) for i in range(512)]))
        outsmi = pd.DataFrame(decoded_smiles_list,
                                  index=df[~df.new_smiles.isna()].index,
                                  columns=["decoded_smiles"])
        outsmi.to_csv(FLAGS.output,header=False, index=False)
    else:
#        df = df.join(pd.DataFrame(decoded_smiles_list,
#                                  index=df.index,
#                                  columns=["decoded_smiles"]))
        output = pd.DataFrame(descriptors,
                                  index=df.index,
                                  columns=["cddd_" + str(i+1) for i in range(512)])
#    print("writing descriptors to file...")
    output.to_csv(FLAGS.output,header=False, index=False)
    # np.save(FLAGS.output,output.to_numpy())

def main_wrapper():
    global FLAGS
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
    
if __name__ == "__main__":
    main_wrapper()
