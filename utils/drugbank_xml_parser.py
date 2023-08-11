# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2023-01-09 09:59:44
LastModifiedBy: Rui Wang
LastEditTime: 2023-01-09 10:25:13
Email: wangru25@msu.edu
FilePath: /FokkerPlanckAutoEncoder/utils/drugbank_xml_parser.py
Description: 
'''
import untangle
import pandas as pd
import numpy as np
import os

#takes 5 minutes
#xml_path = '/Users/xiaoqi/Downloads/full_database.xml' # DrugBank Version 4.5.0 (release date: 2016.04.20) 
xml_path = '/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/drugbank/full_database.xml'
obj=untangle.parse(xml_path)


i=-1
#iterate over drug entries to extract information
for drug in obj.drugbank.drug:
    drug_type= str(drug["type"])
    
    # select for small molecule drugs
    if drug_type in ["small molecule", "Small Molecule", "Small molecule"]:
        i=i+1    
        if i < 1:
            print(drug.name)
            print(drug.groups)
            for element in drug.groups.children:
                print(element.cdata)
            #print(drug.groups.children.
        
        #Drug CAS
        #df_drugbank_sm.loc[i, "cas"]=drug.cas_number.cdata
        
        #Drug group
        #df_drugbank_sm.loc[i, "group"]=drug.groups.cdata

#Data Frame of DrugBank Small Molecule Type Drugs
df_drugbank_sm=pd.DataFrame(columns=["drugbank_id","name", "group", "cas","smiles","logP ALOGPS", "logP ChemAxon", "solubility ALOGPS", "pKa (strongest acidic)", "pKa (strongest basic)"])
print(df_drugbank_sm)

# Takes around 10 minutes to run.
i=-1
#iterate over drug entries to extract information
for drug in obj.drugbank.drug:
    drug_type= str(drug["type"])
    
    # select for small molecule drugs
    if drug_type in ["small molecule", "Small Molecule", "Small molecule"]:
        i=i+1    
        
        #Get drugbank_id
        for id in drug.drugbank_id:
            if str(id["primary"])=="true":
                df_drugbank_sm.loc[i, "drugbank_id"]=id.cdata
        #Drug name
        df_drugbank_sm.loc[i,"name"]=drug.name.cdata
        
        #Drug CAS
        df_drugbank_sm.loc[i, "cas"]=drug.cas_number.cdata
        
        #Drug group
        group = []
        for element in drug.groups.children:
            group.append(element.cdata)
        df_drugbank_sm.loc[i, "group"]=group
        
        #Get SMILES, logP, Solubility
        #Skip drugs with no structure. ("DB00386","DB00407","DB00702","DB00785","DB00840",
        #                                            "DB00893","DB00930","DB00965", "DB01109","DB01266",
        #                                           "DB01323", "DB01341"...)
        if len(drug.calculated_properties.cdata)==0: #If there is no calculated properties
            continue
        else:
            for property in drug.calculated_properties.property:
                if property.kind.cdata == "SMILES":
                    df_drugbank_sm.loc[i, "smiles"]=property.value.cdata
                    
                if property.kind.cdata == "logP":
                    if property.source.cdata == "ALOGPS":
                        df_drugbank_sm.loc[i, "logP ALOGPS"]=property.value.cdata
                    if property.source.cdata == "ChemAxon":
                        df_drugbank_sm.loc[i, "logP ChemAxon"]=property.value.cdata
                
                if property.kind.cdata == "Water Solubility":
                    df_drugbank_sm.loc[i, "solubility ALOGPS"]=property.value.cdata
                
                if property.kind.cdata == "pKa (strongest acidic)":
                    df_drugbank_sm.loc[i, "pKa (strongest acidic)"]=property.value.cdata
                
                if property.kind.cdata == "pKa (strongest basic)":
                    df_drugbank_sm.loc[i, "pKa (strongest basic)"]=property.value.cdata
            
#Drop drugs without SMILES from the dataframe
df_drugbank_smiles = df_drugbank_sm.dropna()
df_drugbank_smiles= df_drugbank_smiles.reset_index(drop=True)
print(df_drugbank_smiles.shape)

write_file = open('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/drugbank/full_drugbank.smi','w')
for i in range(df_drugbank_smiles.shape[0]):
    smi = df_drugbank_smiles['smiles'][i]
    write_file.write('%s\n'%smi)
write_file.close()

df_drugbank_smiles.to_pickle('/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/drugbank/drugbank.pkl')
test = pd.read_pickle("/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/data/drugbank/drugbank.pkl")
print(type(test))