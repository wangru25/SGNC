<!--
 * @Author: Rui Wang
 * @Date: 2022-12-16 14:30:41
 * @LastModifiedBy: Rui Wang
 * @LastEditTime: 2022-12-23 01:56:56
 * @Email: wangru25@msu.edu
 * @FilePath: /FokkerPlanckAutoEncoder/README.md
 * @Description: 
-->
Stochastic-based Generative Network Complex

A python package for the Stochastic-based Generative Network Complex

#### 1 Preperation for Predictive Models

##### 1.1 Training datasets 

##### 1.2 Train predictive model

#### 2 Preperation for Reference and Init Vector

##### 2.1 Reference 

##### 2.2 Init vector

#### 3 Generator Model

1. Submit thousands of jobs to generate latent space vectors.

   ```bash
   cd sbatch
   python submit_generator.py 20221209
   ```

2. Divide generated latent space vectors to sub-files. Each sub-file has 2000 records.

   ```bash
   cd ..
   python ./utils/divide_generated_ls.py 20221209
   ```

3. Decode all generated latent space vectors to smiles.

   ```bash
   cd sbatch
   python submit_decode.py 20221209
   ```

4. Drop duplicated and unlikely smiles. 

   ```bash
   cd ..
   python ./utils/drop_duplicates.py 20221209
   ```

#### 4 Filtered Model

1. Encode generated smiles to latent space vectors

   ```bash
   cd sbatch
   python submit_encoder.py 20221209
   ```

2. Binding affinity test

   ```bash
   cd ..
   python ./src/filtered.py --date 20221209
   ```

3. ADMET and SAS test

   Test ADMET on a online server: [ADMET](https://admetmesh.scbdd.com/service/screening/cal) and download a csv file. Then transfer this file to server

   ```bash
   scp ADMET.csv wangru25@hpcc.msu.edu:/mnt/research/guowei-search.8/RuiWang/FokkerPlanckAutoEncoder/results/generator_20221209
   ```

   Then check if there is a molecule that falls in the optimal range. 

   ```bash
   python ./src/properties.py 20221209
   ```

#### 5 Check the Reproduction Rate
1. Decode latent space vectors (from encoder) to smiles. 
2. Make comparasion with the generated smiles. Check the reproduction rate.

### Reference
[1] Wang, R., Feng, H. and Wei, G.W., 2023. ChatGPT in Drug Discovery: A Case Study on Anticocaine Addiction Drug Development with Chatbots. Journal of Chemical Information and Modeling, 63(22), pp.7189-7209.
