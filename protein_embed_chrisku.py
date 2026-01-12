import torch
import esm
import numpy as np
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import argparse
import os

class ProteinDataset(Dataset):  
    def __init__(self,fasta_path):
        self.data=[(rec.id,str (rec.seq)[:1022]) for rec in SeqIO.parse(fasta_path, "fasta")]
    def __len__(self):
         return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

def embedding(input_fasta, output_dat):
    batch_size=16
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.to(device) # loading the model to the device, in our case since we dont have an nvidia gpu its the cpu
    model.eval()
    batch_converter=alphabet.get_batch_converter()
    dataset= ProteinDataset(input_fasta) 
    dataloader= DataLoader(dataset,batch_size=batch_size,shuffle=False) 

    all_embeddings=[]
    all_ids=[]
    total_proteins=len(dataset)
    processed_proteins=0
    print(f"using {device}")
    with torch.no_grad():
        for batch in dataloader:
            labels, strs, tokens = batch_converter(list(zip(batch[0], batch[1]))) 
            tokens = tokens.to(device)
            results= model(tokens, repr_layers=[6])
            representations=results["representations"][6]
            for i, token_str in enumerate(strs):
                mean_embedding=representations[i, 1: len(token_str)+1].mean(0)
                all_embeddings.append(mean_embedding.cpu().numpy())
                all_ids.append(batch[0][i])
            processed_proteins += len(strs)
            print(f"Progress: {processed_proteins}/{total_proteins} proteins processed...", end='\r')
    print("saving the data")
    np.save(output_dat, np.vstack(all_embeddings))
    with open("ids.txt", "w") as ids:
        for protein_id in all_ids:
            ids.write(f"{protein_id}\n")         
    print(f"Done! Embeddings saved to {output_dat}")
if __name__=="__main__":
    parser=argparse.ArgumentParser()  
    parser.add_argument("-i",required=True)
    parser.add_argument("-o",required=True)
    args=parser.parse_args()    
    try:
        embedding(args.i,args.o)
    except Exception as e:
        print(f"we have an error {e}")
    
