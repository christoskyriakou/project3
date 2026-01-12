#!/usr/bin/env python3
"""
protein_embed.py
Generates ESM-2 embeddings from protein sequences
"""

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import struct
import os


def mean_pool(token_embeddings, attention_mask):
    """
    Mean pooling: average embeddings across sequence length
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def generate_embeddings(fasta_path, model_name="facebook/esm2_t6_8M_UR50D", batch_size=8):
    """
    Generate embeddings for all sequences in a FASTA file
    
    Returns:
        embeddings: numpy array (n, embedding_dim)
        ids: list of sequence IDs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EMBED] Using device: {device}")
    
    # Load ESM-2 model
    print(f"[EMBED] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # Parse FASTA
    print(f"[EMBED] Reading sequences from {fasta_path}")
    sequences = []
    seq_ids = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        seq_ids.append(record.id)
    
    print(f"[EMBED] Loaded {len(sequences)} sequences")
    
    # Generate embeddings in batches
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"[EMBED] Processing batch {i // batch_size + 1}/{(len(sequences) + batch_size - 1) // batch_size}")
            
            # Tokenize
            encoded = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(device)
            
            # Forward pass
            outputs = model(**encoded)
            
            # Mean pooling
            embeddings = mean_pool(
                outputs.last_hidden_state,
                encoded['attention_mask']
            )
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    
    print(f"[EMBED] Generated embeddings: {embeddings.shape}")
    
    return embeddings, seq_ids


def save_as_fvecs(embeddings, output_path):
    """
    Save embeddings in .fvecs format (compatible with SIFT format)
    Format: [dim(int32)][vector(float32 x dim)] repeated
    """
    n, d = embeddings.shape
    print(f"[EMBED] Saving {n} vectors of dimension {d} to {output_path}")
    
    with open(output_path, "wb") as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(embeddings[i].tobytes())
    
    print(f"[EMBED] Saved to {output_path}")


def save_id_mapping(seq_ids, output_path):
    """
    Save sequence ID mapping
    """
    mapping_path = output_path.replace(".fvecs", "_ids.txt").replace(".dat", "_ids.txt")
    
    with open(mapping_path, "w") as f:
        for i, seq_id in enumerate(seq_ids):
            f.write(f"{i}\t{seq_id}\n")
    
    print(f"[EMBED] ID mapping saved to {mapping_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings from protein sequences")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output file (.fvecs or .dat)")
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D",
                        help="ESM-2 model name")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for embedding generation")
    
    args = parser.parse_args()
    
    # Generate embeddings
    embeddings, seq_ids = generate_embeddings(
        args.input,
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    # Save embeddings
    if args.output.endswith(".dat"):
        # Convert to .fvecs
        output_fvecs = args.output.replace(".dat", ".fvecs")
        save_as_fvecs(embeddings, output_fvecs)
    else:
        save_as_fvecs(embeddings, args.output)
    
    # Save ID mapping
    save_id_mapping(seq_ids, args.output)
    
    print("[EMBED] Done!")


if __name__ == "__main__":
    main()