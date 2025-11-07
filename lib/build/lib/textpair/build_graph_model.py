import os
import sys
from multiprocessing import Process, Queue

import lz4.frame
import networkx as nx
import numpy as np
import orjson
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GAE, GATConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Define Model Hyperparameters ---
AUTHOR_EMBED_DIM = 64  # Initial learnable dim for authors
OUT_EMBED_DIM = 32     # The final dim for clustering
BATCH_SIZE = 2048     # Batch size for SBERT encoding


class GNNEncoder(nn.Module):
    def __init__(self, num_authors, author_embed_dim, sbert_dim, out_embed_dim):
        super().__init__()
        self.author_embedding = nn.Embedding(num_authors, author_embed_dim)
        self.conv1 = GATConv(author_embed_dim,
                             32,
                             edge_dim=sbert_dim,
                             add_self_loops=False)
        self.conv2 = GATConv(32,
                             out_embed_dim,
                             edge_dim=sbert_dim,
                             add_self_loops=False)

    def forward(self, x, edge_index, edge_attr):
        author_features = self.author_embedding(x)
        h = self.conv1(author_features, edge_index, edge_attr=edge_attr).relu()
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        return h


def build_author_to_id_mapping(alignments: str, counts: int) -> dict:
    author_to_id = {}
    current_id = 0
    with lz4.frame.open(alignments, 'rb') as f:
        for line in tqdm(f, total=counts, desc="Building author to ID mapping"):
            alignment = orjson.loads(line)
            if alignment["source_author"] not in author_to_id:
                author_to_id[alignment["source_author"]] = current_id
                current_id += 1
            if alignment["target_author"] not in author_to_id:
                author_to_id[alignment["target_author"]] = current_id
                current_id += 1
    return author_to_id

def data_producer(queue: Queue, alignments_file: str, counts: int, author_to_id: dict):
    """
    Reads the file and puts data into the queue. Runs in its own process.
    """
    with lz4.frame.open(alignments_file, 'rb') as f:
        for _ in tqdm(range(counts), desc="[CPU Producer] Reading file"):
            line = f.readline()
            if not line:
                break
            alignment = orjson.loads(line)

            # Get the data we need
            passage = alignment["source_passage"]
            source_id = author_to_id[alignment["source_author"]]
            target_id = author_to_id[alignment["target_author"]]

            # Put the raw data into the queue for the consumer
            queue.put((passage, source_id, target_id))

    # Put a "sentinel" value to signal the end
    queue.put(None)


def create_alignment_embeddings(alignments: str, author_to_id: dict, model_name: str, counts: int):
    """Build the alignment embeddings using a producer-consumer pattern."""
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    edge_attr_memmap = np.memmap("edge_features.memmap", dtype='float32', mode='w+', shape=(counts, embedding_dim))

    edge_index_list = []
    batch_idx = 0
    passages_batch = []

    data_queue = Queue(maxsize=BATCH_SIZE * 2)
    producer_process = Process(target=data_producer,
                               args=(data_queue, alignments_file, alignment_counts, author_to_id))
    producer_process.start()

    with tqdm(total=alignment_counts, desc="[GPU Consumer] Encoding passages") as pbar:
        while True:
            # Get data from the CPU process
            item = data_queue.get()
            if item is None:
                break

            passage, source_id, target_id = item
            passages_batch.append(passage)
            edge_index_list.append((source_id, target_id))

            if len(passages_batch) == BATCH_SIZE:
                embeddings = model.encode(passages_batch,
                                                convert_to_tensor=True,
                                                device=device)
                start = batch_idx * BATCH_SIZE
                end = start + len(passages_batch)
                edge_attr_memmap[start:end] = embeddings.cpu().numpy()
                passages_batch = []
                batch_idx += 1

            pbar.update(1)

    if len(passages_batch) > 0:
        embeddings = model.encode(passages_batch, convert_to_tensor=True, device=device)
        start = batch_idx * BATCH_SIZE
        end = start + len(passages_batch)
        edge_attr_memmap[start:end] = embeddings.cpu().numpy()

    producer_process.join() # Clean up the producer process
    del model

    # Flush and close the memmap file
    edge_attr_memmap.flush()
    del edge_attr_memmap

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    node_ids = torch.arange(len(author_to_id))
    data = Data(x=node_ids, edge_index=edge_index)
    data.num_nodes = len(author_to_id)

    del edge_index_list

    # 1. Initialize the Encoder and GAE Model
    encoder = GNNEncoder(num_authors=len(author_to_id),
                     author_embed_dim=AUTHOR_EMBED_DIM,
                     sbert_dim=embedding_dim,
                     out_embed_dim=OUT_EMBED_DIM)
    model = GAE(encoder)

    model = model.to(device)

    # 2. Load the on-disk edge features (this does NOT load 77GB into RAM)
    edge_attr_tensor = torch.from_numpy(
        np.load('edge_features.memmap', mmap_mode='r')
    ).to(device)

    # 3. Create the Mini-Batch Loader
    # This is the key. It will sample the graph for link prediction.
    from torch_geometric.loader import LinkNeighborLoader

    loader = LinkNeighborLoader(
        data=data,
        batch_size=512,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 5],  # Sample 10 1-hop neighbors and 5 2-hop neighbors
        edge_label_index=data.edge_index,
        edge_label=torch.ones(data.num_edges),
        num_workers=4
    )

    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(5): # Fewer epochs are needed for large graphs
        for batch in tqdm(loader):
            batch = batch.to(device)

            # --- Key Step: Get Edge Features for this Subgraph ---
            # The 'batch' object contains the computation subgraph.
            # 'batch.e_id' stores the *original* indices of the edges
            # that are part of this computation subgraph.
            batch_edge_attr = edge_attr_tensor[batch.e_id]

            # --- Run the Encoder ---
            # Get embeddings for the nodes in this subgraph
            z = model.encode(batch.x, batch.edge_index, batch_edge_attr)

            # --- Calculate Loss ---
            # Get embeddings for *only* the positive and negative sample edges
            pos_out = model.decode(z, batch.edge_label_index[:, batch.edge_label == 1])
            neg_out = model.decode(z, batch.edge_label_index[:, batch.edge_label == 0])

            # Combine scores and create labels for loss calculation
            out = torch.cat([pos_out, neg_out], dim=0)
            labels = torch.cat([torch.ones(pos_out.size(0)),
                                torch.zeros(neg_out.size(0))], dim=0).to(device)

            loss = nn.BCEWithLogitsLoss()(out.view(-1), labels)

            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}")

    # --- Save your trained encoder ---
    torch.save(model.encoder.state_dict(), "author_encoder.pth")


def preprocess_and_train(alignments_file: str, alignment_counts: int, model_name: str):

    # --- 1. PRE-PROCESSING ---
    print("--- Building Author Map ---")
    author_to_id = build_author_to_id_mapping(alignments_file, alignment_counts)
    num_authors = len(author_to_id)

    # --- Load SBERT model and get dims ---
    sbert_model = SentenceTransformer(model_name, device='cuda')
    sbert_dim = sbert_model.get_sentence_embedding_dimension()

    # --- Create memmap and edge_index ---
    edge_attr_memmap = np.memmap("edge_features.memmap", dtype='float32', mode='w+', shape=(alignment_counts, sbert_dim))
    edge_index_list = []
    passages_batch = []
    batch_idx = 0
    BATCH_SIZE = 128 # Define batch size

    print("--- Streaming Alignments and Building Features ---")
    with lz4.frame.open(alignments_file, 'rb') as f:
        for i, line in enumerate(tqdm(f, total=alignment_counts, desc="Creating alignment embeddings")):
            alignment = orjson.loads(line)
            source_id = author_to_id[alignment["source_author"]]
            target_id = author_to_id[alignment["target_author"]]
            edge_index_list.append((source_id, target_id))
            passages_batch.append(alignment["source_passage"])

            if len(passages_batch) == BATCH_SIZE or i == alignment_counts - 1:
                embeddings = sbert_model.encode(passages_batch, convert_to_tensor=True, device='cuda')
                start = batch_idx * BATCH_SIZE
                end = start + len(passages_batch)
                edge_attr_memmap[start:end] = embeddings.cpu().numpy()
                passages_batch = []
                batch_idx += 1

    edge_attr_memmap.flush()
    del edge_attr_memmap  # Close the file

    # --- Create the PyG Data object (structure only) ---
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    node_ids = torch.arange(num_authors) # Node features are just their IDs
    data = Data(x=node_ids, edge_index=edge_index)
    data.num_nodes = num_authors

    del edge_index_list # Free memory

    # --- 2. TRAINING ---
    print("--- Initializing GNN Model ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = GNNEncoder(num_authors=num_authors,
                         author_embed_dim=AUTHOR_EMBED_DIM,
                         sbert_dim=sbert_dim,
                         out_embed_dim=OUT_EMBED_DIM)

    model = GAE(encoder).to(device) # Wrap in GAE and move to device

    # --- Load on-disk edge features (RAM-efficient) ---
    memmap_path = 'edge_features.memmap'
    memmap_shape = (alignment_counts, sbert_dim)
    edge_attr_numpy = np.memmap(memmap_path,
                                dtype='float32',
                                mode='r',
                                shape=memmap_shape)

    # --- Create the Mini-Batch Loader ---
    loader = LinkNeighborLoader(
        data=data,
        batch_size=512,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 5], # <-- Fixed
        edge_label_index=data.edge_index,
        edge_label=torch.ones(data.num_edges),
        num_workers=4,
        pin_memory=True, # Helps speed up GPU data transfer
    )

    optimizer = Adam(model.parameters(), lr=0.001)

    print("--- Starting GNN Training ---")
    for epoch in range(5): # 5 epochs is a good start for massive graphs
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/5"):
            batch = batch.to(device)

            edge_ids = batch.e_id.cpu()
            batch_edge_attr_numpy = edge_attr_numpy[edge_ids].copy()

            # Get the edge features for this specific subgraph
            batch_edge_attr = torch.from_numpy(batch_edge_attr_numpy).to(device)

            # Run Encoder
            z = model.encode(batch.x, batch.edge_index, batch_edge_attr)

            # Get positive and negative link predictions
            pos_out = model.decode(z, batch.edge_label_index[:, batch.edge_label == 1])
            neg_out = model.decode(z, batch.edge_label_index[:, batch.edge_label == 0])

            out = torch.cat([pos_out, neg_out], dim=0)
            labels = torch.cat([torch.ones(pos_out.size(0)),
                                torch.zeros(neg_out.size(0))], dim=0).to(device)

            loss = nn.BCEWithLogitsLoss()(out.view(-1), labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch: {epoch:03d}, Avg. Loss: {total_loss / len(loader):.4f}")

    # --- Save your trained encoder ---
    print("Training Complete. Saving Encoder and config...")
    output_path = os.path.join(os.path.dirname(alignments_file), "graph_model")

    os.makedirs(output_path, exist_ok=True)

    torch.save(model.encoder.state_dict(), os.path.join(output_path, "author_encoder.pth"))

    config = {
        "num_authors": num_authors,
        "author_embed_dim": AUTHOR_EMBED_DIM,
        "sbert_dim": sbert_dim,
        "out_embed_dim": OUT_EMBED_DIM
    }
    with open(os.path.join(output_path, "model_config.json"), "wb") as f:
        f.write(orjson.dumps(config))

    with open(os.path.join(output_path, "author_to_id.json"), "wb") as f:
        f.write(orjson.dumps(author_to_id))

    torch.save(data.edge_index, os.path.join(output_path, "edge_index.pt"))

if __name__ == "__main__":
    alignments_file = sys.argv[1]
    alignment_counts = int(sys.argv[2])
    sbert_model_name = "all-MiniLM-L6-v2"
    preprocess_and_train(alignments_file, alignment_counts, sbert_model_name)