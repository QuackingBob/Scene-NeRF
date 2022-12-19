import torch.nn as nn
import torch
import numpy as np

class OctreeNode:
    node_id = 0

    def __init__(self, x, y, z, size, data=None):
        self.x = x
        self.y = y
        self.z = z
        self.size = size
        self.data = data
        self.children = [None, None, None, None, None, None, None, None]
        self.id = OctreeNode.node_id
        OctreeNode.node_id += 1

class Octree:
    def __init__(self, x, y, z, size):
        self.nodes = {}
        self.root = OctreeNode(x, y, z, size)
        self.nodes[self.root.id] = self.root

    def insert(self, x, y, z, data):
        self._insert(self.root, x, y, z, data)

    def _insert(self, node, x, y, z, data):
        if node.children[0] is None:
            # Node is a leaf, so add the data here
            node.data = data
        else:
            # Node is not a leaf, so figure out which child to insert into
            index = 0
            if x > node.x:
                index += 1
            if y > node.y:
                index += 2
            if z > node.z:
                index += 4
            if node.children[index] is None:
                node.children[index] = OctreeNode(node.x + (index & 1) * node.size / 2,
                                                  node.y + ((index & 2) >> 1) * node.size / 2,
                                                  node.z + ((index & 4) >> 2) * node.size / 2,
                                                  node.size / 2)
                self.nodes[node.children[index].id] = node.children[index]
            self._insert(node.children[index], x, y, z, data)
        return

    def search(self, x, y, z):
        return self._search(self.root, x, y, z)

    def _search(self, node, x, y, z):
        if node.data is not None:
            # Node is a leaf, so return the data
            return node.data
        else:
            # Node is not a leaf, so figure out which child to search in
            index = 0
            if x > node.x:
                index += 1
            if y > node.y:
                index += 2
            if z > node.z:
                index += 4
            if node.children[index] is None:
                return None
            return self._search(node.children[index], x, y, z)

class PositionalEmbedder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_emb = torch.zeros(x.size(0), self.embedding_dim)
        for i in range(self.embedding_dim):
            if i % 2 == 0:
                pos_emb[:, i] = x[:, 0] / np.power(10000, i / self.embedding_dim)
            else:
                pos_emb[:, i] = x[:, 1] / np.power(10000, (i - 1) / self.embedding_dim)
        return pos_emb



class VQAD(nn.Module):
    def __init__(self, embedding_dim: int, codebook_size: int) -> None:
        super().__init__()
        self.positional_embedder = PositionalEmbedder(embedding_dim)
        self.linear = nn.Linear(embedding_dim, codebook_size)
        self.softmax = nn.Softmax(dim=1)
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.octree = Octree(0, 0, 0, 1)
        self.features = []

    def _search_node(self, node, x, y, z):
        if node.children[0] is None:
            # Node is a leaf, so return the data
            return node.data
        else:
            # Node is not a leaf, so figure out which child to search in
            index = 0
            if x > node.x:
                index += 1
            if y > node.y:
                index += 2
            if z > node.z:
                index += 4
            if node.children[index] is None:
                return None
            return self._search_node(node.children[index], x, y, z)

    def _search(self, node, x, y, z, distances):
        if node.data is not None:
            # Node is a leaf, so add the distance to the list
            distances.append(np.linalg.norm([x - node.x, y - node.y, z - node.z]))
        else:
            # Node is not a leaf, so search its children
            for child in node.children:
                if child is not None:
                    self._search(child, x, y, z, distances)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate distances to all nodes in the octree
        distances = []
        self._search(self.octree.root, x[:, 0], x[:, 1], x[:, 2], distances)
        # Convert distances to probabilities using the softmax function
        probs = self.softmax(torch.tensor(distances))
        # Find the closest feature index in the codebook using the probabilities
        _, indices = probs.max(dim=0)
        # Return the corresponding feature vector from the codebook
        return self.codebook(indices)


class DensityHiddenEncoding(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_size)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DensityFunction(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class RGBAFunction(nn.Module):
    def __init__(self, density_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(density_dim + 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_r = nn.Linear(64, 1)
        self.fc_g = nn.Linear(64, 1)
        self.fc_b = nn.Linear(64, 1)
        self.fc_a = nn.Linear(64, 1)

    def forward(self, density: torch.Tensor, heading: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([density, heading], dim=1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        r = self.fc_r(x)
        g = self.fc_g(x)
        b = self.fc_b(x)
        a = self.fc_a(x)
        return r, g, b, a
