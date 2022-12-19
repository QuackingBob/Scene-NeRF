import torch.nn as nn
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
    def __init__(self) -> None:
        super().__init__()


class VQAD(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class DensityFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class RGBAFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()