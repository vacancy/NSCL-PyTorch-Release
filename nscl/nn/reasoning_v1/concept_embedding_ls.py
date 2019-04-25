#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : concept_embedding_ls.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch
import jactorch.nn as jacnn

from jacinle.utils.cache import cached_property

__all__ = [
    'ConceptEmbeddingLS'
]

class ConceptEmbeddingLS(nn.Module):
    def __init__(self, input_dim, concept_embedding_dim, attribute_embedding_dim):
        super().__init__()

        self.input_dim = input_dim
        self.concept_embedding_dim = concept_embedding_dim
        self.attribute_embedding_dim = attribute_embedding_dim

        self.all_attributes = list()
        self.all_concepts = list()

        self.input_transform = nn.Linear(self.input_dim, self.concept_embedding_dim)
        self.prior = nn.Sequential(
            nn.Linear(self.concept_embedding_dim + self.attribute_embedding_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )

    @property
    def nr_attributes(self):
        return len(self.all_attributes)

    @property
    def nr_concepts(self):
        return len(self.all_concepts)

    @cached_property
    def attribute2id(self):
        return {a: i for i, a in enumerate(self.all_attributes)}

    @cached_property
    def concept2id(self):
        return {a: i for i, a in enumerate(self.all_concepts)}

    def init_attributes(self, all_attributes, embeddings=None):
        assert self.nr_attributes == 0
        self.all_attributes.extend([str(x) for x in all_attributes])
        self.all_attributes.sort()
        if self.nr_attributes == 0:
            return

        self.attribute_embeddings = nn.Embedding(len(self.all_attributes), self.attribute_embedding_dim)
        if embeddings is not None:
            attr_embeddings = torch.tensor([embeddings[k] for k in self.all_attributes], dtype=torch.float32)
            self.attribute_embeddings.weight.data.copy_(attr_embeddings)

    def init_concepts(self, all_concepts, embeddings=None):
        assert self.nr_concepts == 0
        self.all_concepts.extend([str(x) for x in all_concepts])
        self.all_concepts.sort()
        self.concept_embeddings = nn.Embedding(len(self.all_concepts), self.concept_embedding_dim)
        if embeddings is not None:
            conc_embeddings = torch.tensor([embeddings[k] for k in self.all_concepts], dtype=torch.float32)
            self.concept_embeddings.weight.data.copy_(conc_embeddings)

    def normalized_embedding(self, value):
        """L2-normalized embedding in all spaces."""
        embedding = value / value.norm(2, dim=-1, keepdim=True)
        return embedding

    def get_concept(self, concept):
        device = self.concept_embeddings.weight.device
        return self.normalized_embedding(self.concept_embeddings(torch.tensor([self.concept2id[concept]], dtype=torch.long, device=device))[0])

    def get_concepts(self, concepts, normalize=True):
        device = self.concept_embeddings.weight.device
        indices = torch.tensor([self.concept2id[c] for c in concepts], dtype=torch.long, device=device)
        if normalize:
            return self.normalized_embedding(self.concept_embeddings(indices))
        return self.concept_embeddings(indices)

    def get_all_concepts(self, normalize=True):
        if normalize:
            return self.normalized_embedding(self.concept_embeddings.weight)
        return self.concept_embeddings.weight

    def get_attribute(self, concept):
        device = self.attribute_embeddings.weight.device
        return self.attribute_embeddings(torch.tensor([self.attribute2id[concept]], dtype=torch.long, device=device))[0]

    _margin = 0.85
    _margin_cross = 0.5
    _tau = 0.25

    def similarity(self, query, identifier):
        concept = self.get_concept(identifier)

        # shape: [batch, channel] or [channel]
        query_mapped = self.input_transform(query)
        query_mapped = query_mapped / query_mapped.norm(2, dim=-1, keepdim=True)
        reference = jactorch.add_dim_as_except(concept, query_mapped, -1)

        margin = self._margin
        logits = ((query_mapped * reference).sum(dim=-1) - 1 + margin) / margin / self._tau

        return logits

    def query_attribute(self, query, identifier):
        query = self.input_transform(query)
        query = query / query.norm(2, dim=-1, keepdim=True)
        query = query.unsqueeze(-2)

        concept_embeddings = self.get_all_concepts()
        concept_embeddings = jactorch.add_dim_as_except(concept_embeddings, query, -2, -1)
        margin = self._margin
        mask = ((query * concept_embeddings).sum(dim=-1) - 1 + margin) / margin / self._tau

        concept_embeddings = self.get_all_concepts(normalize=False)
        attribute_embedding = self.get_attribute(identifier)
        prior = self.prior(torch.cat([concept_embeddings, jactorch.add_dim(attribute_embedding, 0, concept_embeddings.size(0))], dim=-1)).squeeze(-1)
        prior = jactorch.add_dim_as_except(prior, mask, -1)

        answer = jactorch.logits_and(mask, prior)

        return answer, self.concept2id

    def query_attribute_mc(self, query, identifier, concepts):
        query = self.input_transform(query)
        query = query / query.norm(2, dim=-1, keepdim=True)
        query = query.unsqueeze(-2)

        concept2id = {k: i for i, k in enumerate(concepts)}

        concept_embeddings = self.get_concepts(concepts)
        concept_embeddings = jactorch.add_dim_as_except(concept_embeddings, query, -2, -1)
        margin = self._margin
        mask = ((query * concept_embeddings).sum(dim=-1) - 1 + margin) / margin / self._tau

        concept_embeddings = self.get_concepts(concepts, normalize=False)
        attribute_embedding = self.get_attribute(identifier)
        prior = self.prior(torch.cat([concept_embeddings, jactorch.add_dim(attribute_embedding, 0, concept_embeddings.size(0))], dim=-1)).squeeze(-1)
        prior = jactorch.add_dim_as_except(prior, mask, -1)

        answer = jactorch.logits_and(mask, prior)

        return answer, concept2id
