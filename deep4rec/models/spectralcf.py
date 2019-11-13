"""
Implementation of Spectral Collaborative Filtering

Paper: Spectral Collaborative Filtering
link: https://dl.acm.org/citation.cfm?id=3240343
Authors: Lei Zheng, Chun-Ta Lu, Fei Jiang, Jiawei Zhang, Philip S. Yu
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

from deep4rec.models.model import Model


class SpectralCF(Model):
    def __init__(self, ds, K=3, emb_dim=16, lr=0.001, batch_size=1024, decay=0.001):
        super(SpectralCF, self).__init__()

        self.graph = ds.build_graph()
        self.num_items = ds.num_items
        self.num_users = ds.num_users
        self.K = K
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.decay = decay
        self.ds = ds

        self.matrix_adj = self.adjacent_matrix()
        self.matrix_d = self.degree_matrix()
        self.matrix_l = self.laplacian_matrix(normalized=True)

        self.lamda, self.U = np.linalg.eig(self.matrix_l)
        self.lamda = np.diag(self.lamda)

        self.user_embeddings = tfe.Variable(
            tf.random_normal(
                [self.num_users, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32
            ),
            name="user_embeddings",
        )
        self.item_embeddings = tfe.Variable(
            tf.random_normal(
                [self.num_items, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32
            ),
            name="item_embeddings",
        )

        self.A_hat = np.dot(self.U, self.U.T) + np.dot(
            np.dot(self.U, self.lamda), self.U.T
        )
        self.A_hat = self.A_hat.astype(np.float32)

        self.filters = []
        for k in range(self.K):
            self.filters.append(
                tfe.Variable(
                    tf.random_normal(
                        [self.emb_dim, self.emb_dim],
                        mean=0.01,
                        stddev=0.02,
                        dtype=tf.float32,
                    )
                )
            )

    def adjacent_matrix(self, self_connection=False):
        matrix_adj = np.zeros(
            [self.num_users + self.num_items, self.num_users + self.num_items],
            dtype=np.float32,
        )

        matrix_adj[: self.num_users, self.num_users :] = self.graph
        matrix_adj[self.num_users :, : self.num_users] = self.graph.T

        if self_connection:
            return (
                np.identity(self.n_users + self.n_items, dtype=np.float32) + matrix_adj
            )

        return matrix_adj

    def degree_matrix(self):
        matrix_d = np.sum(self.matrix_adj, axis=1, keepdims=False)
        matrix_d[matrix_d == 0] = 1e-8
        return matrix_d

    def laplacian_matrix(self, normalized=False):
        if not normalized:
            return self.D - self.A

        tmp = np.dot(np.diag(np.power(self.matrix_d, -1)), self.matrix_adj)
        return np.identity(self.num_users + self.num_items, dtype=np.float32) - tmp

    def create_bpr_loss(self, users, pos_items, neg_items):
        def calculate_loss():
            pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
            neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

            regularizer = (
                tf.nn.l2_loss(users)
                + tf.nn.l2_loss(pos_items)
                + tf.nn.l2_loss(neg_items)
            )
            regularizer = regularizer / self.batch_size

            maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
            loss = tf.negative(tf.reduce_mean(maxi)) + self.decay * regularizer
            print(loss)
            return loss

        return calculate_loss
    
    def predict(self, users):
        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.K):
            embeddings = tf.matmul(self.A_hat, embeddings)
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings, self.filters[k]))
            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        self.u_embeddings, self.i_embeddings = tf.split(
            all_embeddings, [self.num_users, self.num_items], 0
        )

        self.u_embeddings = tf.nn.embedding_lookup(self.u_embeddings, users)
        all_ratings = tf.matmul(
            self.u_embeddings, self.i_embeddings, transpose_a=False, transpose_b=True
        )

        return all_ratings


    def call(self, one_hot_features, training=False, features=None, **kwargs):
        """Forward pass.

        Args:
            one_hot_features: A dense tensor of shape [batch_size, self._num_features]
                that indicates which features are present in this input.
            training: A boolean indicating if is training or not.
            features: A dense tensor of shape [batch_size, self._num_features] that indicates
                the value of each feature.

        Returns:
            ratings.
        """
        features = [[], []]

        for feature in one_hot_features:
            features[0].append(feature[0])
            features[1].append(feature[1])

        users, items = features
        users = list(map(lambda user: self.ds.index_user_id[user.numpy()], users))
        items = list(map(lambda item: self.ds.index_item_id[item.numpy()], items))

        all_ratings = self.predict(users)

        users_items = list(enumerate(items))
        ratings = tf.gather_nd(all_ratings, users_items)

        return tf.reshape(ratings, shape=(len(users), 1))
