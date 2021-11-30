import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

@registry.Query.register("R2D2")
class R2D2(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @inproceedings{bertinetto2018meta,
            title={Meta-learning with differentiable closed-form solvers},
            author={Bertinetto, Luca and Henriques, Joao F and Torr, Philip and Vedaldi, Andrea},
            booktitle={International Conference on Learning Representations},
            year={2018}
        }

        https://github.com/kjunelee/MetaOptNet/
        """

        self.cfg = cfg
        self.l2_regularizer_lambda = 50.0
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.criterion = nn.CrossEntropyLoss()

    def pooling(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        query = F.adaptive_avg_pool2d(query_xf.view(-1, c, h, w), 1).view(b, q, c)
        support = F.adaptive_avg_pool2d(support_xf.view(-1, c, h, w), 1).view(b, s, c)
        return query, support

    def computeGramMatrix(self, A, B):
        """
        Constructs a linear kernel matrix between A and B.
        We assume that each row in A and B represents a d-dimensional feature vector.
        
        Parameters:
          A:  a (n_batch, n, d) Tensor.
          B:  a (n_batch, m, d) Tensor.
        Returns: a (n_batch, n, m) Tensor.
        """
        
        assert(A.dim() == 3)
        assert(B.dim() == 3)
        assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

        return torch.bmm(A, B.transpose(1,2))

    def binv(self, b_mat):
        """
        Computes an inverse of each matrix in the batch.
        Pytorch 0.4.1 does not support batched matrix inverse.
        Hence, we are solving AX=I.

        Parameters:
          b_mat:  a (n_batch, n, n) Tensor.
        Returns: a (n_batch, n, n) Tensor.
        """

        id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
        b_inv, _ = torch.solve(id_matrix, b_mat)

        return b_inv

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        query, support = self.pooling(support_xf, support_y, query_xf, query_y, n_way, k_shot)

        tasks_per_batch = query.size(0)
        n_support = support.size(1)

        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(n_support == n_way * k_shot)      # n_support must equal to n_way * k_shot

        support_labels_one_hot = F.one_hot(support_y.view(tasks_per_batch * n_support), n_way).float()
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

        id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        # Compute the dual form solution of the ridge regression.
        # W = X^T(X X^T - lambda * I)^(-1) Y
        ridge_sol = self.computeGramMatrix(support, support) + self.l2_regularizer_lambda * id_matrix
        ridge_sol = self.binv(ridge_sol)
        ridge_sol = torch.bmm(support.transpose(1,2), ridge_sol)
        ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)

        # Compute the classification score.
        # score = W X
        logits = torch.bmm(query, ridge_sol)
        logits = self.scale * logits.view(-1, n_way)

        query_y = query_y.view(-1)
        if self.training:
            loss = self.criterion(logits, query_y)
            return {"R2D2_loss": loss}
        else:
            _, predict_labels = torch.max(logits, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
