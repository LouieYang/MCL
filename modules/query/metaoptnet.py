try:
    from qpth.qp import QPFunction
except ImportError:
    print("Can't find qpth, MetaOptNet won't work properly")
    print("Hint: pip install cvxpy qpth")
    pass
from torch.autograd import Variable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

@registry.Query.register("MetaOptNet")
class MetaOptNet(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @inproceedings{lee2019meta,
            title={Meta-Learning with Differentiable Convex Optimization},
            author={Kwonjoon Lee and Subhransu Maji and Avinash Ravichandran and Stefano Soatto},
            booktitle={CVPR},
            year={2019}
        }

        https://github.com/kjunelee/MetaOptNet/
        """

        self.cfg = cfg

        self.C_reg = 0.1
        self.maxIter = 3
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

    def batched_kronecker(self, matrix1, matrix2):
        matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
        matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
        return torch.bmm(
            matrix1_flatten.unsqueeze(2), 
            matrix2_flatten.unsqueeze(1)).reshape(
                [matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        query, support = self.pooling(support_xf, support_y, query_xf, query_y, n_way, k_shot)

        tasks_per_batch = query.size(0)
        n_support = support.size(1)
        n_query = query.size(1)

        assert(query.dim() == 3)
        assert(support.dim() == 3)
        assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert(n_support == n_way * k_shot)      # n_support must equal to n_way * k_shot

        #Here we solve the dual problem:
        #Note that the classes are indexed by m & samples are indexed by i.
        #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
        #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

        #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        #and C^m_i = C if m  = y_i,
        #C^m_i = 0 if m != y_i.
        #This borrows the notation of liblinear.

        #\alpha is an (n_support, n_way) matrix
        kernel_matrix = self.computeGramMatrix(support, support)

        id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
        block_kernel_matrix = self.batched_kronecker(kernel_matrix, id_matrix_0)
        #This seems to help avoid PSD error from the QP solver.
        block_kernel_matrix += 1.0 * torch.eye(n_way*n_support).expand(tasks_per_batch, n_way*n_support, n_way*n_support).cuda()

        support_labels_one_hot = F.one_hot(support_y.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_support)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)

        G = block_kernel_matrix
        e = -1.0 * support_labels_one_hot
        #print (G.size())
        #This part is for the inequality constraints:
        #\alpha^m_i <= C^m_i \forall m,i
        #where C^m_i = C if m  = y_i,
        #C^m_i = 0 if m != y_i.
        id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
        C = Variable(id_matrix_1)
        h = Variable(self.C_reg * support_labels_one_hot)
        #print (C.size(), h.size())
        #This part is for the equality constraints:
        #\sum_m \alpha^m_i=0 \forall i
        id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        A = Variable(self.batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
        b = Variable(torch.zeros(tasks_per_batch, n_support))
        #print (A.size(), b.size())
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        qp_sol = QPFunction(verbose=False, maxIter=self.maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

        # Compute the classification score.
        compatibility = self.computeGramMatrix(support, query)
        compatibility = compatibility.float()
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
        qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
        logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
        logits = logits * compatibility
        logits = torch.sum(logits, 1)

        logits = self.scale * logits.view(-1, n_way)
        query_y = query_y.view(-1)
        if self.training:
            loss = self.criterion(logits, query_y)
            return {"MetaOptNet_loss": loss}
        else:
            _, predict_labels = torch.max(logits, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
