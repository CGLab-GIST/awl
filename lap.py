import torch
from torch_scatter import scatter, scatter_max, scatter_min
import numpy as np

class Geometry:

    def __init__(self, v, f):
        '''
        for the caching
        '''
        self._v = v
        self._f = f
        self._nv = v.shape[0]
        self._nf = f.shape[0]
        self.device = v.device
        self._v_deg = None

        self._vertex_dualareas = None
        self._vertex_bandwidth_cotan = None
        self._vertex_bandwidth_uniform = None

        self._face_areas = None

        self._e = None
        self._ne = None
        self._edge_lengths = None

        self._indices = None
        self._coalesce_indices = None
        self._lap_uniform_values = None
        self._lap_cotangent_values = None
        self._lap_kernel_fix = None
        self._lap_kernel_ada = None
        self._lap_kernel_mix = None

        self._laplacian_uniform = None
        self._laplacian_cotangent = None
        self._laplacian_kernel = None
        self._laplacian_ada = None
        self._laplacian_mix = None
        self._laplacian_mixture = None

        self._adj = None
        self._mesh_length = None
        self._shape_scale = None

        self.local_min = None
        self.local_max = None
        self.local_scale = None

    def __calc_adjacency_list__(self):
        if self._adj is not None:
            return

        ii = self._f[:, [1, 2, 0]].flatten()
        jj = self._f[:, [2, 0, 1]].flatten()
        self._adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)

    def __calc_vertex_degree__(self):
        if self._v_deg is not None:
            return 
        self.__calc_adjacency_list__()

        self._v_deg = torch.unique(self._adj[0], return_counts=True)[1]

    def __calc_edges__(self):
        """
        from Continous remeshing. Palfinger 2022 CGI
        returns tuple of
        - edges E,2 long, 0 for unused, lower vertex index first
        - face_to_edge F,3 long
        - (optional) edge_to_face shape=E,[left,right],[face,side]

        o-<-----e1     e0,e1...edge, e0<e1
        |      /A      L,R....left and right face
        |  L /  |      both triangles ordered counter clockwise
        |  / R  |      normals pointing out of screen
        V/      |      
        e0---->-o     
        """
        if self._e is not None:
            return

        F = self._nf
        
        # make full edges, lower vertex index first
        face_edges = torch.stack((self._f, self._f.roll(-1,1)),dim=-1) #F*3,3,2
        full_edges = face_edges.reshape(F*3,2)
        sorted_edges,_ = full_edges.sort(dim=-1) #F*3,2 TODO min/max faster?

        # make unique edges
        self._e, full_to_unique = torch.unique(input=sorted_edges, sorted=True, return_inverse=True, dim=0) #(E,2),(F*3)
        self._ne = self._e.shape[0]
        self._face_to_edge = full_to_unique.reshape(F,3) #F,3

        is_right = full_edges[:,0]!=sorted_edges[:,0] #F*3
        edge_to_face = torch.zeros((self._ne,2,2), dtype=torch.long, device=self.device) #E,LR=2,S=2
        scatter_src = torch.cartesian_prod(torch.arange(0,F,device=self.device),torch.arange(0,3,device=self.device)) #F*3,2
        edge_to_face.reshape(2*self._ne,2).scatter_(dim=0,index=(2*full_to_unique+is_right)[:,None].expand(F*3,2),src=scatter_src) #E,LR=2,S=2
        edge_to_face[0] = 0
        self._edge_to_face = edge_to_face

    def __calc_edge_lengths__(self):
        if self._edge_lengths is not None:
            return 

        self.__calc_edges__()
        
        full_vertices = self._v[self._e] #E,2,3
        a,b = full_vertices.unbind(dim=1) #E,3
        self._edge_lengths = torch.norm(a-b,p=2,dim=-1)

    def __calc_mesh_length__(self):
        if self._mesh_length is not None:
            return 
        
        self.__calc_edge_lengths__()
        self._mesh_length = torch.sum(self._edge_lengths)/self._ne

    def __calc_faces_areas__(self):
        if self._face_areas is not None:
            return
        face_verts = self._v[self._f]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        s = 0.5 * (A + B + C)
        self._face_areas = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()
    
    def __calc_vertex_dualareas__(self):
        if self._vertex_dualareas is not None:
            return
        self.__calc_faces_areas__()
        vertex_dualareas = torch.zeros((self._nv, 3),dtype=self._v.dtype,device=self.device) #V,C=3,3
        vertex_dualareas.scatter_add_(dim=0,index=self._f,src=self._face_areas[:, None].expand(self._nf, 3))
        self._vertex_dualareas = vertex_dualareas.sum(dim=1) / 3.0 #V,3

    def __calc_shape_scale__(self):
        if self._shape_scale is not None:
            return
        self.__calc_faces_areas__()
        self._shape_scale = torch.sqrt(torch.sum(self._face_areas))

    def __calc_sparse_coo_indices__(self):
        if self._indices is not None:
            return
        self.__calc_adjacency_list__()
        diag_idx = self._adj[0]
        self._indices = torch.cat((self._adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    
    def __calc_laplacian_uniform_indices_and_values(self, normalize=True, return_matrix=True):
        if self._lap_uniform_values is not None:
            return

        self.__calc_adjacency_list__()
        self.__calc_vertex_degree__()
        self.__calc_sparse_coo_indices__()

        adj = self._adj
        adj_values = torch.ones(adj.shape[1], dtype=torch.float, device=self.device)

        # normalization
        if normalize:        
            deg = self._v_deg
            adj_values = torch.div(adj_values[adj[1]], deg[adj[0]])

        # Diagonal indicess
        self._lap_uniform_values = torch.cat((-adj_values, adj_values))
        self._laplacian_uniform = torch.sparse_coo_tensor(self._indices, self._lap_uniform_values, (self._nv,self._nv))
        if self._coalesce_indices is None:
            self._coalesce_indices = self._laplacian_uniform.indices()
    
    def __calc_laplacian_cotangent_indices_and_values__(self, normalize=True):
        """
        https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
        """
        if self._laplacian_cotangent is not None:
            return
        
        face_verts = self._v[self._f]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)
        
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2)
        cotb = (A2 + C2 - B2)
        cotc = (A2 + B2 - C2)
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 2.0

        ii = self._f[:, [1, 2, 0]]
        jj = self._f[:, [2, 0, 1]]
        idx = torch.stack([ii, jj], dim=0).view(2, self._nf * 3)
        L = torch.sparse.FloatTensor(idx, cot.view(-1), (self._nv, self._nv))
        L += L.t()
        L = L.coalesce()

        values = torch.sparse.sum(L, dim=0).to_dense()
        sym_indices = torch.arange(self._nv, device=self.device)
        sym_indices = torch.stack([sym_indices, sym_indices], dim=0)

        if normalize:
            values = 1./values
            L_indices = L.indices()
            D = torch.sparse.FloatTensor(L_indices, L.values()*values[L_indices[0]], (self._nv, self._nv))
            L = torch.sparse.FloatTensor(sym_indices, torch.ones_like(values), (self._nv, self._nv)) - D
            self._laplacian_cotangent = L.coalesce()
            self._lap_cotangent_values = self._laplacian_cotangent.values()
            if self._coalesce_indices is None:
                self._coalesce_indices = self._laplacian_cotangent.indices()
        else:
            L = torch.sparse.FloatTensor(sym_indices, values, (self._nv, self._nv)) - L
            self._laplacian_cotangent = L.coalesce()
            self._lap_cotangent_values = self._laplacian_cotangent.values()
            if self._coalesce_indices is None:
                self._coalesce_indices = self._laplacian_cotangent.indices()
    
    def __calc_vertex_normalize_local__(self):

        if self.local_min is not None:
            return

        adj_verts = self._v[self._coalesce_indices[1]]

        mins, _ = scatter_min(src=adj_verts, index=self._coalesce_indices[0][:, None].expand(-1, 3), dim=0)
        maxs, _ = scatter_max(src=adj_verts, index=self._coalesce_indices[0][:, None].expand(-1, 3), dim=0)

        range_ = (maxs-mins)/2.0
        mins = mins-range_
        maxs = maxs+range_

        self.local_min, self.local_max = mins, maxs
    
    def __calc_laplacian_kernelized_indices_and_values__(self, bandwidth=None, normalize=True):
        
        if self._laplacian_kernel is not None:
            return

        self.__calc_sparse_coo_indices__()
        self.__calc_vertex_normalize_local__()
        
        h = bandwidth
        if h is None:
            self.__calc_vertex_degree__()
            self.__calc_edge_lengths__()
            h = scatter(self._edge_lengths, index=self._e[:, 1], dim=0)/self._v_deg
            h = torch.max(((h[self._adj[1]] + h[self._adj[0]])/2.0)**(0.4), torch.tensor([1e-8], device=self.device))

        mmrange = 1.0
        if normalize:
            mmrange = 2.0/(self.local_max[self._adj[0]] - self.local_min[self._adj[0]])
        local_v = (self._v[self._adj[0]]-self._v[self._adj[1]])*mmrange
        distance = (local_v).square().sum(dim=1)
        values = self.__gaussian(h, distance)/h

        summ = scatter(values, index=self._adj[1], dim=0)
        values = torch.div(values, summ[self._adj[1]])

        values = torch.cat((-values, values))
        L = torch.sparse.FloatTensor(self._indices, values, (self._nv, self._nv))

        self._laplacian_kernel = L.coalesce()
        self._lap_kernel_fix = self._laplacian_kernel.values()
        if self._coalesce_indices is None:
            self._coalesce_indices = self._laplacian_kernel.indices()
        return

    def __calc_asymtotic_bandwidth__(self, coeff, scale):
        self.__calc_vertex_dualareas__()
        self.__calc_mesh_length__()
        self.__calc_laplacian_cotangent_indices_and_values__(True)
        self.__calc_laplacian_kernelized_indices_and_values__(None, True)

        # local normalized coordinates
        g_scale = torch.norm(self.local_max-self.local_min, dim=1, p=2) * scale
        h_common = 3 * coeff * self._vertex_dualareas / (torch.pi)
        mmrange = 2.0/(self.local_max[self._coalesce_indices[0]] - self.local_min[self._coalesce_indices[0]])
        local_v = (self._v[self._coalesce_indices[0]]-self._v[self._coalesce_indices[1]])*mmrange

        # gather edge weights
        Wker = self._lap_kernel_fix
        Wcot = self._lap_cotangent_values

        Lu = scatter(src=Wker[:, None].expand(-1, 3)*local_v, index=self._coalesce_indices[0], dim=0, reduce='sum')
        Ln = scatter(src=Wcot[:, None].expand(-1, 3)*local_v, index=self._coalesce_indices[0], dim=0, reduce='sum') 

        Lu = torch.sum(torch.square(Lu), dim=1)
        hu = h_common/torch.max(torch.tensor([1e-8], device=self.device), Lu)
        hu = torch.pow(hu, 1.0/7.0)

        Ln = torch.sum(torch.square(Ln), dim=1)
        hn = h_common/torch.max(torch.tensor([1e-8], device=self.device), Ln)
        hn = torch.pow(hn, 1.0/7.0) 

        self._vertex_bandwidth_uniform = hu * g_scale
        self._vertex_bandwidth_cotan = hn * g_scale

    def __gaussian(
        self,
        h:torch.tensor,   # input bandwidth
        dist:torch.tensor # L2 distance between two vertices
        )->torch.tensor:

        const = 1.0/(4*torch.pi*h);
        return torch.exp(-1.0 * dist/(4.0*h)) * const
    
    def __calc_laplacian_adaptive_indices_and_values__(self, coeff, scale_range, cotan):
        '''
        None mixed version
        '''
        if self._laplacian_ada is not None:
            return

        self.__calc_asymtotic_bandwidth__(coeff, scale_range)
        self.__calc_sparse_coo_indices__()
        
        if cotan == True:
            h = self._vertex_bandwidth_cotan
        else:
            h = self._vertex_bandwidth_uniform

        distance = (self._v[self._adj[0]] - self._v[self._adj[1]]).square().sum(dim=1)
        w0 = self.__gaussian(h[self._adj[0]], distance)
        w1 = self.__gaussian(h[self._adj[1]], distance)

        val0 = w0/h[self._adj[0]]*self._vertex_dualareas[self._adj[0]]
        val1 = w1/h[self._adj[1]]*self._vertex_dualareas[self._adj[1]]
        adj_values = (val0*val1).sqrt()

        values = adj_values
        values = torch.cat((-values, values))
        L = torch.sparse.FloatTensor(self._indices, values, (self._nv, self._nv))

        self._laplacian_ada = L.coalesce()
        self._lap_kernel_ada = self._laplacian_ada.values()
        if self._coalesce_indices is None:
            self._coalesce_indices = self._laplacian_ada.indices()

    def __calc_laplacian_mixed_indices_and_values__(self, coeff, scale):
        '''
        weight mix
        '''
        if self._laplacian_mix is not None:
            return

        self.__calc_asymtotic_bandwidth__(coeff, scale)
        self.__calc_sparse_coo_indices__()

        h0 = self._vertex_bandwidth_cotan
        h1 = self._vertex_bandwidth_uniform

        distance = (self._v[self._adj[0]] - self._v[self._adj[1]]).square().sum(dim=1)
        w00 = self.__gaussian(h0[self._adj[0]], distance)
        w01 = self.__gaussian(h0[self._adj[1]], distance)
        w10 = self.__gaussian(h1[self._adj[0]], distance)
        w11 = self.__gaussian(h1[self._adj[1]], distance)

        w002, w012 = w00, w01
        w102, w112 = w10, w11
        Wsum0, Wsum1 = w002+w102, w012+w112 
 
        val1 = (w00*w002/Wsum0/h0[self._adj[0]] + w10*w102/Wsum0/h1[self._adj[0]])*self._vertex_dualareas[self._adj[0]]
        val2 = (w01*w012/Wsum1/h0[self._adj[1]] + w11*w112/Wsum1/h1[self._adj[1]])*self._vertex_dualareas[self._adj[1]]
        adj_values = torch.sqrt(val1*val2)

        values = adj_values
        values = torch.cat((-values, values))
        L = torch.sparse.FloatTensor(self._indices, values, (self._nv, self._nv))

        self._laplacian_mix = L.coalesce()
        self._lap_kernel_mix = self._laplacian_mix.values()
        if self._coalesce_indices is None:
            self._coalesce_indices = self._laplacian_mix.indices()

    def __calc_sparse_coo__(self, indices, values):
        return torch.sparse_coo_tensor(indices, values, (self._nv, self._nv)).coalesce()

    '''    
    public functions for real usage
    '''
    def laplacian_uniform(self):
        if self._laplacian_uniform is not None:
            return self._laplacian_uniform

        self.__calc_laplacian_uniform_indices_and_values()
        return self._laplacian_uniform
    
    def laplacian_cotangent(self):
        if self._laplacian_cotangent is not None:
            return self._laplacian_cotangent

        self.__calc_laplacian_cotangent_indices_and_values__()
        return self._laplacian_cotangent
    
    def laplacian_adaptive(self, weight, scale):
        if self._laplacian_mix is not None:
            return self._laplacian_mix.coalesce()
        self.__calc_laplacian_mixed_indices_and_values__(weight, scale)
        return self._laplacian_mix.coalesce()

# accessing functions: use here
def vertex_dualareas(verts, faces):
    geom = Geometry(verts, faces)
    geom.__calc_vertex_dualareas__()
    return geom._vertex_dualareas

def asymtotic_bandwidth(verts, faces, smoothing_weight):
    geom = Geometry(verts, faces)
    geom.__calc_asymtotic_bandwidth__(smoothing_weight)
    return geom._vertex_bandwidth_cotan, geom._vertex_bandwidth_uniform

def laplacian_uniform(verts, faces):
    geom = Geometry(verts, faces)
    return geom.laplacian_uniform()

def laplacian_cotangent(verts, faces, weight=None):
    geom = Geometry(verts, faces)
    return geom.laplacian_cotangent()

def laplacian_adaptive(verts, faces, weight, scale):
    geom = Geometry(verts, faces)
    return geom.laplacian_adaptive(weight, scale)

def csc_cpu_to_coo_gpu(csc_cpu):
    coo = csc_cpu.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    ind = torch.tensor(indices, dtype=torch.long)
    val = torch.tensor(values, dtype=torch.float64)
    L = torch.sparse_coo_tensor(ind, val, coo.shape).cuda()
    return L.coalesce()

# testing
if __name__ == "__main__":
    pos = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    ind = [[0, 1, 2]]

    v, f = torch.tensor(pos, dtype=torch.float64), torch.tensor(ind, dtype=torch.long)
    
    cot = laplacian_cotangent(v, f)
    ada = laplacian_adaptive(v, f, 0.96, 0.1)