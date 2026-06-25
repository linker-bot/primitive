# Single object surface mesh (vertices + triangle faces)
string label
uint32 instance_id
string frame_id
uint32 num_vertices
uint32 num_faces
float32[] vertices    # Nx3 flatten: [x0,y0,z0, x1,y1,z1, ...]
uint32[] faces        # Mx3 flatten: [i0,j0,k0, i1,j1,k1, ...]
