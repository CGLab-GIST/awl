# Adaptively weighted discrete Laplacian
@@todo more writing here

## Implementation
1. Official implementation of ""
2. based on pytorch framework, inspried idea from "Continuos Remehing for Inverse Rendering" and "Geometry-central"

## Role

## Usage
```python
from lap import laplacian_cot, laplacian_mis

    L_cotangent = laplacian_cot(mesh.verts, mesh.faces)
    L_mix = laplacian_mis(mesh.verts, mesh.faces, LAMBDA, [LAMBDA_SCALE_MIN, LAMBDA_SCALE_MAX])
```

Here, LAMBDA, $\lambda$ is a scalar diffusion coefficient, smoothing factor for 
$$ f^{n+1} = (I+\lambda L_{mix})^{step} \cdot f^{n}$$

and LAMBDA_SCALE_MIN and MAX is a scale factor for the specific integration method and diffusion step(forward or backward)


