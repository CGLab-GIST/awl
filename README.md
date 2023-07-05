# [Adaptively weighted discrete Laplacian for inverse rendering](https://cglab.gist.ac.kr/visualcomputer23direct/)
[Hyeonjang An](https://github.com/hyeonjang), [Wonjun Lee](https://cglab.gist.ac.kr/people/), [Bochang Moon](https://cglab.gist.ac.kr/people/bochang.html)

## Overview
This code is the official implementation of The Visual Computer paper, [Adaptively weighted discrete Laplacian for inverse rendering](https://cglab.gist.ac.kr/visualcomputer23direct/).
For the further informations, please refer to the project page. 

## Requirements
We recommend running the code under conda environment.
```bash
conda create -f environment.yml
conda activate awl
```

## Usage
We have tested our discrete Laplacian in existing differentiable rendering frameworks, [Large Steps in Inverse Rendering of Geometry](https://github.com/rgl-epfl/large-steps-pytorch), and [Continuous Remeshing For Inverse Rendering](https://github.com/Profactor/continuous-remeshing). Please replace the existing Laplacian in the frameworks with ours for the test.

## Example code

```python
from lap import laplacian_cotangent, laplacian_adaptive
L_c = laplacian_cotangent(mesh.verts, mesh.faces)                   # cotangent Laplacian
L_a = laplacian_adaptive(mesh.verts, mesh.faces, LAMBDA, SCALE)     # adaptively weighted Laplacian
```

Here, LAMBDA, $\lambda$ is the smoothing factor of Laplacian smoothing and SCALE is global scaling parameter for different domains.
We add the a framework-specific parameter SCALE. For example, SCALE is set as 0.1 in LSIG.

## Issue
If there is an issue, please send an email to this address, <hyeonjang2021@gmail.com>

## License
All source codes are released under a BSD License

## Citation
```
@article{An2023,
  title={Adaptively weighted discrete Laplacian for inverse rendering},
  author={An, Hyeonjang and Lee, Wonjun and Moon, Bochang},
  journal={The Visual Computer},
  year={2023},
  issn={1432-2315},
  doi={10.1007/s00371-023-02955-2},
  url={https://doi.org/10.1007/s00371-023-02955-2}
}
```
