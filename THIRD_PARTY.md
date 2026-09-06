# Third-party assets and dependencies

Exact resolved versions, source archives, and hashes are preserved in `artifact/nommd-arena/uv.lock` and `artifact/nommd-arena/opponent-sequence-lab/uv.lock`. This inventory identifies direct research dependencies and bundled third-party assets; transitive dependencies remain governed by their own upstream terms.

| Asset | Version or revision | Creator or source | Upstream terms and release treatment |
| --- | --- | --- | --- |
| GLEE framework, platform, and SDK | SDK 0.0.5 | Shapira et al.; GLEE organizers | Cited in the paper. The SDK is an external dependency, is not vendored, and had no declared repository or package license in the inspected release metadata; this artifact does not redistribute or relicense it. |
| Mamba-2 and `mamba-ssm` | 2.3.2.post1 at revision `e9594ce1c732d97440f0332fdc43170a2294dbfa` | Tri Dao, Albert Gu, and contributors | [Apache-2.0](https://github.com/state-spaces/mamba/blob/main/LICENSE); dependency source is not vendored. |
| `causal-conv1d` | 1.6.2.post1 | Dao-AILab contributors | [BSD-3-Clause](https://github.com/Dao-AILab/causal-conv1d/blob/main/LICENSE); wheel is fetched from the pinned upstream release. |
| PyTorch | 2.10.0+cu130 | PyTorch contributors | [BSD-style license and third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE); dependency is not vendored. |
| NumPy | 2.5.2 | NumPy developers | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt); dependency is not vendored. |
| Polars | 1.43.2 | Polars contributors | [MIT](https://github.com/pola-rs/polars/blob/main/LICENSE); dependency is not vendored. |
| Pydantic | 2.13.4 | Pydantic contributors | [MIT](https://github.com/pydantic/pydantic/blob/main/LICENSE); dependency is not vendored. |
| Requests | 2.34.2 | Kenneth Reitz and contributors | [Apache-2.0](https://github.com/psf/requests/blob/main/LICENSE); dependency is not vendored. |
| pytest | 9.1.1 | pytest contributors | [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE); test dependency is not vendored. |
| Hatchling | resolved by each lockfile | PyPA contributors | [MIT](https://github.com/pypa/hatch/blob/master/LICENSE.txt); build dependency is not vendored. |
| NeurIPS 2026 style | revision dated 2026-01-29 | Roman Garnett and prior NeurIPS template authors | Bundled only to build the paper, with its attribution header preserved; no license grant is inferred, and the file is excluded from this project's licenses. |

Competition-derived raw records are source material rather than a project-authored software dependency. Raw corpora and participant-derived learned artifacts remain outside the public release because they contain or encode participant identifiers, messages, interaction records, and inferred account linkages; `LICENSE.md` grants no rights in that content.
