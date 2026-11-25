# Inputs

Run `python data/download_data.py` from inside `final/`. Select one clip with `--datasets checker` or `--datasets middle`. The downloader fetches individual files at CompressiveFlows commit 78d9320, verifies length and Git blob SHA-1, and atomically installs verified files in data/real. A locked ledger counts transferred bytes across attempts. Retry the same command after interruption. The recorded cloud download was 19,967,544 bytes. A second invocation verified all files without increasing the ledger.

| Clip | Video bytes | Reference bytes | Native width × height | Decoded frames |
|---|---:|---:|---|---:|
| Checker | 16,341,942 | 317,842 | 752 × 512 | 101 |
| Middle | 3,263,352 | 44,408 | 293 × 253 | 101 |

James et al. identify Checker as a real tank recording with a matching still-water photograph (Real1). Middle is a Tian–Narasimhan text sequence with a reference (Real2). Both belong to development. The authors distribute the PNGs under GroundTruthImages. We keep the source videos locally and store compact derived comparisons. Consult upstream terms before redistributing footage. [Authors’ dataset](https://github.com/jeringeo/CompressiveFlows/tree/78d9320d50f00803b085c2a095094fc1796db241/Dataset)

The CLI decodes grayscale, uses the first requested observations in order, center-crops the largest square and resizes to the requested side. The PNG receives identical spatial processing. Area resampling handles reduction and bilinear resampling handles enlargement. Custom footage can be a video or a directory of lexically ordered, zero-padded image names passed with `--source`. Custom inputs have no assigned clean reference. Generated inputs use `--dataset synthetic --amplitude 3` with seeded smooth pull fields and scikit-image's cameraman image. They are regenerated in memory.

The original RGB, MFIR and paired still/wavy loaders are kept with the historical development code at the repository root, in [src/unripple/data.py](../../src/unripple/data.py) (outside `final/`, not used by it). MFIR development uses sorted backgrounds 0–2 on shallow profile_6; test uses backgrounds 8–15 on that profile; other water profiles and wave types are reserved for unseen-water evaluation. The published archive totals 43.8 GB and was excluded from these runs. [Official benchmark](https://github.com/iafoss/refractive-mfir-benchmark) · [Dataset record](https://doi.org/10.5281/zenodo.19390086)
