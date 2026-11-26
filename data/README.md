# Obtaining the inputs

Run from the repository root. The default downloader selects Checker and Middle. For the full course experiment:

```sh
python data/download_data.py --datasets checker middle cartoon eye small vision
```

The pinned manifest records all 14 CompressiveFlows video/reference pairs at commit 78d9320d50f00803b085c2a095094fc1796db241. Only explicitly selected sequences are transferred beyond the default two. Files are checked against their length and Git blob hash before atomic installation. A locked ledger under real/ reserves bytes before requests and permits a verified restart. The default cumulative cap is 500 MB. [Pinned manifest](manifest.json) · [Original release](https://github.com/jeringeo/CompressiveFlows/tree/78d9320d50f00803b085c2a095094fc1796db241/Dataset)

| Location | Contents |
|---|---|
| real/ | Original captured AVI files and still PNG references |
| public_synthetic/ | Authors’ simulated AVI files and reference PNGs |
| synthetic/ | Reserved for local synthetic data; the course sweep regenerates in memory |
| manifest.json, download_data.py | Dataset categories, exact case-sensitive names, sizes, hashes and downloader |

Custom videos and sorted frame directories use `--source PATH`; full-reference measures are unavailable for those inputs. Data remain external to the submission. The earlier MFIR implementation is preserved in Git history. [Inventory, native dimensions, coverage and sizes](../docs/dataset_inventory.md)
