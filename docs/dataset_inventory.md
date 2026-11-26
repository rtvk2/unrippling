# Dataset inventory and verified coverage

Six public source videos cover five captured clips (Checker, Middle, Cartoon, Eye and Small) and one authors’ simulated clip (Vision). They supply 606 original frames; the first 100 of each gave 600 unique public input frames. Four scaled distortions of one clean cameraman image add 80 generated frame instances. They share one seed and motion realization and are four conditions, not four independent scenes. Across ten cases the release contains 40 primary images and 40 baseline images. It also includes 39 additional V1/V3 images and 18 refined reconstructions, for **137 scored image outputs**. Figures, PNG/float copies, earlier trials and smoke tests are excluded from this count. The two primary branches produced 1,360 registered frame instances; six additional final refined stacks contributed 600. Intermediate arrays can be regenerated locally.

| Sequence | Origin | Width | Height | Frames | Used | AVI bytes | PNG bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BlueTiles | Synthetic | 512 | 512 | 101 | 0 | 18533332 | 480681 |
| BrickWall | Synthetic | 512 | 512 | 101 | 0 | 18197076 | 606069 |
| Bricks | Real1 | 688 | 512 | 101 | 0 | 18918380 | 517966 |
| Cartoon | Real1 | 612 | 512 | 101 | 100 | 8655756 | 139565 |
| Checker | Real1 | 752 | 512 | 101 | 100 | 16341942 | 317842 |
| Dices | Real1 | 726 | 512 | 101 | 0 | 17025398 | 545795 |
| Elephant | Real1 | 512 | 518 | 101 | 0 | 14898840 | 384174 |
| Eye | Real1 | 890 | 512 | 101 | 100 | 7715026 | 165118 |
| HandWritten | Synthetic | 512 | 512 | 101 | 0 | 15922156 | 346016 |
| Math | Real1 | 726 | 512 | 101 | 0 | 12359620 | 181093 |
| Middle | Real2 | 293 | 253 | 101 | 100 | 3263352 | 44408 |
| Small | Real2 | 284 | 255 | 101 | 100 | 3423478 | 46422 |
| Tiny | Real2 | 282 | 272 | 101 | 0 | 2735970 | 44300 |
| Vision | Synthetic | 512 | 512 | 101 | 100 | 5029142 | 141590 |

Real1 comprises posters recorded through a wave tank with matching photographs under still water. Real2 is credited to Tian and Narasimhan. All 14 upstream entries have a paired PNG. We inspected the headers of the eight unevaluated AVIs (256 bytes each) and decoded the six selected videos. References matched native dimensions and scene content for all six. After the common center crop and resize, the median/reference correlation ranged from 0.620 to 0.897. The best median translations were within one pixel on each axis. None of these shifts were used for scoring, registration or parameter tuning. [Per-clip checks](../comparison/results/course/alignment.csv)

The original four files totaled 19,967,544 bytes. The eight additional full assets totaled 25,316,097 bytes. The final decoded-file inventory totals 45,283,641 bytes; the separate header probes read 2,048 further payload bytes. Large arrays, originals and pretrained weights are excluded from delivery. No private water-tray videos were supplied. Two public resources were investigated: CompressiveFlows (14 published videos) and MFIR (procedurally generated). Our generator supplies a third input source.

| Method family | Verified public | Generated conditions | Additional refinement cases |
|---|---:|---:|---:|
| Each of the eight main methods | 6 | 4 | separate below |
| Optical V1 and V3 separately | 6 each | 4 each | — |
| Fourier V1 / V3 | 6 each | 4 / 3 | — |
| Refined Fourier average, RPCA, V2 fusion | — | — | 6 each |
| Official MFIR, consolidated methods | 0 | — | — |

Main reports 128 MFIR conditions on four background identities with six earlier methods and 14 James videos in a separate analysis. No independently inspectable per-clip sweep results were committed. The official compressed archive is 43.813 GB. Its accessible GitHub assets are composite GIFs, not independently downloadable official wave/reference pairs. The full compressed archive was not present in this cloud workspace. See the [dataset-claim audit](branch_comparison.md) for the official subset, reported extraction, scope and evidence limitations.
