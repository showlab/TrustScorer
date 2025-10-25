## ACM MM 2025: Can I Trust You? Advancing GUI Task Automation with Action Trust Score

**[Show Lab @ NUS](https://sites.google.com/view/showlab)**

[Haiyang Mei](https://mhaiyang.github.io/), [Difei Gao](https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en), Xiaopeng Wei, Xin Yang, [Mike Zheng Shou](https://sites.google.com/view/showlab)

[[`Paper`](https://dl.acm.org/doi/10.1145/3746027.3755618)] [[`BibTeX`](#citation)]

- [Table of Contents](#0-table-of-contents)
  * [1. TrustScorer](#1-trustscorer)
  * [2. TrustBench](#2-trustbench)
  * [3. Implementation](#3-implementation)
  * [4. Acknowledgements](#4-acknowledgements)
  * [5. Citation](#5-citation)
  * [6. License](#6-license)
  * [7. Contact](#7-contact)

### 1. TrustScorer

**TrustScorer** evaluates the trustworthiness of GUI agent actions for selective
human intervention when action trust score is low, to help mingling
human precision with AI efficiency.

<p align="center">
  <img src="assets/teaser.png?raw=true" width="400"/>
</p>

**TrustScorer** takes as input the user query _q_, subtask description _d_, action sequence _s_, and state observation _o_, and outputs a trustworthiness label _l_ indicating the likelihood that the action sequence can accomplish the specified subtask

### 2. TrustBench
**TrustBench** includes 106 specific tasks from 9 commonly used applications as well as 718 agent action sequences along with the corresponding ground-truth annotations.
<p align="center">
  <img src="assets/trustbench.png?raw=true" width="800"/>
</p>

One TrustBench example on PPT:
<p align="center">
  <img src="assets/ppt.png?raw=true" width="700"/>
</p>

The annotation pipeline:
<p align="center">
  <img src="assets/annotation.png?raw=true" width="800"/>
</p>

The TrustBench will be released at December 2025.

### 3. Implementation
We will release the training/testing/evaluation codes around the end of November 2025.

### 4. Acknowledgements

Our work builds upon [AssistGUI](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_AssistGUI_Task-Oriented_PC_Graphical_User_Interface_Automation_CVPR_2024_paper.pdf).

### 5. Citation

If you use TrustScorer/TrustBench in your research, please use the following BibTeX entry.

```bibtex
@InProceedings{Mei_2025_MM_TrustScorer,
    author    = {Mei, Haiyang and Gao, Difei and Wei, Xiaopeng and Yang, Xin and Shou, Mike Zheng},
    title     = {Can I Trust You? Advancing GUI Task Automation with Action Trust Score},
    booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM)},
    year      = {2025},
}
```

### 6. License

Please see `LICENSE`

### 7. Contact
E-Mail: Haiyang Mei (haiyang.mei@outlook.com)


**[⬆ back to top](#1-trustscorer)**
