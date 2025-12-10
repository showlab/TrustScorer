<p align="center">
    <h1 align="center">
        Can I Trust You?
        <br>
        Advancing GUI Task Automation with Action Trust Score
    </h1>
</p>

<p align="center">
  <a href="https://mhaiyang.github.io/">Haiyang Mei</a>&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en">Difei Gao</a>&nbsp;&nbsp;&nbsp;   
  <a href="https://scholar.google.com/citations?user=LRDKx5cAAAAJ&hl=en">Xiaopeng Wei</a>&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=YU2ujh4AAAAJ&hl=en">Xin Yang</a>&nbsp;&nbsp;&nbsp;  
  <a href="https://sites.google.com/view/showlab">Mike Zheng Shou</a><sup>✉️</sup>  
  <br>
  Show Lab, National University of Singapore
</p>

<div align="center">
  <p>
    <a href="https://dl.acm.org/doi/10.1145/3746027.3755618" target="_blank"> <img src="https://img.shields.io/badge/ACM%20DL-Paper-grey?logo=acm&logoColor=white&labelColor=blue">
    </a>
    <a href="https://dl.acm.org/doi/10.1145/3746027.3755618" target="_blank">
      <img src="https://img.shields.io/badge/ACM%20DL-Video-grey?logo=youtube&logoColor=white&labelColor=red">
    </a>
  </p>
</div>

<p align="center">
  <a href="https://dl.acm.org/doi/10.1145/3746027.3755618" target="_blank"><img src="assets/promo-video.gif" alt="Watch the video" width="800">
    </a>
</p>

- [Table of Contents](#0-table-of-contents)
  * [1. TrustScorer](#1-trustscorer)
  * [2. TrustBench](#2-trustbench)
  * [3. Implementation](#3-implementation)
  * [4. Acknowledgements](#4-acknowledgements)
  * [5. Citation](#5-citation)
  * [6. License](#6-license)
  * [7. Contact](#7-contact)

### 1. TrustScorer

We introduce **TrustScorer**, which evaluates the trustworthiness of GUI agent actions for selective
human intervention when action trust score is low, to help mingling human precision with AI efficiency.

<p align="center">
  <img src="assets/teaser.png" width="460"/>
</p>

**TrustScorer** takes as input the user query _q_, subtask description _d_, action sequence _s_, and state observation _o_, and outputs a trustworthiness label _l_ indicating the likelihood that the action sequence can accomplish the specified subtask

### 2. TrustBench
**TrustBench** includes 106 specific tasks from 9 commonly used applications as well as 718 agent action sequences along with the corresponding ground-truth annotations.
<p align="center">
  <img src="assets/trustbench.png" width="800"/>
</p>

One TrustBench example on PPT:
<p align="center">
  <img src="assets/ppt.png" width="700"/>
</p>

The annotation pipeline:
<p align="center">
  <img src="assets/annotation.png" width="800"/>
</p>

### 3. Implementation
xxx

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
