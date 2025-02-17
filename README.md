# YouDream

### YouDream: Generating Anatomically Controllable Consistent Text-to-3D Animals  
[Sandeep Mishra<sup>✝︎</sup>](https://sandeep-sm.github.io/), [Oindrila Saha<sup>✝︎</sup>](http://oindrilasaha.github.io), [Alan C. Bovik](https://www.ece.utexas.edu/people/faculty/alan-bovik)  
Accepted at NeurIPS 2024

---

<h3 align="center">
  <a href="https://arxiv.org/abs/2406.16273v1">[arXiv]</a> |
  <a href="https://youdream3d.github.io">[Project Page]</a>
</h3>

---

![Unseen Animal Generation](https://github.com/YouDream3D/YouDream/blob/main/assets/unseen-animals.gif)

![Anatomically Controlled 3D Outputs](https://github.com/YouDream3D/YouDream/blob/main/assets/seen-animals.gif)

---

### Overview
**YouDream** introduces a novel approach for generating 3D animal models with anatomically accurate control, ensuring consistency and detail across generated outputs. This work combines text-to-3D generation with precise anatomical structures to cater to tasks requiring high-quality 3D assets in research and industry.

---

### Git Log
#### To-Do List
- [x] Release codebase
- [ ] Release a detailed documentation
- [ ] Release more Animal Prior configurations
- [ ] Release Pose Editor (manual pose editing -- suitable for pose-refinement and unseen animal generation)
- [ ] Release GPT-pose-editor (automatic pose editing -- suitable for known animals)

---

### Code Documentation
#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YouDream3D/YouDream.git
   cd YouDream
   ```
2. Create and activate the conda environment:
   ```bash
   conda create -n youdream python=3.9 -y
   conda activate youdream
   ```
3. Install dependencies:
   ```bash
   conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
   conda install -c iopath iopath
   conda install pytorch3d -c pytorch3d
   pip3 install -r requirements.txt
   ```
4. Download the Animal Pose ControlNet Checkpoint:
   Visit the [Hugging Face model page](https://huggingface.co/battleMaster/animal_pose_controlnet) to download the animal_pose_controlnet checkpoint.
   Place the downloaded checkpoint folder: "animal_pose_controlnet"  in the root directory of YouDream.

#### Running the Code
1. Run the shell scripts to generate 3D models:
   ```bash
   ./run_all_tiger.sh 
   ```

---

### Citation
If you find this work useful in your research or projects, please cite:
```
@inproceedings{NEURIPS2024_6fe5d7a2,
 author = {Mishra, Sandeep and Saha, Oindrila and Bovik, Alan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {60669--60698},
 publisher = {Curran Associates, Inc.},
 title = {YouDream: Generating Anatomically Controllable Consistent Text-to-3D Animals},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/6fe5d7a2de090168917425fe89a6c1b8-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
---

### Acknowledgements
Our Code is based on [DreamWaltz](https://github.com/IDEA-Research/DreamWaltz). We thank the DreamWaltz contributors for making the code available.

<sup>✝︎</sup> Equal contribution.

