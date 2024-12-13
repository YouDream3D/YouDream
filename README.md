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

![Unseen Animal Generation](https://github.com/YouDream3D/YouDream/assets/172318216/ed597c4b-02a6-4078-bfc8-9d892e5d0731)

<div align="center">
  <img src="https://github.com/YouDream3D/YouDream/assets/172318216/fe9cab6a-55a8-4eb3-b9ca-e9a09fc89b8c" alt="Anatomically Controlled 3D Outputs"/>
</div>

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
1. Prepare the input text prompt file or dataset.
2. Run the main script to generate 3D models:
   ```bash
   ./run_all_tiger.sh 
   ```

---

### Citation
If you find this work useful in your research or projects, please cite:

```
@article{mishra2024youdream,
  title={YouDream: Generating Anatomically Controllable Consistent Text-to-3D Animals},
  author={Mishra, Sandeep and Saha, Oindrila and Bovik, Alan C},
  journal={arXiv preprint arXiv:2406.16273},
  year={2024}
}
```

---

<sup>✝︎</sup> Equal contribution.

