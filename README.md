# ✏️ SketchAnim: Real-time sketch animation transfer from videos

### 🌟 Overview

Animation of hand-drawn sketches is an adorable art. It allows the animator to generate animations with expressive freedom and requires significant expertise. In this work, we introduce a novel sketch animation framework designed to address inherent challenges, such as motion extraction, motion transfer, and occlusion. The framework takes an exemplar video input featuring a moving object and utilizes a robust motion transfer technique to animate the input sketch. We show comparative evaluations that demonstrate the superior performance of our method over existing sketch animation techniques. Notably, our approach exhibits a higher level of user accessibility in contrast to conventional sketch-based animation systems, positioning it as a promising contributor to the field of sketch animation.

### ✨ **Key highlights**:

- 🎥 Video‑driven motion transfer
- 🧩 Robust handling of occlusions
- 🦴 Minimal user input (just a skeleton on the first frame)
- 🧍🐕🪑 Works for bipeds, quadrupeds, and even inanimate objects



<img src="assets/pipeline.png" style="zoom:25%;" />



### ⚙️ Setup

Clone the repository:

```
git clone git@github.com:graphics-research-group/SketchAnim.git
cd SketchAnim
```



### 📦 Installation

Create the conda environment:

```
conda env create -f environment.yml
```

Next, Install required submodules: [co-tracker](https://github.com/facebookresearch/co-tracker.git) and [DeformationPyramid](https://github.com/rabbityl/DeformationPyramid.git).

```
git clone https://github.com/facebookresearch/co-tracker.git
git clone https://github.com/rabbityl/DeformationPyramid.git
```



### 🦴 Draw a Skeleton

Draw the skeleton on the first frame (rest pose) of the video. 

```
python gui.py
```

| ![](assets/skeleton.gif) |
| -------------------------------------------------- |



### 🎨 Animate a Sketch

Run the following notebook to generate the final animation with their intermediate results. Users can use custom hand-drawn sketches.

```
python main.ipynb
```



## 🖼️ Results

**Supports:**

- 🧍 Bipeds
- 🐕 Quadrupeds
- 🪑 Inanimate objects

| ![](assets/biped1.gif)     | ![](assets/biped2.gif)     |
| -------------------------- | -------------------------- |
| ![](assets/quadruped2.gif) | ![](assets/quadruped1.gif) |
| ![](assets/inanimate1.gif) | ![](assets/inanimate2.gif) |



### 📖 Citation

If you find SketchAnim useful in your research, please cite:

```
@inproceedings{rai2024sketchanim,
  title={SketchAnim: Real-time sketch animation transfer from videos},
  author={Rai, Gaurav and Gupta, Shreyas and Sharma, Ojaswa},
  booktitle={Computer Graphics Forum},
  volume={43},
  number={8},
  pages={e15176},
  year={2024},
  organization={Wiley Online Library}
}
```

------



### 🙏 Acknowledgements

We have borrowed the code from [co-tracker](https://github.com/facebookresearch/co-tracker.git), [DeformationPyramid](https://github.com/rabbityl/DeformationPyramid.git) and [BBW](https://github.com/libigl/libigl/blob/main/include/igl/bbw.h) in our work.  We sincerely thank the authors for their valuable contributions to the research community.