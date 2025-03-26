<div align="center">

## Unraveling the Hessian: A Key to Smooth Convergence in Loss Function Landscapes

[**Nikita Kiselev**](https://kisnikser.github.io/)&nbsp;&nbsp;&nbsp;&nbsp;
[**Andrey Grabovoy**](https://intsystems.github.io/people/grabovoy_av/index.html)<br>
Moscow Institute of Physics and Technology

[![arXiv](https://img.shields.io/badge/arXiv-2409.11995-b31b1b.svg)](https://arxiv.org/abs/2409.11995)
[![paper](https://img.shields.io/badge/paper-Doklady_Mathematics-blue.svg)](https://doi.org/10.1134/S1064562424601987)
[![slides](https://img.shields.io/badge/slides-presentation-orange.svg)](https://github.com/kisnikser/landscape-hessian/blob/main/slides/main.pdf)

<br>

<img alt="overview" width=700 src="paper/losses_difference.svg">

</div>

<br>

> **Abstract:** *The loss landscape of neural networks is a critical aspect of their behavior, and understanding its properties is essential for improving their performance. 
> In this paper, we investigate how the loss surface changes when the sample size increases, a previously unexplored issue. 
> We theoretically analyze the convergence of the loss landscape in a fully connected neural network and derive upper bounds for the difference in loss function values when adding a new object to the sample. 
> Our empirical study confirms these results on various datasets, demonstrating the convergence of the loss function surface for image classification tasks. 
> Our findings provide insights into the local geometry of neural loss landscapes and have implications for the development of sample size determination techniques.*

## 🔥 News

- [2025/03/22] [Paper](https://doi.org/10.1134/S1064562424601987) was published in Doklady Mathematics journal (🤗 Open Access).
- [2025/10/10] [Slides](https://github.com/kisnikser/landscape-hessian/blob/main/slides/main.pdf) for presentation were added.
- [2024/09/27] Paper was accepted to be published in Doklady Mathematics journal.
- [2024/09/18] [Preprint](https://arxiv.org/abs/2409.11995) was added to arXiv.
- [2024/08/16] Paper was accepted to the AI Journey 2024 conference.
- [2024/08/13] [Paper](https://github.com/kisnikser/landscape-hessian/blob/main/paper/main.pdf) and [Code](https://github.com/kisnikser/landscape-hessian/tree/main/code) were released.

## 🛠️ Repository Structure
This repository is structured as follows:
- `code`: The computational experiments code with its own `README.md`
- `paper`: Preprint `main.pdf` with source LaTeX file `main.tex`.
- `slides`: Presentation slides `main.pdf` with source LaTeX file `main.tex`.

## 📖 Citation
If you find our paper to be useful for your research or applications, please cite us:

```BibTeX
@article{kiselev2025unraveling,
  title={Unraveling the Hessian: A Key to Smooth Convergence in Loss Function Landscapes},
  author={Kiselev, Nikita and Grabovoy, Andrey},
  journal={Doklady Mathematics},
  number={1},
  pages={S49--S61},
  volume={110},
  year={2024}
}
```

```BibTeX
@article{kiselev2024unraveling,
  title={Unraveling the Hessian: A Key to Smooth Convergence in Loss Function Landscapes},
  author={Kiselev, Nikita and Grabovoy, Andrey},
  journal={arXiv preprint arXiv:2409.11995},
  year={2024}
}
```

We also appreciate it if you could give a star ⭐ to this repository. Thanks a lot!
