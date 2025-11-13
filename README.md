![G-Wizard](G-Wizard.png)

# GPUMD-Wizard
Material structure processing software based on [ASE](https://wiki.fysik.dtu.dk/ase/index.html) (Atomic Simulation Environment) providing automation capabilities for calculating various properties of materials. Additionally, it aims to run and analyze molecular dynamics (MD) simulations using [GPUMD](https://github.com/brucefan1983/GPUMD).

## Features
* Based on the ASE package, MetalProperties-Automator supports different calculators.
* Allows for automated batch calculations of material properties.
* Enables batch processing of files in the XYZ format.
* Integrated with [GPUMD](https://github.com/brucefan1983/GPUMD) for performing molecular dynamics simulations, such as radiation damage.

## Installation


### Requirements


|  Package  | version |
|  ----  | ----  |
| [Python](https://www.python.org/) | >=     3.8 |
| [ase](https://wiki.fysik.dtu.dk/ase/index.html)|>=     3.18.0|
| [calorine](https://gitlab.com/materials-modeling/calorine)|>=     2.2.1|
| [phonopy](http://phonopy.github.io/phonopy/)|>=     v2.43.6|


### By pip 

```shell
$ pip install gpumd-wizard
```

 ### From Source

```shell
$ git clone --recursive https://github.com/Jonsnow-willow/GPUMD-Wizard.git
```

Add `GPUMD-Wizard` to your [`PYTHONPATH`](https://wiki.fysik.dtu.dk/ase/install.html#envvar-PYTHONPATH) environment variable in your `~/.bashrc` file.

```shell
$ export PYTHONPATH=<path-to-GPUMD-Wizard-package>:$PYTHONPATH
```

## Authors:

| Name                  | contact                           |
| --------------------- | --------------------------------- |
| Jiahui Liu            | jiahui.liu.willow@gmail.com       |

## Citations

| Reference             | cite for what?                    |
| --------------------- | --------------------------------- |
| [1]                   | NEP + ZBL |
| [2]                   | UNEP |
| [3]                   | MoNbTaVW |
| [4]                   | for any work that used `GPUMD`    |

## References

[1] Jiahui Liu, Jesper Byggmästar, Zheyong Fan, Ping Qian, and Yanjing Su,
[Large-scale machine-learning molecular dynamics simulation of primary radiation damage in tungsten](https://doi.org/10.1103/PhysRevB.108.054312),
Phys. Rev. B **108**, 054312 (2023).

[2] Keke Song, Rui Zhao, Jiahui Liu, Yanzhou Wang, Eric Lindgren, Yong Wang, Shunda Chen, Ke Xu, Ting Liang, Penghua Ying, Nan Xu, Zhiqiang Zhao, Jiuyang Shi, Junjie Wang, Shuang Lyu, Zezhu Zeng, Shirong Liang, Haikuan Dong, Ligang Sun, Yue Chen, Zhuhua Zhang, Wanlin Guo, Ping Qian, Jian Sun, Paul Erhart, Tapio Ala-Nissila, Yanjing Su, Zheyong Fan,
[General-purpose machine-learned potential for 16 elemental metals and their alloys](https://doi.org/10.1038/s41467-024-54554-x),
Nature Communications **15**, 10208 (2024).

[3] Jiahui Liu, Jesper Byggmästar, Zheyong Fan, Bing Bai, Ping Qian, and Yanjing Su,
[Utilizing a machine-learned potential to explore enhanced radiation tolerance in the MoNbTaVW high-entropy alloy](https://www.sciencedirect.com/science/article/pii/S0022311525003988),
Journal of Nuclear Materials, 156004 (2025).

[4] Ke Xu, Hekai Bu, Shuning Pan, Eric Lindgren, Yongchao Wu, Yong Wang, Jiahui Liu, Keke Song, Bin Xu, Yifan Li, Tobias Hainer, Lucas Svensson, Julia Wiktor, Rui Zhao, Hongfu Huang, Cheng Qian, Shuo Zhang, Zezhu Zeng, Bohan Zhang, Benrui Tang, Yang Xiao, Zihan Yan, Jiuyang Shi, Zhixin Liang, Junjie Wang, Ting Liang, Shuo Cao, Yanzhou Wang, Penghua Ying, Nan Xu, Chengbing Chen, Yuwen Zhang, Zherui Chen, Xin Wu, Wenwu Jiang, Esme Berger, Yanlong Li, Shunda Chen, Alexander J. Gabourie, Haikuan Dong, Shiyun Xiong, Ning Wei, Yue Chen, Jianbin Xu, Feng Ding, Zhimei Sun, Tapio Ala-Nissila, Ari Harju, Jincheng Zheng, Pengfei Guan, Paul Erhart, Jian Sun, Wengen Ouyang, Yanjing Su, Zheyong Fan, [GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials]( https://doi.org/10.1002/mgea.70028), MGE Advances **3**, e70028 (2025).
