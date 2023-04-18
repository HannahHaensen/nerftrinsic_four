

**Hannah Schieber (1), Fabian Deuser (2), Bernhard Egger (3),**
**Norbert Oswald (2) and Daniel Roth (1)**

- (1) Human-Centered Computing and Extended Reality, Friedrich-Alexander University (FAU) Erlangen-Nurnberg, Erlangen, Germany 
- (2) Institute for Distributed Intelligent Systems University of the Bundeswehr Munich Munich, Germany
- (3) Lehrstuhl fur Graphische Datenverarbeitung (LGDV) Friedrich-Alexander Universität (FAU) Erlangen-Nürnberg Erlangen, Germany

contact e-mail: hannah.schieber[at]fau.de

# Overview

[Paper](https://arxiv.org/pdf/2303.09412.pdf) | [iFF](https://drive.google.com/file/d/1deYczPDEcsInCD4MkSKeH_ZMbq_TGGi4/view)

# Visual improvements

Our NeRFtrinsic Four can handle divers camera intrinsics which leads to a better result.

![image](https://user-images.githubusercontent.com/22636930/231704734-de5774b9-7af6-4f77-ade1-92b5431bfe0a.png) 

# Architecture

![image](https://user-images.githubusercontent.com/22636930/231704527-8c070d6b-0ac8-4432-9bd2-17725a04d191.png)

# Results
## LLFF

<img src="https://user-images.githubusercontent.com/22636930/231704938-ae113dbc-d15f-4540-91fc-3deb95ebf8c8.png" width="700"/> <img src="https://user-images.githubusercontent.com/22636930/231704635-d8697cbe-bce4-4907-a306-04b9ea654e96.png" width="300"/> 


## BLEFF  
![image](https://user-images.githubusercontent.com/22636930/231705973-028f4b1e-27c3-4d3e-ba24-038afd04ce6c.png)  

## iFF

![image](https://user-images.githubusercontent.com/22636930/231706040-cb0ef15e-f923-419c-a71c-4d910c5220b4.png)  

You want to work with varying intrinsics as well, check out [iFF](https://drive.google.com/file/d/1deYczPDEcsInCD4MkSKeH_ZMbq_TGGi4/view).


### iFF Example Images

![image](https://user-images.githubusercontent.com/22636930/231706690-bfa8a920-4800-48aa-9104-6dc0c33d4c4b.png)

# Citation

You do something similar? or use parts from our code here you can cite our paper:

```
@misc{schieber2023nerftrinsic,
      title={NeRFtrinsic Four: An End-To-End Trainable NeRF Jointly Optimizing Diverse Intrinsic and Extrinsic Camera Parameters}, 
      author={Hannah Schieber and Fabian Deuser and Bernhard Egger and Norbert Oswald and Daniel Roth},
      year={2023},
      eprint={2303.09412},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

