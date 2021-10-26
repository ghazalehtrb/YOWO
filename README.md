# You Only Watch Once 53 (YOWO53)

PyTorch implementation of the YOWO53 network from "[Joint Detection And Activity Recognition Of Construction Workers Using Convolutional Neural Networks](https://ec-3.org/publications/conferences/2021/paper/?id=197)" paper. YOWO53 is a variation of the original "[You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization](https://github.com/wei-tim/YOWO/blob/master/examples/YOWO_updated.pdf)"  YOWO network with Darknet53 as the 2D backbone. YOWO53 showed better detection performanace for small objects in our studies compared to YOWO. Please refer to "[https://github.com/wei-tim/YOWO](https://github.com/wei-tim/YOWO)" for more details. 


<br/>

Activity recognition results on construction workers!
<br/>
<div align="center" style="width:image width px;">
  <img  src="https://github.com/ghazalehtrb/YOWO/blob/master/examples/Media2.gif" width=400 alt="activity recognition of construction workers">
</div>

## Installation, public datasets, and pretrained models   
Please refer to "[https://github.com/wei-tim/YOWO](https://github.com/wei-tim/YOWO)" for details on installation, to download publicly available datasets and pretrained models for the 3D backbones. The 2D Darknet53 pretrained backbone is included in 


## Installation
```bash
git clone https://github.com/wei-tim/YOWO.git
cd YOWO
```

### Citation
If you use the code for YOWO53, please cite the following:

```@inproceedings{Torabi_2021,
	doi = {10.35490/ec3.2021.197},
	url = {https://doi.org/10.35490/ec3.2021.197},
	year = 2021,
	month = {jul},
	publisher = {University College Dublin},
	author = {Ghazaleh Torabi and Amin Hammad and Nizar Bouguila},
	title = {Joint Detection And Activity Recognition Of Construction Workers Using Convolutional Neural Networks},
	booktitle = {Proceedings of the 2021 European Conference on Computing in Construction},
	volume = {2},
	isbn = {978-3-907234-54-9},
	address = {Online},
	series  = {{Computing} in {Construction}},
	language = {en-GB},
	pages = {212--219},
	abstract = {Manually gathering information about activities on construction sites for project management purposes is labor-intensive and time-consuming. As a result, several works leveraged the already installed surveillance cameras to automate this process. However, the recent learning-based methods discretize continuous activities by assigning a single label to multiple consecutive frames. They do not fully leverage the contextual cues in the scene, and are not optimized end-to-end. A variation of the YOWO network, called YOWO53, is proposed in this paper to address these limitations. YOWO53 shows better classification and detection results over YOWO and allows using smaller input frames with real-time speed.},
	Organisation = {European Council on Computing in Construction}, 
	Editors = {James O{\textquotesingle}Donnell, Daniel Hall, Dragana Nikolic, Athanasios Chassiakos}
}
```
