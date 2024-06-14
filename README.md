# ML
This GPT model follows closely to the GPT model generated by Andrej Karparthy https://github.com/karpathy/ng-video-lecture

Changes to the model are 
- changing the initial weight distribution for linear layers to kaiming_uniform 
- implementing an encapsulated multihead attention layer similar to the one found here https://github.com/karpathy/nanoGPT

This GPT model implements self attention following the paper Attention Is All You Need 
changes that deviate from the structure set here:
- only a decoder has been implemented
- the layer normalisation has been applied both before and after the self attention and feed forward layers

Within this model in the stats.txt different experiments have been conducted 