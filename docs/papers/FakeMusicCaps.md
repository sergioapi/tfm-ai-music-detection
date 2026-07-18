Journal of **_Imaging_** 



### _Article_ 

# **FakeMusicCaps: A Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models** 

**Luca Comanducci * , Paolo Bestagini and Stefano Tubaro** 

Department of Electronics, Information and Bioengineering (DEIB), Politecnico di Milano, 20133 Milano, Italy; paolo.bestagini@polimi.it (P.B.); stefano.tubaro@polimi.it (S.T.) 

***** Correspondence: luca.comanducci@polimi.it 

### **Abstract** 

Text-to-music (TTM) models have recently revolutionized the automatic music generation research field, specifically by being able to generate music that sounds more plausible than all previous state-of-the-art models and by lowering the technical proficiency needed to use them. For these reasons, they have readily started to be adopted for commercial uses and music production practices. This widespread diffusion of TTMs poses several concerns regarding copyright violation and rightful attribution, posing the need of serious consideration of them by the audio forensics community. In this paper, we tackle the problem of detection and attribution of TTM-generated data. We propose a dataset, FakeMusicCaps, that contains several versions of the music-caption pairs dataset MusicCaps regenerated via several state-of-the-art TTM techniques. We evaluate the proposed dataset by performing initial experiments regarding the detection and attribution of TTM-generated audio considering both closed-set and open-set classification. 

**Keywords:** music generation; text-to-music; audio forensics; DeepFake 

## **1. Introduction** 

Academic Editor: Pier Luigi Mazzeo Received: 6 June 2025 Revised: 9 July 2025 Accepted: 14 July 2025 Published: 18 July 2025 

**Citation:** Comanducci, L.; Bestagini, P.; Tubaro, S. FakeMusicCaps: A Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models. _J. Imaging_ **2025** , _11_ , 242. https://doi.org/10.3390/ jimaging11070242 

**Copyright:** © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/). 

Deep learning-based music generation [1] has been recently revolutionized by the introduction of text-to-music models. TTM models are usually based on a language model that decodes continuous or discrete tokenized embeddings obtained via some neural audio codec [2,3], such as MusicLM [4], MusicGEN [5], MAGNeT [6], and JASCO [7], or on latent diffusion models operating on some compressed form of audio, such as AudioLDM [8], AudioLDM2 [9], MusicLDM [10], Noise2Music [11], and Mustango [12]. 

These models are characterized by being able to generate sufficiently realistic music and are simple to use, lowering the technical proficiency needed to successfully interact with them [13]. This combination of factors has made them extremely attractive to the general public and of interest to private industries. 

Several commercial TTMs have been proposed, such as Suno [14] (setting the record for the biggest investment ever in an AI music startup, namely $125 million) and Udio [15]. Recently, both companies have been sued by major record companies and have consecutively admitted to copyright infringement, by training their respective models also using unlicensed music. As both the capabilities and commercial interest of these models grow, it is becoming increasingly necessary to begin to develop forensic approaches to be able to detect and analyze music generated via TTMs [16]. 

https://doi.org/10.3390/jimaging11070242 

_J. Imaging_ **2025** , _11_ , 242 

_J. Imaging_ **2025** , _11_ , 242 

2 of 14 

The multimedia forensics research field is well mature in image [17–21] and video [22] deepfake detection and model attribution. Concerning audio, forensics approaches have been focusing almost exclusively on speech signals [23–25]. 

In the music domain, most efforts have focused on singing voice detection [26–29] with the development of specific challenges [30] to foster research in this direction. More specifically, in [26], the authors present SingFake, a dataset for singing voice detection and use it to evaluate four state-of-the-art speech deepfake detectors. In [27], the authors present a dataset of fake chinese songs and analyze the performance of SOTA detectors trained on speech signals and on the proposed dataset. The SingGraph model [28] leverages MERT [31] and wav2vec 2.0 [32] to detect fake singing voices by merging lyrics and audio analysis, while in [29], the authors use singer-level contrastive learning and demonstrate difficulties in detecting cloned-singers. 

More recently, other works have tackled the fake music research problem, considering also non-vocal audio parts. An overview of the problem is presented in [33]. In [34], the authors focus on detecting which neural audio codec was used to compress real music tracks and, obtaining surprisingly accurate results, point out several issues that might make the fake music detection problem too easy and rapidly worsen in out-of-domainscenarios. Moreover, as a detector, they apply a simple convolutional model composed of just six layers, showing that performance is not the only thing to take into account when considering fake music detection. Singing voice is again considered in [35], where it is demonstrated how including background music could enhance the accuracy when classifying fake singing voices. Specifically, the authors apply a hybrid front-end model that extracts features separately from vocals and background music, before feeding the output to a backend network based on Rawnet2 [36] and AASIST [37]. 

Research in this field is also limited by economic factors, since most models are developed by tech giants that often do not release the code and/or weights. Additionally, available paired text-music datasets are scarce. Notable example datasets include MusicCaps [4], containing 5500 musiclips extracted from AudioSet [38] and annotated by human musicians, Song Describer [39], containing 1100 human-made captions of 706 music recordings, MusicBench [12], which contains music obtained by augmenting and modifying the MusicCaps dataset, obtaining 52,000 samples, and the recently proposed JamendoMaxCaps [40], where 362,000 captions from the Jamendo dataset are described using an audio captioning system [41]. 

In this paper, we propose the FakeMusicCaps dataset, with the objective of encouraging research in the detection of music deepfakes. To build FakeMusicCaps, we replicate the MusicCaps dataset by using its captions as input to five state-of-the-art TTM models, namely MusicGen [5], MusicLDM [10], AudioLDM2 [9], Stable Audio Open [42], and Mustango [12]. The nature of FakeMusicCaps makes it easy to incorporate future TTMs by simply generating music examples using the same procedure. We perform a simple benchmark study, on FakeMusicCaps, by studying if it is possible to perform detection and attribution, i.e., classifying the input music as either real or belonging to one of the chosen TTM models, using state-of-the-art models. We analyze how the models perform both in closed set and open set scenarios, where the latter also includes data belonging to generators not seen during training, specifically, belonging to the SunoCaps dataset [43]. 

At the same time of this work, a similar dataset, named SONICS, has been proposed [44], which however considers only the commercially-available models Suno and Udio and performs only real/fake music detection. More specifically, in [44] the authors explore the classification of real and fake music by proposing the Spectro-Temporal Tokens Transformer (SpecTTTra) architecture to perform fake music detection. 

_J. Imaging_ **2025** , _11_ , 242 

3 of 14 

We instead focus on open-source TTMs and consider commercial ones only in the open set classification. The reasoning behind this is that open-source techniques are possibly available to a wider part of the research community, which could use them to integrate FakeMusicCaps as they see fit. Moreover, open-source models allow researchers to have complete knowledge of the entire pipeline used to generate the audio tracks, enabling them to make stronger assumptions on the behaviors observed when classifying the data. This is particularly useful when dealing with open-set scenarios, since without knowing the audio generation process it is impossible to deem audio signals as belonging to different generators. 

Since its release, the FakeMusicCaps dataset has already been used to foster the research in fake music detection. In fact, in [45], the authors performed a study aimed at understanding the behavior of classifiers operating on the FakeMusicCaps dataset, by applying eXplainable Artificial Intelligence (XAI) techniques. Inspired by FakeMusicCaps and SONICS, the authors of [46] propose the M6 dataset, which aggregates various types of audio content from existing datasets for what concerns real music and generates fake music using simple custom prompts. 

Our contributions can then be summarized as follows: 

- In this paper, we release FakeMusicCaps, the first dataset specifically designed for both detection and attribution of fake music. The dataset is created using only open-source text-to-music models, making the generation process fully transparent. 

- Through the use of simple network architectures, we analyze the detection and (for the first time) attribution of fake music generated via TTM models. We consider both closed-set and open-set classification scenarios, taking into account models generated via Suno in the latter. 

The remainder of the paper is organized as follows. In Section 2, we introduce the attribution problem for TTM models. In Section 3, we describe how the FakeMusicCaps dataset was created. Section 4 presents the experimental setup used to conduct the experiments, while in Section 5, we present the results to analyze the complexity of TTM attribution and the effectiveness of the proposed dataset. Finally, in Section 7, we draw some conclusions. The code used to generate FakeMusicCaps and perform the experiments (https://github.com/polimi-ispl/FakeMusicCaps, accessed on 13 July 2025) as well as the complete dataset (https://zenodo.org/records/15063698, accessed on 13 July 2025) are publicly available. 

## **2. Problem Formulation** 

Given some kind of text representation _τ_ and a composite model _T_ ( _·_ ), the TTM techniques model the function **x** = _T_ ( _τ_ ), where **x** _∈_ R<sup>1</sup><sup>_×N_</sup> is an audio waveform containing music that corresponds to the textual description provided in _τ_ . 

The text-to-music attribution problem, schematically shown in Figure 1, can be formally defined as follows. Given the discrete-time music signal **x** and a set of _I_ TTM models _{T_ 0, . . . , _TI−_ 1 _}_ , the objective is to determine which generator _Ti_ has been used to generate **x** . This is done by training a classifier that takes as input **x** and outputs the probabilities _pi_ , _i_ = 0, . . . , _I −_ 1 of **x** generated using each of the known TTM models. 

The attribution problem is often considered both in closed- and open-set scenarios. In the former, all generators are seen both during training and testing, while in the latter, some TTMs are unknown during training and seen only at testing time, posing the need to develop specific classification strategies. 



<!-- Start of picture text -->
cb<br>‘eaap<br>db<br>etap<br><!-- End of picture text -->





<!-- Start of picture text -->
ap<br>db<br>anap<br><!-- End of picture text -->









<!-- Start of picture text -->
—<br>_.<br><!-- End of picture text -->



<!-- Start of picture text -->
Captions wy<br>(MusicCaps)|*!<br>TTMO1 TTMO02 TTM03 TTMO05<br>MusicGen MusicLDM AudioLDM2 Mustango<br>Audio clip Audio clip Audio clip Audio clip Audio clip<br>alu atu atu atu alu<br><!-- End of picture text -->

_J. Imaging_ **2025** , _11_ , 242 

5 of 14 

- **TTM01-MusicGen** [5] is an autoregressive language model, based on a single-stage transformer that decodes discrete audio tokens obtained via Encodec [3]. It was trained over an undisclosed dataset of over 20,000 h of music. We use the _medium_ checkpoint consisting of 1.5 _B_ parameters. 

- **TTM02-MusicLDM** [10] is a latent diffusion model operating on compressed audio representations extracted via HiFi-GAN [48]. It adapts AudioLDM to the musical domain, by introducing beat-synchronous audio mixup and beat-synchronous latent mixup strategies, to augment the quantity of data used for training. The text conditioning is provided via CLAP [49], which the authors fine-tune on music for a total of 20,000 h. The MusicLDM model is then trained on the Audiostock dataset [49], containing 455.6 h of music. 

- **TTM03-AudioLDM2** [9] is a latent diffusion model where the audio is compressed via a Variational AutoEncoder (VAE) and HiFiGAN, similarly to the AudioLDM pipeline. However, the major difference of AudioLDM2 with respect to the previous version, is that the diffusion model is conditioned through AudioMAE [50] that enables the adoption of a “Language of Audio”, to generate a wide variety of types of audio. We use the _audioldm2-music_ checkpoint to build FakeMusicCaps, specifically trained for text-to-music generation. 

- **TTM04-Stable Audio Open** [42] is a latent-diffusion architecture generating stereo data at 44.1 kHz based on a variant of Stable Audio [51] that uses T5 [52] as a text encoder. The model is trained only on Creative Commons-licensed audio data for a total of 7.3 K hours of audio. 

- **TTM05-Mustango** [12] is a diffusion-based TTM model that through a Music-domainknowledge-informed UNet (MuNet) injects music concepts such as chord, beats, key, or tempo in the generated music, during the reverse diffusion process. Through data augmentation, the authors generate the MusicBench dataset, composed of 53.168 tracks, to train the model. The model generates at 16 kHz 

### _3.2. Generation Strategy_ 

In this section, we describe the strategy used to generate the FakeMusicCaps dataset. We derive the inspiration for the generation procedure from the MusicCaps [4] dataset, consisting of 5500 music clips, each 10 s long, extracted from AudioSet [38]. Each track is supplied with an annotation by a professional musician. MusicCaps has rapidly become the benchmark dataset for the evaluation of TTM models. 

To create FakeMusicCaps, we use the captions from MusicCaps, and for each one of them, we generate a corresponding 10 s audio track using models (TTM01-TTM05) for a total of 27,605 music tracks corresponding to almost 77 h. 

Since the objective of the dataset is to provide an initial dataset for the analysis of the detection and attribution of music generated via TTM models, we adopt an audio pipeline that ensures that all audios are represented using the same format. Specifically, each track is first converted to mono and downsampled to the sampling frequency _Fs_ = 16 kHz. Finally, we save each track using the 32-bit float wav format. 

## **4. Experimental Analysis** 

In this section, we describe the experiments performed with the aim of showing a first validation of the FakeMusicCaps dataset, considering both closed set and open set scenarios. 

_J. Imaging_ **2025** , _11_ , 242 

6 of 14 

### _4.1. Dataset_ 

We used the FakeMusicCaps dataset during the training and test procedures. We made sure that the training and test datasets were disjoint. Specifically, we built the test set by selecting 320 tracks from FakeMusicCaps, as a selection criterion, for each TTM model, we chose those having the same captions as the SunoCaps [43] dataset. This choice was operated in order to be able to coherently use the Suno-generated music excerpts from SunoCaps to perform the open-set scenario experiments. 

### _4.2. Baselines_ 

We use three classification models as simple benchmarks of the FakeMusicCaps for deepfake music detection and attribution. 

We first consider a very simple network that operates on raw audio, namely M5 [53]. This network consists of only 0.5 M parameters and leverages the adoption of several consecutive layers. We use this simple network as an initial experiment, in order to understand the level of hardness of the music deepfake detection and attribution problem. 

Then, we selected a more complicated method operating on raw audio, namely RawNet2 [36]. This model, is an end-to-end neural network that has been used as a baseline for several antispoofing challenges such as ASVspoof 2021 and consists of Sinc Filters, followed by residual blocks and a Gated Recurrent Unit (GRU). 

We also consider a model operating on log-spectrograms, namely ResNet18+Spec [54]. This model is a modified version of ResNet18, consisting of 18-layer deep convolutional layer with residual connections. The modifications make it suitable to work with 1-channel log-spectrograms. 

All methods were modified by adding a fully connected layer with a number of neurons corresponding to the number of considered classes at the end of each network. This modification is necessary in order to be able to discriminate correctly between the considered TTM models. 

### _4.3. Training_ 

All models were trained to discriminate between 6 different classes, comprising the 5 known TTM models and the real music signals belonging to MusicCaps. 

We trained all models using cross-entropy as a loss function and the Adam optimizer with a learning rate of 1 _×_ 10<sup>_−_4</sup> . 

All networks were trained for a maximum of 100 epochs, ending the training earlier if the loss did not improve for more than 10 consecutive epochs. We used a batch size of 32 for M5 and 16 for RawNet2 and ResNet18 + Spec. In the case of ResNet18 + Spec, we computed the STFT with 512 frequency points, using a Hann window of length 512 samples with a hop size of 128 samples. 

### _4.4. Classification Techniques_ 

In the _closed set_ classification problem, given a raw audio waveform corresponding to music, we want to identify the generation method from the set of _known_ (i.e., a set of models included in the training dataset) TTM models. 

Differently, in the case of _open set_ classification, we want also to determine if some audio tracks belong to a TTM model that it is _unknown_ , i.e., not included in the training dataset. 

If we consider _pi_ as the output of the softmax layer of the models, then in the closed set case, class attribution can simply be performed by computing arg max _i pi_ . For open set classification, instead, we follow two different approaches. In the open set _(threshold)_ technique [55], we compute the two highest values of _pi_ , defined as _p_ 1 and _p_ 2, and then classify the input example as unknown if the ratio _p_ 1/ _p_ 2 between these values is higher 

_J. Imaging_ **2025** , _11_ , 242 

7 of 14 

than a threshold. The rationale is that, if the TTM method used to generate the audio track was known from the training set, only one _pi_ value should be high. More formally, the predicted TTM model _T_<sup>ˆ</sup> can be obtained as 



where _γ_ is a threshold that should be determined empirically, following [24], we choose _γ_ = 2. 

In the open set _SVM_ technique, instead, we train a one-class Support Vector Machine (SVM), using radial basis functions kernel, over the _pi_ values computed from the training data. The output of the classification is binary: either the class is known or not. 

## **5. Results** 

In this section, we present preliminary results aimed at demonstrating the suitability of FakeMusicCaps as an initial dataset for text-to-music model detection and attribution. Specifically, we test the performance of state-of-the art models for fake music detection and attribution, both in the closed- and open-set scenarios. Then we perform an additional experiment aimed at understanding the impact of the length of the considered audio tracks. 

More specifically, in Section 5.1, we consider the simpler scenario where the TTM seen during training and test phases are the same. In Section 5.2, instead, we consider the more challenging and realistic open-set scenario where the detection models are trained on TTM01, TTM02, TTM03, TTM03, TTM04, and TTM05 and are tested on the SunoCaps [43] dataset, whose audio files are generated using the commercial TTM model Suno, which is not used to generate audio files used during training. 

### _5.1. Closed-Set Performances_ 

Despite closed-set classification on a single dataset is often considered a trivial task in forensic applications, it is worth investigating the performance of the tested methods in this scenario. 

Table 1 reports closed-set classification results in terms of balanced accuracy ACC _B_ [56], Precision, Recall, and F1 Score. Additionally, the left column of Figure 3 shows the confusion matrix corresponding to M5, RawNet2 and ResNet18+Spec, respectively. 

In all metrics, ResNet18 + Spec provides the best performance, while RawNet2 obtains slightly worse results than M5. From the inspection of the confusion matrices, we can see that ResNet18 slightly confounds TTM03 (AudioLDM2) with TTM05 (Mustango), it is interesting to notice that both are diffusion-based models. M5 has a slightly lower performance in detecting the ground-truth data, while RawNet2 struggles more to detect model TTM02 (MusicLDM). 

**Table 1.** Closed-set classification performances. 

|**Model**|**ACCB** **_↓_**|**Precision**|**Recall**|**F1 Score**|
|---|---|---|---|---|
|M5|0.90|0.90|0.90|0.90|
|RawNet2|0.88|0.89|0.88|0.88|
|ResNet18 + Spec|**1**.**00**|**1**.**00**|**1**.**00**|**1**.**00**|





<!-- Start of picture text -->
reat MONE 0.13 0.055 0 0.036 0.091<br>run OEE coislioeel oo loos<br>oe<br>B rrmo2 40.018 0.035 MEM 0 0 0.018<br>4<br>a3 t1wo34 0 0 0 ME 0 0.053<br>TTMo#) 0 0.035 0.018 0 FO<br>TTM05 40.018 0.035 0 0 0 0.95<br>SF~ eS KSeo SK2 SL- 2<br>Predicted Labels<br><!-- End of picture text -->



<!-- Start of picture text -->
REAL {MUNN 0.091 0.036 0 0.018 0.11 0.15<br>TTMo1 40.035 0.84 0 0 0 oO 012<br>w» TTMO27 0 0. 018 ee 0 0 0.018 0.053<br>3<br>5 trwo3} 0 0) (0 FRM 0 0.035 0.053<br>E2 TTM04 3 0 0 0 O Cee 0 012<br>trmos} 0 0.018 0 0 0 [keeso.oss<br>UNKWN 438) 0.24 0 0 0 0 0.25<br>nO~ » § SS F$ S<br>Predicted Labels<br><!-- End of picture text -->



<!-- Start of picture text -->
REAL Jer) 10.091 0.055 0 0.018 0.055 [ORR<br>TTMo1 | 0.07 001s o 0 oO [aR19<br>w TTMO27 0 0.035 0 0 0<br>3<br>Str} 0 0 0 (ORM o oor IRR<br>Fr22 Trmo44 0 0.0180.018 0 (ORES 0 [OR<br>TTM05} 0 0.035 0 0.018 0 [UEP INUER<br>UNKWN 40:29 0.11 0.016 0 0.016 0 [yg<br>CEL~ - «+ ES¢€ SSS$ ss<br>Predicted Labels<br><!-- End of picture text -->



<!-- Start of picture text -->
REAL MOAI 0.055 0.036 0.073 0.018 0<br>two! o WEioos ons 0 0<br>o<br>B Trmo2 40.053 0.11 (MWAM 0.088 0.035 0.018<br>s<br>r3 TTM03 0 0.12 0.018 Bieta 0 0<br>TTMo#) 0 0.018 0 0 Bam °<br>TTMO05 0 0 0.018 0.035 0 0.95<br>SS. > SFes CS2 KS- °<br>Predicted Labels<br><!-- End of picture text -->



<!-- Start of picture text -->
REAL (UEMHO.055 0 0.055 0 0 0.091<br>9<br>trmor, 0 [@RNMo.053 0 0 0 0.035<br>a TTMo2 40.018 0.088 flsey 0.053 0.035 0.018 0.16<br>3<br>S rrmo3} 0 0.11 0.0180 0 0 0.053<br>E2 timo] 0 0.018 0 0 PRE o 0<br>twos} 0 0.0180.0180.018 0 [OREM 0<br>UNKWN 40.095 [070.048 0.11 0.032 0 0.17<br>a~ » § SS F$ S<br>Predicted Labels<br><!-- End of picture text -->



<!-- Start of picture text -->
REAL 40.33 0.036 0 0 Oo oO |iXas<br>tro} 0 os 0 0 o o [BG<br> TTo24 0 0 0.23 0.0180.018 0 [ag<br>3<br>S trmo3} 0 0.018 0 033 0 Oo [ay<br>F22 toe} 0 0 0 0 o a<br>105} 0 0 0 0 0 [UE<br>UNKWN 40.016 0.095 0.032 0.032 0 fie 0.83<br>a~ - «+ ¢€ S$ S<br>Predicted Labels<br><!-- End of picture text -->



<!-- Start of picture text -->
REAL Sat 0 0 0 0 0<br>tro} 0 fm 0 0 0 0<br>BtTMo2;o 0 0 1 0 0 0<br>4sraF2 TTM034 0 0 me 0.98 OE)<br>Tr; 0 0 o 0 Fim 0<br>TTMO5 0 0 0 0 0 1<br>&~ EF> SSo > & 2<br>Predicted Labels<br><!-- End of picture text -->



<!-- Start of picture text -->
RAL fmm oO 6©6006€00:«6C0C<br>tro} 0 Mm 0 0 0 0 0<br>Tmo, 0 0 Mmm o 0 0 0<br>3<br>ra5F= rrmo3}toe} 0o0 00 00 POR0 oMm ooOo ooO<br>Tro} 0 0 0 0 0 [QREMo.o1s<br>UNKWN Rg 0 0 0 0 0 0<br>re~ Se eo e e e SS<br>Predicted Labels<br><!-- End of picture text -->



<!-- Start of picture text -->
Coe 0.44 i 0.56<br>tro} 0 [Eo o o oo (en<br>Tmo, 0 0 fm o 0 0 idee<br>3<br>Str}Fra2 toe} 00 00 o0 (a0 0.o | oo [OR(ig<br>Tro} 0 0 0 0 0 [OER<br>Ne 0.52 ae 0 0 0 le 0.48<br>re~ & eo e e e S<br>Predicted Labels<br><!-- End of picture text -->

_J. Imaging_ **2025** , _11_ , 242 

9 of 14 

As expected, the open-set scenario is much more challenging than the closed-set one for all the classification models considered. If we look at the results reported in Tables 2 and 3, we can see that again ResNet18 + Spec achieves the best performance in both cases and that the results obtained via the SVM technique are much worse than the ones obtained via the thresholding approach. However, the analysis of the results becomes different if we look at the confusion matrices. When considering the thresholding method (corresponding to the middle column), we can see that ResNet18 + Spec obtains the best performance when classifying the known methods, but misclassifies all audio excerpts belonging to the class not seen during training and named _UNKNWN_ in the image. Interestingly enough, these are confounded with the real examples, which is somewhat expected, given that the commercial model Suno is probably the most realistic of the considered TTM models. M5 and RawNet2 obtain somehow a similar performance, with the former confounding UNKNWN examples with real and MusicGen-generated ones, while the latter mostly confounding them with MusicGen. 

In the case of the SVM open-set approach, all models behave differently. Approximately half the time, the classification models mistake the known TTM techniques for the unknown one. Interestingly, RawNet2 obtains the highest accuracy of 0.82 for what concerns the unknown class, and even in this case, ResNet18 mistakes it for the real one. 

While more complicated techniques for open-set classification could be used [57,58], the results included here are only intended to provide an initial benchmark of tackling the fake music detection problem on the FakeMusicCaps dataset. More complicated approaches will be considered in future works. 

### _5.3. Impact of Window Size_ 

We also performed a small experiment to verify how much the impact of the temporal window length used as input to the models changes their performance. This is important, especially for the design of further datasets, i.e., do we need to create longer musical excerpts or not? 

We consider four window lengths, namely 10 s, 7.5 s, 5 s, and 2.5 s and report the results in terms of balanced accuracy in Figure 4. As we can see, the variations in accuracy are not extreme in all classification scenarios. M5 seems to have an increase in accuracy passing from 7.5 to 10 s window length for both closed set and open set (threshold) methods. ResNet18+Spec does not have major improvements, while a slight increase in accuracy is seen for RawNet2. Results in the case of the Open set (SVM) show a less clear trend, but the impact of the window size does not seem to be relevant even in this case. 



<!-- Start of picture text -->
1.0 ee<br>aa<br>S 0.9 a<br><<br>0.8<br>3 4 5 6 7 8 9 10<br>Window length |s|<br><!-- End of picture text -->



<!-- Start of picture text -->
0.85 —— ee<br>© 0.80<br>S 0.75 “a<br><x 0.70<br>3 4 5 6 7 8 9 10<br>Window length |s|<br><!-- End of picture text -->



<!-- Start of picture text -->
O<br>— 0.45<br>eeEee<br>3 4 5 6 7 8 9 10<br>Window length |s|<br><!-- End of picture text -->

_J. Imaging_ **2025** , _11_ , 242 

11 of 14 

Moreover, the widespread diffusion of fake music also poses some practical problems. Fake music detectors should be easily deployable on on-line streaming services and in any case where music is streamed live. In order to be able to do this, it is important to create models that are both lightweight enough to be usable in such scenarios and scalable in terms of inference to a high quantity of data. 

## **7. Conclusions** 

In this paper, we tackled the problem of detecting and attributing music generated via text-to-music models. Specifically, we introduced the FakeMusicCaps dataset, created by replicating the MusicCaps dataset via five state-of-the-art TTM models. By applying simple audio forensics techniques, we demonstrate that the dataset could be used as an initial benchmark to tackle TTM detection and attribution. Future developments will also include extending the dataset to contain captions from datasets other than MusicCaps. Our results are not to be considered definitive, instead, our objective is to further motivate the research in forensics techniques for the analysis of generated music. In fact, while the problem of fake music detection and attribution is now relatively simple, it is guaranteed to grow more extremely complicated day by day. 

**Author Contributions:** Conceptualization, L.C. and P.B.; methodology, L.C. and P.B.; software, L.C.; validation, L.C.; data curation, L.C.; writing L.C., P.B. and S.T.; supervision, P.B. and S.T.; funding acquisition, P.B. and S.T. All authors have read and agreed to the published version of the manuscript. 

**Funding:** This research was funded by the Defense Advanced Research Projects Agency (DARPA) and the Air Force Research Laboratory (AFRL) under agreement number FA8750-20-2-1004. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of DARPA and AFRL or the U.S. Government. This work is supported by the European Union under the Italian National Recovery and Resilience Plan (NRRP) of NextGenerationEU (PE00000001—program “RESTART”, PE00000014—program “SERICS”). This work is supported by the FOSTERER project, funded by the Italian Ministry of University, and Research within the PRIN 2022 program. 

**Institutional Review Board Statement:** Not applicable. 

#### **Informed Consent Statement:** Not applicable. 

**Data Availability Statement:** The original dataset presented in the study is openly available at https://zenodo.org/records/15063698, accessed on 13 July 2025, while the code used to perform the experiments is available at https://github.com/polimi-ispl/FakeMusicCaps, accessed on 13 July 2025. 

**Conflicts of Interest:** The authors declare no conflicts of interest. 

## **References** 

1. Briot, J.P.; Hadjeres, G.; Pachet, F.D. _Deep Learning Techniques For Music Generation_ ; Springer: Berlin/Heidelberg, Germany, 2020; Volume 1. 

2. Kumar, R.; Seetharaman, P.; Luebs, A.; Kumar, I.; Kumar, K. High-fidelity audio compression with improved rvqgan. In Proceedings of the 37th International Conference on Neural Information Processing Systems, New Orleans, LA, USA, 10–16 December 2023; Volume 36. 

3. Défossez, A.; Copet, J.; Synnaeve, G.; Adi, Y. High Fidelity Neural Audio Compression. _arXiv_ **2022** , arXiv:2210.13438. [CrossRef] 4. Agostinelli, A.; Denk, T.I.; Borsos, Z.; Engel, J.; Verzetti, M.; Caillon, A.; Huang, Q.; Jansen, A.; Roberts, A.; Tagliasacchi, M.; et al. Musiclm: Generating music from text. _arXiv_ **2023** , arXiv:2301.11325. [CrossRef] 

_J. Imaging_ **2025** , _11_ , 242 

12 of 14 

5. Copet, J.; Kreuk, F.; Gat, I.; Remez, T.; Kant, D.; Synnaeve, G.; Adi, Y.; Défossez, A. Simple and controllable music generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, New Orleans, LA, USA, 10–16 December 2023; Volume 36. 

6. Ziv, A.; Gat, I.; Lan, G.L.; Remez, T.; Kreuk, F.; Défossez, A.; Copet, J.; Synnaeve, G.; Adi, Y. Masked audio generation using a single non-autoregressive transformer. _arXiv_ **2024** , arXiv:2401.04577. [CrossRef] 

7. Tal, O.; Ziv, A.; Gat, I.; Kreuk, F.; Adi, Y. Joint Audio and Symbolic Conditioning for Temporally Controlled Text-to-Music Generation. _arXiv_ **2024** , arXiv:2406.10970. 

8. Liu, H.; Chen, Z.; Yuan, Y.; Mei, X.; Liu, X.; Mandic, D.; Wang, W.; Plumbley, M.D. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. In Proceedings of the International Conference on Machine Learning, Honolulu, HI, USA, 23–29 July 2023. 

9. Liu, H.; Yuan, Y.; Liu, X.; Mei, X.; Kong, Q.; Tian, Q.; Wang, Y.; Wang, W.; Wang, Y.; Plumbley, M.D. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. _IEEE/ACM Trans. Audio Speech Lang. Process_ **2024** , _32_ , 2871–2883. [CrossRef] 

10. Chen, K.; Wuderak, Y.; Liu, H.; Nezhurina, M.; Berg-Kirkpatrick, T.; Dubnov, S. Musicldm: Enhancing novelty in text-to-music generation using beat-synchronous mixup strategies. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Republic of Korea, 14–19 April 2024; pp. 1206–1210. 

11. Huang, Q.; Park, D.S.; Wang, T.; Denk, T.I.; Ly, A.; Chen, N.; Zhang, Z.; Zhang, Z.; Yu, J.; Frank, C.; et al. Noise2music: Text-conditioned music generation with diffusion models. _arXiv_ **2023** , arXiv:2302.03917. 

12. Melechovsky, J.; Guo, Z.; Ghosal, D.; Majumder, N.; Herremans, D.; Poria, S. Mustango: Toward Controllable Text-to-Music Generation. In _Proceedings of the NAACL, Mexico City, Mexico, 16–21 June 2024_ ; Association for Computational Linguistics: Vienna, Austria, 2024; pp. 8293–8316. [CrossRef] 

13. Ronchini, F.; Comanducci, L.; Perego, G.; Antonacci, F. PAGURI: A user experience study of creative interaction with text-to-music models. _arXiv_ **2024** , arXiv:2407.04333. 

14. Suno — suno.com. Available online: https://suno.com/ (accessed on 12 September 2024). 

15. Udio|AI Music Generator—Official Website — udio.com. Available online: https://www.udio.com/ (accessed on 12 September 2024). 

16. Feffer, M.; Lipton, Z.C.; Donahue, C. DeepDrake ft. BTS-GAN and TayloRVC: An Exploratory Analysis of Musical Deepfakes and Hosting Platforms. In Proceedings of the HCMIR@ ISMIR, Milan, Italy, 5–9 November 2023. 

17. Sha, Z.; Li, Z.; Yu, N.; Zhang, Y. De-fake: Detection and attribution of fake images generated by text-to-image generation models. In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, Copenhagen, Denmark, 26–30 November 2023; pp. 3418–3432. 

18. Yu, N.; Davis, L.; Fritz, M. Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints. In Proceedings of the International Conference on Computer Vision (ICCV), Seoul, Republic of Korea, 27 October–2 November 2019. 

19. Corvi, R.; Cozzolino, D.; Zingarini, G.; Poggi, G.; Nagano, K.; Verdoliva, L. On the detection of synthetic images generated by diffusion models. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 4–10 June 2023; pp. 1–5. 

20. Abady, L.; Wang, J.; Tondi, B.; Barni, M. A siamese-based verification system for open-set architecture attribution of synthetic images. _Pattern Recognit. Lett._ **2024** , _180_ , 75–81. [CrossRef] 

21. Wißmann, A.; Zeiler, S.; Nickel, R.M.; Kolossa, D. Whodunit: Detection and Attribution of Synthetic Images by Leveraging Model-specific Fingerprints. In Proceedings of the ACM International Workshop on Multimedia AI against Disinformation (MAD), Phuket, Thailand, 10–14 June 2024. 

22. Mandelli, S.; Bestagini, P.; Verdoliva, L.; Tubaro, S. Facing Device Attribution Problem for Stabilized Video Sequences. _IEEE Trans. Inf. Forensics Secur._ **2019** , _15_ , 14–27. [CrossRef] 

23. Wu, H.; Tseng, Y.; Lee, H.y. CodecFake: Enhancing Anti-Spoofing Models Against Deepfake Audios from Codec-Based Speech Synthesis Systems. In Proceedings of the Interspeech, Kos Island, Greece, 1–5 September 2024. 

24. Salvi, D.; Bestagini, P.; Tubaro, S. Exploring the Synthetic Speech Attribution Problem Through Data-Driven Detectors. In Proceedings of the IEEE International Workshop on Information Forensics and Security (WIFS), Shanghai, China, 12–16 December 2022. 

25. Bhagtani, K.; Bartusiak, E.R.; Yadav, A.K.S.; Bestagini, P.; Delp, E.J. Synthesized Speech Attribution Using The Patchout Spectrogram Attribution Transformer. In Proceedings of the ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec), Chicago, IL, USA, 28–30 June 2023. 

26. Zang, Y.; Zhang, Y.; Heydari, M.; Duan, Z. Singfake: Singing voice deepfake detection. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Republic of Korea, 14–19 April 2024; pp. 12156–12160. 

27. Xie, Y.; Zhou, J.; Lu, X.; Jiang, Z.; Yang, Y.; Cheng, H.; Ye, L. FSD: An initial chinese dataset for fake song detection. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Republic of Korea, 14–19 April 2024; pp. 4605–4609. 

_J. Imaging_ **2025** , _11_ , 242 

13 of 14 

28. Chen, X.; Wu, H.; Jang, J.S.R.; Lee, H.y. Singing Voice Graph Modeling for SingFake Detection. In Proceedings of the Interspeech, Kos Island, Greece, 1–5 September 2024. 

29. Desblancs, D.; Meseguer-Brocal, G.; Hennequin, R.; Moussallam, M. From Real to Cloned Singer Identification. InProceedings of the 25th International Society for Music Information Retrieval Conference, San Francisco, CA, USA, 10–14 November 2024. 

30. Guragain, A.; Liu, T.; Pan, Z.; Sailor, H.B.; Wang, Q. Speech Foundation Model Ensembles for the Controlled Singing Voice Deepfake Detection (CtrSVDD) Challenge 2024. In Proceedings of the 2024 IEEE Spoken Language Technology Workshop, Macao, 2–5 December 2024. 

31. Yizhi, L.; Yuan, R.; Zhang, G.; Ma, Y.; Chen, X.; Yin, H.; Xiao, C.; Lin, C.; Ragni, A.; Benetos, E.; et al. MERT: Acoustic music understanding model with large-scale self-supervised training. In Proceedings of the The Twelfth International Conference on Learning Representations, Singapore, 24–28 April 2023. 

32. Baevski, A.; Zhou, Y.; Mohamed, A.; Auli, M. wav2vec 2.0: A framework for self-supervised learning of speech representations. _Adv. Neural Inf. Process. Syst._ **2020** , _33_ , 12449–12460. 

33. Li, Y.; Milling, M.; Specia, L.; Schuller, B.W. From Audio Deepfake Detection to AI-Generated Music Detection–A Pathway and Overview. _arXiv_ **2024** , arXiv:2412.00571. 

34. Afchar, D.; Meseguer-Brocal, G.; Hennequin, R. AI-Generated Music Detection and its Challenges. In Proceedings of the ICASSP 2025—2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Kothaguda, India, 6–11 April 2025; pp. 1–5. 

35. Wei, Z.; Ye, D.; Deng, J.; Lin, Y. From voices to beats: Enhancing music deepfake detection by identifying forgeries in background. In Proceedings of the ICASSP 2025—2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Kothaguda, India, 6–11 April 2025; pp. 1–5. 

36. Tak, H.; Patino, J.; Todisco, M.; Nautsch, A.; Evans, N.; Larcher, A. End-to-end anti-spoofing with rawnet2. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Toronto, ON, Canada, 6–11 June 2021. 

37. Jung, J.w.; Heo, H.S.; Tak, H.; Shim, H.j.; Chung, J.S.; Lee, B.J.; Yu, H.J.; Evans, N. AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, 22–27 May 2022. 

38. Gemmeke, J.F.; Ellis, D.P.; Freedman, D.; Jansen, A.; Lawrence, W.; Moore, R.C.; Plakal, M.; Ritter, M. Audio set: An ontology and human-labeled dataset for audio events. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA, USA, 5–9 March 2017; pp. 776–780. 

39. Manco, I.; Weck, B.; Doh, S.; Won, M.; Zhang, Y.; Bogdanov, D.; Wu, Y.; Chen, K.; Tovstogan, P.; Benetos, E.; et al. The Song Describer Dataset: A Corpus of Audio Captions for Music-and-Language Evaluation. In Proceedings of the Machine Learning for Audio Workshop at NeurIPS, New Orleans, LA, USA, 10–16 December 2023. 

40. Roy, A.; Liu, R.; Lu, T.; Herremans, D. JamendoMaxCaps: A Large-Scale Music-Caption Dataset with Imputed Metadata. _arXiv_ **2025** , arXiv:2502.07461. 

41. Chu, Y.; Xu, J.; Yang, Q.; Wei, H.; Wei, X.; Guo, Z.; Leng, Y.; Lv, Y.; He, J.; Lin, J.; et al. Qwen2-audio technical report. _arXiv_ **2024** , arXiv:2407.10759. [CrossRef] 

42. Evans, Z.; Parker, J.D.; Carr, C.; Zukowski, Z.; Taylor, J.; Pons, J. Stable Audio Open. _arXiv_ **2024** , arXiv:2407.14358. [CrossRef] 

43. Civit, M.; Drai-Zerbib, V.; Lizcano, D.; Escalona, M.J. SunoCaps: A novel dataset of text-prompt based AI-generated music with emotion annotations. _Data Brief_ **2024** , _55_ , 110743. [CrossRef] [PubMed] 

44. Rahman, M.A.; Hakim, Z.I.A.; Sarker, N.H.; Paul, B.; Fattah, S.A. SONICS: Synthetic Or Not—Identifying Counterfeit Songs. In Proceedings of the Thirteenth International Conference on Learning Representations, Las Vegas, NV, USA, 11–13 August 2025. 

45. Li, Y.; Sun, Q.; Li, H.; Specia, L.; Schuller, B.W. Detecting Machine-Generated Music with Explainability–A Challenge and Early Benchmarks. _arXiv_ **2024** , arXiv:2412.13421. 

46. Li, Y.; Li, H.; Specia, L.; Schuller, B.W. M6: Multi-generator, Multi-domain, Multi-lingual and cultural, Multi-genres, Multiinstrument Machine-Generated Music Detection Databases. _arXiv_ **2024** , arXiv:2412.06001. 

47. Kim, C.D.; Kim, B.; Lee, H.; Kim, G. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Minneapolis, MN, USA, 2–7 June 2019; Volume 1 (Long and Short Papers), pp. 119–132. 

48. Kong, J.; Kim, J.; Bae, J. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. _Adv. Neural Inf. Process. Syst._ **2020** , _33_ , 17022–17033. 

49. Wu, Y.; Chen, K.; Zhang, T.; Hui, Y.; Berg-Kirkpatrick, T.; Dubnov, S. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Ialysos, Greece, 4–10 June 2023; pp. 1–5. 

50. Huang, P.Y.; Xu, H.; Li, J.; Baevski, A.; Auli, M.; Galuba, W.; Metze, F.; Feichtenhofer, C. Masked autoencoders that listen. _Adv. Neural Inf. Process. Syst._ **2022** , _35_ , 28708–28720. 

_J. Imaging_ **2025** , _11_ , 242 

14 of 14 

51. Evans, Z.; Parker, J.D.; Carr, C.; Zukowski, Z.; Taylor, J.; Pons, J. Long-form music generation with latent diffusion. _arXiv_ **2024** , arXiv:2404.10301. [CrossRef] 

52. Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; Liu, P.J. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ **2020** , _21_ , 1–67. 

53. Dai, W.; Dai, C.; Qu, S.; Li, J.; Das, S. Very deep convolutional neural networks for raw waveforms. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA, USA, 5–9 March 2017; pp. 421–425. 

54. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 27–30 June 2016; pp. 770–778. 

55. Hendrycks, D.; Gimpel, K. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. In Proceedings of the International Conference on Learning Representations, Toulon, France, 24–26 April 2017. 

56. Kelleher, J.D.; Mac Namee, B.; D’arcy, A. _Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies_ ; MIT Press: Cambridge, MA, USA, 2020. 

57. Sridhar, S.; Cartwright, M. Multi-Label Open-Set Audio Classification. In Proceedings of the 8th Detection and Classification of Acoustic Scenes and Events 2023 Workshop (DCASE2023), Tampere, Finland, 20–22 September 2023; pp. 171–175. 

58. You, J.; Wu, W.; Lee, J. Open set classification of sound event. _Sci. Rep._ **2024** , _14_ , 1282. [CrossRef] [PubMed] 

**Disclaimer/Publisher’s Note:** The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content. 

