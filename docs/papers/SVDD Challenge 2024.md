# **SVDD 2024: THE INAUGURAL SINGING VOICE DEEPFAKE DETECTION CHALLENGE** 

_You Zhang_<sup>1</sup> _, Yongyi Zang_<sup>1</sup> _, Jiatong Shi_<sup>2</sup> _, Ryuichi Yamamoto_<sup>3</sup> _, Tomoki Toda_<sup>3</sup> _, Zhiyao Duan_<sup>1</sup> 

1University of Rochester, Rochester, NY, USA 

2Carnegie Mellon University, Pittsburgh, PA, USA 3Nagoya University, Nagoya, Japan 

## **ABSTRACT** 

With the advancements in singing voice generation and the growing presence of AI singers on media platforms, the inaugural Singing Voice Deepfake Detection (SVDD) Challenge aims to advance research in identifying AI-generated singing voices from authentic singers. This challenge features two tracks: a controlled setting track (CtrSVDD) and an in-the-wild scenario track (WildSVDD). The CtrSVDD track utilizes publicly available singing vocal data to generate deepfakes using state-of-the-art singing voice synthesis and conversion systems. Meanwhile, the WildSVDD track expands upon the existing SingFake dataset, which includes data sourced from popular user-generated content websites. For the CtrSVDD track, we received submissions from 47 teams, with 37 surpassing our baselines and the top team achieving a 1.65% equal error rate. For the WildSVDD track, we benchmarked the baselines. This paper reviews these results, discusses key findings, and outlines future directions for SVDD research. 

**_Index Terms_ —** singing voice deepfake detection, audio deepfake detection, anti-spoofing, singing analysis 

## **1. INTRODUCTION** 

The use of AI tools in music production has sparked considerable debate, introducing new issues and challenges [1]. Notably, the development of advanced singing voice synthesis (SVS) and conversion (SVC) techniques has reached a significant milestone in AIgenerated songs, where synthesized voices sound remarkably natural and align seamlessly with music scores. These synthesized voices can now emulate the vocal characteristics of any singer with minimal training data. While this technological advancement is impressive, it has raised widespread concerns among artists, record labels, and publishing houses [2]. The potential for unauthorized synthetic reproductions that mimic well-known singers poses a real threat to the commercial value and intellectual property rights of original artists, prompting urgent calls for efficient and accurate methods to detect these deepfake singing voices. 

In response to these concerns, research has emerged towards detecting AI-generated songs. Our prior research SingFake [3] has laid the groundwork for the emerging field of singing voice deepfake detection (SVDD). We introduced the SingFake dataset, a comprehensive collection of authentic and deepfake song clips featuring a variety of languages and singers. Concurrently, Xie et al. [4] curated FSD, a Chinese fake song detection dataset in a controlled setting. Both works found that speech-trained deepfake detection 

This work is supported in part by a New York State Center of Excellence in Data Science award, National Institute of Justice (NIJ) Graduate Research Fellowship Award 15PNIJ-23-GG-01933-RESS, National Science Foundation (NSF) grants 1846184 and 2222129, synergistic activities funded by NSF grant DGE-1922591, and JST CREST JPMJCR19A3, Japan. 

models experience significant performance degradation when tested on the SVDD task, highlighting the unique challenges brought by synthesized singing voices. 

Our SingFake evaluations [3] highlighted several challenges, including dealing with unseen singers, various communication codecs, diverse languages and musical contexts, and interference from accompaniment tracks. This underscores the distinct nature of SVDD and the necessity for specialized SVDD systems. SingGraph [5] has been a novel model that achieves state-of-the-art performance within the SingFake dataset. They utilized an acoustic understanding model and speech self-supervised learning (SSL) models, underscoring the synergy between music understanding and voice analysis for SVDD in the wild. A recent work [6] proposed a general-purpose music deepfake detector that also includes generated instrumental parts, highlighting robustness and generalization issues. 

To advance the field of SVDD, we introduce the SVDD challenge [7], the inaugural research initiative specifically dedicated to exploring SVDD. This challenge targets both controlled and inthe-wild settings, aiming to distinguish bonafide and AI-generated singing voice recordings within CtrSVDD and WildSVDD tracks, respectively. For the CtrSVDD track, we generated a dataset CtrSVDD [8] that exclusively uses clean, unaccompanied vocals provided by our data contributors, thereby mitigating the interference of voice separation algorithms. This two-track approach allows participants to tackle the challenges of identifying deepfake singing voices under different and realistic conditions. The WildSVDD track follows the same approach as our SingFake project [3], dealing with deepfakes as they typically appear in online media, complete with background music. 

The SVDD challenge has received much attention, and the results have been promising. For the CtrSVDD track, we received submissions from 47 teams, with 37 surpassing our baselines and the top team achieving a 1.65% equal error rate. Common strategies included self-supervised feature and ensemble models. For the WildSVDD track, we have not received any submissions yet, but we have benchmarked the baselines. Future research endeavors can focus on improving generalization ability with single systems and further exploring in-the-wild scenarios. We look forward to continued progress in this critical area of research. 

## **2. OVERVIEW OF CHALLENGE SETUPS** 

## **2.1. Two tracks: CtrSVDD and WildSVDD** 

For singing voice deepfakes, these artificial creations are typically presented with background music, mirroring the authentic song experience. Our previous work on SingFake [3] followed this approach by collecting deepfakes from web videos. However, this method introduces a significant challenge: the process of separating vocals from music can create artifacts that obscure the differences between bonafide and deepfake vocals. 

To address the above issues comprehensively, the SVDD challenge is structured into two distinct tracks: _Controlled_ (CtrSVDD) and _In-the-Wild_ (WildSVDD) settings. The WildSVDD track mirrors the approach of the SingFake project, dealing with deepfakes as they typically appear in online media, including background instrumental interference and sometimes background noises. Conversely, the CtrSVDD track uses clean, unaccompanied vocals provided by our data contributors, thereby minimizing the interference caused by voice separation algorithms. The data construction process was described in [8]. We also train deepfake algorithms ourselves and thus can have all metadata of training configurations as well as source and target utterances. This double-track approach allows participants to tackle the challenges of identifying deepfake singing voices under both lab-controlled and realistic conditions. 

For CtrSVDD, we generate singing vocals with existing SVS and SVC systems. The setting mitigates the artifacts by the singing voice separation algorithms, and we expect it to be easier than the in-the-wild settings. For WildSVDD, we continued the data collection process of SingFake to collect more data from the video platforms. This collection from the web inevitably introduces additional difficulties in the systematic analysis of SVDD systems due to unknown backend synthesis/conversion systems and post-processing strategies. The compression codecs used by the web platforms are also unknown and may change over time, causing inconsistencies across different downloads. 

**Table 1** . Overview of the deepfake methods included in the CtrSVDD database. Please refer to [8] for detailed descriptions. 

|**System**|**Model**|**Type**|**Description**|
|---|---|---|---|
|A01|XiaoiceSing|SVS|Cascaded Transformer model with a HiFi-GAN vocoder|
|A02|VISinger|SVS|End-to-end VAE with a HiFi-GAN vocoder|
|A03|VISinger2|SVS|End-to-end VAE with a DDSP vocoder|
|A04|NNSVS|SVS|Cascaded diffusion model with a source-flter HiFi-GAN|
|A05|Naive RNN|SVS|Cascaded LSTM model with a HiFi-GAN vocoder|
|A06|NU-SVC|SVC|NNSVS model with ContentVec linguistic features|
|A07|Soft-VITS-SVC|SVC|Soft-VITS model with WavLM linguistic features|
|A08|Soft-VITS-SVC|SVC|Soft-VITS model with ContentVec linguistic features|
|A09|Soft-VITS-SVC|SVC|Soft-VITS model with additional source-flter HiFi-GAN|
|A10|Soft-VITS-SVC|SVC|Soft-VITS model with MR-HuBERT linguistic features|
|A11|Soft-VITS-SVC|SVC|Soft-VITS model with WavLabLM linguistic features|
|A12|DiffSinger|SVS|Cascaded Transformer model with a post diffusion module|
|A13|Soft-VITS-SVC|SVC|Soft-VITS model with Chinese HuBERT linguistic features|
|A14|ACESinger|SVS|Blackbox commercial system with manual tuning|



**Table 2** . Summary of the CtrSVDD database [8]. 

|Partition|# Singers|Bonafde|Dee|pfake|
|---|---|---|---|---|
|||# Clips|# Clips|Methods|
|Training|59|12,169|72,235|A01_∼_A08|
|Development|55|6,547|37,078|A01_∼_A08|
|Evaluation|48|13,596|79,173|A09_∼_A14|



(A01-A08), while the evaluation set employs a different set (A09A14). The database is released under a CC-BY-NC-ND 4.0 license, aligned with the sourcing corpora. We released the training and development set<sup>2</sup> and the evaluation set<sup>3</sup> , while the keys and labels for the evaluation set were withheld until the submission deadline. 

## **2.2. Evaluation metric: EER** 

We expect each SVDD system to generate a score file for every segmented clip, with higher scores indicating greater confidence that the clip is bonafide. The Equal Error Rate (EER) is achieved when the false acceptance and rejection rates are equal by varying the decision threshold, hence EER is independent of the threshold. A lower EER indicates a better system for distinguishing between bonafide and deepfake singing voices. 

## **3. CTRSVDD: CONTROLLED SINGING VOICE DEEPFAKE DETECTION** 

## **3.1. CtrSVDD database** 

To construct the CtrSVDD database, we first collected bonafide samples from open-source singing recordings. These include<sup>1</sup> : 

- Mandarin datasets: Opencpop [9], M4Singer [10], KiSing [11], and the official ACE-Studio release [12]. 

- Japanese datasets: Ofuton-P, Oniku Kurumi, Kiritan [13], and JVS-MuSiC [14]. 

We segmented these public dataset recordings into vocal clips. Next, we generated deepfake singing vocals using 14 existing SVS and SVC systems, as described in Table 1. The SVS systems employed were ESPnet-Muskits [15], NNSVS [16], DiffSinger [17], and ACESinger [12]. For SVC, we utilized NU-SVC [18] and variants of So-VITS-SVC. Specially for ACESinger, we asked the ACE Studio company to provide the bonafide utterances for a list of singers; we then generated deepfake data using the ACE Studio tool. 

The CtrSVDD database is partitioned into training, development, and evaluation splits, following the methodology of the ASVspoof2019 speech anti-spoofing benchmark. The training and development sets use the same set of deepfake generation algorithms 

## **3.2. Protocols** 

Note that the training and development sets available on Zenodo are incomplete because of licensing issues with some bonafide datasets. To fully retrieve the dataset, we asked the participants to first download all the remaining bonafide datasets on their own by agreeing to the terms and conditions and then follow the instructions we provided<sup>4</sup> . Participants can refer to the statistics in Table 2 as a guide to verify the completion of their downloads and generation. 

Participants are asked to ask teams to score each vocal clip. Using the submitted scores, we calculate and rank participant systems using EER. During our baseline analysis [8], we found the A14 attack (originated from ACESinger) shows strong out-of-domain attributes [8]. As such, we exclude the A14 attack and the bonafide utterances of the ACESinger during the official ranking of the CtrSVDD track. Firstly, unlike other deepfake attacks (A09-A13) that were trained using open-source datasets, the A14 attack was trained on unknown data sources. Additionally, the bonafide portion of ACESinger was not used to train the other attack systems (A09-A13). The bonafide portion of ACESinger was provided by Timedomain, a company that may apply additional processing steps to the audio output, while the deepfake portion was generated using their ACE Studio desktop application. 

## **3.3. Submission rules** 

We used CodaBench [19] for CtrSVDD results submission. Each team is allowed a maximum of three submissions for the entire duration of the CtrSVDD challenge for official ranking purposes. This limit is in place to ensure fairness and to encourage strategic submissions. It is important to note that CodaBench’s daily submis- 

> 2https://zenodo.org/records/10467648 

> 1For comprehensive details of CtrSVDD, including links to source datasets and generation methods, please refer to [8]. 

> 3https://zenodo.org/records/12703261 

> 4https://github.com/SVDDChallenge/CtrSVDD ~~U~~ tils 

sion limit is separate; our three-submission limit refers to the total allowable submissions for the challenge. After the three initial submissions are used, participants may utilize an additional CodaBench dedicated to research. This allows them to submit scores and obtain per-attack, per-dataset, and overall EERs for research purposes. 

Participants are welcome to use any publicly available datasets for training in addition to the CtrSVDD we provide, but of course, exclude any datasets used in our test set. Specifically, for the CtrSVDD track, participants must not use M4Singer, KiSing, any open-sourced deepfake models based on M4Singer and/or KiSing, or the commercial software ACE Studio [12]. We refer the participants to the list of available datasets in the evaluation plan [7]. 

## **3.4. Baseline solutions** 

We have developed two baseline systems for the challenge: one that uses raw waveforms and another that employs linear frequency cepstral coefficients (LFCCs) as front-end features: The raw waveform system is an AASIST [20]-based system. The LFCC system uses 60 coefficients, with a 512 sample window and 160 sample hop size. The LFCC features pass through several downsampling residual convolution blocks and a linear layer connecting it to the graph attention network backend of [20]. 

We refer to the LFCC system as **B01** and the raw waveform model as **B02** . For both systems, we conducted training over 100 epochs using a fixed random seed, exclusively on the CtrSVDD training partition. We then selected the checkpoint that achieved the lowest validation EER on the CtrSVDD development partition for evaluation. During training and evaluation, the models processed 4-second random audio segments from each utterance. Details of the implementation are available on the challenge GitHub repository<sup>5</sup> . 

## **4. WILDSVDD: IN-THE-WILD SINGING VOICE DEEPFAKE DETECTION** 

## **4.1. WildSVDD database** 

We gathered data annotations from social media platforms following a method similar to the SingFake project [3]. The WildSVDD dataset has been expanded to approximately double the original size of SingFake, now featuring 97 singers with 2007 deepfake and 1216 bonafide songs. The annotators, who were familiar with the singers they covered, manually verified the user-specified labels during the annotation process to ensure accuracy, especially in cases where the singer(s) did not actually perform certain songs. We cross-checked the annotations against song titles and descriptions, and manually reviewed any discrepancies for further verification. We verified the accessibility of all URLs in the dataset as of March 28th and removed any that were inaccessible. The WildSVDD dataset now includes Korean singers, making Korean the third most represented language in the dataset. To help track changes between the SingFake and WildSVDD datasets, we have added a ”SingFake ~~S~~ et” column that indicates the original partition of an annotation in the SingFake dataset. Annotations that lack a value in this column are new additions to the WildSVDD dataset. We form two test sets: Test A denotes a newly formed testing dataset including new samples, while Test B denotes the hardest test set T04 detailed in [3]. 

Due to potential copyright issues, we only released the annotations<sup>6</sup> under a CC-BY 4.0 license. Consequently, participants might 

**Table 3** . Comparison of EERs of different baseline front-end processing methods and settings on WildSVDD track. 

|Front-end|WildSVDD|Test A|WildSVDD|Test B|
|---|---|---|---|---|
||Mixtures|Vocals|Mixtures|Vocals|
|Raw Waveform|10.50|8.48|16.85|14.91|
|Spectrogram|27.93|20.55|30.97|24.41|
|Mel-Spectrogram|29.27|27.35|32.18|30.78|
|MFCC|17.78|19.14|22.92|23.31|
|LFCC|22.60|23.25|26.82|26.94|
|Wav2vec2 XLS-R|9.57|6.09|21.45|24.09|



acquire slightly different media files that correspond to the same annotations, depending on the specifics of their download process. Due to this variability, self-reported metrics from participants can, at best, be used as a rough reference and cannot be directly used to compare systems. As such, we encourage participants to report the success rate of URL downloads per partition and, if possible, the actual files used for training and testing. This transparency allows researchers to make fairer comparisons. Additionally, participants are encouraged to open source their model implementations to facilitate the reproduction of results with the WildSVDD dataset. 

## **4.2. Protocols** 

We provide the training and test partitions, allowing participants the flexibility to carve out a validation set from the training data for model development. We provide labels of SingFake [3] partitions for annotations that appeared in the SingFake dataset for easy comparison with previous systems. The test set is divided into parts A and B, with part B considered more challenging due to its inclusion of unseen musical contexts [3]. 

We recommend that participants further segment the songs into clips using our tool available in the SingFake GitHub repository<sup>7</sup> . Evaluations should be conducted at the segment level rather than at the song level. We adopt the self-reported EER and do not accurately rank the results. We encourage the participants to submit the score files listing the URLs, segment start and end timestamps, and the corresponding scores output from their systems. 

## **4.3. Baseline solutions** 

To establish baseline solutions, we implement the architecture described in [8], applying both mixtures and vocals configurations as outlined in [3]. We also incorporate the self-supervised learning model XLS-R [21] based system proposed in [22], given its popularity among top performers in the CtrSVDD track. Existing research and CtrSVDD track outcomes indicate that data augmentation significantly enhances performance with self-supervised frontends. Consequently, we present results both with and without augmentation. We adhere to the training schedule, learning rate, and optimizer specified in [8], with the exception of XLS-R based models, where we adopt training settings same as [22]. 

Notably, we observe that the raw waveform system’s performance on WildSVDD Test B, which is equivalent to SingFake T04, shows significant improvement compared to [3], despite highly similar system architectures. This improvement suggests that datacentric approaches may prove most effective in the long run. The most substantial difference between these two system versions lies in the volume of training data available, highlighting the potential impact of increased data resources on model performance. 

> 5https://github.com/SVDDChallenge/CtrSVDD2024 ~~B~~ aseline 

> 6https://zenodo.org/records/10893604 

> 7https://github.com/yongyizang/SingFake 

## **5. CHALLENGE RESULTS** 

We received 89 registration forms, with 84 teams expressing interest in participating in the CtrSVDD track and 65 teams showing interest in the WildSVDD track. Out of these, 47 teams submitted their results to the CtrSVDD track. Unfortunately, we have not received any submissions for the WildSVDD track, probably due to time constraints and copyright concerns. 

## **5.1. CtrSVDD results from the team submissions** 

Table 4 presents the results from the CtrSVDD track of the SVDD challenge, ranking the participating teams based on their EER without ACESinger and providing additional EER values for all attacks. Most teams used their three allowed entries for submission, while some did not but still achieved commendable ranks. We ranked the teams based on the best submission (lowest EER) among their entries. Among the 47 submissions (excluding two baselines), 37 surpassed both baselines, demonstrating substantial progress in developing SVDD systems capable of distinguishing between bonafide and deepfake singing voices. 

The baseline systems B01 and B02 achieved EERs of 12.03% and 11.16%, respectively. The raw-waveform-based system outperformed the LFCC-based system, which was consistent with observations in the speech anti-spoofing task. The top-ranked team achieved an impressive EER of 1.65%, significantly lower than the baselines. The top teams’ results were very close, highlighting the competitive nature of the challenge. However, several teams’ performances fell between or even below the baseline results, underscoring the challenge’s difficulty. These results emphasize the complexity of SVDD, especially when considering diverse unseen attack methods. 

The overall EERs that include the ACESinger bonafide portion and A14 attack are consistently higher than those that exclude them, indicating the difficulty of detecting deepfakes generated from outof-domain commercial SVS systems. While the trend of increasing EERs is generally similar, some systems achieved much lower or higher overall EERs than expected. This leads to our detailed analysis of per-attack EER and per-dataset EER in the next subsection. 

## **5.2. Per-attack and per-dataset analysis** 

For the per-dataset EER, we calculate the EER based on the metadata of the source bonafide datasets, specifically identifying which dataset the target speaker belongs to. For the per-attack EER, A09A13 is calculated using all bonafide data except ACESinger, along with the deepfake data generated by the corresponding methods. The A14 EER is calculated using the ACESinger bonafide portion and the deepfakes generated by the ACE Studio tool, making it identical to the per-dataset EER for ACESinger. Note that we used a slightly different metric during the challenge but changed post-challenge. Previously, per-attack EERs on A09-A14 were calculated using all bonafide song clips against deepfake clips for each category [8], which could be what was reported by participants. 

Table 5 provides a detailed analysis of the top eight ranked submissions. The per-attack results do not show a consistent decline across all attacks as rankings improve. Notably, the performance on A11 and A12 suggests that the top-performing systems may not necessarily be more robust to different types of attacks. This observation is consistent with findings in speech anti-spoofing research [23]. Besides, The systems’ performance on A12 is generally worse compared to other attacks, which is consistent with our baseline analysis [8]. Furthermore, although the top four systems achieved similar 

**Table 4** . Summary of the CtrSVDD challenge results. The EER without ACESinger is used as the evaluation metric to rank the submissions, while the EER for all attacks is listed for analysis. The rows for both baseline systems are shaded. Teams with bolded ranks submitted the system description. 

|**Rank #**|**Team Name**|**# Entries**|**EER**<br>|**EER**<br>|
|---|---|---|---|---|
||||**(w/o ACESinger)**|**(overall)**|
|**1**|Fosafer Speech|3|1.65|4.32|
|**2**|NBU<br>~~M~~ISL|3|2.00|8.41|
|**3**|I2R-ASTAR|3|2.22|4.86|
|**4**<br>|Qishan|2|2.32|4.45<br>|
|5|Breast waves|3|2.73|5.38|
|6|MediaForensics|3|2.75|5.83|
|7|beyond|3|2.99|5.68|
|8|Star|2|3.31|5.21|
|9|shrinep|3|3.53|5.91|
|10|HUBENMINZU|3|3.61|5.63|
|**11**|Pindrop|3|3.85|6.27|
|12|SVDD-Xin|2|4.05|8.17|
|13|DDD|3|5.33|7.57|
|14|rudraprasad|2|5.45|9.87|
|15|beautifulboy|3|5.79|7.50|
|16|WinterIsComing|2|6.71|7.37|
|17|bokingchen|3|6.76|17.98|
|18|UCAS<br>2024|2|6.81|7.48|
|18|ColdLightXXX|3|6.81|9.97|
|**20**|xieyuankun|1|6.90|7.95|
|21|xyyuan|3|7.36|8.35|
|22|tarrifn|3|7.56|17.62|
|23|LLLSLin|3|7.90|8.27|
|24|EEGBrain|3|7.92|10.46|
|25|MelodyAI|1|8.66|9.66|
|26|Outlaw Monkeys|1|8.73|13.24|
|27|<br>starbucks|3|8.82|12.69|
|28|ForgeryMark|3|8.85|8.84|
|29|test<br>~~a~~cc|3|8.87|14.53|
|**30**|notepad|3|8.99|11.88|
|31|Feathers|2|9.22|14.32|
|32|asada|1|9.64|10.66|
|33|fengchuippshuang|1|10.14|9.91|
|34|USC|2|10.31|13.05|
|35|JAIST|2|10.39|12.68|
|36|Polimi-Unibs|3|10.50|22.58|
|37|liuziyi|3|10.80|19.76|
|38|B02||11.16|13.58|
|38|jiaruiliu|1|11.16|14.70|
|40|SnthSound|3|1121|1454|
|41|y<br>zzgww|3|.<br>11.27|.<br>12.60|
|42|AIS|3|11.67|15.52|
|43|B01||12.03|16.10|
|44|B401|3|12.39|15.92|
|45|datajedi23|2|13.50|18.32|
|46|Zhejiang University|3|15.91|18.48|
|**47**|<br>vitas|3|16.05|28.10|
|48|Dashlab|3|17.04|27.13|
|49|jiachengdeng|3|17.44|20.45|



EERs, their robustness differs significantly. The “NBU ~~M~~ ISL” system is less robust to KiSing and ACESinger compared to other top systems, dropping to 19th place when ACESinger is factored in. 

A similar phenomenon is evident in the per-dataset results. The best performance on KiSing was achieved by the team “Star”, which was ranked 8th. While KiSing and M4Singer datasets yield overall low EERs, ACESinger consistently produced EERs around 50%, approximating random guessing. The performance gap observed with ACESinger is likely due to inconsistencies between ACESinger and the rest of the CtrSVDD dataset. Notably, two teams achieved 93% and 84% on ACESinger, indicating some ability to establish a decision boundary on this dataset, albeit with incorrect decisions. 

Both the per-attack and per-dataset EERs highlight the ongoing challenge of generalization to unseen generation methods and acoustic channels. This underscores the need for increasing research efforts focused on developing systems that are more robust to these out-of-distribution conditions. 

**Table 5** . Overview of the top-8 ranked submission results. The best performance in each column is **bolded** and the second is underlined. 

|**Team Name**<br>Results (w/o ACESinger)<br>Results (overall)|Per-Attack EER<br>Per-Dataset EER<br>ACESinger (A14)|
|---|---|
|EER (%)<br>Rank<br>EER (%)<br>Rank<br>A09|<br>A10<br>A11<br>A12<br>A13<br>KiSing<br>M4Singer|
|Fosafer Speech<br>**1.65**<br>1<br>**4.32**<br>1<br>0.23|**0.06**<br>**0.37**<br>**4.19**<br>**0.07**<br>2.66<br>**1.69**<br>49.67|
|<br>NBU<br>~~M~~ISL<br>2.00<br>2<br>8.41<br>19<br>**0.13**|0.11<br>0.94<br>5.17<br>0.10<br>8.98<br>2.07<br>50.02|
|I2R-ASTAR<br>2.22<br>3<br>4.86<br>3<br>0.65|0.51<br>2.49<br>4.57<br>0.64<br>6.01<br>2.16<br>50.02|
|Qishan<br>2.32<br>4<br>4.45<br>2<br>1.02|0.69<br>2.54<br>4.42<br>0.76<br>2.82<br>2.32<br>50.05|
|Breast waves<br>2.73<br>5<br>5.38<br>5<br>1.50|0.76<br>2.03<br>6.14<br>0.88<br>3.56<br>2.84<br>50.44|
|MediaForensics<br>2.75<br>6<br>5.83<br>8<br>0.56<br><br><br><br><br><br>|0.38<br>3.90<br>4.45<br>1.02<br>10.56<br>2.56<br>49.91<br><br><br><br><br><br><br>|
|beyond<br>2.99<br>7<br>5.68<br>7<br>0.45<br><br><br><br><br><br>|0.26<br>4.56<br>4.37<br>0.85<br>9.12<br>2.85<br>**49.53**<br><br><br><br><br><br><br>|
|Star<br>3.31<br>8<br>5.21<br>4<br>1.64|0.19<br>1.11<br>7.30<br>0.23<br>**1.79**<br>3.51<br>49.70|
|AASIST<br>Wav2vec2 XLSR<br> <br> <br><br>|domain adaptation in the third system.<br>The team “NBU|
|RawNet Encoder<br>Raw Waveform<br>Score Summation<br>ResNet<br>AASIST*<br>**#1**Fosafer Speech|developed two subsystems focused on low and high frequ|
|<br>**Data note:**No data augmentation was used. Additional datasets were incorporated.|respectively. Both used feature aggregation from Chinese H<br>and the corresponding frequency band. They augmented the|
|Chinese HuBERT<br>(first 12 layers)<br>ResBlocks+AASIST|with HiFi-GAN [30] vocoded audio. The team “I2R-ASTAR|
|<br>Pooling<br> <br>Chinese HuBERT<br> <br>Feature Concat<br>|oped fve subsystems with various layer aggregation strateg|
|+1DConv ResBlocks<br>(last 13 layers)<br> <br>+FC layer|and variants of the RawBoost data augmentation method f|
|2DConv ResBlocks<br>Low-frequency<br>Subband w/o F0|subsystem. Please refer to [32] for more details of their|
|Low-frequency<br>|The team “Qishan” developed two subsystems with differe<br>features. Each subsystem follows a Sensitive Layer Selec|
|Subsystem<br> <br>Score Averaging<br>**#2**NBU_MISL|classifer that uses an adaptive weight allocation method|
|High-frequency<br>Subsystem<br> <br>|aggregate SSL features and pool the feature map to a scor<br>|
|**Data note:**Augmented with HiFi-GAN vocoded audio. No additional datasets were incorporated.|score with a larger absolute value is selected for submission.<br>The team “Pindrop” developed three subsystems base|
|AASIST<br>Wav2vec2 XLSR<br> <br>|vector, ResNetSE34, and wav2vec2 XLSR. Additional datase|
|**#3**I2R-ASTAR<br>Wav2vec2 XLSR<br>WavLM<br>Score Averaging<br>AASIST<br>AASIST|incorporated. The team “xieyuankun” submitted a single sys<br>|
|WavLM†<br>AASIST<br><br>|ing wav2vec2 XLSR combined with AASIST, without any da|
|WavLM†<br>AASIST|mentation. This approach could serve as an addition to our b|
|**Data note:**Augmented with RawBoost variations. No additional datasets were incorporated.|systems. More systems have been submitted to the researc|
|Wav2vec2 XLSR<br>SLS classifier<br>Score selection w/<br>|of CodaBench, some of which demonstrate better performan|
|<br> <br>WavLM<br>SLS classifier<br> <br>larger absolute value<br>**#4**Qishan|their challenge submissions. We look forward to paper subm|
|**Data note:**No data augmentation was used. No additional datasets were incorporated.<br>**Fig. 1**. Illustration of the strategies employed by the top-4 ranked<br>system submissions for the CtrSVDD track. An asterisk (*) indicates<br>the additional use of adversarial training strategies for AASIST. A<br>dagger (†) denotes different layer aggregation strategies proposed<br>for WavLM, as opposed to the weighted sum method.<br>**5.3. Solution strategies**<br>|to learn more about their strategies.<br>**6. DISCUSSIONS AND CONCLUSIONS**<br>The CtrSVDD track of the SVDD challenge was a notable s<br>attracting 47 submissions, with 37 surpassing the baseline <br>mance. The top teams employed diverse and advanced tech<br>such as self-supervised learning, ensemble learning, and adv<br>training, demonstrating signifcant innovation in the feld. D<br>system descriptions from eight teams provided valuable insi|



domain adaptation in the third system. The team “NBU ~~M~~ ISL” developed two subsystems focused on low and high frequencies, respectively. Both used feature aggregation from Chinese HuBERT and the corresponding frequency band. They augmented the dataset with HiFi-GAN [30] vocoded audio. The team “I2R-ASTAR” developed five subsystems with various layer aggregation strategies [31] and variants of the RawBoost data augmentation method for each subsystem. Please refer to [32] for more details of their system. The team “Qishan” developed two subsystems with different SSL features. Each subsystem follows a Sensitive Layer Select (SLS) classifier that uses an adaptive weight allocation method [33] to aggregate SSL features and pool the feature map to a score. The score with a larger absolute value is selected for submission. 

The team “Pindrop” developed three subsystems based on x- vector, ResNetSE34, and wav2vec2 XLSR. Additional datasets were incorporated. The team “xieyuankun” submitted a single system using wav2vec2 XLSR combined with AASIST, without any data augmentation. This approach could serve as an addition to our baseline systems. More systems have been submitted to the research track of CodaBench, some of which demonstrate better performance than their challenge submissions. We look forward to paper submissions to learn more about their strategies. 

The CtrSVDD track of the SVDD challenge was a notable success, attracting 47 submissions, with 37 surpassing the baseline performance. The top teams employed diverse and advanced techniques, such as self-supervised learning, ensemble learning, and adversarial training, demonstrating significant innovation in the field. Detailed system descriptions from eight teams provided valuable insights for future research. This success highlights the progress in deepfake detection for singing voices and sets the stage for further advancements and improvements. 

Among all submissions, 8 teams submitted system descriptions, with their ranks bolded in Table 4. Based on the submitted strategies, most teams utilized self-supervised learning (SSL) frontends and ensemble learning. For features, both raw waveform and SSL features were extensively explored. The most popular SSL feature used is wav2vec2 XLSR [21], a cross-lingual representation. Popular backend choices included ResNet and AASIST [20], while score averaging was the favored ensemble method. 

The lack of submissions for the WildSVDD track can be attributed to concerns over copyright issues, the complexities of the download process, and the time-consuming nature of data preparation. Additionally, challenges with the reproducibility of baseline methods and the limited time frame may have discouraged participation. Teams might have prioritized the CtrSVDD track, where their efforts would be more recognized and impactful. 

All the top four teams share similar strategies. Their general schemes are illustrated in Figure 1. The top-ranked team, “Fosafer Speech,” used additional datasets besides CtrSVDD: HiFiTTS [24], OpenSinger [25], CSD [26], itako-Singing, JSUT-Song, Namine ~~r~~ itsu ~~u~~ tagoe ~~d~~ b, no7-singing, PJS [27], PoPCS [17], URSing [28], which are all among the list of allowable external training datasets in our evaluation plan [7]. They also generated additional<sup>9</sup> training data, nhv ~~s~~ hare, and made it public<sup>8</sup> , one week before the challenge submission deadline as requested. They fused three subsystems and incorporated adversarial training [29] to improve 

For future work, the CtrSVDD track can be enhanced by including the latest advancements in SVS and SVC techniques. Additionally, cross-database evaluations between the CtrSVDD and WildSVDD datasets could be conducted to assess the generalizability and robustness of SVDD systems, offering an intriguing research avenue. Exploring a broader variety of singing deepfake types is also recommended. Given recent advancements in singing generative models that can produce not only vocals but also accompanying music, as noted in [34], it becomes crucial to develop detection systems capable of identifying AI-generated songs in their entirety. 

> 8https://drive.google.com/file/d/1h36C6mWvywIYXSPErDf2tcYfPuRkGKoQ/ 

> 9https://drive.google.com/file/d/1LhbH2-yNe ~~Z~~ HcmTxqS5q7XQYlL1 ~~J~~ M ~~-~~ / 

### **7. REFERENCES** 

- [1] Emmanuel Deruty, Maarten Grachten, Stefan Lattner, Javier Nistal, and Cyran Aouameur, “On the development and practice of AI technology for contemporary popular music production,” _TISMIR_ , vol. 5, no. 1, pp. 35–50, 2022. 

- [2] Nick Collins and Mick Grierson, “Avoiding an AI-imposed taylor’s version of all music history,” _arXiv preprint arXiv:2402.14589_ , 2024. 

- [3] Yongyi Zang, You Zhang, Mojtaba Heydari, and Zhiyao Duan, “SingFake: Singing voice deepfake detection,” in _Proc. IEEE ICASSP_ . IEEE, 2024, pp. 12156–12160. 

- [4] Yuankun Xie, Jingjing Zhou, Xiaolin Lu, Zhenghao Jiang, Yuxin Yang, Haonan Cheng, and Long Ye, “FSD: An initial chinese dataset for fake song detection,” in _Proc. IEEE ICASSP_ , 2024, pp. 4605–4609. 

- [5] Xuanjun Chen, Haibin Wu, Roger Jang, and Hung yi Lee, “Singing voice graph modeling for SingFake detection,” in _Proc. Interspeech_ , 2024, pp. 4843–4847. 

- [6] Darius Afchar, Gabriel Meseguer Brocal, and Romain Hennequin, “Detecting music deepfakes is easy but actually hard,” _arXiv preprint arXiv:2405.04181_ , 2024. 

- [7] You Zhang, Yongyi Zang, Jiatong Shi, Ryuichi Yamamoto, Jionghao Han, Yuxun Tang, Tomoki Toda, and Zhiyao Duan, “SVDD challenge 2024: A singing voice deepfake detection challenge evaluation plan,” _arXiv preprint arXiv:2405.05244_ , 2024. 

- [8] Yongyi Zang, Jiatong Shi, You Zhang, Ryuichi Yamamoto, Jionghao Han, Yuxun Tang, Shengyuan Xu, Wenxiao Zhao, Jing Guo, Tomoki Toda, and Zhiyao Duan, “CtrSVDD: A benchmark dataset and baseline analysis for controlled singing voice deepfake detection,” in _Proc. Interspeech_ , 2024, pp. 4783–4787. 

- [9] Yu Wang, Xinsheng Wang, Pengcheng Zhu, Jie Wu, Hanzhao Li, Heyang Xue, Yongmao Zhang, Lei Xie, and Mengxiao Bi, “Opencpop: A high-quality open source chinese popular song corpus for singing voice synthesis,” in _Proc. Interspeech_ , 2022, pp. 4242–4246. 

- [10] Lichao Zhang, Ruiqi Li, Shoutong Wang, Liqun Deng, Jinglin Liu, Yi Ren, Jinzheng He, Rongjie Huang, Jieming Zhu, Xiao Chen, and Zhou Zhao, “M4singer: A multi-style, multi-singer and musical score provided mandarin singing corpus,” in _Proc. NeurIPS (Dataset and Benchmarks Track)_ , 2022. 

- [11] Jiatong Shi, Yueqian Lin, Xinyi Bai, Keyi Zhang, Yuning Wu, Yuxun Tang, Yifeng Yu, Qin Jin, and Shinji Watanabe, “Singing voice data scaling-up: An introduction to ACE-Opencpop and ACE-KiSing,” in _Proc. Interspeech_ , 2024, pp. 1880–1884. 

- [12] Timedomain, “ACE Studio,” https://acestudio.ai/. 

- [13] Itsuki Ogawa and Masanori Morise, “Tohoku kiritan singing database: A singing database for statistical parametric singing synthesis using japanese pop songs,” _AST_ , vol. 42, no. 3, pp. 140–145, 2021. 

- [14] Hiroki Tamaru, Shinnosuke Takamichi, Naoko Tanji, and Hiroshi Saruwatari, “JVS-MuSiC: Japanese multispeaker singing-voice corpus,” _arXiv preprint arXiv:2001.07044_ , 2020. 

- [15] Jiatong Shi, Shuai Guo, Tao Qian, Tomoki Hayashi, Yuning Wu, Fangzheng Xu, Xuankai Chang, Huazhe Li, Peter Wu, Shinji Watanabe, and Qin Jin, “Muskits: an end-to-end music processing toolkit for singing voice synthesis,” in _Proc. Interspeech_ , 2022, pp. 4277–4281. 

- [16] Ryuichi Yamamoto, Reo Yoneyama, and Tomoki Toda, “NNSVS: A neural network-based singing voice synthesis toolkit,” in _Proc. IEEE ICASSP_ . IEEE, 2023, pp. 1–5. 

- [17] Jinglin Liu, Chengxi Li, Yi Ren, Feiyang Chen, and Zhou Zhao, “Diffsinger: Singing voice synthesis via shallow diffusion mechanism,” in _Proc. AAAI_ , 2022, vol. 36. 

   - [19] Adrien Pavao, Isabelle Guyon, Anne-Catherine Letournel, Dinh-Tuan Tran, Xavier Baro, Hugo Jair Escalante, Sergio Escalera, Tyler Thomas, and Zhen Xu, “Codalab competitions: An open source platform to organize scientific challenges,” _JMLR_ , vol. 24, no. 198, pp. 1–6, 2023. 

   - [20] Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu, and Nicholas Evans, “AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks,” in _Proc. IEEE ICASSP_ , 2022, pp. 6367–6371. 

   - [21] Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, and Michael Auli, “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,” in _Proc. Interspeech_ , 2022, pp. 2278– 2282. 

   - [22] Hemlata Tak, Massimiliano Todisco, Xin Wang, Jee-weon Jung, Junichi Yamagishi, and Nicholas Evans, “Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation,” in _Proc. Odyssey_ , 2022. 

   - [23] Andreas Nautsch, Xin Wang, Nicholas Evans, Tomi H Kinnunen, Ville Vestman, Massimiliano Todisco, H´ector Delgado, Md Sahidullah, Junichi Yamagishi, and Kong Aik Lee, “Asvspoof 2019: spoofing countermeasures for the detection of synthesized, converted and replayed speech,” _IEEE Transactions on Biometrics, Behavior, and Identity Science_ , vol. 3, no. 2, pp. 252–265, 2021. 

   - [24] Evelina Bakhturina, Vitaly Lavrukhin, Boris Ginsburg, and Yang Zhang, “Hi-Fi multi-speaker english TTS dataset,” _arXiv preprint arXiv:2104.01497_ , 2021. 

   - [25] Rongjie Huang, Feiyang Chen, Yi Ren, Jinglin Liu, Chenye Cui, and Zhou Zhao, “Multi-singer: Fast multi-singer singing voice vocoder with a large-scale corpus,” in _Proc. ACM MM_ , 2021, pp. 3945–3954. 

   - [26] Soonbeom Choi, Wonil Kim, Saebyul Park, Sangeon Yong, and Juhan Nam, “Children’s song dataset for singing voice research,” in _Proc. ISMIR_ , 2020. 

   - [27] Junya Koguchi, Shinnosuke Takamichi, and Masanori Morise, “PJS: Phoneme-balanced japanese singing-voice corpus,” in _Proc. APSIPA_ . IEEE, 2020, pp. 487–491. 

   - [28] Bochen Li, Yuxuan Wang, and Zhiyao Duan, “Audiovisual singing voice separation,” _TISMIR_ , 2021. 

   - [29] Yaroslav Ganin and Victor Lempitsky, “Unsupervised domain adaptation by backpropagation,” in _Proc. ICML_ . PMLR, 2015, pp. 1180– 1189. 

   - [30] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae, “HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis,” _Proc. NeurIPS_ , vol. 33, pp. 17022–17033, 2020. 

   - [31] Zihan Pan, Tianchi Liu, Hardik B. Sailor, and Qiongqiong Wang, “Attentive merging of hidden embeddings from pre-trained speech model for anti-spoofing detection,” in _Proc. Interspeech_ , 2024, pp. 2090– 2094. 

   - [32] Anmol Guragain, Tianchi Liu, Zihan Pan, Hardik B Sailor, and Qiongqiong Wang, “Speech foundation model ensembles for the controlled singing voice deepfake detection (CtrSVDD) challenge 2024,” in _Proc. IEEE SLT_ , 2024. 

   - [33] Qishan Zhang, Shuangbing Wen, and Tao Hu, “Audio deepfake detection with self-supervised XLS-R and SLS classifier,” in _Proc. ACM MM_ , 2024. 

   - [34] Chris Donahue, Antoine Caillon, Adam Roberts, Ethan Manilow, Philippe Esling, Andrea Agostinelli, Mauro Verzetti, Ian Simon, Olivier Pietquin, Neil Zeghidour, and Jesse Engel, “Singsong: Generating musical accompaniments from singing,” in _Proc. ICML_ , 2023. 

- [18] Ryuichi Yamamoto, Reo Yoneyama, Lester Phillip Violeta, Wen-Chin Huang, and Tomoki Toda, “A comparative study of voice conversion models with large-scale speech and singing data: The T13 systems for the singing voice conversion challenge 2023,” in _Proc. IEEE ASRU_ , 2023, pp. 1–6. 

