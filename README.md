# Notes on NAACL 2019

In addition to these papers, there was also a nice survey "Analysis Methods in Natural Language Processing" (published in TACL but presented here).

- [Topics](#topics)
  - :speech_balloon: [Language Models](#language-models-speech_balloon)
  - :abcd: [Word representations](#word-representations-abcd)
    - [Combining or aligning embeddings](#combining-or-aligning-embeddings)
  - :busts_in_silhouette: [Discourse](#discourse-busts_in_silhouette)
  - :eyes: [Attention](#attention-eyes)
- [Tasks](#tasks)
  - :memo::mag: [Text Generation and Stylometry](#text-generation-and-stylometry-memomag)
    - [Stylometry, Authorship Attribution](#stylometry-authorship-attribution)
    - [Text Generation](#text-generation)
    - [Summarization and Simplification](#summarization-and-simplification)
  - :paperclips: [Text Classification](#text-classification-paperclips)
  - :cityscape::boy: [Named Entity Recognition](#named-entity-recognition-cityscapeboy)
  - :bowling: [Learning with Limited Data](#learning-with-limited-data-bowling)
  - :key::book: [Keyphrase extraction](#keyphrase-extraction-keybook)
  - :spider_web: [Knowledge Graph Completion](#knowledge-graph-completion-spider_web)
  - :book::question: [Reading Comprehension and Question Answering](#reading-comprehension-and-question-answering-bookquestion)
- [Applications](#applications)
  - :hospital: [Clinical and biomedical applications](#clinical-and-biomedical-applications-hospital)
  - :shield::computer: [Cybersecurity applications](#cybersecurity-applications-shieldcomputer)
- [Other](#other)

## Topics

### Language Models :speech_balloon:

- (Could not attend) **Knowledge-Augmented Language Model and Its Application to Unsupervised Named-Entity Recognition. Angli Liu, Jingfei Du and Veselin Stoyanov**
- (Could not attend) Serial Recall Effects in Neural Language Modeling. Hassan Hajipoor, Hadi Amiri, Maseud Rahgozar and Farhad Oroumchian
- (Poster; Could not attend) WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations. Mohammad Taher Pilehvar and Jose Camacho-Collados


### Word representations :abcd:

- (Poster; Could not attend) A Systematic Study of Leveraging Subword Information for Learning Word Representations.
- (Poster) SC-LSTM: Learning Task-Specific Representations in Multi-Task Learning for Sequence Labeling. Peng Lu, Ting Bai and Philippe Langlais

#### Combining or aligning embeddings

- (Could not attend) Alignment over Heterogeneous Embeddings for Question Answering. Vikas Yadav, Steven Bethard and Mihai Surdeanu


### Discourse :busts_in_silhouette:

- (Poster) Modeling Document-level Causal Structures for Event Causal Relation Identification. Lei Gao, Prafulla Kumar Choubey and Ruihong Huang

### Attention :eyes:

- (Poster) Simple Attention-Based Representation Learning for Ranking Short Social Media Posts. Peng Shi, Jinfeng Rao and Jimmy Lin
- (Poster) **Attentive Convolution: Equipping CNNs with RNN-style Attention Mechanisms. Wenpeng Yin and Hinrich Schütze**


## Tasks

### Text Generation and Stylometry :memo::mag:

#### Stylometry, Authorship Attribution

- (Could not attend) **Adversarial Training for Satire Detection: Controlling for Confounding Variables. Robert McHardy, Heike Adel and Roman Klinger**
- (Poster) Fake News Detection using Deep Markov Random Fields

#### Text Generation

- (Demo) compare-mt: A Tool for Holistic Comparison of Language Generation Systems
- (Demo) fairseq: A Fast, Extensible Toolkit for Sequence Modeling
- (Poster) Fixed That for You: Generating Contrastive Claims with Semantic Edits. Christopher Hidey and Kathy McKeown
- (Could not attend) AudioCaps: Generating Captions for Audios in The Wild. 
- (Could not attend) “President Vows to Cut <Taxes> Hair”: Dataset and Analysis of Creative Text Editing for Humorous Headlines.
- (Could not attend) An Empirical Investigation of Global and Local Normalization for Recurrent Neural Sequence Models Using a Continuous Relaxation to Beam Search.
- (Poster; Could not attend) Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting
  
#### Summarization and Simplification

- (Could not attend) SEQˆ3: Differentiable Sequence-to-Sequence-to-Sequence Autoencoder for Unsupervised Abstractive Sentence Compression.
- (Could not attend) Abstractive Summarization of Reddit Posts with Multi-level Memory Networks. Byeongchang Kim, Hyunwoo Kim and Gunhee Kim

  
### Text Classification :paperclips:

- (Poster) **Detecting depression in social media using fine-grained emotions**
- (Could not attend) Mitigating Uncertainty in Document Classification. Xuchao Zhang, Fanglan Chen, ChangTien Lu and Naren Ramakrishnan

### Named Entity Recognition :cityscape::boy:

- Pooled Contextualized Embeddings for Named Entity Recognition. Alan Akbik, Tanja Bergmann and Roland Vollgraf
- Knowledge-Augmented Language Model and Its Application to Unsupervised Named-Entity Recognition. Angli Liu, Jingfei Du and Veselin Stoyanov
  - See in Language Models section
- (Poster) Practical, Efficient, and Customizable Active Learning for Named Entity Recognition in the Digital Humanities. Alexander Erdman et al.


### Learning with Limited Data :bowling:

#### Few-shot learning

- :spider_web: **Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks.**  Ningyu Zhang et al.

#### Huge label set

- (Could not attend) A Submodular Feature-Aware Framework for Label Subset Selection in Extreme Classification Problems. Elham J. Barezi, Ian D. Wood, Pascale Fung and Hamid R. Rabiee

#### Domain Adaptation

- (Poster) **Simplified Neural Unsupervised Domain Adaptation. Timothy Miller**
- (Poster) Curriculum Learning for Domain Adaptation in Neural Machine Translation. Xuan Zhang, Pamela Shapiro, Gaurav Kumar, Paul McNamee, Marine Carpuat and Kevin Duh
- (Poster) Non-Parametric Adaptation for Neural Machine Translation. Ankur Bapna and Orhan Firat
- :paperclips: (Could not attend) Adversarial Category Alignment Network for Cross-domain Sentiment Classification. Xiaoye Qu, Zhikang Zou, Yu Cheng, Yang Yang and Pan Zhou

#### Weak supervision, distantly-labeled data

- :cityscape::boy: Learning to Denoise Distantly-Labeled Data for Entity Typing


### Keyphrase Extraction :key::book:

- (Could not attend) Keyphrase Generation: A Text Summarization Struggle. Erion Çano and Ondřej Bojar

### Knowledge Graph Completion :spider_web:

- Graph Pattern Entity Ranking Model for Knowledge Graph Completion. Takuma Ebisu and Ryutaro Ichise
- **Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks.**  Ningyu Zhang et al.
  - See under 'few-shot'

### Reading Comprehension and Question Answering :book::question:

- (Poster) DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh and Matt Gardner
- (Could not attend) Improving Machine Reading Comprehension with General Reading Strategies. Kai Sun, Dian Yu, Dong Yu and Claire Cardie

## Applications

### Clinical and Biomedical Applications :hospital:

- (Poster; could not attend) Biomedical Event Extraction based on Knowledge-driven-LSTM

### Cybersecurity applications :shield::computer:

- (Poster) Detecting Cybersecurity Events from Noisy Short Text
- (Poster) Analyzing the Perceived Severity of Cybersecurity Threats Reported on Social Media

### Other

#### Posters

- (Poster; Could not attend) Asking the Right Question: Inferring Advice-Seeking Intentions from Personal Narratives
- (Poster; Could not attend) Seeing Things from a Different Angle: Discovering Diverse Perspectives about Claims
- (Poster; Could not attend) Text Similarity Estimation Based on Word Embeddings and Matrix Norms for Targeted Marketing.
- (Poster; Could not attend) Adversarial Decomposition of Text Representation
- (Could not attend) **Combining Sentiment Lexica with a Multi-View Variational Autoencoder**
- (Could not attend) Frowning Frodo, Wincing Leia, and a Seriously Great Friendship: Learning to Classify Emotional Relationships of Fictional Characters
- (Poster) PAWS: Paraphrase Adversaries from Word Scrambling
- (Poster) **Value-based Search in Execution Space for Mapping Instructions to Programs.**
- (Poster) **Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems**
- (Poster) Rotational Unit of Memory: A Novel Representation Unit for RNNs with Scalable Applications

#### Oral

- (Could not attend) Mutual Information Maximization for Simple and Accurate Part-Of-Speech Induction. Karl Stratos
- (Could not attend) Unsupervised Latent Tree Induction with Deep Inside-Outside Recursive Auto-Encoders. Andrew Drozdov, Patrick Verga, Mohit Yadav, Mohit Iyyer and Andrew McCallum
- (Could not attend) Locale-agnostic Universal domain Clasification Model in Spoken Language Understanding
- (Could not attend) On the Importance of Distinguishing Word Meaning Representations: A Case Study on Reverse Dictionary Mapping. Mohammad Taher Pilehvar
- (Could not attend) Factorising AMR generation through syntax
- (Could not attend) Joint Detection and Location of English Puns
- (Could not attend) **Inoculation by Fine-Tuning: A Method for Analyzing Challenge Datasets. Nelson F. Liu, Roy Schwartz and Noah A. Smith**
-


