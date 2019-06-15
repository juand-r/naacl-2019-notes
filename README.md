# Notes on NAACL 2019

This is a summary of a selection of papers presented at NAACL 2019, including the NeuralGen workshop.

See also Sebastian Ruder's highlights: http://ruder.io/naacl2019/index.html

- [Topics](#topics)
  - :bowling: [Learning with Limited Data](#learning-with-limited-data-bowling)
      - [Few-shot learning](#few-shot-learning)
      - [Zero-shot learning](#zero-shot-learning)
      - [Huge label set](#huge-label-set)
      - [Domain adaptation](#domain-adaptation)
      - [Weak supervision, distantly-labeled data](#weak-supervision-distantly-labeled-data)
  - :speech_balloon: [Language Models](#language-models-speech_balloon)
  - :abcd: [Word representations](#word-representations-abcd)
    - [Combining or aligning embeddings](#combining-or-aligning-embeddings)
  - :busts_in_silhouette: [Discourse](#discourse-busts_in_silhouette)
  - :eyes: [Attention](#attention-eyes)
- [Tasks](#tasks)
  - :memo: [Text Generation](#text-generation-memo)
  - :mag: [Authorship Attribution, Stylometry, Fake News](#authorship-attribution-stylometry-fake-news-mag)
  - :gem: [Summarization and Simplification](#summarization-and-simplification-gem)
  - :ok::accept: [Machine Translation](#machine-translation-okaccept)
  - :paperclips: [Text Classification](#text-classification-paperclips)
  - :cityscape::boy: [Named Entity Recognition](#named-entity-recognition-cityscapeboy)
  - :repeat: [Relation extraction](#relation-extraction-repeat)
  - :key::bookmark_tabs: [Keyphrase extraction](#keyphrase-extraction-keybookmark_tabs)
  - :spider_web: [Knowledge Graph Completion](#knowledge-graph-completion-spider_web)
  - :book::question: [Reading Comprehension and Question Answering](#reading-comprehension-and-question-answering-bookquestion)
- [Applications](#applications)
  - :hospital: [Clinical and biomedical applications](#clinical-and-biomedical-applications-hospital)
  - :shield::computer: [Cybersecurity applications](#cybersecurity-applications-shieldcomputer)
  - :octopus: [Other applications](#other-applications-octopus)
- [Other](#other)
- [Keynote Lectures](#keynote-lectures)
- [NeuralGen Workshop](#neuralgen-workshop)
- [Coreference and Coherence Revisited](#coreference-and-coherence-revisited)

# Topics

### Learning with Limited Data :bowling:

##### :repeat: Structured Minimally Supervised Learning for Neural Relation Extraction. Fan Bai and Alan Ritter.

---

### Few-shot learning

#####  :spider_web: **Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks.**  Ningyu Zhang et al.

--- 

### Zero-shot learning

---

### Huge label set

#####  (Could not attend) A Submodular Feature-Aware Framework for Label Subset Selection in Extreme Classification Problems. Elham J. Barezi, Ian D. Wood, Pascale Fung and Hamid R. Rabiee

---

### Domain adaptation

#####  :boom: (Poster) **Simplified Neural Unsupervised Domain Adaptation. Timothy Miller**
  - Code: https://github.com/tmills/Neural-SCL-Domain-Adaptation

---

#####  (Poster) Curriculum Learning for Domain Adaptation in Neural Machine Translation. Xuan Zhang, Pamela Shapiro, Gaurav Kumar, Paul McNamee, Marine Carpuat and Kevin Duh

---

#####  (Poster) Non-Parametric Adaptation for Neural Machine Translation. Ankur Bapna and Orhan Firat

---

#####  :paperclips: (Could not attend) Adversarial Category Alignment Network for Cross-domain Sentiment Classification. Xiaoye Qu, Zhikang Zou, Yu Cheng, Yang Yang and Pan Zhou

---

#####  (Could not attend) Joint Learning of Pre-Trained and Random Units for Domain Adaptation in Part-of-Speech Tagging.

---

##### Using Similarity Measures to Select Pretraining Data for NER

---

### Weak supervision, distantly-labeled data

##### :cityscape::boy: Learning to Denoise Distantly-Labeled Data for Entity Typing

---

##### [:repeat: GAN Driven Semi-distant Supervision for Relation Extraction](#gan-driven-semi-distant-supervision-for-relation-extraction)

---


### Language Models :speech_balloon:

##### (Could not attend) **Knowledge-Augmented Language Model and Its Application to Unsupervised Named-Entity Recognition. Angli Liu, Jingfei Du and Veselin Stoyanov**

---

##### (Could not attend) Serial Recall Effects in Neural Language Modeling. Hassan Hajipoor, Hadi Amiri, Maseud Rahgozar and Farhad Oroumchian

##### (Poster) WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations. Mohammad Taher Pilehvar and Jose Camacho-Collados

##### (Could not attend) **Show Some Love to Your n-grams: A Bit of Progress and Stronger n-gram Language Modeling Baselines. Ehsan Shareghi, Daniela Gerz, Ivan Vulić and Anna Korhonen**

---

### Word representations :abcd:

##### Word-Node2Vec: Improving Word Embedding with Document-Level Non-Local Word Co-occurrences

Despite progress in better word representations (word embeddings), the most popular methods are split between purely local and global approaches:

- Local approaches: co-ocurrence of words in a window of fixed size (e.g., word2vec, GloVe)
- Non-local approaches: LSA, LDA

Can we combine the best of both? Words that co-occurr frequently but non-locally within documents may have semantic association that local models are not capturing.

**Idea:** convex combination of local and non-local co-ocurrence weights.

**Evaluation:** concept categorization.

**Code:** https://github.com/procheta/Word-Node2Vec


---


##### :boom: (Poster) :eyes: **Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts**

##### (Poster) A Systematic Study of Leveraging Subword Information for Learning Word Representations.

##### (Poster) SC-LSTM: Learning Task-Specific Representations in Multi-Task Learning for Sequence Labeling. Peng Lu, Ting Bai and Philippe Langlais

##### Augmenting word2vec with latent Dirichlet allocation within a clinical application. Akshay Budhkar and Frank Rudzicz
  - See in "biomedical"


---

#### Combining or aligning embeddings

##### :boom:  (Poster) **Aligning Vector-spaces with Noisy Supervised Lexicon. Noa Yehezkel Lubin, Jacob Goldberger and Yoav Goldberg**


---


### Discourse :busts_in_silhouette:

##### (Poster) Modeling Document-level Causal Structures for Event Causal Relation Identification. Lei Gao, Prafulla Kumar Choubey and Ruihong Huang

---

### Attention :eyes:

##### (Poster) Simple Attention-Based Representation Learning for Ranking Short Social Media Posts. Peng Shi, Jinfeng Rao and Jimmy Lin
##### (Poster) **Attentive Convolution: Equipping CNNs with RNN-style Attention Mechanisms. Wenpeng Yin and Hinrich Schütze**
##### (Could not attend) **Attention is not Explanation**
##### (Could not attend) **Convolutional Self-Attention Networks. Baosong Yang et al.**
##### (Could not attend) Saliency Learning: Teaching the Model Where to Pay Attention. Reza Ghaeini, Xiaoli Fern, Hamed Shahbazi and Prasad Tadepalli
##### (Poster) **Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts**
  - See in "attention"

---

## Tasks

### Text Generation :memo:

##### (Demo) compare-mt: A Tool for Holistic Comparison of Language Generation Systems

##### (Demo) fairseq: A Fast, Extensible Toolkit for Sequence Modeling

##### (Poster) Fixed That for You: Generating Contrastive Claims with Semantic Edits. Christopher Hidey and Kathy McKeown

##### (Could not attend) AudioCaps: Generating Captions for Audios in The Wild.

Examples of generated sound captions: https://audiocaps.github.io/      

##### (Could not attend) An Empirical Investigation of Global and Local Normalization for Recurrent Neural Sequence Models Using a Continuous Relaxation to Beam Search.

##### (Could not attend) **Accelerated Reinforcement Learning for Sentence Generation by Vocabulary Prediction**. Kazuma Hashimoto and Yoshimasa Tsuruoka

##### (Could not attend) Structural Neural Encoders for AMR-to-text Generation. Marco Damonte and Shay B. Cohen

##### (Poster; Could not attend) Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting

---

##### :boom: Answer-based Adversarial Training for Generating Clarification Questions. Sudha Rao and Hal Daumé III

Clarification Questions are context aware questions that ask for missing
Information. For example:

- How to configure path or set environment variables for installation?

**Goal:** generate more useful questions.

Can use context, question and answer triples: try to generate clarification
question "from scratch" (previous work just ranked a list of clarification
questions).

** Baseline**: sequence-to-sequence (maximize likelihood of context, question pairs). Given the context, it generates a question.

**Problem:** This tends to generate a lot of generic questions. Eg:

- is this made in China?

Also a common problem in dialogue generation.

**Solution:** Use RL. Input the question to an an answer generator. Then
use a utility calculator (given question, context and answer) as reward
to retrain the question generator.

But how to train the utility calculator?

- Option 1: pretrain it. "Max-utility"
- Option 2: (a bit better) use GAN ("Gan-utility"). Answer generator + utility calculator as discriminator. Question generator is generator.

Evaluations on these datasets:

- StackExchange (Rao and Daume III 2018)
- Amazon home and kitchen (McAuley and Yang 2018)

Evaluation type:

- Human rating on various scores:
  - How grammatical is it?
  - How useful is it?
  - How relevant is it?
  - Specific to the context?

Example outputs:

- Original: are these pillows firm and keep their shape
- Max-Likelihood: what is the size of the pillow?
- GAN-Utility: does this pillow come with a cover or ...

**Code available:** https://github.com/raosudha89/clarification_question_generation_pytorch

---

##### Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data

Contributions:

- copy-augmented architecture for GEC task
- pre-train with augmented data

Different than ML: since most of the words can be copied. Copy unchanged
and OOV from the source sentence. Incorporate "copy mechanism" into seq2seq
with attention.

**Problem:** not enough training data. **Solution:** pre-training.

Inspired by BERT, ful pre-training with augmented data (randomly deleting
adding, replacing, shuffling). Also tried two auxiliary tasks for multitask
learning.

Experiments:

- Most of improvements come from the copy mechanism.

- Very nice error type classification. Error breakdown by types: determiner, wrong collocation, spelling, punctuation, preposition, noun number, redundancy, verb tense, etc..

Audience Questions:

- Q: Why didn't you use BERT? BERT is everything now!  A: future work.

**Code:**  https://github.com/zhawe01/fairseq-gec

---

##### Topic-Guided Variational Auto-Encoder for Text Generation

Many applications of text generation:

- machine translation (MT)
- dialogue generation
- text generation

VAE are widely used for this. But problems:

- isotropic Gaussian prior cannot indicate the semantic structure.
- "posterior collapse" issue.

Main contribution: new topic-guided VAE (TGVAE) for text generation.

Can generate text conditioned on a given topic, and adapted to summarization.

---


### Authorship Attribution, Stylometry, Fake News :mag:

##### :boom: Generalizing Unmasking for Short Texts

**TL;DR:** Boosting is applied to the "unmasking" method for authorship verification, making the method feasible for "short" texts. It is particularly useful when the goal is high precision. 

Two related tasks:
- **Authorship attribution:** determine the authorship of one text from a limited number of authors.
- **Authorship verification:** given two texts, determine if they are written by the same author.

Authorship verification is often more realistic (when there are millions of possible authors) and in some sense more fundamental than authorship attribution.

This paper generalizes the **unmasking** approach (Koppel and Schler, ICML 2004) to "short" texts (here "short" means about 4 printed pages, rather than book-length text), with a method emphasizing precision.

**Intuition:** texts by the same author differ only in a few superficial features. So the CV accuracy will drop faster when the texts are by the same author than when they are by different authors. 

**Unmasking algorithm:**

- From each text, create non-overlapping chunks of at least 500 words each

- Use BOW features (but with only 250 most frequent words)

- Obtain a curve (features removed against CV accuracy) by iterating the following:
    - Do 10-fold CV between the two texts with a linear SVM.
    - Eliminate the most discriminating features for the model

- Train a classifier on the curves to determine authorship.


**Problem:** doesn't work for short texts because there is not enough material for chunks, and too much noise (see experiments in Sanderson and Guenter, 2006).

**Proposed solution**: Since unmasking uses BOW features anyway, we can use bagging. Create chunks by sampling without replacement to create each chunk.  Then run the unmasking algorithm and average.

**Benefits**: 

- Can achieve very high precision with low recall; (other methods result in balanced precision and recall).  False positives can lead to a wrong conviction!
- Easier to tune hyperparameters than other approaches.
- Must faster than RNN-based approach (Bagnall, 2015).

**Code and data:** https://github.com/webis-de/NAACL-19

---

##### :boom: Adversarial Training for Satire Detection: Controlling for Confounding Variables. Robert McHardy, Heike Adel and Roman Klinger

**Motivation:** Models for satire detection might be learning the characteristics of the publication source. This is bad:

- bad for generalization
- misleading (we don't want Onion detectors, we want satire detectors).

**Goal:** use adversarial training to improve the robustness of the model against confounding variable of publication source.

**Model:** Figure 1 from the paper:

![image](images/adv-satire-model2.png)

The "adversary" (publication identifier) is trying to get the model to perform badly at guessing the publication name.

**Code and data:** www.ims.uni-stuttgart.de/data/germansatire ; https://bitbucket.org/rklinger/adversarialsatire/src/master/

---

##### (Poster) Fake News Detection using Deep Markov Random Fields

### Summarization and Simplification :gem:

##### (Could not attend) SEQˆ3: Differentiable Sequence-to-Sequence-to-Sequence Autoencoder for Unsupervised Abstractive Sentence Compression.
##### (Could not attend) Abstractive Summarization of Reddit Posts with Multi-level Memory Networks. Byeongchang Kim, Hyunwoo Kim and Gunhee Kim
##### (Could not attend) Complexity-Weighted Loss and Diverse Reranking for Sentence Simplification. Reno Kriz et al.

---

### Machine Translation :ok::accept:

##### :boom:  (Poster) Lost in Machine Translation: A Method to Reduce Meaning Loss. Reuben Cohn-Gordon and Noah Goodman
##### (Poster) Understanding and Improving Hidden Representations for Neural Machine Translation.

---


### Text Classification :paperclips:

##### :boom: (Poster) **Vector of Locally-Aggregated Word Embeddings (VLAWE): A Novel Document-level Representation**

##### (Poster) **Detecting depression in social media using fine-grained emotions**

##### (Could not attend) Mitigating Uncertainty in Document Classification. Xuchao Zhang, Fanglan Chen, ChangTien Lu and Naren Ramakrishnan

##### (Could not attend) How Large a Vocabulary Does Text Classification Need? A Variational Approach to Vocabulary Selection.

##### (Could not attend) Rethinking Complex Neural Network Architectures for Document Classification. Ashutosh Adhikari, Achyudh Ram, Raphael Tang and Jimmy Lin

---

### Named Entity Recognition :cityscape::boy:

##### Pooled Contextualized Embeddings for Named Entity Recognition. Alan Akbik, Tanja Bergmann and Roland Vollgraf

##### Knowledge-Augmented Language Model and Its Application to Unsupervised Named-Entity Recognition. Angli Liu, Jingfei Du and Veselin Stoyanov
  - See in Language Models section

##### (Poster) Practical, Efficient, and Customizable Active Learning for Named Entity Recognition in the Digital Humanities. Alexander Erdman et al.

##### [Using Similarity Measures to Select Pretraining Data for NER](#using-similarity-measures-to-select-pretraining-data-for-ner)

---

### Relation extraction :repeat:

##### [GAN Driven Semi-distant Supervision for Relation Extraction](#repeat-gan-driven-semi-distant-supervision-for-relation-extraction)

##### (Could not attend) Structured Minimally Supervised Learning for Neural Relation Extraction. Fan Bai and Alan Ritter

##### (Could not attend) **Document-Level N-ary Relation Extraction with Multiscale Representation Learning**. Robin Jia, Cliff Wong and Hoifung Poon

---

### Keyphrase Extraction :key::bookmark_tabs:

##### (Could not attend) Keyphrase Generation: A Text Summarization Struggle. Erion Çano and Ondřej Bojar

---

### Knowledge Graph Completion :spider_web:

##### Graph Pattern Entity Ranking Model for Knowledge Graph Completion. Takuma Ebisu and Ryutaro Ichise

##### **Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks.**  Ningyu Zhang et al.
  - See under 'few-shot'

---

### Reading Comprehension and Question Answering :book::question:

##### (Poster) DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh and Matt Gardner

##### (Could not attend) Improving Machine Reading Comprehension with General Reading Strategies. Kai Sun, Dian Yu, Dong Yu and Claire Cardie

##### (Could not attend) **Repurposing Entailment for Multi-Hop Question Answering Tasks. Harsh Trivedi et al.**

##### (Could not attend) Understanding Dataset Design Choices for Multi-hop Reasoning. Jifan Chen and Greg Durrett


---


## Applications

### Clinical and Biomedical Applications :hospital:

##### (Poster) Biomedical Event Extraction based on Knowledge-driven-LSTM

##### (Could not attend) Multilingual prediction of Alzheimer’s disease through domain adaptation and concept-based language modelling.

##### (Could not attend) Augmenting word2vec with latent Dirichlet allocation within a clinical application. Akshay Budhkar and Frank Rudzicz

---

### Cybersecurity applications :shield::computer:

##### (Poster) Detecting Cybersecurity Events from Noisy Short Text

##### (Poster) Analyzing the Perceived Severity of Cybersecurity Threats Reported on Social Media

---

### Other applications :octopus:

##### (Poster) Text Similarity Estimation Based on Word Embeddings and Matrix Norms for Targeted Marketing.

---

### Other

#### Posters

##### (Poster) Adversarial Decomposition of Text Representation

##### (Could not attend) **Combining Sentiment Lexica with a Multi-View Variational Autoencoder**

##### (Could not attend) Frowning Frodo, Wincing Leia, and a Seriously Great Friendship: Learning to Classify Emotional Relationships of Fictional Characters

##### (Poster) PAWS: Paraphrase Adversaries from Word Scrambling

##### (Poster) **Value-based Search in Execution Space for Mapping Instructions to Programs.**

##### (Poster) **Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems**

##### (Poster) Rotational Unit of Memory: A Novel Representation Unit for RNNs with Scalable Applications

#### Oral

##### (Could not attend) Mutual Information Maximization for Simple and Accurate Part-Of-Speech Induction. Karl Stratos

##### (Could not attend) Unsupervised Latent Tree Induction with Deep Inside-Outside Recursive Auto-Encoders. Andrew Drozdov, Patrick Verga, Mohit Yadav, Mohit Iyyer and Andrew McCallum

##### (Could not attend) Locale-agnostic Universal domain Clasification Model in Spoken Language Understanding

##### (Could not attend) On the Importance of Distinguishing Word Meaning Representations: A Case Study on Reverse Dictionary Mapping. Mohammad Taher Pilehvar

##### (Could not attend) Factorising AMR generation through syntax

##### (Could not attend) Joint Detection and Location of English Puns

##### (Could not attend) **Inoculation by Fine-Tuning: A Method for Analyzing Challenge Datasets. Nelson F. Liu, Roy Schwartz and Noah A. Smith**

##### **Recursive Routing Networks: Learning to Compose Modules for Language Understanding.**

##### **Benchmarking Hierarchical Script Knowledge. Yonatan Bisk, Jan Buys, Karl Pichotta and Yejin Choi** (text generation?)

---

# Keynote Lectures

#### Keynote 1: Arvind Narayanan, "Data as a Mirror of Society: Lessons from the Emerging Science of Fairness in Machine Learning"

**Motivation**: machine learning models naturally absorb cultural stereotypes. Some examples in the news:

- Hiring:
    - Amazon scraps AI recruiting tool that showed bias against women. The bias came from the data; ML just revealed it.
    - HireVue: uses facial movements, tone, word choice and body language for hiring. Obvious biases here!
- Criminal justice and law enforcement:
    - ML for predictive policing, body cameras, deciding sentencing length. Strong racial bias.
    - UK firm WeSee: reads facial cues to detect "suspicious behavior" - to be used in subways, sports games and other big events. 

##### Exploring bias in NLP

Two good papers on the topic on bias in NLP:

- Caliskan et al Science 2017
- Bolukbasi et al., NIPS 2016 (on debiasing word embeddings)

The big question: how do we measure bias? Look at work in psychology and cognitive science; they have been doing this for a while.

One technique: Implicit Association Test to reveal people's hidden biases. The Word Embedding Association Test (WEAT) was inspired by this to look at biases of word embeddings.

- Very similar bias in pretrained Glove (web) and word2vec (Google news). Surprising! (You might think there would be less bias in news text).
- Some examples of associations:
    - between African American names and "unpleasant" words (eg. filth, murder, etc)
    - between female words (eg. grandmother) and arts; between male words and STEM fields.

Some other examples of measuring and fixing biases:

- Gender biases:
    - Rudinger et al., Gender bias in coreference resolution, NAACL 2018. Example: "The surgeon couldn't operate on {his, her} patient: it was {his, her} son."
    - Gender bias in images: smile removal style transfer tends to also change faces from female to male (Tom White, Sampling generative networks, 2016). Due to correlations in CelebA dataset. Fix: "gender-balanced smile vector"

- Racial biases:
    - Racial bias in language identification: African Americans (question: how do you determine this?) are much more likely to be mis-classified as non-English (Blodget et al., EMNLP 2016).

One solution: use different classifiers for different groups. There is increasing evidence that explicitly modeling sub-populations is helpful (sometimes necessary) for fairness.  But keep in mind:
- Ethical issue: a form of stereotyping?
- Legal issue: would this violate anti-discrimination laws? (disparate treatment in the law

##### Warnings when debiasing

Some problems:
- Word embeddings have poor stability anyway. Get very different lists of nearest neighbors on different runs. (Antoniak and Mimno, NAACL 2018)
- WEAT is highly susceptible to false positives (associations between categories which are not related)

Furthermore, how does debiasing embeddings translate to debiasing in downstream tasks (after all, what we really care about). Perhaps it should be application-specific?

Metaphor: think of AI systems as perception followed by action. Word embeddings are "perception". It might be better to address bias at parts of the system performing the "action".

##### Reverse perspective: ML biases as a lens into human culture

Examples

- Correlation between gender bias in word embeddings over time and women participation in the labor force (Garg et al., "Word embeddings quantify 100 years of gender and ethnic stereotypes" PNAS 2018)
- Does language merely reflect or also cause stereotypes? (M Lewis, G Lupyan PNAS 2019)

---

#### Keynote 2: Rada Mihalcea, "When the Computers Spot the Lie (and People Don’t)"

...

---

#### Keynote 3: Kieran Snyder (Textio) "Leaving the Lab: Building NLP Applications that Real People can Use"

Kieran Snyder is co-founder and CEO of Textio. 

##### From academia to industry

This was partly about her experience transitioning from academia to industry.

When interviewing at Microsoft, she was asked: "When is a product done?".

- Her answer: (Academic answer) it is never done!
- The interviewer's answer: when people pay for it.

In industry: no theoretical biases, use what works!

##### Ethics of the "learning loop" era

Ethics were clear when Kieran worked as an academic (gathering linguistic data) - the participants should be aware of what data is gathered, how it is used and who it is shared with.

But it is common in industry to have a "learning loop": the product creates/collects data, which improves the product, which creates more data (think Waze, Amazon, Spotify, Textio) If the end users stop using it, there is no product.

Suggested data ethics for the "learning loop" era:

- No surprises. Make it clear that you are collecting data up front.
- Use data the say you say you do, and don't sell or share it.
- Use the data for the benefit of those who have provided it.

##### Questions

Philip Resnik's question: isn't there tension between the ethical route (tell users what you collect and why) vs collecting data and figuring it out later (innovate faster)?

---

# NeuralGen Workshop

## Panel Discussion

Remark: these notes are paraphrases of the panel speakers' discussion as I understood them at the time - incomplete and perhaps not entirely correct.

Four themes: evaluation, decoding, pretraining, ethics.

### 1. Evaluation

TL;DR: Evaluation is very hard, there should be more work on it for NLG, and there is likely no silver bullet.

**Moderator:** Many metrics are listed above (BLEU, METEOR, Cider, vector extrema, word mover's deistance, HUSE, RIBES, BERTScore, "sentence mover's similarity"). But shouldn't we measure more semantic notions such as plausibility, acceptability, coherence, entailment..? 

**Yejin Choi:** Maybe the community needs to move to more model-based NLU-based evaluation.

**H. Daume III**: Even human evaluation is tricky! Single black box evaluation is likely a pipe-dream. Evaluation should be task-specific. Focus on what we really want to know. For example: when generating answers on stack exchange: does the generated answer really help?  But this is hard to measure.

**Graham Neubig:** There needs to be more work on evaluation for NLG (as there is in MT) on finding metrics that correlate better with human judgement.  Note BLEU is not as broken for MT as it is for other tasks.

**Audience question:** What about evaluation for high-stakes situtations in industry:
- medical NLG
- false claims, legal trouble
    
**Alexander Rush:** If the only thing you care about is precision, human evaluation should be fine.

### 2.  Decoding

Some types of decoding: argmax, top-k sampling, beam search, nucleus sampling, templates, planning...(I suppose by "decoding" we how the final text is produced, so non-neural methods like templates and planning are included).

**Yejin Choi:** There is no one sampling method to rule them all.

- For short text, familiar methods like beam search can be OK.
- But for longer text, newer methods like nucleus sampling work better, given the current state of technology for language models. Caveat: with future innovation incorporating semantics/pragmatics, beam search could work well again.

**Hal Daume III**: 

1. For longer text it *seems* like one must use something hierarchical.
2. Also I don't like adding noise if you don't have to (e.g. in sampling-based decoding). But it can be useful when you want diversity/creativity. If your model is perfect (theoretically) then left-to-right decoding is all you should need.

**Yejin Choi:** at intuitive level, agree with Hal on needing hierarchy. But *practically* (based on experience with huge language models), it seems like you don't need it...

For better (more robust) modeling of long structure in the long run, it is a good idea to incorporate discourse ideas (latent variables, hierarchy). (NOTE: see paper co-authored by Y. Choi's, Learning to Write)

**Alexander Rush:** Dialogue is hard! We may *never* have enough data! So we need more structure in the models.

**Yejin Choi:** Indeed. We used templates ("secrete sauce") to help win the Alexa prize. Tried RL end-to-end training initially and it wasn't working...

### 3. Pretraining

Is pre-training with giant language models a game-changer?

Several interesting points were raised here.

**Alexander Rush**: This opens up whole new areas, particularly when you don't have enough in-domain data. For example: generating basket ball games from statistics.

**Tatsunori Hashimoto:** There is a problem with using LMS we don't understand:

- robustness
- hard to trust for high-stakes situations
- could have bias we don't understand

**Hal Daume III**: If you don't know on what data the LM was trained you will never know that what you generated was not just simply memorized.

**Yejin Choi**: Actually, that is not true if you use sampling-based decoding. This is guaranteed to avoid plagiarism!

**Graham Neubig:** It is *really* hard to ground LMs to the real world (for example, the famous "unicorns" example from GPT-2).

**Yejin Choi**: There should be more work on world-modelling/fact-checking so when GROVER generates nonsense/propaganda/fake claims, it can be detected.

### 4. Ethics

**Yejin Choi**: End-goal of the GROVER work was detection, not generation, but of course had to start with generation to do detection.

**Graham Neubig**: How many people in the audience have taken ethics courses? (About 30%). We should strive for 100%!

**Hal Daume III**: This is actually not a new topic. There is a nice paper from the 80's by Joshi, Webber and Weischedel, "Preventing false inferences" (referenced in Smiley et al. 2017, "Say the right thing"). The paper distinguishes between:
- never generating false claims
- never generating something that would lead to a false inference

**Graham Neubig**: Another reference ...

**Alexander Rush**: How odd to have a discussion like this not on Twitter! Have a very strong opinion on OpenAI's decision not to release the large GPT-2 model. It bothered me.

**Yejin Choi**: We thought hard about the consequences, and wanted to make sure she didn't create a monster with Grover. Though the can of worms is already open!

There are still practical drawbacks:

- fine-grained control is currently difficult
- very slow

So it is a good time to study it NOW before the generation technology gets better.

**Hal Daume III**: There is another possible danger that comes to mind: "reverse censorship". Flood tactic to drown out dissenting voices! See for example David Graham's article in the Atlantic, "The Age of Reverse Censorship" (June 26, 2018).

---

# Coreference and Coherence Revisited

Lecture from Amir Zeldes (Georgetown University).

**Motivation**
There have been incredible gains in F-scores on coreference in the OntoNotes corpus. But there is some dissatisfaction:

- Scores in the 70's are still not trustworthy enough to be useful.
- System errors are sometimes bizarre (in ways that traditional methods were not).
- Generalizability: out-of-domain performance is often *worse* than older systems.

So it is clear that current methods are missing something. What? Let's look at cohesion and coherence. J. Renkema's definition of cohesion and coherence (from "Introduction to Discourse Studies"):

- cohesion: "connections which have their manifestation in the discourse itself", such as coreference, bridging, connectives...
- coherence: "connections which can be made by the reader" using world knowledge.


**Current (neural) systems**

The good:

- embeddings allow relating OOV items to training data.
- No need to curate KBs. Just plug in a training corpus.

The bad:

- no explicity semantic modeling means the following mistakes will be made:
    - errors that could be fixed given some knowledge of synonymy/antonymy/cardinality
    - overfitting lexical features in the data.
- rely heavily on pre-trained language models:
    - train/test discrepancy: do not account for distributions in current text (of course an issue for ML generally)
    - sensitive to changes in genre/domain


**The train/test paradigm is dead.** Huge neural networks overfit on patterns on both train and test sets, and it is unclear what is really being learned. 


**Some examples of mistakes:**

These from huggingface neural coref and Allen NLP:

- I ate [the good gum]. Mary ate [the bad gum].
- I saw [two myna birds] and a sparrow... When I approach, [the three birds] flew away.
