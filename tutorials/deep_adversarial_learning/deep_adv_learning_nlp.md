# Tutorial: Deep Adversarial Learning for NLP
William Wang and Sameer Singh

All slides available here: http://tiny.cc/adversarial


## Part I: Intro to Adversarial Learning, GANS

(William Wang)

### Background

- Huge growth in last couple years:
  - In 2018 there were 20 times more papers mentioning "adversarial" compared to 2016 (in ACL conferences)
  - CVPR 2018: 1/3 of accepted papers were on adversarial Learning.
- Not only GANS.
- Interdisciplinary: NLP, computer vision, computer security, game theory. Within NLP some popular sub-areas are:
  - Adversarial examples, attacks
  - Adversarial training (adding noise)
  - Adversarial language generation (huge growth, particularly with the SOTA deep learning language models)
  - Other applications, including denoising, **domain adaptation**
- Some success stories:
  - In computer vision
    - CycleGAN (Zhu et al, 2017): video of horse -> video of zebra.
    - GauGAN: turn a sketch into a photo-realistic image (Park et al., 2019)
  - In NLP: still not as successful... but some progress.
    - Adversarial examples:
      - *Alzantot et al., EMNLP 2018*. Adv. examples for NLI. Ex: replace word with
     its synonym to change the classification from entailment to contradiction.
      - *Coavoux et al., EMNLP 2018*. Privacy attack. Attacker tries to recover
    some private information contained in the text from the latent representation used by the main classifier.
    - Adversarial Training: make trained model more robust
      - Wu, Bamman, Russell (EMNLP 2017): Adv. training for relation classification. Interpretation: regularization in the feature space by adding adversarial noise to loss function. Small improvement in precision.

### GANS

Game theoretical approach: generator vs discriminator. In computer vision: usually just throw away the discriminator after training.

Very hard to train. Some difficulties (for GANS in general):

- Mode collapse: unable to produce diverse samples. Always generates the same thing!
- People often do pre-training (generator or discriminator or both) for warm start; but the original GAN papers have no advice on how to do this.
  - If you pretrain too much, could lower diversity of examples
  - But with too little pretraining, may not converge.
- Unstable adversarial training. We have two networks/learners/agents. Should we update them at the same time? The discriminator might overpower the generator. Especially hard to find the balance when dealing with advanced network architectures (self-attention, etc).

An NLP-specific difficulty:

- Cannot back-prop through the generated X (because text is made of discrete units). An example (from DR-GAN, Tran et al., CVPR 2017): given a single image of a face, can generate rotations of the face at different angles.  But cannot do this for text.
  - Possible solution: **SeqGAN**: Policy gradient for generating sequences (Yu et al., 2017). Uses simple RL to send reward signal from discriminator back to the generator.

An interesting recent paper:

- de Masson d-Autume et al., 2018 (Deepmind, but warning: not peer reviewed yet): Claims no MLE pre-trainings are needed, so can train language GANS from scratch. Uses dense rewards (more signals -> better quality results so can get away with not pre-training).

Some people are pessimistic. But do not give up on GANs for NLP!  It is unsupervised learning! Or interpreted as self-supervised learning when using dense rewards.

### Some Applications of Adversarial Learning in NLP

A list of some interesting recent papers using GANs in NLP:

- Social media (Wang et al., 2018a, Carton et al., 2018). Social media text is already noisy. (?)

- Contrastic Estimation (Cai and Wang, 2018, Bose et al, 2018)

- **Domain Adaptation** (Kim et al., 2017; Alam et al., 2018; Zou et al., 2018; Chen and Cardie, 2018; Tran and Nguyen 2018; Cao et al., 2018; Li et al., 2018b)

- Data cleaning (denoising): Elazar and Goldberg 2018; Shah et al, 2018; Ryu et al 2018; Zellers et al 2018

- Distant supervision in information extraction (Qin et al ACL 2018, Hong et al., 2018; Wang et al 2018b, Shi et al., 2018a; Bekoulis et al 2018)
  - Qin et al. ACL 2018: DSGAN: Adversarial Learning for Distant Supervision IE.

- IR; Li and Cheng 2018

- Machine translation - adapting idea from SeqGAN (Yang et al NAACL 2018; Wu et al., ACML 2018). Ask the discriminator to tell the difference between human and machine-generated translation.

- **SentiGAN** (Wang and Wan, IJCAI 2018): use **a mixture of generators** and a multi-class discriminator. So rather than detect between real and fake, detect between k real sentiment types and fake.  Can generate text with certain sentiment.

- Inverse RL: No metrics are perfect: Adversarial reward learning (Wang, Chen et al, ACL 2018)

- **AREL Storytelling Evaluation. VIST Dataset (Huang et al., 2016).**

- **KBGAN: Learning to Generate High-Quality Negative Examples** (Cai and Wang, NAACL 2018). Idea: use adversarial learning to iteratively learn better negative examples.

NOTE: 18 papers at NAACL 2019 on adv. learning.

### A beautiful paper: GANS for Dialogue Systems

Paper: **Jiwei Li et al., Adversarial learning for neural dialogue generation, 2017**.

What should rewards for good dialogue be like?  Idea goes all the way back to Turing Test.

- Ex: Q: "how old are you?"
- Answer: "I'm 25" vs " I don't know what you are talking about.". A human can way "I'm 25" is a more realistic response.  Can a computer do this?

Uses a seq2seq, but on top of that a discriminator tries to guess a score (eg. 90% human generated). More realistic response generation.

An example:
- Q: Tell me ... how long have you had this falling sickness ?
- A (vanilla seq2seq) I don't know what you are talking about (it learns its the "safest" response)
- A (mutual information): I'm not a doctor
- A (adversarial learning): A few months, I guess. (Closer to what a human would say)

Many improvements to this paper since then. In particular (huge improvement): Self-Supervised Dialog Learning Wu et al., ACL 2019. Tries to learn higher-level structures in dialog, rather than local response. Discriminator tries to tell the difference between permuted (misordered) response sequence and real response sequence.

### The future

A lot can be done by incorporating recent advances in language models (eg. GPT-2, BERT), self-supervised learning. Etc.

## Part II: Adversarial Examples

(Sameer Singh; AI2)

### Introduction

Example of a real attack: real-world attack on self-driving cars via modified stop signs (Eykholt et al., CVPR 2018).

Applications of adversarial attacks:

- Security: deploy or not? What is the worst that can happen?
- Evaluation of ML models: held-out test error no longer enough.
- Finding bugs: what kind of "adversaries" might happen naturally?
- Interpretability: what does the model care about? What does it ignore?

Challenges in NLP:

1. Change inputs by a small amount in order to change class label. But L_2 is not defined for text. What is "small"/imperceivable change to a sentence?
2. Not so clearly defined for structured prediction problems (dependency parse, NER, language generation, machine translation or summarization?).
3. Biggest technical challenge: text is discrete, so cannot use continuous optimization. How to do search?

"If you combine choices in the above three things you get a paper!"

### Choices in crafting adversaries

Design choices:

#### 1. What is a small change? What to change?

Possibilities:

  - **Characters**: things that change few characters.  **Pros**: easy to miss, easier to search over (character set vs vocabulary set). **Cons**: gibberish, nonsensical words, not useful for interpretability.
    - Ex: "I love movies". -> one-hot representation of each character (**x**).
      Change 'o' (in 'love') to 'i' (to get **x'**). Edit distance: flip,
      insert, delete (see hot-flip papers - they look at the nearest neighbors of the modified words (eg. 'taln' instead of 'talk', and they are of =
      course completely different.
  - **Words**. **Pros**: always from vocabulary; often easy to miss (eg. appropriate synonyms). **Cons:** pretty easy to get ungrammatical changes and meaning changes.
    - Ex: "I like this movie".  Replace "like". But with what?
      - like -> lamp.  Will cause havoc but it does not make any sense.
      - Use idea of word2vec to pick a new word (eg. like -> really). But still
        ungrammatical sometimes.
      - Add a POS constraint to the above idea (eg. like -> eat). But may
        make no sense.
      - Language Model (eg. BERT). But could produce like -> hate and still
        completely change the meaning.
  - **Phrase/sentence** (more recent): **Pros**: most natural/human-like. Test long distance-effects. **Cons:** very hard to guarantee quality (trying to solve text generation); larger space to search.
    - Popular method: **paragraphsing via backtranslation**. **x** and **x'**
      should mean the same thing ("semantically equivalent adversaries"). See:
      Ribeiro et al ACL 2018.
      - Ex: want S("This is a good movie","That is a good movie") to have high
        score.
        But S("This is a good movie","Dogs like cats") should be low.
        - Backtranslation: translate "This is a good movie" to multiple languages.
        Then use back-translators to score candidates.
    - **Sentence embeddings**. Deep representations are supposed to encode meaning in vectors. Use L_2 distance (Zhao et al ICLR 2018) in the encoder space.

#### 2. Search: How do we find the attack?

A spectrum:
- Black box: only have access to the predictions (with unlimited queries).
  - Create x' and test whether model misbehaves.
- Grey box (like black box, but now have probabilities rather than only predictions)
  - Create x' and test whether general direction is correct.
- White box: full access to the model (can compute gradients)
  - Use the gradient to craft x'.

Types of search:

- For White Box: **Gradient-based search**. Compute gradient at embedding   
  layer (because we have the model), and step in that
  direction. But the new embedding probably doesn't correspond to anything.
  Find nearest neighbor that corresponds to {word, sentence, character}.
  Repeat if necessary. Could be greedy, or use beam search.
- For Grey Box: **Sampling**. Generate local perturbations, and select the
  ones that really mess up the classifier. Repeat with these new ones.
  Can use beam search or genetic algorithms with this idea.
- For Black Box: **Enumeration (Trial/Error)**: Make some perturbations.
  See if they work. Optional: pick the best one.


#### 3. Effect: what does it mean to misbehave?

For classification:

- Untargeted attack: any other class.
- Targeted: a specific other class.

Other tasks:

- MT: eg. "Don't attack me" -> "No me ataques".
- NER: change sequence label.

Possible metrics:

-maximize the loss on the example (eg. perplexity/log-loss of the prediction)
- property-based. Test whether a property holds (eg. a certain word or classifier of words is not generated.)


#### 4. Evaluation. Are the attacks "good"?

How to measure them?

- Attack success rate.
- Are the changes perceivable? (Human evaluation)
  - Would it have the same label?
  - **Does it look natural?**
  - Does it mean the same thing?
- Do they help improve the model?
  - Accuracy after data augmentation.
- Look at actual examples.

### A selection of papers

- Noise breaks machine translation (Belinkov, Bisk, ICLR 2018)
    - Change: random character based
    - Search: passive (add and test)
    - Task: MT.
- Hotflip (Ebrahimi et al, ACL 2018, COLING 2018)
    - Change: character-based (extended to words)
    - Search: gradient-based with beam search
    - Tasks: MT, topic classification, sentiment
    - Examples:
      - Topic classification: eg. change "mood" to "moo"
      changed the label from Word to Sci/Tech )
      - MT: German to English. one character turned "psychotherapist" into
     "psychopath"
- Search Using Genetic Algorithms (Alzantot et. al. EMNLP 2018):
  - Change: word based, with language model score
  - Search: GA
  - Tasks: Textual Entailment, Sentiment Analysis.
  - Issue: textual entailment relies too much on lexical overlap
- Natural Adversaries (Zhao et al, ICLR 2018):
  - Change: sentence, with GAN embedding
  - Search: stochastic search (assuming black box access)
  - Tasks: images, Entailment, MT.
  - Advantages:
    - black box!
    - only uses a GAN, so can apply to both images and text.
- Semantic Adversaries (Zhao et al, ICLR 2018). Semantically-equivalent adversary. Also semantically
  Equivalent Adversarial Rules (SEARs): rules that can be used to generated
  many of the examples. Ex. WP VBS -> WP's. What has been cut -> What's been cut.  Ex. What NOUN -> Which NOUN. Ex. What VERB -> And What VERB. Ex:
  film -> movie changes positive to negative sentiment.
  - Change: sentence via backgranslation
  - Search: Enumeration
  - Tasks: Visual Question Answering, SQuAD, Sentiment Analysis.
- Adding a sentence (AddSent).
  - Change: add a sentence.
  - Search: domain knowledge, stochastic search.
  - Tasks: question answering.

### Some more loosely related work

- **CRIAGE: Adversaries for Graph Embeddings**
  - Which link should we add/remove out of a million possible links, so
    that when the model is retrained, it makes a different prediction.
- **"Should not change"** Ribeiro et al, AAAI 2018. How do dialogue systems behave when the inputs are perturbed in specific ways. Should not change: adversarial examples like paraphrase, etc. Should change ("overstability"): eg. add negation, add antonym, randomize inputs.
Anchors: identify the conditions under which the classifier has the same prediction.
  - Ex: Image of woman with banana mustache. Q: What is the mustache made of? A: banana
    - What is the bed made of? A: banana
    - What is the man made of? A: banana
- **Overstability: input reduction** - remove as much of the input as possible without changing the prediction! (Feng et al, EMNLP 2018).
  - Ex: picture with yellow flower.
    - Q: what color is the flower? A: yellow.
    - Q: flower? A: yellow.

### The future

- More realistic threat models (give even less access to model/data, eg.
  Not infinite queries)
- Defenses and fixes:
  - Spell-check based filtering
  - Attack recognition (Pruthi et al. ACL 2019)
  - Data augmentation
  - Novel losses (Zhang, Liang AISTATS 2019)
- **Adversarial attacks beyond the sentences**.
  - Paragraphs, Documents?
  - Semantic equivalency -> coherency across sentences

## Questions

- Work in identifying fake reviews. A growing problem. But "fake" is a spectrum. Who are you fooling? A human, a machine?
- A possible danger: BERT poisoning.  If you download a language model, there is no guarantee it hasn't been tampered with.

## Miscelaneous

See this paper: **Adversarial Examples Are Not Bugs, They Are Features**
  Misalignment between human-specified 'robustness' and the geometry of the data.
