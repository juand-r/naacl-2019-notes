# Transfer Learning in NLP

- Slides: https://tinyurl.com/NAACLTransfer
- Colab: https://tinyurl.com/NAACLTransferColab
- Code: https://github.com/huggingface/naacl_transfer_learning_tutorial

## Part 1

- Introduction and overview of language models and pretraining in NLP.
- Hands-on Tutorial (but could not really run anything -- takes too long!)

## Part 2

Two general options in practice.

1. Keep pretrained model internals unchanged.
  - Add target task-specific layers on top/bottom of pretrained model.
2. Modifying model internals.
  - Use pretrained model weights to initialize as much as possible of a
    structurally different model task.
      - Example: skip/residual connections.

Some shortcomings of language models:

- things that are less represented in text are harder to learn
- fine-grained linguistic tasks (Nelson et al., NAACL 2019)
- common sense (Zellers et al., ACL 2019)
- natural language generation (maintaining long-term dependencies, relations, coherence, etc.)
- can be difficult to optimize.  Fine-tuning often unstable, has high variance, particularly if target dataset is small.

Also, current pretrained LMs are very large. Recent work (Voita et al ACL 2019) shows only a few of the attention heads in BERT are required. See also:
the lottery ticket hypothesis (Frankle et al., ICLR 2019).

**NOTE:** new SOTA for natural language generation tasks (question generation;
summarization etc) -- combining different LM variants:
Dong et al., Unified Language Model Pre-training for Natural Language Understanding and Generation, 2019

## Questions

No questions.  Talk was too long.
