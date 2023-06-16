# Probing for Hyperbole in Pre-trained Language Models

_Accepted as a paper at the Student Research Workshop (SRW) at ACL 2023_

> **Abstract**: Hyperbole is a common figure of speech, which is under-explored in NLP research. In this study, we conduct edge and minimal description length (MDL) probing experiments for three pre-trained language models (PLMs) in an attempt to explore the extent to which hyperbolic information is encoded in these models. We use both word-in-context and sentence-level representations as model inputs as a basis for comparison. We also annotate 63 hyperbole sentences from the HYPO dataset according to an operational taxonomy to conduct an error analysis to explore the encoding of different hyperbole categories. Our results show that hyperbole is to a limited extent encoded in PLMs, and mostly in the final layers. They also indicate that hyperbolic information may be better encoded by the sentence-level representations, which, due to the pragmatic nature of hyperbole, may therefore provide a more accurate and informative representation in PLMs. Finally, the inter-annotator agreement for our annotations, a Cohen's Kappa of 0.339, suggest that the taxonomy categories may not be intuitive and need revision or simplification.

## Notes

This is a fork from the repository associated with the paper [Metaphors in pre-trained Language Models](https://arxiv.org/abs/2203.14139), the original of which can be found [here](https://github.com/EhsanAghazadeh/Metaphors_in_PLMs). We built on top of the existing code and adapted our experiments on hyperbole.
- The HYPO dataset is created by Troiano et Al. as part of their paper, [A Computational Exploration of Exaggeration](https://aclanthology.org/D18-1367.pdf). It can be freely distributed under the [Creative Commons License](https://creativecommons.org/licenses/by/4.0/).

## Running Probings
You can run the probings by running the following command:
```
python3 {EDGE_CODE_PATH/MDL_CODE_PATH} {MODEL_NAME} {TASK_NAME} {SEED}
```
Example:
```
python3 source_code/scripts/edge_probing.py bert-base-uncased trofi 0
python3 source_code/scripts/mdl_probing.py bert-base-uncased trofi 0
```
MODEL_NAME:
```
bert-base-uncased
roberta-base
google/electra-base-discriminator
xlm-roberta-base
```
TASK_NAME:
```
lcc
trofi
vua_verb
vua_pos
lcc_fa
lcc_es
lcc_ru
```
