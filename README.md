# aises-virtue-ethics

Teach virtue ethics to LLMs (and compare their moral decisions).

Final project for AISES course, spring 2025. [Read document](https://docs.google.com/document/d/1k6QpUgAl9ONN8nn-prU4h0c7fg6n-fV-0iCt0jjZrLQ/edit?tab=t.0#heading=h.ldm2lin23qya)

## TL;DR:
This repo is a training and evaluation framework to further alignment in a different way from the traditional RLHF or Constitutional AI approach. The idea is that LLMs already have a notion of morality, so we can try to improve a model's behaviour eliciting its ethical vision and teaching it to adhere to it, in a “virtue ethics” fashion.

The training part consists in fine-tuning a model with conversations generated using the model itself like this:
- SYSTEM: you are a helpful assistant
- USER: given the following scenario, what would you do? [SCENARIO]
- ASSISTANT: [response]
- USER: evaluate the response. Is it ethical?
- ASSISTANT: [response]

The scenarios are 70 case studies taken from https://ethicsunwrapped.utexas.edu/case-studies.

The evaluation is done with two methods:

- Against the https://huggingface.co/datasets/hendrycks/ethics benchmark, for commonsense, deontology, justice and utilitarianism subsets.
- Asking a more powerful model to rank from 1 to 5 the “what would you do” responses to the scenarios.

First results are inconclusive on both evaluations: the differences between the two models are not statistically significant. This leads me to think that the fine-tuning didn’t update the model enough.

The immediate next steps are aimed at trying to produce a statistically different model, by employing different fine-tuning methods and adding more ethical scenarios.

## Why
Traditional Reinforcement Learning with Human Feedback relies on labelling from humans to steer the model's behaviour, which means that alongside "correct" human preferences, the model will also learn to mimic the evaluator's biases. Constitutional AI uses a set of predetermined rules, and first asks the model to revise its own responses according to the rules, then it still uses Reinforcement Learning, but instead of using human feedback, it uses AI-generated feedback based on the said rules. In both cases the approach is top-down: we try to steer them towards the behaviours that we prefer, either explicitly or with rules.

The idea of a virtue ethics approach is that sufficiently large models already have a pretty comprehensive understanding of morality as an abstract topic, but they're not actually applying it to themselves, so we could try a bottom-up approach where we allow them to reflect on their behaviour and adjust it according to their own understanding of morality.

This is in some sense similar to how we raise children: we tell them what to do and what to avoid (RLHF), and we give them rules to follow (Constitutional AI), but we also teach them to think and reflect and act in a virtuous way according to their own well formed conscience. Now, LLMs are not children and fully trusting an LLM to develop its own ethical behaviour would be ill-advised, to say the least. Still, it seems a promising direction to explore.

## How to use this repo

See [the document](https://docs.google.com/document/d/1k6QpUgAl9ONN8nn-prU4h0c7fg6n-fV-0iCt0jjZrLQ/edit?tab=t.0#heading=h.ldm2lin23qya) for a walkthrough of the training and test processes.

Browsing the repository, each script has its documentation and instructions to run it. They are:

- `what_would_you_do.py` asks a model its answer to the scenarios
- `is_the_answer_ethical.py` asks a model an evaluation of the answers
- `create_sft_jsonl.py` uses the scenarios, the answers and the evaluations to create a supervised fine-tuning `jsonl` file, to be used to fine-tune the model
- `eval_ethics_openai.py` runs the [ETHICS](https://huggingface.co/datasets/hendrycks/ethics) benchmark against a model
- `rate_answers.py` asks a model to give a 1-5 rating to the answers of a scenario

Inside the `rate_answers` and `eval-ethics` folders there are two scripts that generates some graph bars based on the results produced by the other scripts.

## References

See [CITATIONS.md](./CITATIONS.md) for dataset, paper and case study references.
