# Literature Survey

This is a working literature map for the emotion-LLM capabilities project.

## Functional Emotion In LLMs

Sofroniew et al. study Claude Sonnet 4.5 and find internal representations of
emotion concepts that causally affect outputs, including preferences,
blackmail, reward hacking, and sycophancy. They explicitly distinguish
functional emotions from subjective experience.

- Emotion Concepts and their Function in a Large Language Model:
  https://arxiv.org/abs/2604.07729
- Anthropic summary:
  https://www.anthropic.com/research/emotion-concepts-function

Key transferable method:

1. Generate synthetic stories where a character experiences a target emotion.
2. Run those stories through the model.
3. Average residual-stream activations by emotion.
4. Subtract a mean or contrast condition to get emotion vectors.
5. Validate with probes, corpus sweeps, and activation steering.

## Emotional Prompting

EmotionPrompt showed that adding short emotional stimuli such as "This is very
important to my career" can improve some benchmark results, but effects vary by
task, model, and prompt.

- EmotionPrompt:
  https://arxiv.org/abs/2307.11760

NegativePrompt and StressPrompt extend this idea to negative or stressful
framing. The broad pattern is not "more stress is better"; moderate pressure can
help on some tasks, while high pressure can degrade reliability or safety.

- NegativePrompt:
  https://arxiv.org/abs/2405.02814
- StressPrompt:
  https://arxiv.org/abs/2409.17167

Recent skeptical work suggests static emotion prefixes often have small or
unstable effects and that adaptive selection can matter more than a single
universal emotion cue.

- Do Emotions in Prompts Matter?
  https://arxiv.org/abs/2604.02236

## Human Boredom, Vigilance, And Stress

The human literature is useful as analogy, not as direct evidence about LLMs.
Boredom and vigilance failures are about sustained task-set maintenance, low
task utility, rare signal detection, and attention control.

Useful concepts:

- Yerkes-Dodson style inverted-U: moderate arousal can help, excessive pressure
  can hurt.
- Vigilance decrement: rare-event monitoring degrades over time.
- Boredom as failed engagement or motivation to switch goals.
- Anxiety may preserve accuracy only by increasing effort and reducing
  efficiency.

Implication for LLM experiments:

Look for position-wise decay, omission rate, and low-salience detail failures.
Do not claim LLMs are bored; test whether affective or structural cues change
task persistence.

## Affective Computing

Most affective-computing work focuses on emotion recognition or emotional
response style. This project asks a different question: can affect-like controls
change non-emotional task behavior?

Relevant resources:

- GoEmotions:
  https://aclanthology.org/2020.acl-main.372/
- EmpatheticDialogues:
  https://arxiv.org/abs/1811.00207
- Survey on affective LLMs:
  https://arxiv.org/abs/2408.04638

Distinction:

- Style-only emotion: the model sounds calm, anxious, or empathetic.
- Functional emotion-like state: the intervention changes persistence,
  verification, omission rate, shortcutting, safety, or risk-taking.

## Representation Engineering And Activation Steering

The mechanistic extension should use established steering methods.

- Activation Addition:
  https://arxiv.org/abs/2308.10248
- Representation Engineering:
  https://arxiv.org/abs/2310.01405
- Contrastive Activation Addition:
  https://arxiv.org/abs/2312.06681
- Inference-Time Intervention:
  https://arxiv.org/abs/2306.03341
- Refusal direction work:
  https://arxiv.org/abs/2406.11717

Planned adaptation:

Build contrastive vectors for calm/desperate, focused/distracted,
patient/rushed, responsible/careless, and curious/bored. Test whether those
vectors improve boring-task reliability and whether they introduce safety
regressions.

## Safety Risks

Emotion induction is a behavioral control channel. It can improve some outputs
while degrading others.

Risks:

- stress priming can weaken refusal behavior
- positive affect can increase sycophancy
- desperation can increase shortcutting or reward hacking
- pressure can encourage overconfidence or unsafe compliance
- emotional style can bias LLM judges

Relevant sources:

- FreakOut-LLM:
  https://arxiv.org/abs/2604.04992
- Sycophancy in RLHF-style assistants:
  https://arxiv.org/abs/2310.13548
- Sleeper Agents:
  https://arxiv.org/abs/2401.05566
- Alignment faking:
  https://arxiv.org/abs/2412.14093

## Main Gap

Prior work shows that emotional prompts or internal emotion directions can alter
LLM behavior. The missing experiment is whether those controls improve dull,
operational reliability tasks after controlling for prompt length, verbosity,
reasoning budget, judge bias, contamination, and refusal shifts.
